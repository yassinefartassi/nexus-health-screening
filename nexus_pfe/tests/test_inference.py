"""
Smoke tests for all three inference modules.

Ocular/dermatology: confirm the ONNX models load and produce valid,
deterministic output shapes. rPPG: confirm the full face-ROI -> CHROM/POS ->
BPM/HRV pipeline runs end-to-end on a short synthetic video clip (a still
face image with an injected periodic color modulation simulating a pulse)
and returns a plausible BPM, without crashing.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.inference.dermatology_infer import predict_skin_condition
from src.inference.ocular_infer import predict_eye_condition


def _random_image(size: int = 224) -> Image.Image:
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestOcularInference:
    def test_returns_valid_predictions(self):
        preds = predict_eye_condition(_random_image(), top_k=5)
        assert 1 <= len(preds) <= 5
        for pred in preds:
            assert isinstance(pred.label, str) and pred.label
            assert 0.0 <= pred.confidence <= 1.0
        # top-1 confidence should be the largest
        assert preds[0].confidence == max(p.confidence for p in preds)

    def test_deterministic_on_same_input(self):
        image = _random_image()
        preds_a = predict_eye_condition(image, top_k=5)
        preds_b = predict_eye_condition(image, top_k=5)
        assert [p.label for p in preds_a] == [p.label for p in preds_b]
        assert [pytest.approx(p.confidence) for p in preds_a] == [
            pytest.approx(p.confidence) for p in preds_b
        ]


class TestDermatologyInference:
    def test_returns_valid_predictions(self):
        preds = predict_skin_condition(_random_image(), top_k=5)
        assert 1 <= len(preds) <= 5
        for pred in preds:
            assert isinstance(pred.label, str) and pred.label
            assert 0.0 <= pred.confidence <= 1.0
        assert preds[0].confidence == max(p.confidence for p in preds)

    def test_deterministic_on_same_input(self):
        image = _random_image()
        preds_a = predict_skin_condition(image, top_k=5)
        preds_b = predict_skin_condition(image, top_k=5)
        assert [p.label for p in preds_a] == [p.label for p in preds_b]


# --------------------------------------------------------------------------
# rPPG: end-to-end pipeline test on a short synthetic video clip
# --------------------------------------------------------------------------

MIN_PLAUSIBLE_BPM = 40
MAX_PLAUSIBLE_BPM = 180


def _make_synthetic_face_video(path: str, target_bpm: float = 72.0, fps: float = 30.0, duration_s: float = 12.0) -> None:
    """Writes a short video of a real face photo (skimage's bundled 'astronaut'
    test image) with a periodic per-channel brightness modulation injected at
    `target_bpm`, simulating the subtle color change a pulse produces in skin."""
    import cv2
    from skimage import data

    base_rgb = data.astronaut()
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR).astype(np.float64)
    size = base_bgr.shape[0]

    n_frames = int(fps * duration_s)
    hr_hz = target_bpm / 60.0
    amp = np.array([0.010, 0.018, 0.006])  # BGR AC amplitude, green strongest

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    rng = np.random.default_rng(0)
    try:
        for i in range(n_frames):
            t = i / fps
            modulation = 1.0 + amp * np.sin(2 * np.pi * hr_hz * t)
            frame = base_bgr * modulation[None, None, :]
            frame = frame + rng.normal(0, 0.5, frame.shape)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()


@pytest.fixture(scope="module")
def synthetic_face_video(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("rppg") / "synthetic_face.mp4")
    _make_synthetic_face_video(path)
    return path


class TestRPPGPipeline:
    def test_end_to_end_returns_plausible_bpm(self, synthetic_face_video):
        from src.inference.rppg_infer import analyze_video_file

        result = analyze_video_file(synthetic_face_video)

        assert MIN_PLAUSIBLE_BPM <= result.bpm <= MAX_PLAUSIBLE_BPM
        assert result.n_samples > 0
        assert result.fps > 0
        assert len(result.bvp_chrom) > 0
        assert len(result.bvp_pos) > 0
        assert not np.isnan(result.bvp_chrom).any()
        assert not np.isnan(result.bvp_pos).any()
        # HRV metrics should be finite when they exist, and never crash the pipeline.
        assert isinstance(result.hrv_sdnn_ms, float)
        assert isinstance(result.hrv_rmssd_ms, float)

    def test_chrom_and_pos_individually_plausible(self, synthetic_face_video):
        from src.inference.rppg_infer import analyze_video_file

        result = analyze_video_file(synthetic_face_video)

        assert MIN_PLAUSIBLE_BPM <= result.bpm_chrom <= MAX_PLAUSIBLE_BPM
        assert MIN_PLAUSIBLE_BPM <= result.bpm_pos <= MAX_PLAUSIBLE_BPM

    def test_face_roi_detects_face_and_extracts_rgb(self, synthetic_face_video):
        import cv2

        from src.preprocessing.face_roi import FaceROITracker

        cap = cv2.VideoCapture(synthetic_face_video)
        ok, frame = cap.read()
        cap.release()
        assert ok

        with FaceROITracker() as tracker:
            result = tracker.process_frame(frame)

        assert result.face_found
        assert result.mean_rgb is not None
        assert len(result.mean_rgb) == 3
        assert all(0 <= c <= 255 for c in result.mean_rgb)
