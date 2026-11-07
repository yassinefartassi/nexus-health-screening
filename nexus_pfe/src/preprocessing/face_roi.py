"""
Real-time face detection and ROI tracking for rPPG, using MediaPipe's
FaceLandmarker (Tasks API, 478-point face mesh).

For each frame, extracts three skin ROIs (forehead, left cheek, right cheek)
from stable facial landmarks and returns their combined mean R/G/B — the raw
per-frame trace that src/inference/rppg_infer.py buffers and feeds to CHROM/POS.

ROI placement is landmark-relative (not just a face bounding box), using
these anchor points from the 478-point mesh:
  10  = hairline midpoint (forehead top)     9   = glabella (forehead bottom)
  152 = chin tip                             4   = nose tip
  234 / 454 = left/right face edge (ear level)
  33  / 263 = left/right eye outer corner
  61  / 291 = left/right mouth corner
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "mediapipe", "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Landmark indices, see module docstring.
_TOP, _CHIN, _LEFT, _RIGHT = 10, 152, 234, 454
_GLABELLA, _NOSE_TIP = 9, 4
_EYE_L, _EYE_R = 33, 263
_MOUTH_L, _MOUTH_R = 61, 291

_BOX_COLORS = {"forehead": (0, 255, 0), "left_cheek": (255, 0, 0), "right_cheek": (0, 0, 255)}


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


def _ensure_model(model_path: str = MODEL_PATH) -> str:
    """Download the official MediaPipe face_landmarker model bundle if it's
    not already present locally (one-time, ~3.7MB)."""
    if os.path.exists(model_path):
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"[face_roi] Downloading MediaPipe face_landmarker model to {model_path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the MediaPipe face landmarker model from {MODEL_URL}.\n"
            f"Download it manually and place it at {model_path}."
        ) from exc
    return model_path


@dataclass
class FaceROIResult:
    face_found: bool
    mean_rgb: Optional[Tuple[float, float, float]]
    overlay_bgr: np.ndarray


class FaceROITracker:
    """Wraps MediaPipe FaceLandmarker to extract a mean-RGB skin trace per frame."""

    def __init__(self, model_path: str = MODEL_PATH, min_detection_confidence: float = 0.5):
        model_path = _ensure_model(model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceROITracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def _roi_boxes(pts: np.ndarray) -> dict:
        """Compute (x1, y1, x2, y2) pixel boxes for forehead + both cheeks
        from a (478, 2) array of pixel-space landmark coordinates."""
        top, chin = pts[_TOP], pts[_CHIN]
        left, right = pts[_LEFT], pts[_RIGHT]
        glabella, nose = pts[_GLABELLA], pts[_NOSE_TIP]
        eye_l, eye_r = pts[_EYE_L], pts[_EYE_R]
        mouth_l, mouth_r = pts[_MOUTH_L], pts[_MOUTH_R]

        fx1, fx2 = _lerp(left[0], right[0], 0.32), _lerp(left[0], right[0], 0.68)
        fy1, fy2 = _lerp(top[1], glabella[1], 0.55), _lerp(top[1], glabella[1], 1.05)

        eye_y = (eye_l[1] + eye_r[1]) / 2
        mouth_y = (mouth_l[1] + mouth_r[1]) / 2
        span = mouth_y - eye_y
        cy1, cy2 = eye_y + 0.30 * span, mouth_y - 0.10 * span

        lc_x1, lc_x2 = _lerp(nose[0], left[0], 0.90), _lerp(nose[0], left[0], 0.15)
        rc_x1, rc_x2 = _lerp(nose[0], right[0], 0.15), _lerp(nose[0], right[0], 0.90)

        return {
            "forehead": (fx1, fy1, fx2, fy2),
            "left_cheek": (lc_x1, cy1, lc_x2, cy2),
            "right_cheek": (rc_x1, cy1, rc_x2, cy2),
        }

    def process_frame(self, frame_bgr: np.ndarray) -> FaceROIResult:
        """Run face detection on a single BGR frame (as read by OpenCV) and
        return the mean RGB over the combined skin ROIs, plus an annotated
        copy of the frame for UI display."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))

        result = self._landmarker.detect(mp_image)
        overlay = frame_bgr.copy()

        if not result.face_landmarks:
            return FaceROIResult(face_found=False, mean_rgb=None, overlay_bgr=overlay)

        landmarks = result.face_landmarks[0]
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
        boxes = self._roi_boxes(pts)

        samples = []
        for name, (x1, y1, x2, y2) in boxes.items():
            x1i, y1i = max(int(x1), 0), max(int(y1), 0)
            x2i, y2i = min(int(x2), w), min(int(y2), h)
            cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), _BOX_COLORS[name], 2)
            if x2i <= x1i or y2i <= y1i:
                continue
            patch = frame_rgb[y1i:y2i, x1i:x2i]
            samples.append(patch.reshape(-1, 3).mean(axis=0))

        if not samples:
            return FaceROIResult(face_found=False, mean_rgb=None, overlay_bgr=overlay)

        mean_rgb = tuple(np.mean(samples, axis=0).tolist())
        return FaceROIResult(face_found=True, mean_rgb=mean_rgb, overlay_bgr=overlay)
