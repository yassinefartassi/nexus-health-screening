"""
Real-time face detection and ROI tracking for rPPG.

Primary backend: MediaPipe's FaceLandmarker (Tasks API, 478-point face mesh).
For each frame, it extracts three skin ROIs (forehead, left cheek, right
cheek) from stable facial landmarks and returns their combined mean R/G/B --
the raw per-frame trace that src/inference/rppg_infer.py buffers and feeds
to CHROM/POS.

Fallback backend: OpenCV Haar cascade (cv2.CascadeClassifier), used
automatically whenever `import mediapipe` fails -- e.g. on a Raspberry Pi 5
running Python 3.13 on ARM64, where MediaPipe currently ships no wheel.
There is no manual flag: the import is attempted once at module load, and
FaceROITracker picks whichever backend is available. Both backends expose
the exact same FaceROIResult contract (face_found, mean_rgb, overlay_bgr),
so rppg_infer.py and app.py work unchanged regardless of which one is active.

MediaPipe ROI placement is landmark-relative (not just a face bounding box),
using these anchor points from the 478-point mesh:
  10  = hairline midpoint (forehead top)     9   = glabella (forehead bottom)
  152 = chin tip                             4   = nose tip
  234 / 454 = left/right face edge (ear level)
  33  / 263 = left/right eye outer corner
  61  / 291 = left/right mouth corner

The Haar fallback only gets a face bounding box (no landmarks), so its ROIs
are fixed proportions of that box instead -- a coarser approximation of the
same forehead/cheek regions, not the whole face.
"""
from __future__ import annotations

import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "mediapipe", "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

CASCADE_PATH = os.path.join(MODELS_DIR, "haarcascades", "haarcascade_frontalface_default.xml")
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"

# MediaPipe landmark indices, see module docstring.
_TOP, _CHIN, _LEFT, _RIGHT = 10, 152, 234, 454
_GLABELLA, _NOSE_TIP = 9, 4
_EYE_L, _EYE_R = 33, 263
_MOUTH_L, _MOUTH_R = 61, 291

_BOX_COLORS = {"forehead": (0, 255, 0), "left_cheek": (255, 0, 0), "right_cheek": (0, 0, 255)}


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


def _ensure_local_file(path: str, url: str, label: str) -> str:
    """Download `url` to `path` if it's not already present locally
    (one-time). Shared by the MediaPipe model bundle and the Haar cascade --
    both are small, static, license-clear files worth vendoring in `models/`
    so a deployed device never needs network access at runtime, only the
    first time this dev machine sets the repo up."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[face_roi] Downloading {label} to {path} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {label} from {url}.\n"
            f"Download it manually and place it at {path} (needed once; it's "
            f"then committed/deployed alongside models/)."
        ) from exc
    return path


@dataclass
class FaceROIResult:
    face_found: bool
    mean_rgb: Optional[Tuple[float, float, float]]
    overlay_bgr: np.ndarray


class _FaceROIBackend(ABC):
    """Common interface both backends implement; FaceROITracker just
    delegates to whichever one was selected at import time."""

    @abstractmethod
    def process_frame(self, frame_bgr: np.ndarray) -> FaceROIResult:
        ...

    def close(self) -> None:
        pass


class _MediaPipeFaceROI(_FaceROIBackend):
    """Wraps MediaPipe FaceLandmarker to extract a mean-RGB skin trace per
    frame from three landmark-relative ROIs."""

    def __init__(self, model_path: str = MODEL_PATH, min_detection_confidence: float = 0.5):
        model_path = _ensure_local_file(model_path, MODEL_URL, "the MediaPipe face landmarker model")
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

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


class _HaarFaceROI(_FaceROIBackend):
    """OpenCV Haar-cascade fallback, used when MediaPipe isn't installed.

    Haar cascades only give a face bounding box, not landmarks, so the
    forehead/cheek ROIs here are fixed proportions of that box -- a coarser
    stand-in for the MediaPipe landmark-relative regions above, but a stable
    skin sub-region (not the whole face box, which would include eyes/mouth
    and dilute the pulse signal)."""

    def __init__(self, cascade_path: str = CASCADE_PATH, min_detection_confidence: float = 0.5):
        cascade_path = _ensure_local_file(cascade_path, CASCADE_URL, "the OpenCV Haar face cascade")
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path} (file present but unreadable).")
        # Haar's detectMultiScale has no direct "confidence" knob; approximate
        # it via minNeighbors (higher = stricter, fewer false positives).
        self._min_neighbors = 3 if min_detection_confidence < 0.5 else 5

    @staticmethod
    def _roi_boxes(x: int, y: int, w: int, h: int) -> dict:
        """Fixed proportions of the face box, tuned to land on forehead
        skin above the eyebrows and cheeks below the eyes / beside the nose,
        avoiding hairline, eyes, and mouth."""
        fx1, fx2 = x + int(0.22 * w), x + int(0.78 * w)
        fy1, fy2 = y + int(0.08 * h), y + int(0.24 * h)

        cy1, cy2 = y + int(0.45 * h), y + int(0.65 * h)
        lc_x1, lc_x2 = x + int(0.10 * w), x + int(0.35 * w)
        rc_x1, rc_x2 = x + int(0.65 * w), x + int(0.90 * w)

        return {
            "forehead": (fx1, fy1, fx2, fy2),
            "left_cheek": (lc_x1, cy1, lc_x2, cy2),
            "right_cheek": (rc_x1, cy1, rc_x2, cy2),
        }

    def process_frame(self, frame_bgr: np.ndarray) -> FaceROIResult:
        h_frame, w_frame = frame_bgr.shape[:2]
        gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        faces = self._detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=self._min_neighbors, minSize=(80, 80)
        )
        overlay = frame_bgr.copy()

        if len(faces) == 0:
            return FaceROIResult(face_found=False, mean_rgb=None, overlay_bgr=overlay)

        # Largest detected face box, in case of spurious extra detections.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes = self._roi_boxes(int(x), int(y), int(w), int(h))

        samples = []
        for name, (x1, y1, x2, y2) in boxes.items():
            x1i, y1i = max(int(x1), 0), max(int(y1), 0)
            x2i, y2i = min(int(x2), w_frame), min(int(y2), h_frame)
            cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), _BOX_COLORS[name], 2)
            if x2i <= x1i or y2i <= y1i:
                continue
            patch = frame_rgb[y1i:y2i, x1i:x2i]
            samples.append(patch.reshape(-1, 3).mean(axis=0))

        if not samples:
            return FaceROIResult(face_found=False, mean_rgb=None, overlay_bgr=overlay)

        mean_rgb = tuple(np.mean(samples, axis=0).tolist())
        return FaceROIResult(face_found=True, mean_rgb=mean_rgb, overlay_bgr=overlay)


class FaceROITracker:
    """Extracts a mean-RGB skin trace per frame for rPPG, using MediaPipe
    when it's importable and falling back to an OpenCV Haar cascade
    automatically otherwise (see module docstring) -- callers never need to
    know or care which backend is active."""

    def __init__(self, model_path: str = MODEL_PATH, min_detection_confidence: float = 0.5):
        if _MEDIAPIPE_AVAILABLE:
            self._backend: _FaceROIBackend = _MediaPipeFaceROI(model_path, min_detection_confidence)
        else:
            self._backend = _HaarFaceROI(CASCADE_PATH, min_detection_confidence)

    def process_frame(self, frame_bgr: np.ndarray) -> FaceROIResult:
        return self._backend.process_frame(frame_bgr)

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "FaceROITracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
