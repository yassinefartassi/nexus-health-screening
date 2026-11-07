"""
Camera abstraction so the inference pipeline (src/inference/, src/preprocessing/)
never has to know whether frames come from a phone IP camera, the RPi5's
rpicam/libcamera stack, or a USB webcam.

Every implementation exposes the same OpenCV-shaped contract:
    ok, frame_bgr = source.read()
so existing call sites that were written against cv2.VideoCapture (e.g. the
rPPG capture loop in app.py, or src/inference/rppg_infer.analyze_video_file)
work unchanged regardless of which CameraSource is behind them.

Selection is driven by config/env vars (see create_camera_source), not by
editing calling code -- the same app.py runs unmodified on the Ubuntu desktop
and the RPi5.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class CameraSource(ABC):
    """Common interface for any frame source. Frames are always BGR uint8,
    matching what cv2.VideoCapture.read() returns, since that's what
    src/preprocessing/face_roi.py's FaceROITracker.process_frame expects."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns (ok, frame_bgr). ok is False on failure/end-of-stream."""

    @abstractmethod
    def isOpened(self) -> bool:
        ...

    @abstractmethod
    def release(self) -> None:
        ...

    def __enter__(self) -> "CameraSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class WebcamSource(CameraSource):
    """USB webcam (or any local video device) via OpenCV, by device index."""

    def __init__(self, index: int = 0):
        import cv2

        self._cap = cv2.VideoCapture(index)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self._cap.read()

    def isOpened(self) -> bool:
        return bool(self._cap.isOpened())

    def release(self) -> None:
        self._cap.release()


class IPCameraSource(CameraSource):
    """Phone or network camera reachable over MJPEG (http://.../video) or
    RTSP (rtsp://...). OpenCV's FFMPEG backend handles both transparently."""

    def __init__(self, url: str):
        import cv2

        self._url = url
        self._cap = cv2.VideoCapture(url)
        # Keep the read buffer shallow so rPPG timing isn't reading stale,
        # queued-up frames from a laggy Wi-Fi link.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self._cap.read()

    def isOpened(self) -> bool:
        return bool(self._cap.isOpened())

    def release(self) -> None:
        self._cap.release()


class RpicamSource(CameraSource):
    """RPi5 camera module via picamera2 (libcamera). Only importable on the
    RPi5 itself -- the import is lazy so this module still loads fine on the
    Ubuntu desktop where picamera2 isn't installed."""

    def __init__(self, size: Tuple[int, int] = (640, 480)):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise ImportError(
                "picamera2 is not installed. On RPi5 OS: "
                "'sudo apt install -y python3-picamera2', and create your venv "
                "with '--system-site-packages' so it can see the apt-installed "
                "package (see requirements-rpi5.txt)."
            ) from exc

        import cv2

        self._cv2 = cv2
        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration(
            main={"size": size, "format": "RGB888"}
        )
        self._picam2.configure(config)
        self._picam2.start()
        self._opened = True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            return False, None
        try:
            frame_rgb = self._picam2.capture_array()
        except Exception:
            return False, None
        # picamera2's "RGB888" format is actually ordered BGR in memory on
        # most builds; be explicit either way by round-tripping through cv2's
        # RGB->BGR swap so downstream code always gets true BGR like OpenCV.
        frame_bgr = self._cv2.cvtColor(frame_rgb, self._cv2.COLOR_RGB2BGR)
        return True, frame_bgr

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        if self._opened:
            self._picam2.stop()
            self._opened = False


def _is_raspberry_pi() -> bool:
    """Best-effort platform check used to pick a sensible default camera
    source when NEXUS_CAMERA_SOURCE isn't set explicitly."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "raspberry pi" in f.read().lower()
    except OSError:
        return False


def create_camera_source(
    source: Optional[str] = None,
    *,
    index: Optional[int] = None,
    url: Optional[str] = None,
) -> CameraSource:
    """Factory: picks a CameraSource from an explicit argument, then
    NEXUS_CAMERA_SOURCE/NEXUS_CAMERA_INDEX/NEXUS_CAMERA_URL env vars, then a
    platform-appropriate default (rpicam on a detected RPi, webcam otherwise).

    source: "webcam" | "ip" | "rpicam"
    """
    source = (source or os.environ.get("NEXUS_CAMERA_SOURCE") or "").strip().lower()
    if not source:
        source = "rpicam" if _is_raspberry_pi() else "webcam"

    if source == "webcam":
        idx = index if index is not None else int(os.environ.get("NEXUS_CAMERA_INDEX", "0"))
        return WebcamSource(index=idx)

    if source == "ip":
        cam_url = url or os.environ.get("NEXUS_CAMERA_URL")
        if not cam_url:
            raise ValueError(
                "IP camera source selected but no URL given. Set NEXUS_CAMERA_URL "
                "(e.g. 'http://192.168.1.50:8080/video' or 'rtsp://192.168.1.50:554/stream')."
            )
        return IPCameraSource(url=cam_url)

    if source == "rpicam":
        return RpicamSource()

    raise ValueError(f"Unknown camera source '{source}' (expected webcam | ip | rpicam)")
