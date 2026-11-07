"""
rPPG inference pipeline: buffers per-frame mean-RGB samples from
src/preprocessing/face_roi.py, runs CHROM and POS
(src/inference/rppg_algorithms.py), and derives BPM + HRV metrics.

Pipeline per analysis call:
  1. Take the last ~10s of buffered (timestamp, mean_rgb) samples.
  2. Run CHROM and POS on the raw RGB trace to get two candidate BVP signals.
  3. Zero-phase Butterworth bandpass (0.7-4 Hz, i.e. 42-240 BPM) each BVP
     signal to the physiological heart-rate range.
  4. Estimate BPM per algorithm via FFT peak search; average the two into a
     single consensus BPM.
  5. Detect BVP peaks (time domain) on the POS signal to get peak-to-peak
     (RR) intervals, from which SDNN and RMSSD are computed.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

from src.inference.rppg_algorithms import chrom_dehaan, pos_wang

BUFFER_SECONDS = 10.0
MIN_SECONDS_TO_ANALYZE = 5.0
MIN_BPM = 40
MAX_BPM = 180
BANDPASS_LOW_HZ = 0.7
BANDPASS_HIGH_HZ = 4.0


@dataclass
class RPPGResult:
    bpm: float
    bpm_chrom: float
    bpm_pos: float
    hrv_sdnn_ms: float
    hrv_rmssd_ms: float
    bvp_chrom: np.ndarray
    bvp_pos: np.ndarray
    fps: float
    n_samples: int


def bandpass_filter(
    sig: np.ndarray, fps: float, low_hz: float = BANDPASS_LOW_HZ, high_hz: float = BANDPASS_HIGH_HZ
) -> np.ndarray:
    """Zero-phase Butterworth bandpass restricted to the physiological HR range."""
    nyquist = fps / 2.0
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.999)
    sos = sp_signal.butter(3, [low, high], btype="band", output="sos")
    return sp_signal.sosfiltfilt(sos, sig)


def bpm_from_bvp(bvp: np.ndarray, fps: float, min_bpm: int = MIN_BPM, max_bpm: int = MAX_BPM) -> float:
    """FFT-based BPM: Hann-windowed spectrum, peak search restricted to [min_bpm, max_bpm]."""
    windowed = bvp * np.hanning(len(bvp))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / fps)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    bpm_axis = freqs * 60.0
    valid = (bpm_axis >= min_bpm) & (bpm_axis <= max_bpm)
    if not np.any(valid):
        return 0.0
    peak_idx = np.argmax(power[valid])
    return float(bpm_axis[valid][peak_idx])


def hrv_metrics(
    bvp: np.ndarray, fps: float, min_bpm: int = MIN_BPM, max_bpm: int = MAX_BPM
) -> Tuple[float, float, np.ndarray]:
    """SDNN and RMSSD (ms) from peak-to-peak intervals detected on a filtered BVP signal."""
    min_distance = max(int(fps * 60.0 / max_bpm), 1)
    peaks, _ = sp_signal.find_peaks(bvp, distance=min_distance, prominence=np.std(bvp) * 0.3)
    if len(peaks) < 2:
        return float("nan"), float("nan"), peaks

    peak_times_ms = (peaks / fps) * 1000.0
    rr = np.diff(peak_times_ms)
    lo, hi = 60000.0 / max_bpm, 60000.0 / min_bpm
    rr = rr[(rr >= lo) & (rr <= hi)]
    if len(rr) < 2:
        return float("nan"), float("nan"), peaks

    sdnn = float(np.std(rr, ddof=1))
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) >= 3 else float("nan")
    return sdnn, rmssd, peaks


class RPPGBuffer:
    """Accumulates a rolling window of per-frame mean-RGB samples and runs
    CHROM + POS on demand once enough data has been collected."""

    def __init__(self, buffer_seconds: float = BUFFER_SECONDS):
        self.buffer_seconds = buffer_seconds
        self._timestamps: Deque[float] = deque()
        self._rgb: Deque[Tuple[float, float, float]] = deque()

    def add_sample(self, mean_rgb: Tuple[float, float, float], timestamp: Optional[float] = None) -> None:
        ts = time.time() if timestamp is None else timestamp
        self._timestamps.append(ts)
        self._rgb.append(mean_rgb)
        while self._timestamps and (ts - self._timestamps[0]) > self.buffer_seconds:
            self._timestamps.popleft()
            self._rgb.popleft()

    def fill_fraction(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        return float(min(span / self.buffer_seconds, 1.0))

    def is_ready(self, min_seconds: float = MIN_SECONDS_TO_ANALYZE) -> bool:
        if len(self._timestamps) < 2:
            return False
        return (self._timestamps[-1] - self._timestamps[0]) >= min_seconds

    def clear(self) -> None:
        self._timestamps.clear()
        self._rgb.clear()

    def analyze(self) -> RPPGResult:
        if not self.is_ready():
            raise ValueError("Not enough buffered samples yet to analyze (need >= %.0fs)" % MIN_SECONDS_TO_ANALYZE)

        timestamps = np.array(self._timestamps)
        rgb = np.array(self._rgb)
        duration = timestamps[-1] - timestamps[0]
        fps = (len(timestamps) - 1) / duration

        bvp_chrom = bandpass_filter(chrom_dehaan(rgb, fps), fps)
        bvp_pos = bandpass_filter(pos_wang(rgb, fps), fps)

        bpm_chrom = bpm_from_bvp(bvp_chrom, fps)
        bpm_pos = bpm_from_bvp(bvp_pos, fps)
        valid_bpms = [b for b in (bpm_chrom, bpm_pos) if b > 0]
        bpm = float(np.mean(valid_bpms)) if valid_bpms else 0.0

        sdnn, rmssd, _ = hrv_metrics(bvp_pos, fps)

        return RPPGResult(
            bpm=bpm,
            bpm_chrom=bpm_chrom,
            bpm_pos=bpm_pos,
            hrv_sdnn_ms=sdnn,
            hrv_rmssd_ms=rmssd,
            bvp_chrom=bvp_chrom,
            bvp_pos=bvp_pos,
            fps=fps,
            n_samples=len(timestamps),
        )


def analyze_video_file(
    path: str, buffer_seconds: float = BUFFER_SECONDS, max_seconds: Optional[float] = None
) -> RPPGResult:
    """Run the full rPPG pipeline (face ROI -> buffer -> CHROM/POS -> BPM/HRV)
    on a recorded video file. Used by tests and offline sanity-checking."""
    import cv2

    from src.preprocessing.face_roi import FaceROITracker

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {path}")
    fps_hint = cap.get(cv2.CAP_PROP_FPS) or 30.0

    buffer = RPPGBuffer(buffer_seconds=buffer_seconds)
    frame_idx = 0
    try:
        with FaceROITracker() as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                roi_result = tracker.process_frame(frame)
                if roi_result.face_found:
                    buffer.add_sample(roi_result.mean_rgb, timestamp=frame_idx / fps_hint)
                frame_idx += 1
                if max_seconds is not None and frame_idx / fps_hint > max_seconds:
                    break
    finally:
        cap.release()

    return buffer.analyze()
