"""
CHROM and POS remote-photoplethysmography (rPPG) signal-extraction algorithms.

Adapted from rPPG-Toolbox (https://github.com/ubicomplab/rPPG-Toolbox):
  - unsupervised_methods/methods/CHROME_DEHAAN.py
  - unsupervised_methods/methods/POS_WANG.py
  - unsupervised_methods/utils.py (detrend)

rPPG-Toolbox is released under the Responsible AI License (RAIL) v1 —
review the license terms (https://github.com/ubicomplab/rPPG-Toolbox/blob/main/LICENSE) before any use
beyond academic/research purposes, and cite the papers below in any written
deliverable that uses these algorithms:

  De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based
  rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.

  Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
  Algorithmic principles of remote PPG. IEEE Transactions on Biomedical
  Engineering, 64(7), 1479-1491.

Only the two algorithm implementations are ported here (not the toolbox's
dataloader/training/eval framework). Unlike the toolbox originals, which take
raw video frames and average each frame internally, these versions take an
already-extracted (T, 3) mean-RGB trace as input, since ROI extraction and
per-frame averaging happen upstream in src/preprocessing/face_roi.py.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal, sparse


def _detrend(input_signal: np.ndarray, lambda_value: int = 100) -> np.ndarray:
    """Smoothness-priors detrending (Tarvainen et al., 2002), as used by POS_WANG."""
    signal_length = input_signal.shape[0]
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    return np.dot((H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)


def chrom_dehaan(rgb: np.ndarray, fs: float) -> np.ndarray:
    """CHROM method (De Haan & Jeanne, 2013).

    Args:
        rgb: (T, 3) array of mean R, G, B values, one row per frame.
        fs: sampling rate (frames per second) of `rgb`.

    Returns:
        (T,) BVP (blood volume pulse) signal, built from overlap-added
        Hann-windowed, bandpass-filtered segments.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6

    FN = rgb.shape[0]
    NyquistF = 0.5 * fs
    B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], "bandpass")

    WinL = math.ceil(WinSec * fs)
    if WinL % 2:
        WinL = WinL + 1
    WinL = max(WinL, 4)
    NWin = math.floor((FN - WinL // 2) / (WinL // 2))
    if NWin < 1:
        raise ValueError(
            f"rgb trace too short for CHROM ({FN} frames at {fs:.1f} fps); "
            f"need at least ~{WinSec * 2:.1f}s of data"
        )

    WinS = 0
    WinM = int(WinS + WinL // 2)
    WinE = WinS + WinL
    total_len = (WinL // 2) * (NWin + 1)
    S = np.zeros(total_len)

    for _ in range(NWin):
        RGBBase = np.mean(rgb[WinS:WinE, :], axis=0)
        RGBNorm = rgb[WinS:WinE, :] / RGBBase
        Xs = 3 * RGBNorm[:, 0] - 2 * RGBNorm[:, 1]
        Ys = 1.5 * RGBNorm[:, 0] + RGBNorm[:, 1] - 1.5 * RGBNorm[:, 2]
        Xf = signal.filtfilt(B, A, Xs, axis=0)
        Yf = signal.filtfilt(B, A, Ys)

        Alpha = np.std(Xf) / np.std(Yf)
        SWin = (Xf - Alpha * Yf) * signal.windows.hann(WinL)

        S[WinS:WinM] = S[WinS:WinM] + SWin[: WinL // 2]
        S[WinM:WinE] = SWin[WinL // 2 :]
        WinS = WinM
        WinM = WinS + WinL // 2
        WinE = WinS + WinL

    return S


def pos_wang(rgb: np.ndarray, fs: float) -> np.ndarray:
    """POS method (Wang et al., 2017), Plane-Orthogonal-to-Skin.

    Args:
        rgb: (T, 3) array of mean R, G, B values, one row per frame.
        fs: sampling rate (frames per second) of `rgb`.

    Returns:
        (T,) detrended + bandpass-filtered BVP signal.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    WinSec = 1.6
    N = rgb.shape[0]
    H = np.zeros(N)
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m < 0:
            continue
        Cn = (rgb[m:n, :] / np.mean(rgb[m:n, :], axis=0)).T  # (3, l)
        S = np.array([[0, 1, -1], [-2, 1, 1]]) @ Cn  # (2, l)
        std0, std1 = np.std(S[0]), np.std(S[1])
        alpha = std0 / std1 if std1 != 0 else 0.0
        h = S[0] + alpha * S[1]
        h = h - np.mean(h)
        H[m:n] += h

    bvp = _detrend(H, 100)
    nyquist = fs / 2.0
    b, a = signal.butter(1, [0.75 / nyquist, 3.0 / nyquist], btype="bandpass")
    return signal.filtfilt(b, a, bvp.astype(np.float64))
