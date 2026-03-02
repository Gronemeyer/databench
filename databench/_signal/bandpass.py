"""Bandpass filtering and envelope extraction."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal


def bandpass_envelope(
    x: np.ndarray,
    fs: float,
    band: Tuple[float, float],
    order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Butterworth bandpass + Hilbert envelope.

    Returns (filtered_signal, envelope).
    """
    nyq = 0.5 * fs
    sos = signal.butter(order, [band[0] / nyq, band[1] / nyq], btype="band", output="sos")
    xf = signal.sosfiltfilt(sos, x)
    env = np.abs(np.asarray(signal.hilbert(xf)))
    return xf, env


def robust_threshold(
    env: np.ndarray,
    k: float,
) -> Tuple[float, float, float]:
    """Median + k × robust_std threshold.

    Returns (threshold, median, robust_std).
    """
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med)))
    robust_std = 1.4826 * mad
    return med + k * robust_std, med, robust_std
