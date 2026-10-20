"""Bandpass filtering and envelope extraction."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import DTypeLike
from scipy import signal


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    band: Tuple[float, float],
    order: int,
    *,
    axis: int = -1,
    dtype: DTypeLike = np.float64,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass (``sosfiltfilt``) of *x* along *axis*.

    Filtering runs in float64; the result is cast to *dtype*.  With
    *chunk_size*, the other dimensions are flattened and filtered that many
    series at a time, which bounds the float64 working copy for large arrays
    (e.g. pixels × frames).
    """
    nyq = 0.5 * fs
    sos = signal.butter(order, [band[0] / nyq, band[1] / nyq], btype="band", output="sos")
    x = np.asarray(x)
    if chunk_size is None:
        return signal.sosfiltfilt(sos, x.astype(np.float64, copy=False), axis=axis).astype(dtype, copy=False)
    series = np.moveaxis(x, axis, -1)
    flat = series.reshape(-1, series.shape[-1])
    out = np.empty(flat.shape, dtype=dtype)
    for start in range(0, len(flat), chunk_size):
        out[start:start + chunk_size] = signal.sosfiltfilt(
            sos, flat[start:start + chunk_size].astype(np.float64), axis=-1)
    return np.moveaxis(out.reshape(series.shape), -1, axis)


def bandpass_envelope(
    x: np.ndarray,
    fs: float,
    band: Tuple[float, float],
    order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Butterworth bandpass (:func:`bandpass_filter`) + Hilbert envelope.

    Returns (filtered_signal, envelope).
    """
    filtered = bandpass_filter(x, fs, band, order)
    envelope = np.abs(np.asarray(signal.hilbert(filtered)))
    return filtered, envelope


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
