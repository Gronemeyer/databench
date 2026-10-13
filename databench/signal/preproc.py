"""Signal preprocessing primitives: smoothing, detrending, outlier removal.

These operate on plain NumPy arrays and are used by both analysis and
plotting code.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from databench.signal.epoching import segments_from_mask


# ── Defaults ───────────────────────────────────────────────────────────────

MEDIAN_FILTER_SIZE: int = 3
"""Kernel size for the median pre-filter in :func:`smooth_dense`."""

SAVGOL_WINDOW: int = 5
"""Window length for the Savitzky-Golay filter in :func:`smooth_dense`."""

SAVGOL_POLYORDER: int = 2
"""Polynomial order for :func:`smooth_dense`."""

OUTLIER_IQR_K: float = 4.0
"""Default multiplier for :func:`remove_outliers_iqr`."""


# ── Smoothing ──────────────────────────────────────────────────────────────

def smooth_savgol(
    sig: np.ndarray,
    window_s: float,
    fs: float,
    polyorder: int = 3,
) -> np.ndarray:
    """Savitzky-Golay smoothing with automatic NaN handling.

    NaN values are linearly interpolated before filtering and restored
    afterward, so they appear as gaps in downstream plots.

    Parameters
    ----------
    sig : array
        1-D signal (may contain NaN).
    window_s : float
        Smoothing window in *seconds*.
    fs : float
        Sampling rate in Hz (used to convert *window_s* to samples).
    polyorder : int
        Polynomial order for the Savitzky-Golay filter.

    Returns
    -------
    np.ndarray
        Smoothed signal, same length as *sig*.
    """
    if sig is None or len(sig) == 0:
        return sig
    window_length = int(round(window_s * fs))
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(3, window_length)
    if len(sig) <= window_length:
        return sig

    nan_mask = np.isnan(sig)
    if nan_mask.all():
        return sig
    if nan_mask.any():
        filled = sig.copy()
        filled[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(~nan_mask),
            sig[~nan_mask],
        )
        out = savgol_filter(filled, window_length=window_length, polyorder=polyorder)
        out[nan_mask] = np.nan
        return out
    return savgol_filter(sig, window_length=window_length, polyorder=polyorder)


def smooth_dense(
    sig: np.ndarray,
    *,
    median_size: int = MEDIAN_FILTER_SIZE,
    window: int = SAVGOL_WINDOW,
    polyorder: int = SAVGOL_POLYORDER,
) -> np.ndarray:
    """Median-filter then Savitzky-Golay smooth a *NaN-free* array.

    Parameters
    ----------
    sig : array
        1-D signal, no NaN expected.
    median_size : int
        Kernel size for ``scipy.ndimage.median_filter``.
    window : int
        Savitzky-Golay window length (must be odd).
    polyorder : int
        Savitzky-Golay polynomial order.

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    if sig is None or len(sig) <= window:
        return sig
    out = median_filter(sig, size=median_size)
    window_length = window
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(3, window_length)
    if len(out) > window_length and window_length >= polyorder + 1:
        out = savgol_filter(out, window_length, polyorder)
    return out


def smooth_median(
    sig: np.ndarray,
    size: int = MEDIAN_FILTER_SIZE,
) -> np.ndarray:
    """Pure median filter.

    Parameters
    ----------
    sig : array
        1-D signal.
    size : int
        Kernel size.
    """
    if sig is None or len(sig) <= size:
        return sig
    return median_filter(sig, size=size)


# ── Outlier removal ────────────────────────────────────────────────────────

def remove_outliers_iqr(
    data: np.ndarray,
    k: float = OUTLIER_IQR_K,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace IQR-based outliers with NaN.

    Parameters
    ----------
    data : array
        1-D signal.
    k : float
        Multiplier on the IQR to define outlier bounds.

    Returns
    -------
    cleaned : np.ndarray
        Copy of *data* with outliers set to NaN.
    inlier_mask : np.ndarray[bool]
        ``True`` for inlier samples.
    """
    data = np.asarray(data, dtype=float).copy()
    if data.ndim > 1:
        data = data.ravel()
    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        return data, np.zeros(len(data), dtype=bool)
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    inlier_mask = (~np.isnan(data)) & (data >= lower) & (data <= upper)
    cleaned = data.copy()
    cleaned[~inlier_mask] = np.nan
    return cleaned, inlier_mask


# ── Detrending & normalization ─────────────────────────────────────────────

def zscore_1d(x: np.ndarray) -> np.ndarray:
    """``(x - mean) / sd`` ignoring NaNs; all-NaN when the trace is flat.

    The per-recording normalisation used for pupil diameter, whose
    millimetre scale differs between camera sessions.
    """
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.full_like(x, np.nan)
    return (x - np.nanmean(x)) / sd


def detrend_zscore_1d(x: np.ndarray) -> np.ndarray:
    """Linear detrend followed by z-score normalization.

    Parameters
    ----------
    x : array
        1-D signal.

    Returns
    -------
    np.ndarray
        Detrended and z-scored signal.
    """
    detrended = sp_signal.detrend(x, type="linear")
    std_dev = detrended.std(ddof=1)
    if std_dev == 0:
        std_dev = 1.0
    return (detrended - detrended.mean()) / std_dev


# ── Artefacts ──────────────────────────────────────────────────────────────

def clean_step_artifacts(
    t: np.ndarray,
    y: np.ndarray,
    *,
    max_step: float,
    bounds: Tuple[float, float] | None = None,
    max_fill_s: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flag frame-to-frame jumps and out-of-bounds values; repair short runs.

    A sample is flagged when it is non-finite, when ``|y[i] - y[i-1]|`` or
    ``|y[i+1] - y[i]|`` exceeds *max_step*, or when it lies outside
    ``bounds = (low, high)`` (``y <= low`` or ``y > high``).  Runs of flagged
    samples lasting at most *max_fill_s* are linearly interpolated from the
    samples either side; longer runs, and runs touching either end, become NaN.

    Returns ``(cleaned, flagged)``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    flagged = ~np.isfinite(y)
    if bounds is not None:
        with np.errstate(invalid="ignore"):
            flagged |= (y <= bounds[0]) | (y > bounds[1])
    with np.errstate(invalid="ignore"):
        jump = np.abs(np.diff(y)) > max_step
    flagged[1:] |= jump
    flagged[:-1] |= jump

    cleaned = np.where(flagged, np.nan, y)
    if y.size < 2:
        return cleaned, flagged
    dt = float(np.nanmedian(np.diff(t)))
    for start, end in segments_from_mask(flagged):
        if start == 0 or end == y.size - 1 or t[end] - t[start] + dt > max_fill_s:
            continue
        cleaned[start:end + 1] = np.interp(t[start:end + 1], [t[start - 1], t[end + 1]],
                                           [y[start - 1], y[end + 1]])
    return cleaned, flagged
