"""Reusable trace-processing and trace-plotting helpers.

This module collects the signal-conditioning logic that is shared across
analysis-specific plotting modules (oscillation, ETA, future plots).
Every function operates on plain NumPy arrays — no analysis-specific
knowledge is embedded here.

Public API
----------
smooth_savgol          Savitzky-Golay smoothing (NaN-safe).
prepare_sparse_trace   Interpolate-and-gap-fill for sparsely-sampled traces
                       (e.g. treadmill speed after merge_asof alignment).
smooth_dense           Median-filter + Savitzky-Golay for dense traces.
remove_outliers_iqr    IQR-based outlier removal.
time_mask              Boolean mask + index slice for a time window.
plot_trace             Draw a single trace on an Axes with standard styling.
"""
from __future__ import annotations

from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


# ── Defaults ───────────────────────────────────────────────────────────────
# Shared defaults for overview and treadmill trace rendering.

GAP_THRESHOLD_S: float = 0.5
"""Gaps longer than this are *recording gaps*, not just irregular samples."""

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

    Use this **after** :func:`prepare_sparse_trace` has already filled gaps.

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


# ── Sparse-trace processing ───────────────────────────────────────────────

def remap_previous_sample(
    reference_time: np.ndarray,
    source_time: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    """Remap source values onto a reference timebase via previous-sample hold.

    Previous-sample hold mapping rule:
    ``idx = searchsorted(t_src, t_ref, side='right') - 1`` (clipped).
    """
    t_ref = np.asarray(reference_time, dtype=float)
    t_src = np.asarray(source_time, dtype=float)
    v_src = np.asarray(source_values, dtype=float)

    if t_ref.size == 0:
        return np.asarray([], dtype=float)
    if t_src.size == 0 or v_src.size == 0:
        return np.zeros_like(t_ref, dtype=float)

    n = min(t_src.size, v_src.size)
    t_src = t_src[:n]
    v_src = v_src[:n]

    valid = np.isfinite(t_src) & np.isfinite(v_src)
    if not np.any(valid):
        return np.zeros_like(t_ref, dtype=float)
    t_src = t_src[valid]
    v_src = v_src[valid]

    order = np.argsort(t_src)
    t_src = t_src[order]
    v_src = v_src[order]

    _, unique_idx = np.unique(t_src, return_index=True)
    t_src = t_src[unique_idx]
    v_src = v_src[unique_idx]

    idx = np.searchsorted(t_src, t_ref, side="right") - 1
    idx = np.clip(idx, 0, len(v_src) - 1)
    return v_src[idx]

def prepare_sparse_trace(
    aligned_time: np.ndarray,
    values: np.ndarray,
    *,
    gap_threshold_s: float | None = GAP_THRESHOLD_S,
    fill_value: float = 0.0,
    smooth: bool = True,
    method: Literal["step_previous", "linear"] = "step_previous",
    median_size: int = MEDIAN_FILTER_SIZE,
    savgol_window: int = SAVGOL_WINDOW,
    savgol_polyorder: int = SAVGOL_POLYORDER,
) -> np.ndarray:
    """Condition a sparse / irregularly-sampled trace for continuous plotting.

    After ``merge_asof`` alignment, sources with a lower sampling rate
    (e.g. treadmill) end up with NaN wherever the tolerance was exceeded.
    This function:

     1. Maps valid (non-NaN) samples onto the full aligned time grid.
         - ``method='step_previous'`` (default): previous-sample hold via
            ``searchsorted(..., side='right') - 1``.
         - ``method='linear'``: linear interpolation via ``np.interp``.
    2. **Gap-fills** — any time point whose nearest original sample is
       farther than *gap_threshold_s* is set to *fill_value* (default 0).
       These are genuine recording gaps, not just jitter.
    3. Optionally **smooths** the result with
       :func:`smooth_dense` (median + savgol).

    Parameters
    ----------
    aligned_time : array
        The (reference-source) time base the signal was merged onto.
    values : array
        Signal values, same length as *aligned_time*.  May contain NaN at
        unmatched samples.
    gap_threshold_s : float or None
        Maximum distance (seconds) to the nearest valid sample before a
        point is considered *in a gap*. ``None`` disables gap-filling.
    fill_value : float
        Value to assign inside gaps (0 is typical for speed).
    smooth : bool
        Whether to apply :func:`smooth_dense` after gap filling.
    method : {"step_previous", "linear"}
        Mapping strategy from sparse samples to the aligned grid.
        ``"step_previous"`` uses previous-sample hold behavior.
    median_size, savgol_window, savgol_polyorder
        Forwarded to :func:`smooth_dense`.

    Returns
    -------
    np.ndarray
        Cleaned, continuous signal ready for plotting.
    """
    aligned_time = np.asarray(aligned_time, dtype=float)
    values = np.asarray(values, dtype=float).copy()
    nan_m = np.isnan(values)

    if not nan_m.any():
        return smooth_dense(values, median_size=median_size, window=savgol_window, polyorder=savgol_polyorder) if smooth else values
    if nan_m.all():
        return np.full_like(values, fill_value)

    # Indices and times of actual (non-NaN) samples
    valid_idx = np.flatnonzero(~nan_m)
    t_valid = aligned_time[valid_idx]
    s_valid = values[valid_idx]

    # Sort and deduplicate valid times for auxiliary traces.
    order = np.argsort(t_valid)
    t_valid = t_valid[order]
    s_valid = s_valid[order]
    if len(t_valid):
        _, unique_idx = np.unique(t_valid, return_index=True)
        t_valid = t_valid[unique_idx]
        s_valid = s_valid[unique_idx]

    if method == "linear":
        interp = np.interp(aligned_time, t_valid, s_valid)
    elif method == "step_previous":
        # idx = searchsorted(t_src, t_ref, side="right") - 1, clipped
        idx = np.searchsorted(t_valid, aligned_time, side="right") - 1
        idx = np.clip(idx, 0, len(s_valid) - 1)
        interp = s_valid[idx]
    else:
        raise ValueError("method must be 'step_previous' or 'linear'")

    if gap_threshold_s is not None:
        # Zero-out (or fill) points that fall inside recording gaps
        # Use searchsorted for O(n log n) instead of O(n²)
        insert_pos = np.searchsorted(t_valid, aligned_time)
        insert_pos = np.clip(insert_pos, 0, len(t_valid) - 1)

        # Distance to nearest valid sample (check both neighbours)
        dist_right = np.abs(t_valid[insert_pos] - aligned_time)
        dist_left = np.abs(
            t_valid[np.clip(insert_pos - 1, 0, len(t_valid) - 1)] - aligned_time
        )
        nearest_dist = np.minimum(dist_right, dist_left)
        interp[nearest_dist > gap_threshold_s] = fill_value

    if smooth:
        interp = smooth_dense(
            interp,
            median_size=median_size,
            window=savgol_window,
            polyorder=savgol_polyorder,
        )
    return interp


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


# ── Time windowing ─────────────────────────────────────────────────────────

def time_mask(
    t: np.ndarray,
    window: Tuple[float, float] | None,
) -> Tuple[np.ndarray | None, slice | None]:
    """Boolean mask and contiguous index slice for a time window.

    Parameters
    ----------
    t : array
        Monotonic time vector.
    window : (lo, hi) or None
        Time bounds.  ``None`` selects the full array.

    Returns
    -------
    mask : np.ndarray[bool] or None
    sl : slice or None
        ``None`` if *t* is empty or window selects nothing.
    """
    if t is None or len(t) == 0:
        return None, None
    if window is None:
        return np.ones(len(t), dtype=bool), slice(None)
    m = (t >= window[0]) & (t <= window[1])
    idx = np.where(m)[0]
    if len(idx) == 0:
        return None, None
    return m, slice(idx[0], idx[-1] + 1)


# ── Trace plotting ─────────────────────────────────────────────────────────

# Sensible default colours for common auxiliary signals.
TRACE_COLORS = {
    "pupil": "#EF553B",
    "speed": "#00CC96",
    "locomotion": "#00CC96",
    "dff": "#1f77b4",
}


def plot_trace(
    ax: plt.Axes,
    time: np.ndarray,
    values: np.ndarray,
    *,
    window: Tuple[float, float] | None = None,
    color: str | None = None,
    label: str | None = None,
    ylabel: str | None = None,
    lw: float = 1.4,
    alpha: float = 0.9,
    fontsize: int = 9,
) -> plt.Axes:
    """Draw a single trace on *ax*, clipping to an optional time window.

    Parameters
    ----------
    ax : matplotlib Axes
    time : array
        Time vector.
    values : array
        Signal values, same length as *time*.
    window : (lo, hi), optional
        Restrict the visible portion.
    color, label, ylabel : str, optional
        Styling.  *color* defaults to matplotlib's cycle.
    lw, alpha : float
        Line width and opacity.
    fontsize : int
        Axis label font size.

    Returns
    -------
    ax : plt.Axes
        The same Axes (for chaining).
    """
    mask, _ = time_mask(time, window)
    if mask is None:
        return ax
    kwargs: dict = {"lw": lw, "alpha": alpha}
    if color is not None:
        kwargs["color"] = color
    if label is not None:
        kwargs["label"] = label
    ax.plot(time[mask], values[mask], **kwargs)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    return ax


# ── Config-driven helpers ──────────────────────────────────────────────────

def remap_to_timebase(
    reference_time: np.ndarray,
    source_time: np.ndarray,
    source_values: np.ndarray,
    *,
    method: str = "step_previous",
    gap_threshold_s: float | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Map *source_values* onto *reference_time* using the given method.

    This is a thin dispatcher that calls either :func:`remap_previous_sample`
    or ``np.interp`` depending on *method*, with optional gap-filling.

    Parameters
    ----------
    reference_time : array
        Target timebase.
    source_time, source_values : array
        Raw source arrays (same length).
    method : {"step_previous", "linear"}
        Mapping strategy.
    gap_threshold_s : float or None
        Max distance to nearest valid sample before a point is gap-filled.
    fill_value : float
        Value inside gaps.

    Returns
    -------
    np.ndarray
        Values on *reference_time*.
    """
    if method == "step_previous":
        out = remap_previous_sample(reference_time, source_time, source_values)
    elif method == "linear":
        t_ref = np.asarray(reference_time, dtype=float)
        t_src = np.asarray(source_time, dtype=float)
        v_src = np.asarray(source_values, dtype=float)
        valid = np.isfinite(t_src) & np.isfinite(v_src)
        if not np.any(valid):
            return np.zeros_like(t_ref, dtype=float)
        out = np.interp(t_ref, t_src[valid], v_src[valid])
    else:
        raise ValueError(f"method must be 'step_previous' or 'linear', got {method!r}")

    if gap_threshold_s is not None:
        t_src = np.asarray(source_time, dtype=float)
        t_ref = np.asarray(reference_time, dtype=float)
        ins = np.searchsorted(t_src, t_ref)
        ins = np.clip(ins, 0, len(t_src) - 1)
        d_r = np.abs(t_src[ins] - t_ref)
        d_l = np.abs(t_src[np.clip(ins - 1, 0, len(t_src) - 1)] - t_ref)
        out[np.minimum(d_r, d_l) > gap_threshold_s] = fill_value

    return out


def prepare_trace_styled(
    reference_time: np.ndarray,
    source_time: np.ndarray | None,
    source_values: np.ndarray,
    style: "TraceStyle",
    *,
    est_fs: float | None = None,
) -> np.ndarray:
    """Condition a trace for plotting according to a :class:`TraceStyle`.

    Performs remap → gap-fill → optional smoothing, all driven by *style*.

    Parameters
    ----------
    reference_time : array
        Target timebase for the plot.
    source_time : array or None
        Raw source timestamps.  Required when ``style.method != "raw"``.
    source_values : array
        Raw signal values.
    style : TraceStyle
        Rendering recipe (from :data:`~databench._plotting.trace_config.TRACE_STYLES`).
    est_fs : float, optional
        Estimated sampling rate (used for ``smooth_savgol_s`` conversion).

    Returns
    -------
    np.ndarray
        Conditioned values on *reference_time*.
    """
    from databench._plotting.trace_config import TraceStyle  # noqa: F811

    values = np.asarray(source_values, dtype=float)

    # 1. Remap onto reference timebase
    if style.method in ("step_previous", "linear"):
        if source_time is None:
            raise ValueError(
                f"source_time is required for method={style.method!r}"
            )
        values = remap_to_timebase(
            reference_time,
            source_time,
            values,
            method=style.method,
            gap_threshold_s=style.gap_threshold_s,
            fill_value=style.fill_value,
        )
    # method == "raw": values stay as-is

    # 2. Optional outlier removal
    if style.outlier_iqr_k is not None:
        values, _ = remove_outliers_iqr(values, k=style.outlier_iqr_k)

    # 3. Optional SG smoothing (seconds-based)
    if style.smooth_savgol_s > 0 and est_fs is not None and est_fs > 0:
        values = smooth_savgol(values, style.smooth_savgol_s, est_fs)

    # 4. Optional median + SG smoothing (sample-based)
    if style.smooth:
        values = smooth_dense(
            values,
            median_size=style.median_size,
            window=style.savgol_window,
            polyorder=style.savgol_polyorder,
        )

    return values


def plot_trace_styled(
    ax: plt.Axes,
    time: np.ndarray,
    values: np.ndarray,
    style: "TraceStyle",
    *,
    window: Tuple[float, float] | None = None,
    label: str | None = None,
    fontsize: int = 9,
) -> plt.Axes:
    """Draw a trace on *ax* using visual parameters from a :class:`TraceStyle`.

    Parameters
    ----------
    ax : matplotlib Axes
    time, values : array
        Timebase and signal (same length, already conditioned).
    style : TraceStyle
        Visual config (color, lw, alpha, drawstyle, ylabel).
    window : (lo, hi), optional
        Restrict the visible portion.
    label : str, optional
        Legend label.
    fontsize : int
        Axis label font size.

    Returns
    -------
    ax
    """
    mask, _ = time_mask(time, window)
    if mask is None:
        return ax
    kwargs: dict = {"lw": style.lw, "alpha": style.alpha, "color": style.color}
    if style.drawstyle is not None:
        kwargs["drawstyle"] = style.drawstyle
    if label is not None:
        kwargs["label"] = label
    ax.plot(time[mask], values[mask], **kwargs)
    if style.ylabel:
        ax.set_ylabel(style.ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    return ax
