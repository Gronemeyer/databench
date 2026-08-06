"""Time-base remapping and sparse-trace conditioning.

Canonical location for functions that align signal data onto a reference
timebase.  These are signal-processing operations, not plotting-specific.
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from databench.signal.preproc import (
    MEDIAN_FILTER_SIZE,
    SAVGOL_WINDOW,
    SAVGOL_POLYORDER,
    smooth_dense,
)


GAP_THRESHOLD_S: float = 0.5
"""Gaps longer than this are *recording gaps*, not just irregular samples."""


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
        if t_src.size == 0:
            # No source samples at all: every reference point is in a gap.
            # Without this guard the clip below yields index -1 into an
            # empty array.  remap_previous_sample already returns
            # fill-shaped zeros for this case.
            return np.full_like(t_ref, fill_value, dtype=float)
        ins = np.searchsorted(t_src, t_ref)
        ins = np.clip(ins, 0, len(t_src) - 1)
        d_r = np.abs(t_src[ins] - t_ref)
        d_l = np.abs(t_src[np.clip(ins - 1, 0, len(t_src) - 1)] - t_ref)
        out[np.minimum(d_r, d_l) > gap_threshold_s] = fill_value

    return out


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
        idx = np.searchsorted(t_valid, aligned_time, side="right") - 1
        idx = np.clip(idx, 0, len(s_valid) - 1)
        interp = s_valid[idx]
    else:
        raise ValueError("method must be 'step_previous' or 'linear'")

    if gap_threshold_s is not None:
        insert_pos = np.searchsorted(t_valid, aligned_time)
        insert_pos = np.clip(insert_pos, 0, len(t_valid) - 1)

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
