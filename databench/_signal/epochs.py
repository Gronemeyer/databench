"""Peri-event epoch extraction and interpolation."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def epoch_indices(
    t: np.ndarray,
    t0: float,
    *,
    window: Tuple[float, float] = (-2.0, 1.0),
) -> Tuple[int, int] | None:
    """Return inclusive index bounds for an epoch; None if window exceeds data range."""
    t_start = t0 + window[0]
    t_end = t0 + window[1]
    i0 = int(np.searchsorted(t, t_start, side="left"))
    i1 = int(np.searchsorted(t, t_end, side="right") - 1)
    return i0, i1


def extract_epoch_interpolated(
    t: np.ndarray,
    y: np.ndarray,
    t0: float,
    *,
    window: Tuple[float, float] = (-2.0, 1.0),
    dt: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """Interpolate an epoch onto a uniform grid.

    Returns (rel_t, y_interp) or (None, None) if data is insufficient.
    """
    valid = np.isfinite(t) & np.isfinite(y)
    t_valid = t[valid]
    y_valid = y[valid]
    if t_valid.size < 2:
        return None, None
    rel_t = np.arange(window[0], window[1] + 1e-12, dt)
    target_times = t0 + rel_t
    interpolated_values = np.interp(target_times, t_valid, y_valid)
    return rel_t, interpolated_values
