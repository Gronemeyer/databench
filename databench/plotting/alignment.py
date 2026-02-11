from __future__ import annotations

from typing import Tuple

import numpy as np

TIME_OFFSET_THRESHOLD_S = 0.5


def maybe_shift_timebase(
    t_source: np.ndarray,
    time_axis: np.ndarray,
    *,
    label: str,
    threshold_s: float = TIME_OFFSET_THRESHOLD_S,
) -> Tuple[np.ndarray, float]:
    if t_source.size == 0 or time_axis.size == 0:
        return t_source, 0.0
    offset = float(t_source[0] - time_axis[0])
    if abs(offset) >= threshold_s:
        print(f"[WARN] {label}: shifting timebase by {offset:.3f}s to match master axis")
        return t_source - offset, offset
    return t_source, 0.0