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
    offset = float(t_source[0] - time_axis[0])
    print(f"[WARN] {label}: shifting timebase by {offset:.3f}s to match master axis")
    return t_source - offset, offset