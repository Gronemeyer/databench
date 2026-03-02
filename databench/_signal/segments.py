"""Contiguous-segment detection, merging, and filtering."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def segments_from_mask(mask: np.ndarray) -> list[Tuple[int, int]]:
    """Convert a boolean mask to a list of (start_idx, end_idx) contiguous segments."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    segments: list[Tuple[int, int]] = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segments.append((start, prev))
            start = i
            prev = i
    segments.append((start, prev))
    return segments


def merge_gaps(
    segments: Iterable[Tuple[int, int]],
    min_gap: int,
) -> list[Tuple[int, int]]:
    """Merge neighbouring segments when gap ≤ *min_gap* samples."""
    merged: list[Tuple[int, int]] = []
    for s, e in segments:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s - pe - 1 <= min_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def apply_min_duration(
    segments: Iterable[Tuple[int, int]],
    min_len: int,
) -> list[Tuple[int, int]]:
    """Drop segments shorter than *min_len* samples."""
    return [(s, e) for s, e in segments if (e - s + 1) >= min_len]


def burst_table(
    t: np.ndarray,
    env: np.ndarray,
    bursts: list[Tuple[int, int]],
    fs: float,
) -> "pd.DataFrame":
    """Build a DataFrame with start_s, end_s, duration_s, peak_env."""
    import pandas as pd

    return pd.DataFrame(
        {
            "start_s": [float(t[s]) for s, _ in bursts],
            "end_s": [float(t[e]) for _, e in bursts],
            "duration_s": [(e - s + 1) / fs for s, e in bursts],
            "peak_env": [float(env[s : e + 1].max()) for s, e in bursts],
        }
    )
