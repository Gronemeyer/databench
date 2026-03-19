"""Locomotion bout detection from speed traces."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd

from databench._signal.segments import segments_from_mask


# ── Low-level helpers ──────────────────────────────────────────────────────

def _merge_gaps_by_time(
    segments: Iterable[Tuple[int, int]],
    t: np.ndarray,
    min_gap_s: float,
) -> list[Tuple[int, int]]:
    merged: list[Tuple[int, int]] = []
    for s, e in segments:
        if not merged:
            merged.append((s, e))
            continue
        _, prev_end = merged[-1]
        gap = float(t[s] - t[prev_end])
        if gap <= min_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _apply_min_duration_by_time(
    segments: Iterable[Tuple[int, int]],
    t: np.ndarray,
    min_duration_s: float,
) -> list[Tuple[int, int]]:
    sample_interval = float(np.nanmedian(np.diff(t))) if t.size > 1 else 0.0
    return [
        (s, e) for s, e in segments
        if float(t[e] - t[s] + sample_interval) >= min_duration_s
    ]


def _locomotion_bouts(
    t: np.ndarray,
    speed_cms: np.ndarray,
    *,
    min_speed_cms: float,
    min_duration_s: float,
    merge_gap_s: float,
) -> list[Tuple[int, int]]:
    mask = speed_cms >= min_speed_cms
    segs = segments_from_mask(mask)
    if not segs:
        return []
    if merge_gap_s > 0:
        segs = _merge_gaps_by_time(segs, t, merge_gap_s)
    if min_duration_s > 0:
        segs = _apply_min_duration_by_time(segs, t, min_duration_s)
    return segs


# ── Public function ────────────────────────────────────────────────────────

def locomotion_bout_events(
    t: np.ndarray | pd.DataFrame,
    speed_cms: np.ndarray | None = None,
    *,
    min_speed_cms: float = 0.5,
    min_duration_s: float = 1.0,
    merge_gap_s: float = 0.5,
    as_table: bool = False,
    context: Mapping[str, Any] | None = None,
    group_cols: tuple[str, ...] = ("Subject", "Session", "Task"),
    time_col: str = "time_elapsed_s",
    speed_col: str = "speed_mm",
    speed_scale_to_cms: float = 10.0,
) -> pd.DataFrame | tuple:
    """Detect locomotion bouts from speed traces.

    Accepts either raw arrays ``(t, speed_cms)`` or a grouped DataFrame.
    Returns a DataFrame if *as_table=True* or when *t* is a DataFrame.
    """
    if isinstance(t, pd.DataFrame):
        tables: list[pd.DataFrame] = []
        for key, group in t.groupby(list(group_cols), sort=False):
            group = group.sort_values(time_col)
            t_raw = group[time_col].to_numpy()
            v_raw = group[speed_col].to_numpy()
            valid = np.isfinite(t_raw) & np.isfinite(v_raw)
            time_valid = t_raw[valid]
            speed_valid = v_raw[valid]
            if time_valid.size < 3:
                continue
            local_context = {
                col: val
                for col, val in zip(group_cols, key if isinstance(key, tuple) else (key,))
            }
            if context:
                local_context.update(dict(context))
            bout_table = locomotion_bout_events(
                time_valid,
                speed_valid / speed_scale_to_cms,
                min_speed_cms=min_speed_cms,
                min_duration_s=min_duration_s,
                merge_gap_s=merge_gap_s,
                as_table=True,
                context=local_context,
            )
            if bout_table.empty:
                continue
            tables.append(bout_table)

        base_cols = ["bout_id", "onset_idx", "offset_idx", "onset_t", "offset_t"]
        extra_cols = list(group_cols)
        if context:
            extra_cols.extend([c for c in context.keys() if c not in extra_cols])
        out_cols = base_cols + extra_cols
        out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=out_cols)
        for col in out_cols:
            if col not in out.columns:
                out[col] = np.nan
        return out[out_cols]

    bouts = _locomotion_bouts(
        t,
        speed_cms,
        min_speed_cms=min_speed_cms,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
    )
    onset_idx = np.array([s for s, _ in bouts], dtype=int)
    offset_idx = np.array([e for _, e in bouts], dtype=int)
    onset_t = t[onset_idx].astype(float) if len(bouts) else np.array([], dtype=float)
    offset_t = t[offset_idx].astype(float) if len(bouts) else np.array([], dtype=float)

    if as_table:
        table = pd.DataFrame(
            {
                "bout_id": np.arange(len(bouts), dtype=int),
                "onset_idx": onset_idx,
                "offset_idx": offset_idx,
                "onset_t": onset_t,
                "offset_t": offset_t,
            }
        )
        if context:
            for key, value in context.items():
                table[key] = value
        return table
    return bouts, onset_idx, offset_idx, onset_t, offset_t
