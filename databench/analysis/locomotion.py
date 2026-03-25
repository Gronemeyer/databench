"""Locomotion bout detection and event generation.

Provides:

* :func:`locomotion_bout_events` — detect locomotion bouts from speed traces
  (accepts raw arrays or grouped DataFrames).
* :func:`locomotion_events` — detect locomotion-bout events across a
  :class:`~databench.session.SessionGroup` and return an events table
  compatible with :meth:`~databench.analysis.eta.EtaAnalysis.run`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from databench._signal.segments import segments_from_mask
from databench._signal.events import make_events, REQUIRED_COLUMNS

if TYPE_CHECKING:
    from databench.session import Session, SessionGroup


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


# ── Locomotion-bout events ─────────────────────────────────────────────────

def locomotion_events(
    sessions: "SessionGroup | list[Session]",
    *,
    speed_source: str = "treadmill",
    speed_column: str = "speed_mm",
    speed_scale_to_cms: float = 10.0,
    time_column: str = "time_elapsed_s",
    min_speed_cms: float = 0.5,
    min_duration_s: float = 1.0,
    merge_gap_s: float = 0.5,
    event_types: tuple[str, ...] = ("onset", "offset"),
    condition_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Detect locomotion-bout events across sessions.

    Wraps :func:`locomotion_bout_events` for each session and returns a
    combined events table ready for :meth:`EtaAnalysis.run`.

    Parameters
    ----------
    sessions : SessionGroup or list of Session
        Sessions to scan.
    speed_source : str
        Source containing the speed trace.
    speed_column : str
        Column name for speed within the source.
    speed_scale_to_cms : float
        Divide raw speed by this to get cm/s.
    time_column : str
        Time column within the speed source.
    min_speed_cms : float
        Minimum speed to qualify as locomotion.
    min_duration_s : float
        Minimum bout duration (seconds).
    merge_gap_s : float
        Merge bouts closer than this (seconds).
    event_types : tuple of str
        Which transitions to include (``"onset"`` and/or ``"offset"``).
    condition_map : dict, optional
        Maps task name (or ``"subject,session,task"`` key) to condition label.

    Returns
    -------
    pd.DataFrame
        Events table with columns ``Subject``, ``Session``, ``Task``,
        ``EventType``, ``event_time`` (and ``Condition`` if *condition_map*
        is provided).
    """
    all_events: list[pd.DataFrame] = []
    for sess in sessions:
        try:
            t = sess.time(speed_source, time_column)
            speed = sess.signal(speed_source, speed_column)
        except Exception:
            continue
        if t is None or speed is None or t.size < 3 or speed.size < 3:
            continue
        speed_cms = speed / speed_scale_to_cms

        _, _, _, onset_t, offset_t = locomotion_bout_events(
            t,
            speed_cms,
            min_speed_cms=min_speed_cms,
            min_duration_s=min_duration_s,
            merge_gap_s=merge_gap_s,
            as_table=False,
        )

        times_by_type: dict[str, np.ndarray] = {}
        for etype in event_types:
            if etype == "onset":
                times_by_type[etype] = onset_t
            elif etype == "offset":
                times_by_type[etype] = offset_t
            else:
                raise ValueError(
                    f"Unknown locomotion event type {etype!r}. "
                    f"Expected 'onset' or 'offset'."
                )

        cond: str | None = None
        if condition_map is not None:
            key = f"{sess.subject},{sess.session},{sess.task}"
            cond = condition_map.get(key, condition_map.get(sess.task, "all"))

        ev = make_events(
            times_by_type,
            subject=sess.subject,
            session=sess.session,
            task=sess.task,
            condition=cond,
        )
        all_events.append(ev)

    if not all_events:
        cols = list(REQUIRED_COLUMNS)
        if condition_map is not None:
            cols.append("Condition")
        return pd.DataFrame(columns=cols)
    # Filter out empty DataFrames before concat to avoid FutureWarning
    non_empty = [df for df in all_events if not df.empty]
    if not non_empty:
        cols = list(REQUIRED_COLUMNS)
        if condition_map is not None:
            cols.append("Condition")
        return pd.DataFrame(columns=cols)
    return pd.concat(non_empty, ignore_index=True)
