"""Locomotion bout detection and event generation.

This module is the **canonical reference for the two-tier portability
pattern** other analyses should follow: a pure-array core, then a thin
SessionGroup wrapper on the side.

Which function do I want?
-------------------------
+--------------------------------+----------------------------------------------------+
| You have…                      | Use…                                               |
+================================+====================================================+
| Plain ``t`` and ``speed_cms``  | :func:`locomotion_bout_events` (returns full       |
| numpy arrays, want bout info   | epoch DataFrame) — pure-array core, no databench   |
|                                | imports required for downstream callers.           |
+--------------------------------+----------------------------------------------------+
| Same arrays, just need         | :func:`locomotion_bouts` — thin wrapper that       |
| ``[(start_idx, end_idx), ...]``| returns index pairs only.                          |
+--------------------------------+----------------------------------------------------+
| Index pairs, want the inverse  | :func:`quiescent_bouts` — non-locomotion intervals |
| (quiet periods)                | between bouts.                                     |
+--------------------------------+----------------------------------------------------+
| A :class:`SessionGroup` and    | :func:`locomotion_events` — scans every session    |
| need events for ETA            | and returns an :data:`~databench.types.EventsTable`|
|                                | with ``EventType`` / ``event_time``.               |
+--------------------------------+----------------------------------------------------+
| A long-format DataFrame        | :class:`LocomotionBoutEventsExtractor` — explicit  |
| ``(Subject, Session, Task,     | extractor object you can compose with other        |
| time, speed)``                 | long-table pipelines.                              |
+--------------------------------+----------------------------------------------------+

A colleague who does not use databench can call
``locomotion_bout_events(t, speed_cms)`` on their own arrays — no
``Project``, ``SessionGroup``, or plotting setup required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
import pandas as pd

from databench.signal.epoching import (
    make_events,
    REQUIRED_COLUMNS,
    detect_epochs,
    START_IDX,
    END_IDX,
)

if TYPE_CHECKING:
    from databench.session import Session, SessionGroup
    from databench.types import BoutEventsTable, EventsTable

SOURCE_COLOR = "#2ca02c"

DIRECTIONS = ("forward", "absolute", "backward")
"""Accepted values of the *direction* argument on every detector here."""

__all__ = [
    "DIRECTIONS",
    "direction_mask",
    "locomotion_bout_events",
    "locomotion_bouts",
    "locomotion_events",
    "quiescent_bouts",
    "LocomotionBoutEventsExtractor",
]


# ── Direction ──────────────────────────────────────────────────────────────

def direction_mask(
    speed_cms: np.ndarray,
    min_speed_cms: float,
    direction: str = "forward",
) -> np.ndarray:
    """Boolean mask of samples counting as locomotion in *direction*.

    The encoder reports signed speed, so the sign convention has to be an
    explicit choice rather than an accident of the comparison operator:

    ``"forward"``
        ``speed_cms >= min_speed_cms`` — forward travel only (default).
    ``"absolute"``
        ``abs(speed_cms) >= min_speed_cms`` — travel in either direction.
    ``"backward"``
        ``speed_cms <= -min_speed_cms`` — reverse travel only.
    """
    speed_cms = np.asarray(speed_cms, dtype=float)
    if direction == "forward":
        return speed_cms >= min_speed_cms
    if direction == "absolute":
        return np.abs(speed_cms) >= min_speed_cms
    if direction == "backward":
        return speed_cms <= -min_speed_cms
    raise ValueError(
        f"Unknown direction {direction!r}. Expected one of {DIRECTIONS}."
    )


# ── Private helpers ────────────────────────────────────────────────────────

def _bout_stats(
    t: np.ndarray,
    speed_cms: np.ndarray,
    epochs: pd.DataFrame,
    *,
    direction: str = "forward",
) -> None:
    """Add ``mean_speed_cms`` and ``distance_m`` columns to an EpochTable *in place*.

    ``mean_speed_cms`` follows the sign convention the bouts were detected
    under: magnitudes for ``direction="absolute"``, signed values otherwise —
    so a backward bout reports a negative mean speed and a negative distance.
    """
    mean_speeds: list[float] = []
    distances_m: list[float] = []
    for _, row in epochs.iterrows():
        s, e = int(row[START_IDX]), int(row[END_IDX])
        window = speed_cms[s : e + 1]
        if direction == "absolute":
            window = np.abs(window)
        mean_speeds.append(float(np.nanmean(window)))
        time_steps = np.diff(t[s : e + 1])
        dist_cm = float(np.nansum(speed_cms[s + 1 : e + 1] * time_steps))
        distances_m.append(dist_cm / 100.0)
    epochs["mean_speed_cms"] = mean_speeds
    epochs["distance_m"] = distances_m


# ── Index-pair detectors (lightweight; for procedural scripts) ────────────

def locomotion_bouts(
    t: np.ndarray,
    speed_cms: np.ndarray,
    *,
    min_speed_cms: float = 0.5,
    min_duration_s: float = 1.0,
    merge_gap_s: float = 0.5,
    direction: str = "forward",
) -> list[tuple[int, int]]:
    """Detect locomotion bouts and return index pairs.

    Convenience wrapper over :func:`locomotion_bout_events` for procedural
    scripts that only need ``[(start_idx, end_idx), ...]`` rather than a
    full EpochTable.  *direction* is passed through — see
    :func:`direction_mask`.
    """
    epochs = locomotion_bout_events(
        t,
        speed_cms,
        min_speed_cms=min_speed_cms,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
        direction=direction,
    )
    if epochs.empty:
        return []
    return [
        (int(s), int(e))
        for s, e in zip(epochs[START_IDX].to_numpy(), epochs[END_IDX].to_numpy())
    ]


def quiescent_bouts(
    t: np.ndarray,
    loco_bouts: list[tuple[int, int]],
    *,
    min_duration_s: float = 0.0,
) -> list[tuple[int, int]]:
    """Return index pairs for non-locomotion periods between *loco_bouts*.

    Parameters
    ----------
    t : np.ndarray
        Timestamps in seconds.
    loco_bouts : list of (start_idx, end_idx)
        Locomotion-bout indices, sorted in time.
    min_duration_s : float
        Drop quiescent intervals shorter than this.
    """
    n = len(t)
    if not loco_bouts:
        return [(0, n - 1)] if n > 0 else []

    quiet: list[tuple[int, int]] = []
    first_start = loco_bouts[0][0]
    if first_start > 0:
        quiet.append((0, first_start - 1))
    for i in range(len(loco_bouts) - 1):
        gap_start = loco_bouts[i][1] + 1
        gap_end = loco_bouts[i + 1][0] - 1
        if gap_end >= gap_start:
            quiet.append((gap_start, gap_end))
    last_end = loco_bouts[-1][1]
    if last_end < n - 1:
        quiet.append((last_end + 1, n - 1))

    if min_duration_s <= 0:
        return quiet
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 0.02
    return [
        (s, e) for s, e in quiet
        if float(t[e] - t[s] + dt) >= min_duration_s
    ]


# ── Bout detection ─────────────────────────────────────────────────────────

def locomotion_bout_events(
    t: np.ndarray | pd.DataFrame,
    speed_cms: np.ndarray | None = None,
    *,
    min_speed_cms: float = 0.5,
    min_duration_s: float = 1.0,
    merge_gap_s: float = 0.5,
    direction: str = "forward",
    context: Mapping[str, Any] | None = None,
    group_cols: tuple[str, ...] = ("Subject", "Session", "Task"),
    time_col: str = "time_elapsed_s",
    speed_col: str = "speed_mm",
    speed_scale_to_cms: float = 10.0,
) -> BoutEventsTable:
    """Detect locomotion bouts and return a standard EpochTable.

    Accepts either raw arrays ``(t, speed_cms)`` or a grouped DataFrame.
    Always returns a DataFrame with the standard epoch columns plus
    ``mean_speed_cms`` and ``distance_m``.

    *direction* selects the sign convention for the speed threshold —
    ``"forward"`` (default), ``"absolute"``, or ``"backward"``; see
    :func:`direction_mask`.
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
                direction=direction,
                context=local_context,
            )
            if bout_table.empty:
                continue
            tables.append(bout_table)

        if not tables:
            return pd.DataFrame()
        return pd.concat(tables, ignore_index=True)

    # ── Array path ─────────────────────────────────────────────────────
    mask = direction_mask(speed_cms, min_speed_cms, direction)
    epochs = detect_epochs(
        mask, t,
        event_type="locomotion_bout",
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
        metadata=dict(context) if context else None,
    )
    if not epochs.empty:
        _bout_stats(t, speed_cms, epochs, direction=direction)
    else:
        epochs["mean_speed_cms"] = []
        epochs["distance_m"] = []
    return epochs


# ── Session-level event generation ─────────────────────────────────────────

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
    direction: str = "forward",
    event_types: tuple[str, ...] = ("onset", "offset"),
    condition_map: dict[str, str] | None = None,
) -> EventsTable:
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
    direction : str
        Sign convention for the speed threshold — ``"forward"`` (default),
        ``"absolute"``, or ``"backward"``.  See :func:`direction_mask`.
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

        epochs = locomotion_bout_events(
            t,
            speed_cms,
            min_speed_cms=min_speed_cms,
            min_duration_s=min_duration_s,
            merge_gap_s=merge_gap_s,
            direction=direction,
        )

        if epochs.empty:
            continue

        times_by_type: dict[str, np.ndarray] = {}
        for etype in event_types:
            if etype == "onset":
                times_by_type[etype] = epochs["start_s"].to_numpy()
            elif etype == "offset":
                times_by_type[etype] = epochs["end_s"].to_numpy()
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


# ── Bout-events extractor (long-table pipelines) ──────────────────────────

@dataclass(frozen=True)
class LocomotionBoutEventsExtractor:
    """Reusable, explicit bout-event extractor for long-table pipelines."""

    min_speed_cms: float = 0.5
    min_duration_s: float = 1.0
    merge_gap_s: float = 0.5
    direction: str = "forward"
    group_cols: tuple[str, ...] = ("Subject", "Session", "Task")
    time_col: str = "time_elapsed_s"
    speed_col: str = "speed_mm"
    speed_scale_to_cms: float = 10.0

    def run(self, long: pd.DataFrame) -> pd.DataFrame:
        return locomotion_bout_events(
            long,
            min_speed_cms=self.min_speed_cms,
            min_duration_s=self.min_duration_s,
            merge_gap_s=self.merge_gap_s,
            direction=self.direction,
            group_cols=self.group_cols,
            time_col=self.time_col,
            speed_col=self.speed_col,
            speed_scale_to_cms=self.speed_scale_to_cms,
        )

    def __call__(self, long: pd.DataFrame) -> pd.DataFrame:
        return self.run(long)
