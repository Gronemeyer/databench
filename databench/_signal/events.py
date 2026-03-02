"""Event detection helpers for ETA and other peri-event analyses.

An **events table** is a plain :class:`~pandas.DataFrame` with columns:

    Subject, Session, Task, EventType, event_time

Any function that produces this schema can be passed to
:meth:`~databench.analysis.eta.EtaAnalysis.run`.

This module provides:

* :func:`make_events` — build an events table from named arrays of times.
* :func:`locomotion_events` — detect locomotion-bout onset/offset across
  a :class:`~databench.session.SessionGroup`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from databench._signal.bouts import locomotion_bout_events

REQUIRED_COLUMNS = ("Subject", "Session", "Task", "EventType", "event_time")
"""Columns that every events table must have."""


# ── Generic event-table builder ────────────────────────────────────────────

def make_events(
    event_times: dict[str, np.ndarray | list[float] | Sequence[float]],
    *,
    subject: str,
    session: str,
    task: str,
    condition: str | None = None,
) -> pd.DataFrame:
    """Build an events table from named event-time arrays.

    Parameters
    ----------
    event_times : dict
        Mapping from event-type label to an array of timestamps.
        Example: ``{"onset": [1.0, 5.0], "offset": [3.0, 7.0]}``.
    subject, session, task : str
        Session identifiers.
    condition : str, optional
        Condition label.  If ``None``, the column is omitted and
        :meth:`EtaAnalysis.run` will fill it from *condition_map*.

    Returns
    -------
    pd.DataFrame
        Columns: ``Subject``, ``Session``, ``Task``, ``EventType``,
        ``event_time`` (and optionally ``Condition``).

    Examples
    --------
    >>> make_events(
    ...     {"onset": [1.0, 5.0], "offset": [3.0, 7.0]},
    ...     subject="GS28", session="ses-01", task="task-spont",
    ... )
    """
    rows: list[dict] = []
    for etype, times in event_times.items():
        for t in np.atleast_1d(times):
            row: dict = {
                "Subject": subject,
                "Session": session,
                "Task": task,
                "EventType": etype,
                "event_time": float(t),
            }
            if condition is not None:
                row["Condition"] = condition
            rows.append(row)
    cols = list(REQUIRED_COLUMNS)
    if condition is not None:
        cols.append("Condition")
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


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

    Wraps :func:`~databench._signal.bouts.locomotion_bout_events` for
    each session and returns a combined events table ready for
    :meth:`EtaAnalysis.run`.

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
