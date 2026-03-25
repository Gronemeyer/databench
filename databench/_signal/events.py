"""Event detection helpers for ETA and other peri-event analyses.

An **events table** is a plain :class:`~pandas.DataFrame` with columns:

    Subject, Session, Task, EventType, event_time

Any function that produces this schema can be passed to
:meth:`~databench.analysis.eta.EtaAnalysis.run`.

This module provides:

* :func:`make_events` — build an events table from named arrays of times.

Analysis-specific event detectors (e.g. locomotion) live in
:mod:`databench.analysis.locomotion`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

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
