"""Template producer function — copy this module, rename, edit the body.

A canonical *event-producing* analysis function.  Mirrors the shape of
``databench.analysis.locomotion.locomotion_events``: takes a
:class:`SessionGroup`, scans each session, returns an
:data:`~databench.types.EventsTable` ready for
:meth:`~databench.analysis.eta.EtaAnalysis.run`.

Workflow for adding a new detector
----------------------------------
1. Copy this file to ``databench/analysis/<name>.py``.
2. Rename ``my_events`` and update the docstring.
3. Replace the body of the per-session loop with your detection logic.
4. Optionally add a new schema stub to :mod:`databench.types` and use it
   as the return annotation.
5. Add a unit test or a smoke script under ``Scripts/<group>/``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from databench.signal.epoching import make_events, REQUIRED_COLUMNS
from databench.utils.logger import get_logger

if TYPE_CHECKING:
    from databench.session import Session, SessionGroup
    from databench.types import EventsTable

_log = get_logger(__name__)


def my_events(
    sessions: "SessionGroup | list[Session]",
    *,
    source: str = "pupil",
    column: str = "radius",
    time_column: str = "time_elapsed_s",
    threshold: float = 1.5,
    min_duration_s: float = 0.2,
    event_types: tuple[str, ...] = ("onset",),
    condition_map: dict[str, str] | None = None,
) -> EventsTable:
    """Detect <DEVELOPER: describe what this finds> across sessions.

    Parameters
    ----------
    sessions
        Sessions to scan.  Pass ``proj.sessions(task=...)``.
    source, column
        Signal to threshold.
    time_column
        Time-axis column name in ``source``.
    threshold
        DEVELOPER: replace with whatever criterion makes sense.
    min_duration_s
        Reject events shorter than this.
    event_types
        Which transitions to emit (e.g. ``("onset",)``).
    condition_map
        Optional mapping ``task -> condition`` (or ``"subject,session,task" -> condition``).

    Returns
    -------
    EventsTable
        Columns: ``Subject``, ``Session``, ``Task``, ``EventType``,
        ``event_time``, and ``Condition`` if *condition_map* is given.
    """
    all_events: list[pd.DataFrame] = []

    for sess in sessions:
        try:
            t = sess.time(source, time_column)
            x = sess.signal(source, column)
        except Exception:
            continue
        if t is None or x is None or t.size < 3:
            continue

        # DEVELOPER: replace this block with your detector.
        # Example: index of first sample crossing `threshold` upward.
        above = np.asarray(x) > threshold
        crossings = np.where(np.diff(above.astype(int)) > 0)[0] + 1
        if crossings.size == 0:
            continue
        onset_times = t[crossings]

        # Drop too-short events using the next crossing as a duration proxy.
        if min_duration_s > 0 and onset_times.size > 1:
            durations = np.diff(np.append(onset_times, t[-1]))
            onset_times = onset_times[durations >= min_duration_s]
        if onset_times.size == 0:
            continue

        times_by_type: dict[str, np.ndarray] = {}
        for etype in event_types:
            if etype == "onset":
                times_by_type[etype] = onset_times
            else:
                raise ValueError(
                    f"Unknown event type {etype!r}.  "
                    f"DEVELOPER: extend the dispatch above."
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

    non_empty = [df for df in all_events if not df.empty]
    if not non_empty:
        cols = list(REQUIRED_COLUMNS)
        if condition_map is not None:
            cols.append("Condition")
        return pd.DataFrame(columns=cols)

    return pd.concat(non_empty, ignore_index=True)
