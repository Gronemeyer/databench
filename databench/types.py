"""Schema-aware DataFrame stubs for databench public APIs.

These names exist purely to enrich IDE hover, Go-to-Definition, and
autocomplete on column access.  At **runtime** every name in this module
is just :class:`pandas.DataFrame` — there is no subclass, no wrapping,
no validation.

Design rules
------------
* Stubs are defined inside ``if TYPE_CHECKING:`` so the column
  annotations exist for Pylance/Pyright but are never evaluated at
  runtime.  The ``else`` branch binds the same name to plain
  ``pd.DataFrame`` so any runtime ``isinstance`` / construction still
  works.
* Consumers import these names under ``if TYPE_CHECKING:`` and rely on
  ``from __future__ import annotations``.  Deleting / renaming /
  breaking anything here degrades only IDE features — no
  ``ImportError`` at runtime.
* The column annotations are *documentation*, not enforced contracts.
  Pandas already supports ``df.column_name`` attribute access at
  runtime when the column name is a valid identifier, so the stubs
  unlock typo-safe autocomplete (``epochs.start_s``) for free.

Adding a new schema
-------------------
Drop a new ``class FooTable(pd.DataFrame)`` block below with one
``: pd.Series`` annotation per column.  No registry, no decorator,
no producer changes required.

Usage in scripts (optional, opt-in)::

    from databench.types import EventsTable
    events: EventsTable = locomotion_events(group, ...)
    events.event_time.min()    # autocomplete + hover
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd


if TYPE_CHECKING:

    class EventsTable(pd.DataFrame):
        """Event table consumed by :meth:`EtaAnalysis.run`.

        One row per detected event, regardless of source detector.
        """

        Subject: pd.Series
        Session: pd.Series
        Task: pd.Series
        EventType: pd.Series   # e.g. "onset", "offset", "burst"
        event_time: pd.Series  # seconds, in session time base
        Condition: pd.Series   # optional; overrides condition_map when present

    class BoutEventsTable(pd.DataFrame):
        """Per-bout epoch table produced by ``locomotion_bout_events``.

        Each row describes one detected bout.
        """

        start_s: pd.Series          # bout onset time (s)
        end_s: pd.Series            # bout offset time (s)
        duration_s: pd.Series       # bout duration (s)
        mean_speed_cms: pd.Series   # mean speed within the bout (cm/s)
        distance_m: pd.Series       # distance covered within the bout (m)
        # In long-table form, the configured group columns are also present:
        Subject: pd.Series
        Session: pd.Series
        Task: pd.Series

    class EtaEventsTable(pd.DataFrame):
        """Per-event ETA traces — one row per (event, ROI, time bin)."""

        Subject: pd.Series
        Session: pd.Series
        Task: pd.Series
        Condition: pd.Series
        EventType: pd.Series
        event_id: pd.Series
        rel_time: pd.Series   # seconds relative to event_time
        ROI: pd.Series
        value: pd.Series      # baselined signal value

    class EtaSubjectMeansTable(pd.DataFrame):
        """Per-subject ETA means (averaged over events per subject)."""

        Subject: pd.Series
        Condition: pd.Series
        EventType: pd.Series
        ROI: pd.Series
        rel_time: pd.Series
        mean: pd.Series

    class EtaGroupMeansTable(pd.DataFrame):
        """Group-level ETA means ± SEM across subjects."""

        Condition: pd.Series
        EventType: pd.Series
        ROI: pd.Series
        rel_time: pd.Series
        mean: pd.Series
        sem: pd.Series

else:
    # Runtime: every schema is just pd.DataFrame.
    EventsTable = pd.DataFrame
    BoutEventsTable = pd.DataFrame
    EtaEventsTable = pd.DataFrame
    EtaSubjectMeansTable = pd.DataFrame
    EtaGroupMeansTable = pd.DataFrame


__all__ = [
    "EventsTable",
    "BoutEventsTable",
    "EtaEventsTable",
    "EtaSubjectMeansTable",
    "EtaGroupMeansTable",
]
