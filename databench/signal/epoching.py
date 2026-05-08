"""Unified epoch detection, event tables, and peri-event extraction.

This module consolidates the former ``segments.py``, ``epochs.py``, and
``events.py`` into a single canonical source for all epoching operations.

**EpochTable** — the standard output format
    A :class:`~pandas.DataFrame` with at least these columns::

        epoch_id  start_s  end_s  duration_s  event_type

    When produced by :func:`detect_epochs`, additional ``start_idx`` and
    ``end_idx`` columns are included.  Arbitrary metadata columns (e.g.
    ``Subject``, ``Session``) can be attached via the *metadata* parameter.

Key public API:

* :func:`detect_epochs` — threshold-based epoch detection from a boolean mask.
* :func:`epochs_around_events` — build epoch intervals around point events.
* :func:`make_events` — build a point-event table for ETA pipelines.
* :func:`epoch_indices` / :func:`extract_epoch_interpolated` — peri-event
  signal extraction on a uniform time grid.

Low-level segment helpers (``segments_from_mask``, ``merge_gaps``, etc.) are
also re-exported for callers that need direct access.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

# ── Column-name constants ─────────────────────────────────────────────────

EPOCH_ID = "epoch_id"
START_S = "start_s"
END_S = "end_s"
DURATION_S = "duration_s"
EVENT_TYPE = "event_type"
START_IDX = "start_idx"
END_IDX = "end_idx"

EPOCH_COLUMNS = (EPOCH_ID, START_S, END_S, DURATION_S, EVENT_TYPE)
"""Minimum required columns in every EpochTable."""

REQUIRED_COLUMNS = ("Subject", "Session", "Task", "EventType", "event_time")
"""Columns required by the ETA point-event table (:func:`make_events`)."""


# ═══════════════════════════════════════════════════════════════════════════
# Segment primitives  (formerly segments.py)
# ═══════════════════════════════════════════════════════════════════════════

def segments_from_mask(mask: np.ndarray) -> list[Tuple[int, int]]:
    """Convert a boolean mask to a list of *(start_idx, end_idx)* contiguous
    segments.  Both indices are **inclusive**."""
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


# ── Sample-based merging / filtering ──────────────────────────────────────

def merge_gaps(
    segments: Iterable[Tuple[int, int]],
    min_gap: int,
) -> list[Tuple[int, int]]:
    """Merge neighbouring segments when gap <= *min_gap* samples."""
    merged: list[Tuple[int, int]] = []
    for s, e in segments:
        if not merged:
            merged.append((s, e))
            continue
        prev_start, prev_end = merged[-1]
        if s - prev_end - 1 <= min_gap:
            merged[-1] = (prev_start, e)
        else:
            merged.append((s, e))
    return merged


def apply_min_duration(
    segments: Iterable[Tuple[int, int]],
    min_len: int,
) -> list[Tuple[int, int]]:
    """Drop segments shorter than *min_len* samples."""
    return [(s, e) for s, e in segments if (e - s + 1) >= min_len]


# ── Time-based merging / filtering ────────────────────────────────────────

def merge_gaps_by_time(
    segments: Iterable[Tuple[int, int]],
    t: np.ndarray,
    min_gap_s: float,
) -> list[Tuple[int, int]]:
    """Merge neighbouring segments when temporal gap <= *min_gap_s* seconds."""
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


def apply_min_duration_by_time(
    segments: Iterable[Tuple[int, int]],
    t: np.ndarray,
    min_duration_s: float,
) -> list[Tuple[int, int]]:
    """Drop segments whose temporal duration is below *min_duration_s*."""
    sample_interval = float(np.nanmedian(np.diff(t))) if t.size > 1 else 0.0
    return [
        (s, e) for s, e in segments
        if float(t[e] - t[s] + sample_interval) >= min_duration_s
    ]


# ═══════════════════════════════════════════════════════════════════════════
# EpochTable builders
# ═══════════════════════════════════════════════════════════════════════════

def _segments_to_table(
    segs: list[Tuple[int, int]],
    t: np.ndarray,
    *,
    event_type: str,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Convert index-pair segments into a standard EpochTable DataFrame."""
    if not segs:
        cols = [EPOCH_ID, START_S, END_S, DURATION_S, EVENT_TYPE, START_IDX, END_IDX]
        if metadata:
            cols.extend(metadata.keys())
        return pd.DataFrame(columns=cols)

    sample_interval = float(np.nanmedian(np.diff(t))) if t.size > 1 else 0.0
    rows: dict[str, list] = {
        EPOCH_ID: list(range(len(segs))),
        START_S: [float(t[s]) for s, _ in segs],
        END_S: [float(t[e]) for _, e in segs],
        DURATION_S: [float(t[e] - t[s] + sample_interval) for s, e in segs],
        EVENT_TYPE: [event_type] * len(segs),
        START_IDX: [s for s, _ in segs],
        END_IDX: [e for _, e in segs],
    }
    if metadata:
        for k, v in metadata.items():
            rows[k] = [v] * len(segs)
    return pd.DataFrame(rows)


def detect_epochs(
    mask: np.ndarray,
    t: np.ndarray,
    *,
    event_type: str,
    min_duration_s: float = 0.0,
    merge_gap_s: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Detect epochs from a boolean mask and return a standard EpochTable.

    Pipeline: ``segments_from_mask`` → ``merge_gaps_by_time`` →
    ``apply_min_duration_by_time`` → EpochTable DataFrame.

    Parameters
    ----------
    mask : array of bool
        Same length as *t*.  ``True`` where the epoch condition holds.
    t : array of float
        Timestamps in seconds, monotonically increasing.
    event_type : str
        Label written to the ``event_type`` column (e.g.
        ``"locomotion_bout"``, ``"oscillation_burst"``).
    min_duration_s : float
        Minimum epoch duration; shorter epochs are dropped.
    merge_gap_s : float
        Maximum inter-epoch gap to fuse.
    metadata : dict, optional
        Scalar key/value pairs broadcast to every row (e.g.
        ``{"Subject": "GS28", "Session": "ses-01"}``).

    Returns
    -------
    pd.DataFrame
        Columns: ``epoch_id``, ``start_s``, ``end_s``, ``duration_s``,
        ``event_type``, ``start_idx``, ``end_idx``, plus any *metadata*.
    """
    segs = segments_from_mask(mask)
    if merge_gap_s > 0:
        segs = merge_gaps_by_time(segs, t, merge_gap_s)
    if min_duration_s > 0:
        segs = apply_min_duration_by_time(segs, t, min_duration_s)
    return _segments_to_table(segs, t, event_type=event_type, metadata=metadata)


def epochs_around_events(
    event_times: np.ndarray | Sequence[float],
    *,
    window: Tuple[float, float],
    event_type: str,
    t: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build an EpochTable from point events and a (pre, post) window.

    Each epoch spans ``[event_time + window[0], event_time + window[1]]``.
    If *t* is provided, ``start_idx`` / ``end_idx`` are computed via
    :func:`epoch_indices`; otherwise those columns are absent.

    Parameters
    ----------
    event_times : array-like of float
        Absolute timestamps of each event.
    window : (float, float)
        ``(pre_offset, post_offset)`` relative to each event, in seconds.
        Typically ``(-2.0, 5.0)`` or similar.
    event_type : str
        Label for the ``event_type`` column.
    t : array of float, optional
        Full time vector for index lookups.
    metadata : dict, optional
        Scalar key/value pairs broadcast to every row.

    Returns
    -------
    pd.DataFrame
        Standard EpochTable.
    """
    times = np.atleast_1d(np.asarray(event_times, dtype=float))
    records: list[dict[str, Any]] = []
    for i, et in enumerate(times):
        row: dict[str, Any] = {
            EPOCH_ID: i,
            START_S: float(et + window[0]),
            END_S: float(et + window[1]),
            DURATION_S: float(window[1] - window[0]),
            EVENT_TYPE: event_type,
        }
        if t is not None:
            bounds = epoch_indices(t, et, window=window)
            if bounds is not None:
                row[START_IDX] = bounds[0]
                row[END_IDX] = bounds[1]
        if metadata:
            row.update(metadata)
        records.append(row)
    if not records:
        cols = list(EPOCH_COLUMNS)
        if t is not None:
            cols += [START_IDX, END_IDX]
        if metadata:
            cols.extend(metadata.keys())
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# Point-event table  (formerly events.py)
# ═══════════════════════════════════════════════════════════════════════════

def make_events(
    event_times: dict[str, np.ndarray | list[float] | Sequence[float]],
    *,
    subject: str,
    session: str,
    task: str,
    condition: str | None = None,
) -> pd.DataFrame:
    """Build a point-event table from named event-time arrays.

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
    """
    rows: list[dict] = []
    for etype, times in event_times.items():
        for tv in np.atleast_1d(times):
            row: dict = {
                "Subject": subject,
                "Session": session,
                "Task": task,
                "EventType": etype,
                "event_time": float(tv),
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


# ═══════════════════════════════════════════════════════════════════════════
# Peri-event extraction  (formerly epochs.py)
# ═══════════════════════════════════════════════════════════════════════════

def epoch_indices(
    t: np.ndarray,
    t0: float,
    *,
    window: Tuple[float, float] = (-2.0, 1.0),
) -> Tuple[int, int] | None:
    """Return inclusive index bounds for an epoch around *t0*."""
    t_start = t0 + window[0]
    t_end = t0 + window[1]
    i0 = int(np.searchsorted(t, t_start, side="left"))
    i1 = int(np.searchsorted(t, t_end, side="right") - 1)
    return i0, i1


def extract_epoch_interpolated(
    t: np.ndarray,
    y: np.ndarray,
    t0: float,
    *,
    window: Tuple[float, float] = (-2.0, 1.0),
    dt: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """Interpolate an epoch onto a uniform grid.

    Returns ``(rel_t, y_interp)`` or ``(None, None)`` if data is
    insufficient.
    """
    valid = np.isfinite(t) & np.isfinite(y)
    t_valid = t[valid]
    y_valid = y[valid]
    if t_valid.size < 2:
        return None, None
    rel_t = np.arange(window[0], window[1] + 1e-12, dt)
    target_times = t0 + rel_t
    interpolated_values = np.interp(target_times, t_valid, y_valid)
    out_of_range = (target_times < t_valid[0]) | (target_times > t_valid[-1])
    interpolated_values[out_of_range] = np.nan
    return rel_t, interpolated_values
