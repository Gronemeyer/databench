"""Locomotion bout detection, event generation, and treadmill features.

Key public API:

* :func:`locomotion_bout_events` — detect locomotion bouts from speed traces
  (accepts raw arrays or grouped DataFrames).
* :func:`locomotion_events` — detect locomotion-bout events across a
  :class:`~databench.session.SessionGroup` for ETA.
* :class:`MeanSpeedCMS`, :class:`StdSpeedCMS`, :class:`TotalDistanceM` —
  simple per-session speed/distance features.
* :class:`LocomotionBoutsCount`, :class:`LocomotionBoutSpeedMeanCMS`,
  :class:`LocomotionBoutDistanceM`, :class:`LocomotionBoutDurationS` —
  bout-based features.
* :class:`LocomotionBoutEventsExtractor` — reusable extractor for
  long-table pipelines.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
import pandas as pd

from databench.analysis._signal.epoching import (
    make_events,
    REQUIRED_COLUMNS,
    detect_epochs,
    START_IDX,
    END_IDX,
)
from databench.analysis.base import FeatureFn
from databench._utils import as_1d, clean_xy, get_first

if TYPE_CHECKING:
    from databench.session import Session, SessionGroup

SOURCE_COLOR = "#2ca02c"

__all__ = [
    "locomotion_bout_events",
    "locomotion_events",
    "LocomotionBoutEventsExtractor",
    "MeanSpeedCMS",
    "StdSpeedCMS",
    "TotalDistanceM",
    "LocomotionBoutFeature",
    "LocomotionBoutsCount",
    "LocomotionBoutSpeedMeanCMS",
    "LocomotionBoutDistanceM",
    "LocomotionBoutDurationS",
]


# ── Private helpers ────────────────────────────────────────────────────────

def _bout_stats(
    t: np.ndarray,
    speed_cms: np.ndarray,
    epochs: pd.DataFrame,
) -> None:
    """Add ``mean_speed_cms`` and ``distance_m`` columns to an EpochTable *in place*."""
    mean_speeds: list[float] = []
    distances_m: list[float] = []
    for _, row in epochs.iterrows():
        s, e = int(row[START_IDX]), int(row[END_IDX])
        mean_speeds.append(float(np.nanmean(np.abs(speed_cms[s : e + 1]))))
        time_steps = np.diff(t[s : e + 1])
        dist_cm = float(np.nansum(speed_cms[s + 1 : e + 1] * time_steps))
        distances_m.append(dist_cm / 100.0)
    epochs["mean_speed_cms"] = mean_speeds
    epochs["distance_m"] = distances_m


def _nanmean_no_warn(values) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(values))


# ── Bout detection ─────────────────────────────────────────────────────────

def locomotion_bout_events(
    t: np.ndarray | pd.DataFrame,
    speed_cms: np.ndarray | None = None,
    *,
    min_speed_cms: float = 0.5,
    min_duration_s: float = 1.0,
    merge_gap_s: float = 0.5,
    context: Mapping[str, Any] | None = None,
    group_cols: tuple[str, ...] = ("Subject", "Session", "Task"),
    time_col: str = "time_elapsed_s",
    speed_col: str = "speed_mm",
    speed_scale_to_cms: float = 10.0,
) -> pd.DataFrame:
    """Detect locomotion bouts and return a standard EpochTable.

    Accepts either raw arrays ``(t, speed_cms)`` or a grouped DataFrame.
    Always returns a DataFrame with the standard epoch columns plus
    ``mean_speed_cms`` and ``distance_m``.
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
                context=local_context,
            )
            if bout_table.empty:
                continue
            tables.append(bout_table)

        if not tables:
            return pd.DataFrame()
        return pd.concat(tables, ignore_index=True)

    # ── Array path ─────────────────────────────────────────────────────
    mask = speed_cms >= min_speed_cms
    epochs = detect_epochs(
        mask, t,
        event_type="locomotion_bout",
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
        metadata=dict(context) if context else None,
    )
    if not epochs.empty:
        _bout_stats(t, speed_cms, epochs)
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

        epochs = locomotion_bout_events(
            t,
            speed_cms,
            min_speed_cms=min_speed_cms,
            min_duration_s=min_duration_s,
            merge_gap_s=merge_gap_s,
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
    group_cols: tuple[str, ...] = ("Subject", "Session", "Task")
    time_col: str = "time_elapsed_s"
    speed_col: str = "speed_mm"
    speed_scale_to_cms: float = 10.0

    @classmethod
    def from_feature(
        cls,
        feat: "LocomotionBoutFeature",
        *,
        group_cols: tuple[str, ...] = ("Subject", "Session", "Task"),
        time_col: str = "time_elapsed_s",
        speed_col: str = "speed_mm",
        speed_scale_to_cms: float = 10.0,
    ) -> "LocomotionBoutEventsExtractor":
        """Build an extractor from a LocomotionBoutFeature's thresholds."""
        return cls(
            min_speed_cms=feat.min_speed_cms,
            min_duration_s=feat.min_duration_s,
            merge_gap_s=feat.merge_gap_s,
            group_cols=group_cols,
            time_col=time_col,
            speed_col=speed_col,
            speed_scale_to_cms=speed_scale_to_cms,
        )

    def run(self, long: pd.DataFrame) -> pd.DataFrame:
        return locomotion_bout_events(
            long,
            min_speed_cms=self.min_speed_cms,
            min_duration_s=self.min_duration_s,
            merge_gap_s=self.merge_gap_s,
            group_cols=self.group_cols,
            time_col=self.time_col,
            speed_col=self.speed_col,
            speed_scale_to_cms=self.speed_scale_to_cms,
        )

    def __call__(self, long: pd.DataFrame) -> pd.DataFrame:
        return self.run(long)


# ── Feature extractors ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class MeanSpeedCMS(FeatureFn):
    """Mean treadmill speed per session in cm/s."""

    name: str = "speed_mean_cms"
    label: str = "Speed (cm/s)"
    color: str = SOURCE_COLOR
    source: str = "treadmill"
    plotter: str = "boxplot"

    def _run_impl(self, row) -> float:
        t, spd_mm = clean_xy(
            get_first(row, [("treadmill", "time_elapsed_s"), ("encoder", "time_elapsed_s")]),
            get_first(row, [("treadmill", "speed_mm"), ("encoder", "speed")]),
        )
        return float(np.nanmean(spd_mm) / 10.0)


@dataclass(frozen=True)
class StdSpeedCMS(FeatureFn):
    """Standard deviation of treadmill speed in cm/s."""

    name: str = "speed_std_cms"
    label: str = "Speed SD (cm/s)"
    color: str = SOURCE_COLOR
    source: str = "treadmill"

    def _run_impl(self, row) -> float:
        t, spd_mm = clean_xy(
            get_first(row, [("treadmill", "time_elapsed_s"), ("encoder", "time_elapsed_s")]),
            get_first(row, [("treadmill", "speed_mm"), ("encoder", "speed")]),
        )
        return float(np.nanstd(spd_mm) / 10.0)


@dataclass(frozen=True)
class TotalDistanceM(FeatureFn):
    """Total distance traveled in meters during the session."""

    name: str = "distance_m"
    label: str = "Distance (m)"
    color: str = SOURCE_COLOR
    source: str = "treadmill"
    plotter: str = "boxplot"

    def _run_impl(self, row) -> float:
        dist_mm = as_1d(get_first(row, [("treadmill", "distance_mm"), ("encoder", "distance")]))
        dist_mm = dist_mm[np.isfinite(dist_mm)]
        return float((dist_mm[-1] - dist_mm[0]) / 1000.0)


@dataclass(frozen=True)
class LocomotionBoutFeature(FeatureFn):
    """Base class for locomotion-bout features with shared thresholds."""

    color: str = SOURCE_COLOR
    source: str = "treadmill"
    plotter: str = "boxplot"
    min_speed_cms: float = 5
    min_duration_s: float = 1.5
    merge_gap_s: float = 1.5

    def _bout_params(self) -> dict[str, float]:
        return {
            "min_speed_cms": self.min_speed_cms,
            "min_duration_s": self.min_duration_s,
            "merge_gap_s": self.merge_gap_s,
        }

    def _get_bouts(self, row):
        t, spd_mm = clean_xy(
            get_first(row, [("treadmill", "time_elapsed_s"), ("encoder", "time_elapsed_s")]),
            get_first(row, [("treadmill", "speed_mm"), ("encoder", "speed")]),
        )
        speed_cms = spd_mm / 10.0
        return locomotion_bout_events(t, speed_cms, **self._bout_params())


@dataclass(frozen=True)
class LocomotionBoutsCount(LocomotionBoutFeature):
    """Count of locomotion bouts per session."""

    name: str = "locomotion_bouts_n"
    label: str = "Locomotion bouts (n)"

    def _run_impl(self, row) -> float:
        epochs = self._get_bouts(row)
        return float(len(epochs))


@dataclass(frozen=True)
class LocomotionBoutSpeedMeanCMS(LocomotionBoutFeature):
    """Mean speed across locomotion bouts in cm/s."""

    name: str = "locomotion_bout_speed_mean_cms"
    label: str = "Bout speed (cm/s)"

    def _run_impl(self, row) -> float:
        epochs = self._get_bouts(row)
        return _nanmean_no_warn(epochs["mean_speed_cms"]) if not epochs.empty else float("nan")


@dataclass(frozen=True)
class LocomotionBoutDistanceM(LocomotionBoutFeature):
    """Mean distance across locomotion bouts in meters."""

    name: str = "locomotion_bout_distance_m"
    label: str = "Bout distance (m)"

    def _run_impl(self, row) -> float:
        epochs = self._get_bouts(row)
        return _nanmean_no_warn(epochs["distance_m"]) if not epochs.empty else float("nan")


@dataclass(frozen=True)
class LocomotionBoutDurationS(LocomotionBoutFeature):
    """Mean duration across locomotion bouts in seconds."""

    name: str = "locomotion_bout_duration_s"
    label: str = "Bout duration (s)"

    def _run_impl(self, row) -> float:
        epochs = self._get_bouts(row)
        return _nanmean_no_warn(epochs["duration_s"]) if not epochs.empty else float("nan")
