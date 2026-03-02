"""Shared event-triggered average (ETA) analysis utilities.

Provides the core ``eta_baselined`` function and three Analysis classes:

* :class:`EtaByConditionAnalysis` — condition-based ETA with subject → group aggregation.
* :class:`EtaLongitudinalAnalysis` — longitudinal day-by-day ETA with pooled and metric aggregation.
* :class:`EtaPrePostDiffAnalysis` — scalar pre/post-event difference analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from databench.analysis.base import Analysis, AnalysisResult
from databench.features.treadmill import (
    LocomotionBoutEventsExtractor,
    extract_epoch_interpolated,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_sem(x: pd.Series) -> float:
    """Standard error of the mean, returning 0.0 for n <= 1."""
    vals = x.dropna().to_numpy(dtype=float)
    n = vals.size
    if n <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(n))


def _session_to_day(session: str) -> int:
    """Parse session string to an integer day number (e.g. ``'ses-03'`` → 3)."""
    digits = "".join(ch for ch in str(session) if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse session number from {session!r}")
    return int(digits)


# ---------------------------------------------------------------------------
# Core ETA function
# ---------------------------------------------------------------------------

def eta_baselined(
    df: pd.DataFrame,
    event_times: np.ndarray,
    roi_columns: Sequence[str],
    *,
    time_column: str = "time_elapsed_s",
    window: tuple[float, float] = (-2.0, 2.0),
    dt: float = 0.05,
    baseline: tuple[float, float] = (-2.0, -1.0),
    bout_intervals: Optional[np.ndarray] = None,
    exclude_events_in_bouts: bool = True,
    baseline_exclude_bouts: bool = False,
    min_clean_baseline_points: int = 3,
    fallback_to_full_baseline: bool = True,
) -> pd.DataFrame:
    """Compute baseline-subtracted ETA traces for each event × ROI.

    Parameters
    ----------
    df : DataFrame
        Long-format table containing at least *time_column* and all
        *roi_columns*.
    event_times : array
        1-D array of event timestamps.
    roi_columns : sequence of str
        Column names to extract per-event epochs from.
    time_column : str
        Name of the time column in *df*.
    window : (float, float)
        Start and end of the peri-event window in seconds.
    dt : float
        Sampling step for the interpolation grid.
    baseline : (float, float)
        Start and end of the baseline window within *window*, used for
        subtraction.
    bout_intervals : ndarray, optional
        ``(N, 2)`` array of ``[onset, offset]`` times.
    exclude_events_in_bouts : bool
        When *True* (default) and *bout_intervals* is given, events whose
        timestamp falls inside any bout interval are **dropped** from output.
    baseline_exclude_bouts : bool
        When *True* and *bout_intervals* is given, baseline timepoints that
        overlap with any bout interval are **masked** before computing the
        baseline mean.  Useful for longitudinal designs where bouts should
        not contaminate the baseline window.
    min_clean_baseline_points : int
        Minimum number of finite, non-bout baseline points required when
        *baseline_exclude_bouts* is *True*.  If fewer are available, the
        function falls back to the full baseline (if *fallback_to_full_baseline*
        is *True*) or uses whatever is available.
    fallback_to_full_baseline : bool
        When *True* and the cleaned baseline has fewer than
        *min_clean_baseline_points*, use the full (unmasked) baseline window.

    Returns
    -------
    DataFrame
        Columns: ``event_id``, ``rel_time``, ``ROI``, ``value``.
    """
    df = df.sort_values(time_column)
    time_values = df[time_column].to_numpy()

    # Optionally filter events that fall inside bout intervals
    if exclude_events_in_bouts and bout_intervals is not None and len(bout_intervals) > 0:
        keep = np.ones(len(event_times), dtype=bool)
        for i, et in enumerate(event_times):
            for onset, offset in bout_intervals:
                if onset <= et <= offset:
                    keep[i] = False
                    break
        event_times = event_times[keep]

    output_frames: list[pd.DataFrame] = []
    for event_id, event_time in enumerate(event_times):
        for roi in roi_columns:
            roi_values = df[roi].to_numpy()
            rel_t, roi_epoch = extract_epoch_interpolated(
                time_values, roi_values, event_time, window=window, dt=dt,
            )
            if rel_t is None:
                continue

            baseline_mask_full = (rel_t >= baseline[0]) & (rel_t <= baseline[1])
            baseline_mask = baseline_mask_full.copy()

            # Mask baseline timepoints overlapping bouts
            if baseline_exclude_bouts and bout_intervals is not None and len(bout_intervals) > 0:
                abs_t = event_time + rel_t
                in_bout = np.zeros(abs_t.shape, dtype=bool)
                for onset_t, offset_t in bout_intervals:
                    in_bout |= (abs_t >= float(onset_t)) & (abs_t <= float(offset_t))
                baseline_mask = baseline_mask & ~in_bout

                clean_count = int(np.sum(baseline_mask & np.isfinite(roi_epoch)))
                if clean_count < min_clean_baseline_points and fallback_to_full_baseline:
                    baseline_mask = baseline_mask_full

            baseline_value = (
                np.nanmean(roi_epoch[baseline_mask])
                if baseline_mask.any()
                else np.nan
            )
            roi_epoch = roi_epoch - baseline_value

            output_frames.append(
                pd.DataFrame(
                    {
                        "event_id": event_id,
                        "rel_time": rel_t,
                        "ROI": roi,
                        "value": roi_epoch,
                    }
                )
            )

    if not output_frames:
        return pd.DataFrame(columns=["event_id", "rel_time", "ROI", "value"])
    return pd.concat(output_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Analysis: condition-based ETA
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EtaByConditionAnalysis(Analysis):
    """Event-triggered average grouped by experimental condition.

    Produces four DataFrames:

    * ``events`` — extracted locomotion bout table
    * ``eta_events`` — per-event ETA traces
    * ``eta_subj`` — mean per (Subject, Condition, EventType, ROI, rel_time)
    * ``eta_group`` — group mean ± SEM over subjects
    """

    name: str = "eta_by_condition"
    roi_cols: tuple[str, ...] = ()
    task: str = "task-spont"
    event_types: tuple[str, ...] = ("onset", "offset")
    window: tuple[float, float] = (-1.0, 3.0)
    dt: float = 0.05
    baseline: tuple[float, float] = (-5.0, 0.0)
    bout_events_extractor: Any = LocomotionBoutEventsExtractor()

    def run(self, long: pd.DataFrame) -> AnalysisResult:
        roi_cols = list(self.roi_cols)
        required = ["Subject", "Session", "Task", "Condition", "time_elapsed_s", "speed_mm", *roi_cols]
        missing = [c for c in required if c not in long.columns]
        if missing:
            raise ValueError(f"Missing required columns in long table: {', '.join(missing)}")

        events = self.bout_events_extractor(long)
        if not isinstance(events, pd.DataFrame):
            raise TypeError("bout_events_extractor must return a pandas DataFrame.")
        required_event_cols = {"Subject", "Session", "Task", "onset_t", "offset_t"}
        missing_event_cols = sorted(required_event_cols.difference(events.columns))
        if missing_event_cols:
            raise ValueError(f"Extracted event table missing columns: {', '.join(missing_event_cols)}")
        events = events.loc[events["Task"] == self.task].copy()

        event_map = {
            (subj, ses, task_name): {
                "onset": grp["onset_t"].to_numpy(),
                "offset": grp["offset_t"].to_numpy(),
            }
            for (subj, ses, task_name), grp in events.groupby(
                ["Subject", "Session", "Task"], sort=False
            )
        }

        rows = []
        for (subj, ses, task_name), g in long.groupby(
            ["Subject", "Session", "Task"], sort=False
        ):
            if task_name != self.task:
                continue
            transitions = event_map.get(
                (subj, ses, task_name),
                {"onset": np.array([]), "offset": np.array([])},
            )
            for event_type in self.event_types:
                event_times = transitions.get(event_type, np.array([]))
                if event_times.size == 0:
                    continue
                eta = eta_baselined(
                    g,
                    event_times,
                    roi_cols,
                    window=self.window,
                    dt=self.dt,
                    baseline=self.baseline,
                )
                if eta.empty:
                    continue
                eta.insert(0, "EventType", event_type)
                eta.insert(0, "Task", task_name)
                eta.insert(0, "Session", ses)
                eta.insert(0, "Subject", subj)
                eta["Condition"] = g["Condition"].iloc[0]
                rows.append(eta)

        eta_events = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

        if eta_events.empty:
            return AnalysisResult(
                name=self.name,
                data={
                    "events": events,
                    "eta_events": eta_events,
                    "eta_subj": pd.DataFrame(),
                    "eta_group": pd.DataFrame(),
                },
                meta={"task": self.task, "baseline": self.baseline, "window": self.window, "dt": self.dt},
            )

        eta_subj = (
            eta_events.groupby(
                ["Subject", "Condition", "Task", "EventType", "ROI", "rel_time"],
                as_index=False,
            )
            .agg(value=("value", "mean"))
        )

        eta_group = (
            eta_subj.groupby(
                ["Condition", "Task", "EventType", "ROI", "rel_time"],
                as_index=False,
            )
            .agg(
                mean=("value", "mean"),
                sem=("value", lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum())),
            )
        )

        return AnalysisResult(
            name=self.name,
            data={
                "events": events,
                "eta_events": eta_events,
                "eta_subj": eta_subj,
                "eta_group": eta_group,
            },
            meta={"task": self.task, "baseline": self.baseline, "window": self.window, "dt": self.dt},
        )


# ---------------------------------------------------------------------------
# Analysis: general-purpose event-triggered average
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventTriggeredAverageAnalysis(Analysis):
    """Event-triggered average from an externally-supplied events table.

    Unlike :class:`EtaByConditionAnalysis`, which extracts locomotion bouts
    internally, this analysis accepts *any* events DataFrame with at least
    ``Subject``, ``Session``, ``Task``, ``onset_t``, ``offset_t`` columns.

    Produces three DataFrames:

    * ``eta_events``  — per-event ETA traces
    * ``eta_session`` — mean per (Subject, Session, EventType, ROI, rel_time)
    * ``eta_group``   — group mean ± SEM across sessions

    Usage::

        analysis = EventTriggeredAverageAnalysis(
            roi_cols=("L_VISp", "pupil_diameter_mm"),
            task="task-widefield",
            window=(-2.0, 3.0),
            dt=0.02,
            baseline=(-2.0, -0.5),
            events=osc_events,
        )
        bench.analyze(analysis)   # auto-supplies bench.long
    """

    name: str = "event_triggered_average"
    roi_cols: tuple[str, ...] = ()
    task: str = ""
    event_types: tuple[str, ...] = ("onset", "offset")
    window: tuple[float, float] = (-2.0, 3.0)
    dt: float = 0.02
    baseline: tuple[float, float] = (-2.0, -0.5)
    events: Optional[pd.DataFrame] = None

    def run(
        self,
        long: pd.DataFrame,
    ) -> AnalysisResult:
        if self.events is None or self.events.empty:
            raise ValueError(
                "EventTriggeredAverageAnalysis requires an events DataFrame. "
                "Pass it at construction: EventTriggeredAverageAnalysis(events=df)."
            )
        events = self.events
        roi_cols = list(self.roi_cols)
        task = self.task

        # Build per-session event map from the supplied events table
        event_map: dict[tuple, dict[str, np.ndarray]] = {}
        for (subj, ses, task_name), grp in events.groupby(
            ["Subject", "Session", "Task"], sort=False,
        ):
            event_map[(subj, ses, task_name)] = {
                "onset": grp["onset_t"].to_numpy(),
                "offset": grp["offset_t"].to_numpy(),
            }

        rows: list[pd.DataFrame] = []
        for (subj, ses, task_name), g in long.groupby(
            ["Subject", "Session", "Task"], sort=False,
        ):
            if task and task_name != task:
                continue
            transitions = event_map.get((subj, ses, task_name))
            if transitions is None:
                continue
            for event_type in self.event_types:
                event_times = transitions.get(event_type, np.array([]))
                if event_times.size == 0:
                    continue
                eta = eta_baselined(
                    g,
                    event_times,
                    roi_cols,
                    window=self.window,
                    dt=self.dt,
                    baseline=self.baseline,
                )
                if eta.empty:
                    continue
                eta.insert(0, "EventType", event_type)
                eta.insert(0, "Task", task_name)
                eta.insert(0, "Session", ses)
                eta.insert(0, "Subject", subj)
                rows.append(eta)

        eta_events = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

        if eta_events.empty:
            return AnalysisResult(
                name=self.name,
                data={
                    "eta_events": eta_events,
                    "eta_session": pd.DataFrame(),
                    "eta_group": pd.DataFrame(),
                },
                meta={"task": task, "window": self.window, "dt": self.dt, "baseline": self.baseline},
            )

        eta_session = (
            eta_events.groupby(
                ["Subject", "Session", "Task", "EventType", "ROI", "rel_time"],
                as_index=False,
            )
            .agg(value=("value", "mean"))
        )

        eta_group = (
            eta_session.groupby(
                ["Task", "EventType", "ROI", "rel_time"],
                as_index=False,
            )
            .agg(
                mean=("value", "mean"),
                sem=("value", _safe_sem),
                n_sessions=("Session", "nunique"),
                n_subjects=("Subject", "nunique"),
            )
        )

        return AnalysisResult(
            name=self.name,
            data={
                "eta_events": eta_events,
                "eta_session": eta_session,
                "eta_group": eta_group,
            },
            meta={"task": task, "window": self.window, "dt": self.dt, "baseline": self.baseline},
        )


# ---------------------------------------------------------------------------
# Analysis: longitudinal day-by-day ETA
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EtaLongitudinalAnalysis(Analysis):
    """Longitudinal ETA analysis across sessions (days).

    Produces seven DataFrames:

    * ``events`` — bout events
    * ``eta_events`` — per-event traces
    * ``eta_subj_day`` — per (Subject, Session, day_n, EventType, ROI, rel_time) mean
    * ``eta_group`` — pooled across all days
    * ``eta_longitudinal`` — per (day_n, EventType, ROI, rel_time) mean ± SEM
    * ``eta_day_metric`` — subject-level scalar metric in *metric_window*
    * ``eta_day_metric_group`` — group-averaged metric over days
    """

    name: str = "eta_widefield_longitudinal"
    roi_cols: tuple[str, ...] = ()
    task: str = "task-widefield"
    event_types: tuple[str, ...] = ("onset", "offset")
    window: tuple[float, float] = (-1.0, 3.0)
    dt: float = 0.05
    baseline: tuple[float, float] = (-5.0, 0.0)
    baseline_exclude_bouts: bool = True
    min_clean_baseline_points: int = 3
    fallback_to_full_baseline: bool = True
    metric_window: tuple[float, float] = (0.0, 1.0)
    bout_events_extractor: Any = LocomotionBoutEventsExtractor()

    def run(self, long: pd.DataFrame) -> AnalysisResult:
        roi_cols = list(self.roi_cols)
        required = ["Subject", "Session", "Task", "time_elapsed_s", "speed_mm", *roi_cols]
        missing = [c for c in required if c not in long.columns]
        if missing:
            raise ValueError(f"Missing required columns in long table: {', '.join(missing)}")

        d = long.loc[long["Task"] == self.task].copy()
        if d.empty:
            raise ValueError(f"No rows found for task={self.task!r}.")

        d["day_n"] = d["Session"].map(_session_to_day)

        events = self.bout_events_extractor(d)
        if not isinstance(events, pd.DataFrame):
            raise TypeError("bout_events_extractor must return a pandas DataFrame.")

        required_event_cols = {"Subject", "Session", "Task", "onset_t", "offset_t"}
        missing_event_cols = sorted(required_event_cols.difference(events.columns))
        if missing_event_cols:
            raise ValueError(f"Extracted event table missing columns: {', '.join(missing_event_cols)}")

        events = events.loc[events["Task"] == self.task].copy()
        if not events.empty:
            events["day_n"] = events["Session"].map(_session_to_day)

        event_map = {
            (subj, ses, task_name): {
                "onset": grp["onset_t"].to_numpy(),
                "offset": grp["offset_t"].to_numpy(),
                "intervals": grp[["onset_t", "offset_t"]].to_numpy(dtype=float),
            }
            for (subj, ses, task_name), grp in events.groupby(
                ["Subject", "Session", "Task"], sort=False
            )
        }

        rows = []
        for (subj, ses, task_name), g in d.groupby(
            ["Subject", "Session", "Task"], sort=False
        ):
            transitions = event_map.get(
                (subj, ses, task_name),
                {"onset": np.array([]), "offset": np.array([])},
            )
            day_n = _session_to_day(ses)
            bout_intervals = transitions.get("intervals", np.empty((0, 2), dtype=float))

            for event_type in self.event_types:
                event_times = transitions.get(event_type, np.array([]))
                if event_times.size == 0:
                    continue

                eta = eta_baselined(
                    g,
                    event_times,
                    roi_cols,
                    window=self.window,
                    dt=self.dt,
                    baseline=self.baseline,
                    bout_intervals=bout_intervals,
                    exclude_events_in_bouts=False,
                    baseline_exclude_bouts=self.baseline_exclude_bouts,
                    min_clean_baseline_points=self.min_clean_baseline_points,
                    fallback_to_full_baseline=self.fallback_to_full_baseline,
                )
                if eta.empty:
                    continue

                eta.insert(0, "EventType", event_type)
                eta.insert(0, "day_n", day_n)
                eta.insert(0, "Task", task_name)
                eta.insert(0, "Session", ses)
                eta.insert(0, "Subject", subj)
                rows.append(eta)

        eta_events = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

        empty_result = AnalysisResult(
            name=self.name,
            data={
                "events": events,
                "eta_events": eta_events,
                "eta_subj_day": pd.DataFrame(),
                "eta_group": pd.DataFrame(),
                "eta_longitudinal": pd.DataFrame(),
                "eta_day_metric": pd.DataFrame(),
                "eta_day_metric_group": pd.DataFrame(),
            },
            meta={
                "task": self.task,
                "baseline": self.baseline,
                "baseline_exclude_bouts": self.baseline_exclude_bouts,
                "min_clean_baseline_points": self.min_clean_baseline_points,
                "fallback_to_full_baseline": self.fallback_to_full_baseline,
                "window": self.window,
                "metric_window": self.metric_window,
                "dt": self.dt,
            },
        )
        if eta_events.empty:
            return empty_result

        eta_subj_day = (
            eta_events.groupby(
                ["Subject", "Session", "day_n", "Task", "EventType", "ROI", "rel_time"],
                as_index=False,
            )
            .agg(value=("value", "mean"))
        )

        eta_group = (
            eta_subj_day.groupby(
                ["Task", "EventType", "ROI", "rel_time"], as_index=False
            )
            .agg(mean=("value", "mean"), sem=("value", _safe_sem), n_subjects=("Subject", "nunique"))
        )

        eta_longitudinal = (
            eta_subj_day.groupby(
                ["day_n", "Task", "EventType", "ROI", "rel_time"], as_index=False
            )
            .agg(mean=("value", "mean"), sem=("value", _safe_sem), n_subjects=("Subject", "nunique"))
            .sort_values(["day_n", "EventType", "ROI", "rel_time"])
        )

        metric_mask = (
            (eta_subj_day["rel_time"] >= self.metric_window[0])
            & (eta_subj_day["rel_time"] <= self.metric_window[1])
        )
        eta_day_metric = (
            eta_subj_day.loc[metric_mask]
            .groupby(
                ["Subject", "Session", "day_n", "Task", "EventType", "ROI"],
                as_index=False,
            )
            .agg(metric_value=("value", "mean"))
        )
        eta_day_metric_group = (
            eta_day_metric.groupby(
                ["day_n", "Task", "EventType", "ROI"], as_index=False
            )
            .agg(
                mean=("metric_value", "mean"),
                sem=("metric_value", _safe_sem),
                n_subjects=("Subject", "nunique"),
            )
            .sort_values(["day_n", "EventType", "ROI"])
        )

        return AnalysisResult(
            name=self.name,
            data={
                "events": events,
                "eta_events": eta_events,
                "eta_subj_day": eta_subj_day,
                "eta_group": eta_group,
                "eta_longitudinal": eta_longitudinal,
                "eta_day_metric": eta_day_metric,
                "eta_day_metric_group": eta_day_metric_group,
            },
            meta={
                "task": self.task,
                "baseline": self.baseline,
                "baseline_exclude_bouts": self.baseline_exclude_bouts,
                "min_clean_baseline_points": self.min_clean_baseline_points,
                "fallback_to_full_baseline": self.fallback_to_full_baseline,
                "window": self.window,
                "metric_window": self.metric_window,
                "dt": self.dt,
            },
        )


# ---------------------------------------------------------------------------
# Analysis: pre/post scalar difference
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EtaPrePostDiffAnalysis(Analysis):
    """Compute scalar pre-vs-post dF/F difference per event.

    Produces four DataFrames:

    * ``events`` — extracted bout events
    * ``event_diff`` — per-event pre/post difference
    * ``subject_diff`` — mean diff per subject
    * ``group_diff`` — group mean ± SEM across subjects
    """

    name: str = "eta_prepost_diff"
    roi_cols: tuple[str, ...] = ("L_MOp", "R_MOp", "L_MOs", "R_MOs")
    task: str = "task-spont"
    condition_col: str = "Condition"
    pre_window: tuple[float, float] = (-1.0, 0.0)
    post_window: tuple[float, float] = (0.0, 1.0)
    dt: float = 0.02
    reverse_for_offset: bool = False
    bout_events_extractor: Any = LocomotionBoutEventsExtractor()

    def run(self, long: pd.DataFrame) -> AnalysisResult:
        roi_cols = list(self.roi_cols)
        required = [
            "Subject", "Session", "Task",
            self.condition_col, "time_elapsed_s", "speed_mm",
            *roi_cols,
        ]
        missing = [c for c in required if c not in long.columns]
        if missing:
            raise ValueError(f"Missing required columns in long table: {', '.join(missing)}")

        d = long.loc[long["Task"] == self.task].copy()
        events = self.bout_events_extractor(d)
        if not isinstance(events, pd.DataFrame):
            raise TypeError("bout_events_extractor must return a pandas DataFrame.")

        required_event_cols = {"Subject", "Session", "Task", "bout_id", "onset_t", "offset_t"}
        missing_event_cols = sorted(required_event_cols.difference(events.columns))
        if missing_event_cols:
            raise ValueError(f"Extracted event table missing columns: {', '.join(missing_event_cols)}")

        key_cols = ["Subject", "Session", "Task"]
        grouped_events = {
            k: g.reset_index(drop=True)
            for k, g in events.groupby(key_cols, sort=False)
        }

        rows: list[dict[str, Any]] = []

        for key, g in d.groupby(key_cols, sort=False):
            subj, ses, task_name = key
            cond = g[self.condition_col].iloc[0]
            ge = grouped_events.get(key)
            if ge is None or ge.empty:
                continue

            g = g.sort_values("time_elapsed_s")
            t = g["time_elapsed_s"].to_numpy()

            for _, event_row in ge.iterrows():
                event_type_to_time = {
                    "onset": float(event_row["onset_t"]),
                    "offset": float(event_row["offset_t"]),
                }
                bout_id = int(event_row["bout_id"])

                for event_type, event_time in event_type_to_time.items():
                    for roi in roi_cols:
                        y = g[roi].to_numpy()
                        rel_t, yy = extract_epoch_interpolated(
                            t, y, event_time,
                            window=(self.pre_window[0], self.post_window[1]),
                            dt=self.dt,
                        )
                        if rel_t is None:
                            continue

                        pre_mask = (rel_t >= self.pre_window[0]) & (rel_t < self.pre_window[1])
                        post_mask = (rel_t >= self.post_window[0]) & (rel_t <= self.post_window[1])
                        if not pre_mask.any() or not post_mask.any():
                            continue

                        pre_mean = float(np.nanmean(yy[pre_mask]))
                        post_mean = float(np.nanmean(yy[post_mask]))
                        if not np.isfinite(pre_mean) or not np.isfinite(post_mean):
                            continue

                        diff = post_mean - pre_mean
                        if self.reverse_for_offset and event_type == "offset":
                            diff = pre_mean - post_mean

                        rows.append({
                            "Subject": subj,
                            "Session": ses,
                            "Task": task_name,
                            self.condition_col: cond,
                            "EventType": event_type,
                            "ROI": roi,
                            "bout_id": bout_id,
                            "event_time": event_time,
                            "pre_mean": pre_mean,
                            "post_mean": post_mean,
                            "diff": float(diff),
                        })

        event_diff = pd.DataFrame(rows)
        if event_diff.empty:
            return AnalysisResult(
                name=self.name,
                data={
                    "events": events,
                    "event_diff": event_diff,
                    "subject_diff": pd.DataFrame(),
                    "group_diff": pd.DataFrame(),
                },
                meta={
                    "task": self.task,
                    "pre_window": self.pre_window,
                    "post_window": self.post_window,
                    "reverse_for_offset": self.reverse_for_offset,
                    "dt": self.dt,
                    "condition_col": self.condition_col,
                },
            )

        subject_diff = (
            event_diff.groupby(
                ["Subject", self.condition_col, "Task", "EventType", "ROI"],
                as_index=False,
            )
            .agg(diff_mean=("diff", "mean"), n_events=("diff", "size"))
        )

        group_diff = (
            subject_diff.groupby(
                [self.condition_col, "Task", "EventType", "ROI"],
                as_index=False,
            )
            .agg(
                mean=("diff_mean", "mean"),
                sem=("diff_mean", lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum())),
                n_subjects=("Subject", "nunique"),
            )
        )

        return AnalysisResult(
            name=self.name,
            data={
                "events": events,
                "event_diff": event_diff,
                "subject_diff": subject_diff,
                "group_diff": group_diff,
            },
            meta={
                "task": self.task,
                "pre_window": self.pre_window,
                "post_window": self.post_window,
                "reverse_for_offset": self.reverse_for_offset,
                "dt": self.dt,
                "condition_col": self.condition_col,
            },
        )
