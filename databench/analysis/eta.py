"""Event-triggered average analysis — public API.

Usage::

    from databench import EtaAnalysis
    from databench._signal.events import locomotion_events

    events = locomotion_events(group, min_speed_cms=0.5)
    eta = EtaAnalysis(
        roi_columns=("L_VISp", "R_VISp", "L_MOs"),
        window=(-2.0, 5.0),
        baseline=(-2.0, -1.0),
    )
    result = eta.run(group, events)
    result.plot(event="onset").save("eta_onset.svg")
    result.save_tables(prefix="eta")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from databench._signal.eta_core import eta_baselined
from databench.config import OutputContext


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_sem(x: pd.Series) -> float:
    """Standard error of the mean, returning 0.0 for n ≤ 1."""
    vals = x.dropna().to_numpy(dtype=float)
    n = vals.size
    if n <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(n))


# ── EtaAnalysis ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EtaAnalysis:
    """Configure and run event-triggered averages across a SessionGroup.

    This class is **event-source agnostic** — it does not detect events
    itself.  Instead, pass a pre-computed events table (a
    :class:`~pandas.DataFrame` with columns ``Subject``, ``Session``,
    ``Task``, ``EventType``, ``event_time``) to :meth:`run`.

    Use helper functions like
    :func:`~databench._signal.events.locomotion_events` to produce events
    from specific detectors.

    Parameters
    ----------
    roi_columns : tuple of str
        Signal names to extract peri-event epochs from.
    window : (float, float)
        Peri-event window in seconds.
    dt : float
        Interpolation step size.
    baseline : (float, float)
        Baseline subtraction window.
    source : str
        Data source for ROI columns.
    reference_source : str
        Reference source for time alignment.
    time_column : str
        Time column name used for alignment and epoch extraction.
    alignment_tolerance_s : float
        Tolerance for ``merge_asof`` alignment.
    """

    roi_columns: Tuple[str, ...] = ()
    window: Tuple[float, float] = (-2.0, 5.0)
    dt: float = 0.05
    baseline: Tuple[float, float] = (-2.0, -1.0)

    # Data access details
    source: str = "mesomap"
    reference_source: str = "mesomap"
    time_column: str = "time_elapsed_s"
    alignment_tolerance_s: float = 0.05

    def run(
        self,
        sessions: "SessionGroup",
        events: "pd.DataFrame",
        *,
        aligned: list["AlignedData"] | None = None,
        condition_map: Mapping[str, str] | None = None,
    ) -> "EtaResult":
        """Run the ETA analysis across all sessions.

        Parameters
        ----------
        sessions : SessionGroup
            Sessions to analyze.
        events : pd.DataFrame
            Events table with at least ``Subject``, ``Session``, ``Task``,
            ``EventType``, ``event_time``.  An optional ``Condition`` column
            overrides *condition_map*.  Produce this with
            :func:`~databench._signal.events.locomotion_events`,
            :func:`~databench._signal.events.make_events`, or any custom
            function.
        aligned : list of AlignedData, optional
            Pre-computed aligned data for each session.  When ``None``,
            alignment is computed automatically using ``session.align()``.
        condition_map : dict, optional
            Maps task name (or ``"subject,session,task"`` key) to condition
            label.  Only used for sessions whose events lack a ``Condition``
            column.

        Returns
        -------
        EtaResult
        """
        from databench.session import AlignedData

        if not self.roi_columns:
            raise ValueError("roi_columns cannot be empty")

        if not isinstance(events, pd.DataFrame):
            raise TypeError(
                f"'events' must be a pandas DataFrame, got {type(events).__name__}. "
                "Use locomotion_events() or make_events() to produce one."
            )

        # Validate events schema
        required = {"Subject", "Session", "Task", "EventType", "event_time"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(
                f"Events table is missing required columns: {missing}. "
                f"Expected: {sorted(required)}"
            )

        has_condition = "Condition" in events.columns

        all_events: list[pd.DataFrame] = []
        all_eta: list[pd.DataFrame] = []

        for i, sess in enumerate(sessions):
            # Filter events for this session
            sess_mask = (
                (events["Subject"] == sess.subject)
                & (events["Session"] == sess.session)
                & (events["Task"] == sess.task)
            )
            sess_events = events.loc[sess_mask].copy()
            if sess_events.empty:
                continue

            # Fill Condition column if missing
            if not has_condition:
                if condition_map is not None:
                    key = f"{sess.subject},{sess.session},{sess.task}"
                    cond = condition_map.get(key, condition_map.get(sess.task, "all"))
                else:
                    cond = "all"
                sess_events["Condition"] = cond

            all_events.append(sess_events)

            # Align if needed
            if aligned is not None:
                ad = aligned[i]
            else:
                sources_dict: dict[str, list[str]] = {
                    self.source: list(self.roi_columns),
                }
                ad = sess.align(
                    sources_dict,
                    reference=self.reference_source,
                    tolerance_s=self.alignment_tolerance_s,
                    time_column=self.time_column,
                )

            df = ad.df

            # Compute ETA for each event type present in this session
            for etype, etype_group in sess_events.groupby("EventType", sort=False):
                event_times = etype_group["event_time"].to_numpy()
                if len(event_times) == 0:
                    continue

                eta_df = eta_baselined(
                    df,
                    event_times,
                    list(self.roi_columns),
                    time_column=self.time_column,
                    window=self.window,
                    dt=self.dt,
                    baseline=self.baseline,
                )
                if eta_df.empty:
                    continue

                cond_val = etype_group["Condition"].iloc[0] if "Condition" in etype_group.columns else "all"
                eta_df["Subject"] = sess.subject
                eta_df["Session"] = sess.session
                eta_df["Task"] = sess.task
                eta_df["Condition"] = cond_val
                eta_df["EventType"] = etype
                all_eta.append(eta_df)

        # Combine
        base_cols = [
            "Subject", "Session", "Task", "Condition", "EventType",
            "event_id", "rel_time", "ROI", "value",
        ]
        if all_eta:
            eta_events = pd.concat(all_eta, ignore_index=True)
        else:
            eta_events = pd.DataFrame(columns=base_cols)

        if all_events:
            combined_events = pd.concat(all_events, ignore_index=True)
        else:
            combined_events = pd.DataFrame(
                columns=["Subject", "Session", "Task", "Condition", "EventType", "event_time"]
            )

        # Infer event types from what was actually found
        event_types = sorted(combined_events["EventType"].unique().tolist()) if not combined_events.empty else []

        # Aggregate: subject-level means
        subject_means = self._aggregate_subject(eta_events)

        # Aggregate: group means ± SEM
        group_means = self._aggregate_group(subject_means)

        return EtaResult(
            events=combined_events,
            eta_events=eta_events,
            subject_means=subject_means,
            group_means=group_means,
            roi_columns=list(self.roi_columns),
            event_types=event_types,
            window=self.window,
            baseline=self.baseline,
            _context=sessions[0]._context if len(sessions) > 0 else None,
        )

    # ── Aggregation helpers ────────────────────────────────────────────────

    @staticmethod
    def _aggregate_subject(eta_events: pd.DataFrame) -> pd.DataFrame:
        """Per-subject mean across events: mean(event_id) per (Subject, Condition, EventType, ROI, rel_time)."""
        if eta_events.empty:
            return pd.DataFrame(
                columns=["Subject", "Condition", "EventType", "ROI", "rel_time", "mean"]
            )
        return (
            eta_events
            .groupby(["Subject", "Condition", "EventType", "ROI", "rel_time"], sort=False)
            .agg(mean=("value", "mean"))
            .reset_index()
        )

    @staticmethod
    def _aggregate_group(subject_means: pd.DataFrame) -> pd.DataFrame:
        """Group mean ± SEM across subjects."""
        if subject_means.empty:
            return pd.DataFrame(
                columns=["Condition", "EventType", "ROI", "rel_time", "mean", "sem"]
            )
        return (
            subject_means
            .groupby(["Condition", "EventType", "ROI", "rel_time"], sort=False)
            .agg(mean=("mean", "mean"), sem=("mean", _safe_sem))
            .reset_index()
        )


# ── EtaResult ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EtaResult:
    """Result of an ETA analysis.

    Attributes
    ----------
    events : DataFrame
        Bout event table with ``Subject``, ``Session``, ``Task``,
        ``Condition``, ``EventType``, ``event_time``.
    eta_events : DataFrame
        Per-event ETA traces (columns: ``event_id``, ``rel_time``,
        ``ROI``, ``value``, plus session/condition columns).
    subject_means : DataFrame
        Mean per (Subject, Condition, EventType, ROI, rel_time).
    group_means : DataFrame
        Group mean ± SEM per (Condition, EventType, ROI, rel_time).
    roi_columns : list of str
    event_types : list of str
    window, baseline : tuple of floats
    """

    events: pd.DataFrame
    eta_events: pd.DataFrame
    subject_means: pd.DataFrame
    group_means: pd.DataFrame
    roi_columns: list[str]
    event_types: list[str]
    window: Tuple[float, float]
    baseline: Tuple[float, float]

    _context: OutputContext | None = field(repr=False, default=None)

    # ── Plotting ───────────────────────────────────────────────────────────

    def plot(
        self,
        event: str = "onset",
        *,
        rois: list[str] | None = None,
        conditions: list[str] | None = None,
        condition_colors: Dict[str, str] | None = None,
        ncols: int = 2,
        task: str | None = None,
    ) -> "SaveableFigure":
        """Plot group-mean ± SEM ETA traces.

        Parameters
        ----------
        event : str
            Event type to display (e.g. ``"onset"``).
        rois : list of str, optional
            ROIs to include. Defaults to all ``roi_columns``.
        conditions : list of str, optional
            Condition ordering.
        condition_colors : dict, optional
            Mapping from condition → color string.
        ncols : int
            Number of subplot columns.
        task : str, optional
            Filter group_means to a specific task.

        Returns
        -------
        SaveableFigure
        """
        from databench._plotting.eta import plot_eta_by_condition
        from databench.session import SaveableFigure

        rois_to_plot = rois if rois is not None else self.roi_columns

        fig = plot_eta_by_condition(
            self.group_means,
            event=event,
            rois=rois_to_plot,
            conditions=conditions,
            condition_colors=condition_colors,
            baseline_s=self.baseline,
            ncols=ncols,
            task=task,
        )
        return SaveableFigure(fig, self._context)

    # ── Saving ─────────────────────────────────────────────────────────────

    def save_tables(self, prefix: str = "eta") -> dict[str, Path]:
        """Save the main DataFrames to CSV.

        Returns
        -------
        dict
            Mapping from table name to saved file path.
        """
        if self._context is None:
            raise RuntimeError("Cannot save — no output context available.")
        out_dir = self._context.stats_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}
        for name, df in [
            ("events", self.events),
            ("eta_events", self.eta_events),
            ("subject_means", self.subject_means),
            ("group_means", self.group_means),
        ]:
            p = out_dir / f"{prefix}_{name}.csv"
            df.to_csv(p, index=False)
            paths[name] = p
        return paths

    def save_summary(self, name: str = "eta_summary.json") -> Path:
        """Save a JSON summary of the analysis."""
        if self._context is None:
            raise RuntimeError("Cannot save — no output context available.")
        out_dir = self._context.run_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name

        n_subj = int(self.events["Subject"].nunique()) if not self.events.empty else 0
        conds = sorted(self.events["Condition"].unique().tolist()) if not self.events.empty else []

        summary = {
            "analysis": "event_triggered_average",
            "created_at": datetime.now().isoformat(),
            "analyst": self._context.analyst,
            "lab": self._context.lab,
            "run_name": self._context.run_name,
            "tag": self._context.tag,
            "roi_columns": self.roi_columns,
            "event_types": self.event_types,
            "window": list(self.window),
            "baseline": list(self.baseline),
            "n_events": int(len(self.events)),
            "n_subjects": n_subj,
            "conditions": conds,
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path

    # ── Reporting ──────────────────────────────────────────────────────────

    def _report_section(self) -> "ReportSection":
        """Build a :class:`~databench._reporting.ReportSection` for this result."""
        from databench._reporting import ReportSection

        n_subj = int(self.events["Subject"].nunique()) if not self.events.empty else 0
        conds = sorted(self.events["Condition"].unique().tolist()) if not self.events.empty else []
        notes = (
            f"**{len(self.events)} events** across **{n_subj} subjects**.\n\n"
            f"Conditions: {', '.join(conds) if conds else 'all'}.\n\n"
            f"ROIs: {', '.join(self.roi_columns)}."
        )
        params = {
            "roi_columns": self.roi_columns,
            "event_types": self.event_types,
            "window": self.window,
            "baseline": self.baseline,
            "n_events": len(self.events),
            "n_subjects": n_subj,
        }
        figures = []
        tables = []
        if self._context is not None:
            figures = sorted(self._context.plots_dir.glob("*.svg")) + sorted(self._context.plots_dir.glob("*.png"))
            tables = sorted(self._context.stats_dir.glob("*.csv"))
        return ReportSection(
            heading="Event-Triggered Average",
            params=params,
            notes=notes,
            figures=figures,
            tables=tables,
        )
