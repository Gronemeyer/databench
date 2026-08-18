"""Event-triggered average analysis — public API.

Usage::

    from databench import EtaAnalysis
    from databench.analysis.locomotion import locomotion_events

    events = locomotion_events(group, min_speed_cms=0.5)
    eta = EtaAnalysis(
        roi_columns=("L_VISp", "R_VISp", "L_MOs"),
        window=(-2.0, 5.0),
        baseline=(-2.0, -1.0),
    )
    result = eta.run(group, events)
    fig = result.plot(event="onset")
    run = proj.run(name="eta")
    run.save_figure(fig, "eta_onset.svg")
    run.save_table(result.events, "events.csv")
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench.signal.epoching import extract_epoch_interpolated
from databench.utils.logger import get_logger, log_run

if TYPE_CHECKING:
    from databench.types import EventsTable

_log = get_logger(__name__)


# ── ETA computation ─────────────────────────

def eta_baselined(
    df: pd.DataFrame,
    event_times: np.ndarray,
    roi_columns: Sequence[str],
    *,
    time_column: str = "time_elapsed_s",
    window: tuple[float, float] = (-2.0, 2.0),
    dt: float = 0.05,
    baseline: Optional[tuple[float, float]] = (-2.0, -1.0),
    baseline_times: Optional[np.ndarray] = None,
    bout_intervals: Optional[np.ndarray] = None,
    exclude_events_in_bouts: bool = True,
    baseline_exclude_bouts: bool = False,
    min_clean_baseline_points: int = 3,
    fallback_to_full_baseline: bool = True,
) -> pd.DataFrame:
    """Compute baseline-subtracted ETA traces for each event \u00d7 ROI.

    Parameters
    ----------
    df : DataFrame
        Long-format table with *time_column* and all *roi_columns*.
    event_times : array
        1-D array of event timestamps.  Epochs are cut around these.
    roi_columns : sequence of str
        Columns to extract per-event epochs from.
    time_column : str
        Name of the time column.
    window : (float, float)
        Peri-event window in seconds.
    dt : float
        Interpolation step size in seconds.
    baseline : (float, float) or None
        Baseline window for subtraction.  ``None`` subtracts nothing and
        returns the epoch in the signal's own units \u2014 use it for a signal
        that already has a meaningful zero (speed, licks, pupil in mm),
        where subtracting a per-event mean would move that zero.
    baseline_times : array, optional
        Times the *baseline* window is measured from, one per event.  By
        default the baseline is taken around the event itself; pass this to
        take it around a different event of the same trial, e.g. epochs cut
        around bout **offset** but referenced to the quiescence before that
        bout's **onset**, so onset- and offset-aligned epochs share one zero.
        Baseline samples are read from the signal directly and may fall
        outside *window*.  ``baseline_exclude_bouts`` applies only to the
        default, in-window baseline.

    Returns
    -------
    DataFrame
        Columns: ``event_id``, ``rel_time``, ``ROI``, ``value``.
    """
    df = df.sort_values(time_column)
    time_values = df[time_column].to_numpy()

    event_times = np.asarray(event_times)
    if baseline_times is not None:
        baseline_times = np.asarray(baseline_times)
        if len(baseline_times) != len(event_times):
            raise ValueError(
                f"baseline_times has {len(baseline_times)} entries for "
                f"{len(event_times)} events; they are paired one to one."
            )

    # Optionally filter events inside bout intervals
    if exclude_events_in_bouts and bout_intervals is not None and len(bout_intervals) > 0:
        keep = np.ones(len(event_times), dtype=bool)
        for i, et in enumerate(event_times):
            for onset, offset in bout_intervals:
                if onset <= et <= offset:
                    keep[i] = False
                    break
        event_times = event_times[keep]
        if baseline_times is not None:
            baseline_times = baseline_times[keep]

    output_frames: list[pd.DataFrame] = []
    for event_id, event_time in enumerate(event_times):
        for roi in roi_columns:
            if roi not in df.columns:
                continue  # ROI not available for this session \u2014 skip
            roi_values = df[roi].to_numpy()
            rel_t, roi_epoch = extract_epoch_interpolated(
                time_values, roi_values, event_time, window=window, dt=dt,
            )
            if rel_t is None:
                continue

            if baseline is None:
                baseline_value = 0.0
            elif baseline_times is not None:
                # Baseline read around its own event, so it can sit outside
                # the epoch window entirely.
                _, bl_epoch = extract_epoch_interpolated(
                    time_values, roi_values, baseline_times[event_id],
                    window=baseline, dt=dt,
                )
                baseline_value = (
                    np.nanmean(bl_epoch)
                    if bl_epoch is not None and np.any(np.isfinite(bl_epoch))
                    else np.nan
                )
            else:
                baseline_mask_full = (rel_t >= baseline[0]) & (rel_t <= baseline[1])
                baseline_mask = baseline_mask_full.copy()

                if baseline_exclude_bouts and bout_intervals is not None and len(bout_intervals) > 0:
                    abs_t = event_time + rel_t
                    in_bout = np.zeros(abs_t.shape, dtype=bool)
                    for onset_t, offset_t in bout_intervals:
                        in_bout |= (abs_t >= float(onset_t)) & (abs_t <= float(offset_t))
                    baseline_mask = baseline_mask & ~in_bout

                    clean_count = int(np.sum(baseline_mask & np.isfinite(roi_epoch)))
                    if clean_count < min_clean_baseline_points and fallback_to_full_baseline:
                        baseline_mask = baseline_mask_full

                bl_samples = roi_epoch[baseline_mask]
                if baseline_mask.any() and np.any(np.isfinite(bl_samples)):
                    baseline_value = np.nanmean(bl_samples)
                else:
                    baseline_value = np.nan
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


# ── ETA plotting (moved from _plotting.eta) ────────────────────────────────

def _mean_sem_line(ax, frame: pd.DataFrame, color: str, label: str | None = None,
                   *, lw: float = 1.0, alpha: float = 0.2) -> None:
    """Draw one mean line with its ± SEM band from a tidy ETA frame."""
    g = frame.sort_values("rel_time")
    if g.empty:
        return
    ax.plot(g["rel_time"], g["mean"], color=color, lw=lw, label=label)
    ax.fill_between(g["rel_time"], g["mean"] - g["sem"], g["mean"] + g["sem"],
                    color=color, alpha=alpha, lw=0)


def plot_eta_traces(
    group_means: pd.DataFrame,
    ax,
    *,
    event: str,
    rois: Sequence[str],
    condition: str | None = None,
    colors: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    lw: float = 1.0,
    zero_line: bool = True,
) -> "plt.Axes":
    """Draw mean ± SEM ETA traces for *rois* into an existing *ax*, one line per ROI.

    The axes-level counterpart of :func:`plot_eta_by_condition`, for composing
    ETA panels into a larger figure.  *group_means* is the tidy table
    :class:`EtaAnalysis` produces — columns ``EventType``, ``ROI``,
    ``rel_time``, ``mean``, ``sem``, and optionally ``Condition``.

    Parameters
    ----------
    group_means : DataFrame
        Tidy mean ± SEM table.  Rows are selected by *event*, and by
        *condition* when the table carries a ``Condition`` column.
    ax : matplotlib Axes
        Axes to draw into.
    event : str
        ``EventType`` to plot.
    rois : sequence of str
        ROI names to draw, in drawing order.
    condition : str, optional
        Restrict to one ``Condition``.
    colors, labels : mapping, optional
        Per-ROI colour and legend label, keyed by ROI name.  ROIs absent from
        *colors* fall back to the active theme's palette.
    lw : float
        Line width for the mean.
    zero_line : bool
        Draw a dotted vertical line at ``rel_time`` 0.

    Returns
    -------
    matplotlib Axes
    """
    from databench.plotting import get_theme, style_axes

    rows = group_means[group_means["EventType"] == event]
    if condition is not None and "Condition" in rows.columns:
        rows = rows[rows["Condition"] == condition]

    palette = get_theme().colors
    for i, roi in enumerate(rois):
        color = (colors or {}).get(roi, palette[i % len(palette)])
        _mean_sem_line(ax, rows[rows["ROI"] == roi], color,
                       (labels or {}).get(roi, roi), lw=lw)

    if zero_line:
        ax.axvline(0.0, color=get_theme().fg, lw=0.6, ls=":")
    style_axes(ax)
    return ax


def plot_eta_by_condition(
    group_means: pd.DataFrame,
    *,
    event: str,
    rois: list[str],
    conditions: list[str] | None = None,
    condition_colors: Dict[str, str] | None = None,
    baseline_s: tuple[float, float] | None = None,
    ncols: int = 2,
    task: str | None = None,
) -> plt.Figure:
    """Plot group mean \u00b1 SEM ETA traces: one subplot per ROI, lines per condition."""

    # 1) Prep data
    d = group_means.query("EventType == @event and ROI in @rois").copy()
    if task is not None and "Task" in d.columns:
        d = d.query("Task == @task")

    if conditions is None:
        conditions = sorted(d["Condition"].unique())

    if condition_colors is None:
        from databench.plotting import get_theme

        default_palette = get_theme().colors
        condition_colors = {c: default_palette[i % len(default_palette)] for i, c in enumerate(conditions)}

    # 2) Create canvas
    n = len(rois)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.0 * ncols, 3.2 * nrows + 0.6),
        sharex=True, sharey="row",
    )
    axes = np.atleast_1d(axes).ravel()

    # 3) Draw panels
    from databench.plotting import style_axes

    for ax, roi in zip(axes, rois):
        roi_data = d[d["ROI"] == roi]
        for cond in conditions:
            _mean_sem_line(ax, roi_data[roi_data["Condition"] == cond],
                           condition_colors.get(cond, "#333333"), cond,
                           lw=plt.rcParams["lines.linewidth"])
        ax.axvline(0, color="k", lw=1)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_title(roi)
        style_axes(ax)

    for ax in axes[n:]:
        ax.axis("off")

    # 4) Finalize figure
    handles, labels = axes[0].get_legend_handles_labels()
    has_legend = bool(handles)
    if has_legend:
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.93),
            ncol=min(6, len(labels)), frameon=False,
        )

    baseline_label = (
        "no baseline subtraction"
        if baseline_s is None
        else f"baseline-subtracted [{baseline_s[0]:g}, {baseline_s[1]:g}] s"
    )
    fig.supxlabel(f"Time relative to {event} (s)", y=0.03)
    fig.supylabel("Group mean \u00b1 SEM (subject-averaged)", x=0.02)
    fig.suptitle(
        f"ETA across ROIs \u2014 {event} ({baseline_label})",
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        bottom=0.12,
        top=0.84 if has_legend else 0.90,
        wspace=0.15,
        hspace=0.18,
    )
    return fig


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
    :func:`~databench.signal.epoching.locomotion_events` to produce events
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

    @log_run
    def run(
        self,
        sessions: "SessionGroup",
        events: EventsTable,
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
            :func:`~databench.signal.epoching.locomotion_events`,
            :func:`~databench.signal.epoching.make_events`, or any custom
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
        n_sessions = len(sessions)
        n_skipped = 0

        for i, sess in enumerate(sessions):
            # Filter events for this session
            sess_mask = (
                (events["Subject"] == sess.subject)
                & (events["Session"] == sess.session)
                & (events["Task"] == sess.task)
            )
            sess_events = events.loc[sess_mask].copy()
            if sess_events.empty:
                n_skipped += 1
                continue

            _log.debug(
                f"Session {i + 1}/{n_sessions} {sess.label} | "
                f"{len(sess_events)} events"
            )

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
        _log.info(
            f"Processed {n_sessions - n_skipped}/{n_sessions} sessions | "
            f"{len(eta_events)} epoch rows | "
            f"event types: {event_types}"
        )

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
    ):
        """Plot group-mean ± SEM ETA traces.  Returns a matplotlib ``Figure``.

        Persist via ``run.save_figure(...)``.
        """
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
        return fig

    def plot_traces(self, ax, *, event: str = "onset",
                    rois: list[str] | None = None, **kwargs):
        """Draw mean ± SEM traces for this result into *ax*.

        Thin wrapper over :func:`plot_eta_traces`, for composing an ETA panel
        into a larger figure.
        """
        return plot_eta_traces(self.group_means, ax, event=event,
                               rois=list(rois) if rois else self.roi_columns,
                               **kwargs)

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        """All result DataFrames keyed by name."""
        return {
            "events": self.events,
            "eta_events": self.eta_events,
            "subject_means": self.subject_means,
            "group_means": self.group_means,
        }

    # ── Per-event window aggregation ───────────────────────────────────────

    def window_mean(
        self,
        window: tuple[float, float],
        *,
        name: str = "mean",
    ) -> pd.DataFrame:
        """Mean of ``value`` over ``rel_time`` ∈ ``window`` per event × ROI.

        Returns one row per (Subject, Session, Task, Condition, EventType,
        event_id, ROI) with column ``name`` holding the windowed mean.
        Rows whose group has no samples in the window are dropped.
        """
        lo, hi = window
        ev = self.eta_events
        mask = (ev["rel_time"] >= lo) & (ev["rel_time"] <= hi)
        key_cols = [
            c for c in
            ("Subject", "Session", "Task", "Condition", "EventType", "event_id", "ROI")
            if c in ev.columns
        ]
        return (
            ev.loc[mask]
            .groupby(key_cols, sort=False)
            .agg(**{name: ("value", "mean")})
            .reset_index()
        )

    # ── Plotter factory ────────────────────────────────────────────────────

    def condition_plotter(self, **kwargs):
        """Return an :class:`EtaConditionPlotter` configured for this result."""
        return EtaConditionPlotter(**kwargs)


# --- Plotter ---

from dataclasses import dataclass as _dataclass
from databench.plotting.base import Plotter as _Plotter


@_dataclass(frozen=True)
class EtaConditionPlotter(_Plotter):
    name: str = "eta_by_condition"
    event: str = "onset"
    rois: tuple = ()
    conditions: Optional[tuple] = None
    ncols: int = 2
    task: Optional[str] = None

    def plot(self, result):
        rois = list(self.rois) if self.rois else None
        conditions = list(self.conditions) if self.conditions else None
        return result.plot(
            event=self.event,
            rois=rois,
            conditions=conditions,
            ncols=self.ncols,
            task=self.task,
        )
