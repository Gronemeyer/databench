#%%
"""
Event-based ETA analysis for longitudinal widefield recordings.

This script demonstrates:
1) explicit procedural flow (load -> preflight -> analyze -> plot -> save)
2) event-triggered averages pooled across all 10 days
3) longitudinal day-by-day visualizations across sessions
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench import Bench
from databench.analysis.base import Analysis, AnalysisResult
from databench.features.treadmill import (
    LocomotionBoutEventsExtractor,
    extract_epoch_interpolated,
)
from databench.plotting.base import Plotter


def _session_to_day(session: str) -> int:
    s = str(session)
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse session number from {session!r}")
    return int(digits)


def _safe_sem(x: pd.Series) -> float:
    vals = x.dropna().to_numpy(dtype=float)
    n = vals.size
    if n <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(n))


def eta_baselined(
    df: pd.DataFrame,
    event_times: np.ndarray,
    roi_columns: list[str],
    *,
    time_column: str = "time_elapsed_s",
    window: tuple[float, float] = (-2.0, 2.0),
    dt: float = 0.05,
    baseline: tuple[float, float] = (-2.0, -1.0),
    bout_intervals: np.ndarray | None = None,
    baseline_exclude_bouts: bool = True,
    min_clean_baseline_points: int = 3,
    fallback_to_full_baseline: bool = True,
) -> pd.DataFrame:
    df = df.sort_values(time_column)
    time_values = df[time_column].to_numpy()

    output_frames = []
    for event_id, event_time in enumerate(event_times):
        for roi in roi_columns:
            roi_values = df[roi].to_numpy()
            rel_t, roi_epoch = extract_epoch_interpolated(
                time_values,
                roi_values,
                event_time,
                window=window,
                dt=dt,
            )
            if rel_t is None:
                continue

            baseline_mask_full = (rel_t >= baseline[0]) & (rel_t <= baseline[1])
            baseline_mask = baseline_mask_full.copy()

            if baseline_exclude_bouts and bout_intervals is not None and bout_intervals.size > 0:
                abs_t = event_time + rel_t
                in_bout = np.zeros(abs_t.shape, dtype=bool)
                for onset_t, offset_t in bout_intervals:
                    in_bout |= (abs_t >= float(onset_t)) & (abs_t <= float(offset_t))
                baseline_mask &= ~in_bout

            clean_count = int(np.sum(baseline_mask & np.isfinite(roi_epoch)))
            if clean_count < int(min_clean_baseline_points) and fallback_to_full_baseline:
                baseline_mask = baseline_mask_full

            baseline_value = np.nanmean(roi_epoch[baseline_mask]) if baseline_mask.any() else np.nan
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


@dataclass(frozen=True)
class EtaWidefieldLongitudinalAnalysis(Analysis):
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
            for (subj, ses, task_name), grp in events.groupby(["Subject", "Session", "Task"], sort=False)
        }

        rows = []
        for (subj, ses, task_name), g in d.groupby(["Subject", "Session", "Task"], sort=False):
            transitions = event_map.get((subj, ses, task_name), {"onset": np.array([]), "offset": np.array([])})
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

        if eta_events.empty:
            return AnalysisResult(
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

        eta_subj_day = (
            eta_events.groupby(
                ["Subject", "Session", "day_n", "Task", "EventType", "ROI", "rel_time"], as_index=False
            )
            .agg(value=("value", "mean"))
        )

        eta_group = (
            eta_subj_day.groupby(["Task", "EventType", "ROI", "rel_time"], as_index=False)
            .agg(mean=("value", "mean"), sem=("value", _safe_sem), n_subjects=("Subject", "nunique"))
        )

        eta_longitudinal = (
            eta_subj_day.groupby(["day_n", "Task", "EventType", "ROI", "rel_time"], as_index=False)
            .agg(mean=("value", "mean"), sem=("value", _safe_sem), n_subjects=("Subject", "nunique"))
            .sort_values(["day_n", "EventType", "ROI", "rel_time"])
        )

        metric_mask = (
            (eta_subj_day["rel_time"] >= self.metric_window[0])
            & (eta_subj_day["rel_time"] <= self.metric_window[1])
        )
        eta_day_metric = (
            eta_subj_day.loc[metric_mask]
            .groupby(["Subject", "Session", "day_n", "Task", "EventType", "ROI"], as_index=False)
            .agg(metric_value=("value", "mean"))
        )
        eta_day_metric_group = (
            eta_day_metric.groupby(["day_n", "Task", "EventType", "ROI"], as_index=False)
            .agg(mean=("metric_value", "mean"), sem=("metric_value", _safe_sem), n_subjects=("Subject", "nunique"))
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


@dataclass(frozen=True)
class EtaAllDaysAveragePlotter(Plotter):
    name: str = "eta_all_days_avg"
    rois: tuple[str, ...] = ()
    task: str = "task-widefield"
    event_type: str = "onset"
    baseline: tuple[float, float] | None = None
    ncols: int = 2

    def plot(self, result: AnalysisResult):
        rois = list(self.rois)
        eta_group = result.data["eta_group"]
        d = eta_group.query("Task == @self.task and EventType == @self.event_type and ROI in @rois").copy()

        n = len(rois)
        nrows = int(np.ceil(n / self.ncols))
        fig, axes = plt.subplots(
            nrows,
            self.ncols,
            figsize=(7 * self.ncols / 2, 3 * nrows),
            sharex=True,
            sharey="row",
        )
        axes = np.atleast_1d(axes).ravel()

        line_color = "#1f77b4"
        for ax, roi in zip(axes, rois):
            g = d[d["ROI"] == roi].sort_values("rel_time")
            if not g.empty:
                ax.plot(g["rel_time"], g["mean"], color=line_color, lw=1.8)
                ax.fill_between(g["rel_time"], g["mean"] - g["sem"], g["mean"] + g["sem"], alpha=0.2, color=line_color)
            ax.axvline(0, color="k", lw=1)
            ax.axhline(0, color="k", lw=0.5, alpha=0.5)
            ax.set_title(roi, fontsize=10)
            ax.tick_params(axis="both", labelsize=9)

        for ax in axes[n:]:
            ax.axis("off")

        baseline_label = "no baseline subtraction" if self.baseline is None else f"baseline-subtracted [{self.baseline[0]:g}, {self.baseline[1]:g}] s"
        fig.supxlabel(f"Time relative to run {self.event_type} (s)", y=0.02)
        fig.supylabel("Group mean ± SEM (pooled across days)", x=0.03)
        fig.suptitle(f"ETA across ROIs — {self.task} — run {self.event_type}\n{baseline_label}", y=0.98)
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.09, top=0.87, hspace=0.28, wspace=0.35)
        return fig, axes


@dataclass(frozen=True)
class EtaLongitudinalHeatmapPlotter(Plotter):
    name: str = "eta_longitudinal_heatmap"
    rois: tuple[str, ...] = ()
    task: str = "task-widefield"
    event_type: str = "onset"
    ncols: int = 2
    cmap: str = "RdBu_r"
    vmin: float | None = None
    vmax: float | None = None

    def plot(self, result: AnalysisResult):
        rois = list(self.rois)
        eta_long = result.data["eta_longitudinal"]
        d = eta_long.query("Task == @self.task and EventType == @self.event_type and ROI in @rois").copy()

        n = len(rois)
        nrows = int(np.ceil(n / self.ncols))
        fig, axes = plt.subplots(nrows, self.ncols, figsize=(7 * self.ncols / 2, 3 * nrows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()

        image = None
        used_axes = []
        for ax, roi in zip(axes, rois):
            dd = d[d["ROI"] == roi]
            if dd.empty:
                ax.axis("off")
                continue

            days = np.array(sorted(dd["day_n"].unique()), dtype=float)
            rel_times = np.array(sorted(dd["rel_time"].unique()), dtype=float)
            mat = (
                dd.pivot(index="day_n", columns="rel_time", values="mean")
                .reindex(index=days, columns=rel_times)
                .to_numpy(dtype=float)
            )

            image = ax.imshow(
                mat,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=[rel_times.min(), rel_times.max(), days.min() - 0.5, days.max() + 0.5],
                cmap=self.cmap,
                vmin=self.vmin,
                vmax=self.vmax,
            )
            ax.axvline(0, color="k", lw=1)
            ax.set_yticks(days)
            ax.set_title(roi, fontsize=10)
            ax.tick_params(axis="both", labelsize=9)
            used_axes.append(ax)

        for ax in axes[n:]:
            ax.axis("off")

        fig.supxlabel(f"Time relative to run {self.event_type} (s)", y=0.03)
        fig.supylabel("Day / session number", x=0.03)
        fig.suptitle(f"Longitudinal ETA heatmap — {self.task} — run {self.event_type}", y=0.98)
        if image is not None and used_axes:
            fig.colorbar(image, ax=used_axes, shrink=0.9, label="Mean dF/F")
        fig.subplots_adjust(left=0.11, right=0.93, bottom=0.11, top=0.88, hspace=0.30, wspace=0.30)
        return fig, axes


@dataclass(frozen=True)
class EtaLongitudinalMetricPlotter(Plotter):
    name: str = "eta_longitudinal_metric"
    rois: tuple[str, ...] = ()
    task: str = "task-widefield"
    event_type: str = "onset"
    metric_window: tuple[float, float] = (0.0, 1.0)
    ncols: int = 2

    def plot(self, result: AnalysisResult):
        rois = list(self.rois)
        metric_group = result.data["eta_day_metric_group"]
        d = metric_group.query("Task == @self.task and EventType == @self.event_type and ROI in @rois").copy()

        n = len(rois)
        nrows = int(np.ceil(n / self.ncols))
        fig, axes = plt.subplots(nrows, self.ncols, figsize=(7 * self.ncols / 2, 3 * nrows), sharex=True, sharey="row")
        axes = np.atleast_1d(axes).ravel()

        line_color = "#2ca02c"
        for ax, roi in zip(axes, rois):
            g = d[d["ROI"] == roi].sort_values("day_n")
            if not g.empty:
                ax.plot(g["day_n"], g["mean"], marker="o", color=line_color, lw=1.8)
                ax.fill_between(g["day_n"], g["mean"] - g["sem"], g["mean"] + g["sem"], alpha=0.2, color=line_color)
            ax.axhline(0, color="k", lw=0.5, alpha=0.5)
            ax.set_title(roi, fontsize=10)
            ax.tick_params(axis="both", labelsize=9)

        for ax in axes[n:]:
            ax.axis("off")

        fig.supxlabel("Day / session number", y=0.03)
        fig.supylabel("Mean ETA value (subject-averaged)", x=0.03)
        fig.suptitle(
            f"Longitudinal ETA summary — {self.task} — run {self.event_type}\n"
            f"metric window [{self.metric_window[0]:g}, {self.metric_window[1]:g}] s",
            y=0.98,
        )
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.11, top=0.85, hspace=0.30, wspace=0.30)
        return fig, axes


def _infer_default_rois(df: pd.DataFrame, preferred: list[str], fallback_n: int = 6) -> list[str]:
    if not isinstance(df.columns, pd.MultiIndex):
        return preferred

    mesomap_cols = [
        feat
        for source, feat in df.columns
        if source == "mesomap" and feat != "time_elapsed_s"
    ]
    mesomap_cols = list(dict.fromkeys(mesomap_cols))
    selected = [r for r in preferred if r in mesomap_cols]
    if selected:
        return selected
    return mesomap_cols[:fallback_n]


#%%
# Procedural workflow
pickle_path = Path(r"/Users/jakegronemeyer/Desktop/4jake/260212_ETOH-HFSA_dataset.pkl")
project_root = Path(__file__).resolve().parents[1]
output_root = project_root / "outputs"

bench = Bench()
bench.setup(pickle_path, output_root=output_root, run_name="260217", tag="widefield-10day-eta_2s")

paths = bench.output_paths
print(f"[databench] run_dir: {paths.run_dir}")
df = bench.load()
bouts_feature = bench.get_feature("locomotion_bouts_n")

preferred_rois = ["L_MOp", "R_MOp", "L_MOs", "R_MOs", "L_VISp", "R_VISp"]
roi_cols = _infer_default_rois(df, preferred_rois, fallback_n=6)
if not roi_cols:
    raise ValueError("No mesomap ROI columns were found in the dataset.")
plot_rois = roi_cols

analysis = EtaWidefieldLongitudinalAnalysis(
    roi_cols=tuple(roi_cols),
    task="task-widefield",
    event_types=("onset", "offset"),
    window=(-1.0, 1.0),
    dt=0.02,
    baseline=(-2, 0.0),
    baseline_exclude_bouts=True,
    min_clean_baseline_points=3,
    fallback_to_full_baseline=True,
    metric_window=(0.0, 2.0),
    bout_events_extractor=LocomotionBoutEventsExtractor(
        min_speed_cms=bouts_feature.min_speed_cms,
        min_duration_s=bouts_feature.min_duration_s,
        merge_gap_s=bouts_feature.merge_gap_s,
        group_cols=("Subject", "Session", "Task"),
        time_col="time_elapsed_s",
        speed_col="speed_mm",
        speed_scale_to_cms=10.0,
    ),
)

plot_avg_onset = EtaAllDaysAveragePlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="onset",
    baseline=analysis.baseline,
)
plot_avg_offset = EtaAllDaysAveragePlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="offset",
    baseline=analysis.baseline,
)

plot_long_heatmap_onset = EtaLongitudinalHeatmapPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="onset",
)
plot_long_heatmap_offset = EtaLongitudinalHeatmapPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="offset",
)

plot_long_metric_onset = EtaLongitudinalMetricPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="onset",
    metric_window=analysis.metric_window,
)
plot_long_metric_offset = EtaLongitudinalMetricPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="offset",
    metric_window=analysis.metric_window,
)

source_features = [
    ("mesomap", roi_cols),
    ("treadmill", ["speed_mm"]),
]
long = bench.build_long(
    df,
    source_features=source_features,
    tol=0.25,
    time_column="time_elapsed_s",
    reference_source="mesomap",
)

bench.preflight(
    analysis=analysis,
    plotter=plot_avg_onset,
    df=long,
    required_columns=["Subject", "Session", "Task", "time_elapsed_s", "speed_mm", *roi_cols],
)

res = bench.analyze(analysis, long)

fig_avg_onset, _ = bench.plot(plot_avg_onset, res)
avg_onset_plot_path = bench.save_figure(fig_avg_onset, "eta_onset_rois_10day_avg.png")
plt.close(fig_avg_onset)

fig_avg_offset, _ = bench.plot(plot_avg_offset, res)
avg_offset_plot_path = bench.save_figure(fig_avg_offset, "eta_offset_rois_10day_avg.png")
plt.close(fig_avg_offset)

fig_long_heatmap_onset, _ = bench.plot(plot_long_heatmap_onset, res)
long_heatmap_onset_path = bench.save_figure(fig_long_heatmap_onset, "eta_onset_longitudinal_heatmap.png")
plt.close(fig_long_heatmap_onset)

fig_long_heatmap_offset, _ = bench.plot(plot_long_heatmap_offset, res)
long_heatmap_offset_path = bench.save_figure(fig_long_heatmap_offset, "eta_offset_longitudinal_heatmap.png")
plt.close(fig_long_heatmap_offset)

fig_long_metric_onset, _ = bench.plot(plot_long_metric_onset, res)
long_metric_onset_path = bench.save_figure(fig_long_metric_onset, "eta_onset_longitudinal_metric.png")
plt.close(fig_long_metric_onset)

fig_long_metric_offset, _ = bench.plot(plot_long_metric_offset, res)
long_metric_offset_path = bench.save_figure(fig_long_metric_offset, "eta_offset_longitudinal_metric.png")
plt.close(fig_long_metric_offset)

saved_tables = bench.save_analysis_result_tables(res, prefix="eta_widefield_10day")

run_summary_path = bench.save_run_summary(
    notes=(
        "Widefield event-based ETA workflow for a longitudinal 10-day dataset. "
        "Includes pooled 10-day averages and day-wise longitudinal visualizations."
    ),
)

saved_paths = [
    avg_onset_plot_path,
    avg_offset_plot_path,
    long_heatmap_onset_path,
    long_heatmap_offset_path,
    long_metric_onset_path,
    long_metric_offset_path,
    *saved_tables.values(),
    run_summary_path,
]
for p in saved_paths:
    print(f"[databench] saved: {p}")

missing_paths = [str(p) for p in saved_paths if not Path(p).exists()]
if missing_paths:
    raise RuntimeError(f"Expected output files were not created: {missing_paths}")

# %%