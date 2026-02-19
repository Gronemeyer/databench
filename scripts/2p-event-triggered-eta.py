#%%
"""
Event-triggered averages for ROI traces aligned to locomotion onset/offset.

This script mirrors the event-based workflow:
- explicit procedural flow (load -> preflight -> analyze -> plot -> save)
- databench Analysis + Plotter classes
- locomotion bout event extraction
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
from databench.features.treadmill import LocomotionBoutEventsExtractor, extract_epoch_interpolated
from databench.plotting.base import Plotter


def eta_baselined(
    df: pd.DataFrame,
    event_times: np.ndarray,
    roi_columns: list[str],
    *,
    time_column: str = "time_elapsed_s",
    window: tuple[float, float] = (-2.0, 2.0),
    dt: float = 0.05,
    baseline: tuple[float, float] = (-2.0, -1.0),
) -> pd.DataFrame:
    df = df.sort_values(time_column)
    time_values = df[time_column].to_numpy()

    output_frames = []
    for event_id, event_time in enumerate(event_times):
        for roi in roi_columns:
            roi_values = df[roi].to_numpy()
            rel_time, roi_epoch = extract_epoch_interpolated(
                time_values, roi_values, event_time, window=window, dt=dt
            )
            if rel_time is None:
                continue

            baseline_mask = (rel_time >= baseline[0]) & (rel_time <= baseline[1])
            baseline_value = np.nanmean(roi_epoch[baseline_mask]) if baseline_mask.any() else np.nan
            roi_epoch = roi_epoch - baseline_value

            output_frames.append(
                pd.DataFrame(
                    {
                        "event_id": event_id,
                        "rel_time": rel_time,
                        "ROI": roi,
                        "value": roi_epoch,
                    }
                )
            )

    if not output_frames:
        return pd.DataFrame(columns=["event_id", "rel_time", "ROI", "value"])
    return pd.concat(output_frames, ignore_index=True)


@dataclass(frozen=True)
class EtaByLocomotionAnalysis(Analysis):
    name: str = "eta_by_locomotion"
    roi_cols: tuple[str, ...] = ()
    task: str = "task-spont"
    event_types: tuple[str, ...] = ("onset", "offset")
    window: tuple[float, float] = (-1.0, 3.0)
    dt: float = 0.05
    baseline: tuple[float, float] = (-2.0, -0.5)
    bout_events_extractor: Any = LocomotionBoutEventsExtractor()

    def run(self, long: pd.DataFrame) -> AnalysisResult:
        roi_cols = list(self.roi_cols)
        required = [
            "Subject",
            "Session",
            "Task",
            "time_elapsed_s",
            "speed_mm",
            *roi_cols,
        ]
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
            for (subj, ses, task_name), grp in events.groupby(["Subject", "Session", "Task"], sort=False)
        }

        rows = []
        for (subj, ses, task_name), g in long.groupby(["Subject", "Session", "Task"], sort=False):
            if task_name != self.task:
                continue

            transitions = event_map.get((subj, ses, task_name), {"onset": np.array([]), "offset": np.array([])})
            for event_type in self.event_types:
                event_times = transitions.get(event_type, np.array([]))
                if event_times.size == 0:
                    continue

                eta = eta_baselined(g, event_times, roi_cols, window=self.window, dt=self.dt, baseline=self.baseline)
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
            eta_events.groupby(["Subject", "Condition", "Task", "EventType", "ROI", "rel_time"], as_index=False)
            .agg(value=("value", "mean"))
        )

        eta_group = (
            eta_subj.groupby(["Condition", "Task", "EventType", "ROI", "rel_time"], as_index=False)
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


@dataclass(frozen=True)
class EtaLocomotionPlotter(Plotter):
    name: str = "eta_locomotion"
    rois: tuple[str, ...] = ()
    task: str = "task-spont"
    event_type: str = "onset"
    baseline: tuple[float, float] | None = None
    ncols: int = 2

    def plot(self, result: AnalysisResult):
        rois = list(self.rois)
        eta_subj = result.data["eta_subj"]
        d = eta_subj.query("Task == @self.task and EventType == @self.event_type and ROI in @rois").copy()

        cond_colors = {
            "baseline": "#bbabab",
            "saline": "#4289e6",
            "ethanol_low": "#ffa251",
            "ethanol_high": "#ce1818",
            "low": "#ffa251",
            "high": "#ce1818",
        }
        cond_order = ["baseline", "saline", "low", "high"]

        subjects = d["Subject"].dropna().unique().tolist()
        n = len(rois) * len(subjects)
        nrows = int(np.ceil(n / self.ncols))
        fig, axes = plt.subplots(
            nrows,
            self.ncols,
            figsize=(7 * self.ncols / 2, 3 * nrows),
            sharex=True,
            sharey="row",
        )
        axes = np.atleast_1d(axes).ravel()

        panel_specs = [(roi, subj) for roi in rois for subj in subjects]
        for ax, (roi, subj) in zip(axes, panel_specs):
            dd = d[(d["ROI"] == roi) & (d["Subject"] == subj)]
            for cond in cond_order:
                g = dd[dd["Condition"] == cond]
                if g.empty:
                    continue
                color = cond_colors.get(cond)
                ax.plot(g["rel_time"], g["value"], label=cond, color=color)

            ax.axvline(0, color="k", lw=1)
            ax.axhline(0, color="k", lw=0.5, alpha=0.5)
            ax.set_title(f"{subj} | {roi}", fontsize=10)
            ax.tick_params(axis="both", labelsize=9)

        for ax in axes[n:]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.92),
                ncol=min(4, len(labels)),
                frameon=False,
            )

        baseline_label = (
            "no baseline subtraction"
            if self.baseline is None
            else f"baseline-subtracted [{self.baseline[0]:g}, {self.baseline[1]:g}] s"
        )
        fig.supxlabel(f"Time relative to run {self.event_type} (s)", y=0.02)
        fig.supylabel("Subject mean (event-averaged)", x=0.03)
        fig.suptitle(f"ETA by subject - {self.task} - run {self.event_type}\n{baseline_label}", y=0.98)
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.09, top=0.84, hspace=0.28, wspace=0.35)
        return fig, axes


#%%
# Procedural workflow

pickle_path = Path(r'/Users/jakegronemeyer/Desktop/4jake/260212_ACUTEVIS_dataset.pkl')
project_root = Path(__file__).resolve().parents[1]
output_root = project_root / "outputs"

bench = Bench()
bench.setup(pickle_path, output_root=output_root, run_name="260217", tag="2p-locomotion-eta")

paths = bench.output_paths
print(f"[databench] run_dir: {paths.run_dir}")

# Load and prepare data
raw = bench.load()

roi_feature = "deltaf_f"
mean_feature = f"{roi_feature}_mean"
roi_cols = [mean_feature]
plot_rois = roi_cols

mean_col = ("suite2p", mean_feature)
if mean_col not in raw.columns:
    def mean_suite2p_trace(row: pd.Series) -> np.ndarray | float:
        trace_matrix = row.get(("suite2p", roi_feature))
        if trace_matrix is None:
            return np.nan
        arr = np.asarray(trace_matrix)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            return np.nanmean(arr, axis=0)
        return np.nan

    raw[mean_col] = raw.apply(mean_suite2p_trace, axis=1)

task = "task-gratings"
min_speed_cms = 0.05
min_duration_s = 1.0
merge_gap_s = 0.5

analysis = EtaByLocomotionAnalysis(
    roi_cols=tuple(roi_cols),
    task=task,
    event_types=("onset", "offset"),
    window=(-1.0, 3.0),
    dt=0.02,
    baseline=(-5.0, 0.0),
    bout_events_extractor=LocomotionBoutEventsExtractor(
        group_cols=("Subject", "Session", "Task"),
        time_col="time_elapsed_s",
        speed_col="speed_mm",
        speed_scale_to_cms=10.0,
        min_speed_cms=min_speed_cms,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
    ),
)

plot_onset = EtaLocomotionPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="onset",
    baseline=analysis.baseline,
)
plot_offset = EtaLocomotionPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="offset",
    baseline=analysis.baseline,
)

source_features = [
    ("suite2p", [mean_feature]),
    #("pupil", ["pupil_diameter_mm"]),
    ("encoder", ["speed_mm"]),
]

long = bench.build_long(
	raw,
	source_features=source_features,
	tol=0.25,
	time_column="time_elapsed_s",
	reference_source="suite2p",
)

# Pick which session_config columns you want
config_cols = ["injection"]  # or ["Condition", "Drug", "Cohort"]

cfg = raw["session_config"][config_cols].copy()
cfg = cfg.rename(columns={"injection": "Condition"})
cfg["Condition"] = cfg["Condition"].astype(str).str.strip().str.lower()
condition_map = {
    "baseline": "baseline",
    "saline": "saline",
    "ethanol_low": "low",
    "ethanol_high": "high",
    "low": "low",
    "high": "high",
}
cfg["Condition"] = cfg["Condition"].map(condition_map)
cfg["Condition"] = pd.Categorical(
    cfg["Condition"],
    categories=["baseline", "saline", "low", "high"],
    ordered=True,
)

# Ensure index alignment with long's (Subject, Session, Task)
long = long.join(cfg, on=["Subject", "Session", "Task"])

print(f"[debug] long rows: {len(long)}")
print(f"[debug] long tasks: {sorted(long['Task'].unique().tolist())}")
print(f"[debug] roi columns present: {[c for c in roi_cols if c in long.columns]}")
print(f"[debug] roi columns missing: {[c for c in roi_cols if c not in long.columns]}")
if task not in set(long["Task"].unique()):
    raise ValueError(f"Requested task {task!r} not found in long table.")

task_conditions = (
    long.loc[long["Task"] == task, "Condition"]
    .dropna()
    .astype(str)
    .value_counts()
    .to_dict()
)
print(f"[debug] condition counts ({task}): {task_conditions}")

task_speed = long.loc[long["Task"] == task, "speed_mm"].astype(float)
speed_valid = task_speed.replace([np.inf, -np.inf], np.nan).dropna()
print(f"[debug] speed_mm range ({task}): {speed_valid.min():.4g}..{speed_valid.max():.4g}")
if speed_valid.size:
    speed_cms = speed_valid / 10.0
    frac_moving = float((speed_cms >= min_speed_cms).mean())
    print(f"[debug] fraction >= {min_speed_cms} cm/s: {frac_moving:.3f}")

# bench.preflight(
#     analysis=analysis,
#     plotter=plot_onset,
#     df=long,
#     required_columns=[
#         "Subject",
#         "Session",
#         "Task",
#         "injection",
#         "time_elapsed_s",
#         "speed_mm",
#     ],
# )

res = bench.analyze(analysis, long)

fig_onset, _ = bench.plot(plot_onset, res)
onset_plot_path = bench.save_figure(fig_onset, "eta_locomotion_onset.png")
plt.close(fig_onset)

fig_offset, _ = bench.plot(plot_offset, res)
offset_plot_path = bench.save_figure(fig_offset, "eta_locomotion_offset.png")
plt.close(fig_offset)

saved_tables = bench.save_analysis_result_tables(res, prefix="eta_locomotion")

run_summary_path = bench.save_run_summary(
    notes="Event-triggered ETA for ROI traces aligned to locomotion onset/offset.",
)

saved_paths = [
    onset_plot_path,
    offset_plot_path,
    *saved_tables.values(),
    run_summary_path,
]
for p in saved_paths:
    print(f"[databench] saved: {p}")

missing_paths = [str(p) for p in saved_paths if not Path(p).exists()]
if missing_paths:
    raise RuntimeError(f"Expected output files were not created: {missing_paths}")

# %%
