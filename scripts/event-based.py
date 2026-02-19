#%%
"""
Event-based analysis in blessed databench style.

This script demonstrates:
1) explicit procedural flow (load -> preflight -> analyze -> plot -> save)
2) decorator-registered custom analysis and plotter
3) DataFrame-based locomotion bout event API
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
            relative_time_grid, roi_epoch = extract_epoch_interpolated(time_values, roi_values, event_time, window=window, dt=dt)
            if relative_time_grid is None:
                continue

            baseline_mask = (relative_time_grid >= baseline[0]) & (relative_time_grid <= baseline[1])
            baseline_value = np.nanmean(roi_epoch[baseline_mask]) if baseline_mask.any() else np.nan
            roi_epoch = roi_epoch - baseline_value

            output_frames.append(
                pd.DataFrame(
                    {
                        "event_id": event_id,
                        "rel_time": relative_time_grid,
                        "ROI": roi,
                        "value": roi_epoch,
                    }
                )
            )

    if not output_frames:
        return pd.DataFrame(columns=["event_id", "rel_time", "ROI", "value"])
    return pd.concat(output_frames, ignore_index=True)


@dataclass(frozen=True)
class EtaByConditionAnalysis(Analysis):
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
                data={"events": events, "eta_events": eta_events, "eta_subj": pd.DataFrame(), "eta_group": pd.DataFrame()},
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
class EtaConditionPlotter(Plotter):
    name: str = "eta_condition"
    rois: tuple[str, ...] = ()
    task: str = "task-movies"
    event_type: str = "onset"
    baseline: tuple[float, float] | None = None
    ncols: int = 2

    def plot(self, result: AnalysisResult):
        rois = list(self.rois)
        eta_group = result.data["eta_group"]
        d = eta_group.query("Task == @self.task and EventType == @self.event_type and ROI in @rois").copy()

        cond_colors = {
            "baseline": "#bbabab",
            "saline": "#4289e6",
            "ethanol_low": "#ffa251",
            "ethanol_high": "#ce1818",
        }
        cond_order = ["baseline", "saline", "ethanol_low", "ethanol_high"]

        n = len(rois)
        nrows = int(np.ceil(n / self.ncols))
        fig, axes = plt.subplots(nrows, self.ncols, figsize=(7 * self.ncols / 2, 3 * nrows), sharex=True, sharey="row") # sharey by row to allow better comparison of onset/offset across conditions within each ROI
        axes = np.atleast_1d(axes).ravel()

        for ax, roi in zip(axes, rois):
            dd = d[d["ROI"] == roi]
            for cond in cond_order:
                g = dd[dd["Condition"] == cond]
                if g.empty:
                    continue
                color = cond_colors.get(cond)
                ax.plot(g["rel_time"], g["mean"], label=cond, color=color)
                ax.fill_between(g["rel_time"], g["mean"] - g["sem"], g["mean"] + g["sem"], alpha=0.2, color=color)

            ax.axvline(0, color="k", lw=1)
            ax.axhline(0, color="k", lw=0.5, alpha=0.5)
            ax.set_title(roi, fontsize=10)
            ax.tick_params(axis="both", labelsize=9)

        for ax in axes[n:]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=min(4, len(labels)), frameon=False)

        baseline_label = "no baseline subtraction" if self.baseline is None else f"baseline-subtracted [{self.baseline[0]:g}, {self.baseline[1]:g}] s"
        fig.supxlabel(f"Time relative to run {self.event_type} (s)", y=0.02)
        fig.supylabel("Group mean ± SEM (subject-averaged)", x=0.03)
        fig.suptitle(f"ETA across ROIs — {self.task} — run {self.event_type}\n{baseline_label}", y=0.98)
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.09, top=0.84, hspace=0.28, wspace=0.35) # make spacing wider
        return fig, axes


#%%
# Procedural workflow

pickle_path = Path(r"/Users/jakegronemeyer/Desktop/4jake/260211_ETOH_dataset.pkl")
project_root = Path(__file__).resolve().parents[1]
output_root = project_root / "outputs"

bench = Bench()
bench.setup(pickle_path, output_root=output_root, run_name="260218", tag="spont-mop-eta")

paths = bench.output_paths
print(f"[databench] run_dir: {paths.run_dir}")
df = bench.load()
bouts_feature = bench.get_feature("locomotion_bouts_n")
bench._usage["features"].append({"names": [bouts_feature.name]})

roi_cols = ["L_MOp", "R_MOp", "L_MOs", "R_MOs"]#, "L_VISp", "R_VISp", "L_SSp-ll", "R_SSp-ll", "L_SSp-m", "R_SSp-m"]
plot_rois = ["L_MOp", "R_MOp", "L_MOs", "R_MOs",]#] "L_VISp", "R_VISp", "L_SSp-ll", "R_SSp-ll", "L_SSp-m", "R_SSp-m",]

analysis = EtaByConditionAnalysis(
    roi_cols=tuple(roi_cols),
    task="task-spont",
    event_types=("onset", "offset"),
    window=(-1.0, 3.0),
    dt=0.02,
    baseline=(-5.0, 0.0),
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

plot_onset = EtaConditionPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="onset",
    baseline=analysis.baseline,
)
plot_offset = EtaConditionPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="offset",
    baseline=analysis.baseline,
)

source_features = [
    ("mesomap", roi_cols),
    ("pupil", ["pupil_diameter_mm"]),
    ("treadmill", ["speed_mm"]),
]
long = bench.build_long(
    df,
    source_features=source_features,
    tol=0.25,
    time_column="time_elapsed_s",
    reference_source="mesomap",
)

ses_to_cond = {
    "ses-01": "baseline",
    "ses-02": "saline",
    "ses-03": "ethanol_low",
    "ses-04": "ethanol_high",
}
long["Condition"] = long["Session"].map(ses_to_cond)

bench.preflight(
    analysis=analysis,
    plotter=plot_onset,
    df=long,
    required_columns=["Subject", "Session", "Task", "Condition", "time_elapsed_s", "speed_mm", *roi_cols],
)

res = bench.analyze(analysis, long)

fig_onset, _ = bench.plot(plot_onset, res)
#onset_plot_path = bench.save_figure(fig_onset, "eta_onset_rois.png")
bench.save_feature_plot(
    fig_onset,
    "eta_onset_rois.png",
    feature_name="ETA onset-aligned",
    plotter=plot_onset,
)
plt.close(fig_onset)

fig_offset, _ = bench.plot(plot_offset, res)
#offset_plot_path = bench.save_figure(fig_offset, "eta_offset_rois.png")
bench.save_feature_plot(
    fig_offset,
    "eta_offset_rois.png",
    feature_name="ETA offset-aligned",
    plotter=plot_offset,
)
plt.close(fig_offset)

saved_tables = bench.save_analysis_result_tables(res, prefix="eta")

run_summary_path = bench.save_run_summary(
    notes="Event-based ETA workflow using registered analysis/plotter and preflight guardrails.",
)
bench.save_provenance()

saved_paths = [
    # onset_plot_path,
    # offset_plot_path,
    *saved_tables.values(),
    run_summary_path,
]
for p in saved_paths:
    print(f"[databench] saved: {p}")

missing_paths = [str(p) for p in saved_paths if not Path(p).exists()]
if missing_paths:
    raise RuntimeError(f"Expected output files were not created: {missing_paths}")

# %%
