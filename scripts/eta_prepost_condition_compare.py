#%%
"""
Compare ETA pre/post dF/F differences for locomotion onset vs offset.

This script demonstrates:
1) explicit procedural flow (load -> preflight -> analyze -> plot -> save)
2) custom analysis + plotter classes reusing databench interfaces
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


@dataclass(frozen=True)
class EtaPrePostDiffAnalysis(Analysis):
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
            "Subject",
            "Session",
            "Task",
            self.condition_col,
            "time_elapsed_s",
            "speed_mm",
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
                            t,
                            y,
                            event_time,
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

                        rows.append(
                            {
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
                            }
                        )

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
            event_diff.groupby(["Subject", self.condition_col, "Task", "EventType", "ROI"], as_index=False)
            .agg(diff_mean=("diff", "mean"), n_events=("diff", "size"))
        )

        group_diff = (
            subject_diff.groupby([self.condition_col, "Task", "EventType", "ROI"], as_index=False)
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


@dataclass(frozen=True)
class EtaPrePostDiffBoxplot(Plotter):
    name: str = "eta_prepost_diff_boxplot"
    rois: tuple[str, ...] = ("L_MOp", "R_MOp", "L_MOs", "R_MOs")
    task: str = "task-spont"
    condition_col: str = "Condition"
    condition_order: tuple[str, ...] = ("baseline", "saline", "ethanol_low", "ethanol_high")
    event_order: tuple[str, ...] = ("onset", "offset")

    def plot(self, result: AnalysisResult):
        d = result.data["subject_diff"].copy()
        d = d.loc[d["Task"] == self.task]

        rois = list(self.rois)
        conds = list(self.condition_order)
        events = list(self.event_order)

        cond_colors = {
            "baseline": "#bbabab",
            "saline": "#4289e6",
            "ethanol_low": "#ffa251",
            "ethanol_high": "#ce1818",
        }

        nrows = len(events)
        ncols = len(rois)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.1 * ncols, 2.9 * nrows),
            sharex=False,
            sharey="row",
        )
        axes = np.atleast_2d(axes)
        x = np.arange(len(conds), dtype=float)

        for i, event_type in enumerate(events):
            for j, roi in enumerate(rois):
                ax = axes[i, j]
                dd = d[(d["EventType"] == event_type) & (d["ROI"] == roi)]

                data_by_cond = []
                for cond in conds:
                    y = dd.loc[dd[self.condition_col] == cond, "diff_mean"].dropna().to_numpy()
                    data_by_cond.append(y)

                bp = ax.boxplot(
                    data_by_cond,
                    positions=x,
                    widths=0.62,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.2},
                    whiskerprops={"color": "#444444", "linewidth": 1.0},
                    capprops={"color": "#444444", "linewidth": 1.0},
                    boxprops={"linewidth": 1.0, "edgecolor": "#444444"},
                )

                for patch, cond in zip(bp["boxes"], conds):
                    patch.set_facecolor(cond_colors.get(cond, "#999999"))
                    patch.set_alpha(0.45)

                for k, cond in enumerate(conds):
                    y = data_by_cond[k]
                    if y.size == 0:
                        continue
                    jitter = (np.random.rand(y.size) - 0.5) * 0.20
                    ax.scatter(
                        np.full(y.size, x[k], dtype=float) + jitter,
                        y,
                        s=20,
                        color=cond_colors.get(cond, "#999999"),
                        edgecolors="white",
                        linewidths=0.4,
                        alpha=0.9,
                        zorder=3,
                    )

                ax.axhline(0, color="k", lw=0.7, alpha=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(conds, rotation=25, ha="right")
                ax.tick_params(axis="both", labelsize=8)
                ax.grid(axis="y", alpha=0.25)

                if i == 0:
                    ax.set_title(roi, fontsize=10)
                if j == 0:
                    ax.set_ylabel(event_type, fontsize=9)

        fig.suptitle(
            "ETA pre/post difference by condition\n"
            "onset: [0,1] - [-1,0), offset: [-1,0) - [0,1]",
            y=0.99,
        )
        fig.supxlabel("Condition", y=0.04)
        fig.supylabel("Mean dF/F difference", x=0.02)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.14, hspace=0.35, wspace=0.18)
        return fig, axes


#%%
# Procedural workflow

pickle_path = Path(r"/Volumes/ake.bin/4jake/260211_ETOH_dataset.pkl")
project_root = Path(__file__).resolve().parents[1]
output_root = project_root / "outputs"

bench = Bench()
bench.setup(pickle_path, output_root=output_root, run_name="260215", tag="eta-prepost-condition_vis-primary-secondary")

paths = bench.output_paths
print(f"[databench] run_dir: {paths.run_dir}")
df = bench.load()
bouts_feature = bench.get_feature("locomotion_bouts_n")

#roi_cols = ["L_MOp", "R_MOp", "L_MOs", "R_MOs"]
roi_cols = ["L_VISp", "L_VISa", "L_SSp-ll", "L_SSp-bfd"]  # alternate set of ROIs to compare
source_features = [
    ("mesomap", roi_cols),
    ("treadmill", ["speed_mm"]),
]
long = bench.build_long(df, source_features=source_features, tol=0.25)

ses_to_cond = {
    "ses-01": "baseline",
    "ses-02": "saline",
    "ses-03": "ethanol_low",
    "ses-04": "ethanol_high",
}
long["Condition"] = long["Session"].map(ses_to_cond)

analysis = EtaPrePostDiffAnalysis(
    roi_cols=tuple(roi_cols),
    task="task-spont",
    pre_window=(-1.0, 0.0),
    post_window=(0.0, 1.0),
    dt=0.02,
    reverse_for_offset=False,
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

plotter = EtaPrePostDiffBoxplot(
    rois=tuple(roi_cols),
    task=analysis.task,
    condition_col=analysis.condition_col,
)

# TODO: preflight did not catch missing "L_SS-ll" column - empty plots
bench.preflight(
    analysis=analysis,
    plotter=plotter,
    df=long,
    required_columns=["Subject", "Session", "Task", "Condition", "time_elapsed_s", "speed_mm", *roi_cols],
)

res = bench.analyze(analysis, long)

fig, _ = bench.plot(plotter, res)
plot_path = bench.save_figure(fig, "eta_prepost_diff_boxplot.png")
plt.close(fig)

saved_tables = bench.save_analysis_result_tables(res, prefix="eta_prepost_diff")

run_summary_path = bench.save_run_summary(
    notes="ETA pre/post difference comparison across conditions for onset/offset locomotion events.",
)

saved_paths = [
    plot_path,
    *saved_tables.values(),
    run_summary_path,
]
for p in saved_paths:
    print(f"[databench] saved: {p}")


# %%