#%%
"""
Compare locomotion bout statistics across conditions and tasks.

This script demonstrates:
1) explicit procedural flow (load -> preflight -> analyze -> plot -> save)
2) custom analysis + plotter classes for locomotion comparison
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
from databench.features.treadmill import LocomotionBoutEventsExtractor
from databench.plotting.base import Plotter


def _sem(x: pd.Series) -> float:
    n = int(x.notna().sum())
    if n <= 1:
        return np.nan
    return float(x.std(ddof=1) / np.sqrt(n))


@dataclass(frozen=True)
class LocomotionByConditionTaskAnalysis(Analysis):
    name: str = "locomotion_by_condition_task"
    task_filter: tuple[str, ...] = ("task-spont", "task-movies")
    condition_col: str = "Condition"
    group_cols: tuple[str, ...] = ("Subject", "Session", "Task")
    time_col: str = "time_elapsed_s"
    speed_col: str = "speed_mm"
    speed_scale_to_cms: float = 100.0
    bout_events_extractor: Any = LocomotionBoutEventsExtractor()

    def run(self, long: pd.DataFrame) -> AnalysisResult:
        required = [*self.group_cols, self.condition_col, self.time_col, self.speed_col]
        missing = [c for c in required if c not in long.columns]
        if missing:
            raise ValueError(f"Missing required columns in long table: {', '.join(missing)}")

        d = long.copy()
        if self.task_filter:
            d = d[d["Task"].isin(self.task_filter)]

        events = self.bout_events_extractor(d)
        if not isinstance(events, pd.DataFrame):
            raise TypeError("bout_events_extractor must return a pandas DataFrame.")

        required_event_cols = {
            "Subject",
            "Session",
            "Task",
            "bout_id",
            "onset_idx",
            "offset_idx",
            "onset_t",
            "offset_t",
        }
        missing_event_cols = sorted(required_event_cols.difference(events.columns))
        if missing_event_cols:
            raise ValueError(f"Extracted event table missing columns: {', '.join(missing_event_cols)}")

        key_cols = ["Subject", "Session", "Task"]

        if self.condition_col not in events.columns:
            cond_map = (
                d[["Subject", "Session", "Task", self.condition_col]]
                .drop_duplicates(subset=key_cols)
                .set_index(key_cols)[self.condition_col]
            )
            events = events.copy()
            events[self.condition_col] = [
                cond_map.get((s, sess, task), np.nan)
                for s, sess, task in zip(events["Subject"], events["Session"], events["Task"])
            ]

        session_rows: list[dict[str, Any]] = []
        bout_rows: list[dict[str, Any]] = []

        min_speed_cms = float(self.bout_events_extractor.min_speed_cms)

        grouped_events = {
            k: g.reset_index(drop=True)
            for k, g in events.groupby(key_cols, sort=False)
        }

        for key, g in d.groupby(key_cols, sort=False):
            subj, ses, task = key
            cond = g[self.condition_col].iloc[0]

            g = g.sort_values(self.time_col)
            t_raw = g[self.time_col].to_numpy()
            v_raw = g[self.speed_col].to_numpy()
            valid = np.isfinite(t_raw) & np.isfinite(v_raw)
            tt = t_raw[valid]
            vv_mm = v_raw[valid]

            if tt.size < 3:
                continue

            speed_cms = vv_mm / self.speed_scale_to_cms
            dt_med = float(np.nanmedian(np.diff(tt))) if tt.size > 1 else 0.0
            session_duration_s = float(tt[-1] - tt[0] + dt_med)
            session_duration_min = session_duration_s / 60.0 if session_duration_s > 0 else np.nan
            running_fraction_time = float(np.mean(speed_cms >= min_speed_cms))

            ge = grouped_events.get(key)
            if ge is None or ge.empty:
                session_rows.append(
                    {
                        "Subject": subj,
                        "Session": ses,
                        "Task": task,
                        self.condition_col: cond,
                        "session_duration_s": session_duration_s,
                        "session_duration_min": session_duration_min,
                        "running_fraction_time": running_fraction_time,
                        "n_bouts": 0,
                        "bout_rate_per_min": 0.0,
                        "mean_bout_duration_s": np.nan,
                        "mean_bout_speed_cms": np.nan,
                        "total_bout_distance_m": 0.0,
                    }
                )
                continue

            durations: list[float] = []
            mean_speeds: list[float] = []
            distances_m: list[float] = []

            for _, row in ge.iterrows():
                s = int(row["onset_idx"])
                e = int(row["offset_idx"])
                if s < 0 or e < s or e >= tt.size:
                    continue

                duration_s = float(tt[e] - tt[s] + dt_med)
                segment_speed_cms = np.abs(speed_cms[s : e + 1])
                mean_speed_cms = float(np.nanmean(segment_speed_cms))

                if e > s:
                    dt = np.diff(tt[s : e + 1])
                    dist_cm = float(np.nansum(segment_speed_cms[1:] * dt))
                else:
                    dist_cm = 0.0
                distance_m = dist_cm / 100.0

                durations.append(duration_s)
                mean_speeds.append(mean_speed_cms)
                distances_m.append(distance_m)

                bout_rows.append(
                    {
                        "Subject": subj,
                        "Session": ses,
                        "Task": task,
                        self.condition_col: cond,
                        "bout_id": int(row["bout_id"]),
                        "onset_t": float(row["onset_t"]),
                        "offset_t": float(row["offset_t"]),
                        "bout_duration_s": duration_s,
                        "bout_mean_speed_cms": mean_speed_cms,
                        "bout_distance_m": distance_m,
                    }
                )

            n_bouts = len(durations)
            bout_rate_per_min = (n_bouts / session_duration_min) if session_duration_min and np.isfinite(session_duration_min) and session_duration_min > 0 else np.nan

            session_rows.append(
                {
                    "Subject": subj,
                    "Session": ses,
                    "Task": task,
                    self.condition_col: cond,
                    "session_duration_s": session_duration_s,
                    "session_duration_min": session_duration_min,
                    "running_fraction_time": running_fraction_time,
                    "n_bouts": n_bouts,
                    "bout_rate_per_min": bout_rate_per_min,
                    "mean_bout_duration_s": float(np.nanmean(durations)) if durations else np.nan,
                    "mean_bout_speed_cms": float(np.nanmean(mean_speeds)) if mean_speeds else np.nan,
                    "total_bout_distance_m": float(np.nansum(distances_m)) if distances_m else 0.0,
                }
            )

        bout_table = pd.DataFrame(bout_rows)
        session_stats = pd.DataFrame(session_rows)

        if session_stats.empty:
            return AnalysisResult(
                name=self.name,
                data={
                    "events": events,
                    "bout_table": bout_table,
                    "session_stats": session_stats,
                    "subject_stats": pd.DataFrame(),
                    "group_stats": pd.DataFrame(),
                },
                meta={"tasks": self.task_filter, "condition_col": self.condition_col},
            )

        stat_cols = [
            "n_bouts",
            "bout_rate_per_min",
            "mean_bout_duration_s",
            "mean_bout_speed_cms",
            "total_bout_distance_m",
            "running_fraction_time",
        ]

        subject_stats = (
            session_stats.groupby(["Subject", self.condition_col, "Task"], as_index=False)[stat_cols]
            .mean(numeric_only=True)
        )

        group_rows: list[pd.DataFrame] = []
        for metric in stat_cols:
            g = (
                subject_stats.groupby([self.condition_col, "Task"], as_index=False)
                .agg(
                    mean=(metric, "mean"),
                    sem=(metric, _sem),
                    n_subjects=("Subject", "nunique"),
                )
            )
            g.insert(0, "metric", metric)
            group_rows.append(g)

        group_stats = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()

        return AnalysisResult(
            name=self.name,
            data={
                "events": events,
                "bout_table": bout_table,
                "session_stats": session_stats,
                "subject_stats": subject_stats,
                "group_stats": group_stats,
            },
            meta={"tasks": self.task_filter, "condition_col": self.condition_col},
        )


@dataclass(frozen=True)
class LocomotionConditionTaskPlotter(Plotter):
    name: str = "locomotion_condition_task"
    condition_col: str = "Condition"
    condition_order: tuple[str, ...] = ("baseline", "saline", "ethanol_low", "ethanol_high")
    task_order: tuple[str, ...] = ("task-spont", "task-movies")
    metrics: tuple[str, ...] = (
        "n_bouts",
        "bout_rate_per_min",
        "mean_bout_duration_s",
        "mean_bout_speed_cms",
        "total_bout_distance_m",
        "running_fraction_time",
    )
    ncols: int = 2

    def plot(self, result: AnalysisResult):
        subject_stats = result.data["subject_stats"]
        d = subject_stats.copy()

        metric_labels = {
            "n_bouts": "Bout count",
            "bout_rate_per_min": "Bouts / min",
            "mean_bout_duration_s": "Mean bout duration (s)",
            "mean_bout_speed_cms": "Mean bout speed (cm/s)",
            "total_bout_distance_m": "Total bout distance (m)",
            "running_fraction_time": "Running time fraction",
        }

        cond_colors = {
            "baseline": "#bbabab",
            "saline": "#4289e6",
            "ethanol_low": "#ffa251",
            "ethanol_high": "#ce1818",
        }

        n = len(self.metrics)
        nrows = int(np.ceil(n / self.ncols))
        conds = list(self.condition_order)
        tasks = list(self.task_order)
        x = np.arange(len(conds), dtype=float)
        task_figs: dict[str, tuple[Any, Any]] = {}

        for task in tasks:
            dt = d[d["Task"] == task].copy()

            fig, axes = plt.subplots(
                nrows,
                self.ncols,
                figsize=(7.2 * self.ncols / 2, 3.1 * nrows),
                sharex=False,
                sharey=False,
            )
            axes = np.atleast_1d(axes).ravel()

            for ax, metric in zip(axes, self.metrics):
                data_by_cond = []
                for cond in conds:
                    y = dt.loc[dt[self.condition_col] == cond, metric].dropna().to_numpy()
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

                for i, cond in enumerate(conds):
                    y = data_by_cond[i]
                    if y.size == 0:
                        continue
                    jitter = (np.random.rand(y.size) - 0.5) * 0.20
                    ax.scatter(
                        np.full(y.size, x[i], dtype=float) + jitter,
                        y,
                        s=20,
                        color=cond_colors.get(cond, "#999999"),
                        edgecolors="white",
                        linewidths=0.4,
                        alpha=0.9,
                        zorder=3,
                    )

                ax.set_title(metric_labels.get(metric, metric), fontsize=10)
                ax.set_xticks(x)
                ax.set_xticklabels(conds, rotation=25, ha="right")
                ax.tick_params(axis="both", labelsize=9)
                ax.grid(axis="y", alpha=0.25)

            for ax in axes[n:]:
                ax.axis("off")

            fig.suptitle(f"Locomotion comparison across conditions — {task}", y=0.99)
            fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.10, hspace=0.40, wspace=0.22)
            task_figs[task] = (fig, axes)

        return task_figs


#%%
# Procedural workflow

pickle_path = Path(r"D:\4jake\260211_ETOH_dataset.pkl")
project_root = Path(__file__).resolve().parents[1]
output_root = project_root / "outputs"

bench = Bench()
bench.setup(pickle_path, output_root=output_root, run_name="260214", tag="locomotion-condition-task")

paths = bench.output_paths
print(f"[databench] run_dir: {paths.run_dir}")
df = bench.load()
bouts_feature = bench.get_feature("locomotion_bouts_n")

source_features = [("treadmill", ["speed_mm"])]
long = bench.build_long(
    df,
    source_features=source_features,
    tol=0.25,
    time_column="time_elapsed_s",
    reference_source="treadmill",
)

ses_to_cond = {
    "ses-01": "baseline",
    "ses-02": "saline",
    "ses-03": "ethanol_low",
    "ses-04": "ethanol_high",
}
long["Condition"] = long["Session"].map(ses_to_cond)

analysis = LocomotionByConditionTaskAnalysis(
    task_filter=("task-spont", "task-movies"),
    condition_col="Condition",
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

plotter = LocomotionConditionTaskPlotter(
    condition_col=analysis.condition_col,
    task_order=analysis.task_filter,
)

bench.preflight(
    analysis=analysis,
    plotter=plotter,
    df=long,
    required_columns=["Subject", "Session", "Task", "Condition", "time_elapsed_s", "speed_mm"],
)

res = bench.analyze(analysis, long)

task_figs = bench.plot(plotter, res)
plot_paths: list[Path] = []
for task_name, (fig, _) in task_figs.items():
    safe_task = task_name.replace("task-", "")
    p = bench.save_figure(fig, f"locomotion_condition_summary_{safe_task}.png")
    plot_paths.append(p)
    plt.close(fig)

saved_tables = bench.save_analysis_result_tables(res, prefix="locomotion_compare")

run_summary_path = bench.save_run_summary(
    notes="Locomotion bout/stats comparison across conditions and tasks using DataFrame event API.",
)

saved_paths = [
    *plot_paths,
    *saved_tables.values(),
    run_summary_path,
]
for p in saved_paths:
    print(f"[databench] saved: {p}")

missing_paths = [str(p) for p in saved_paths if not Path(p).exists()]
if missing_paths:
    raise RuntimeError(f"Expected output files were not created: {missing_paths}")

# %%
