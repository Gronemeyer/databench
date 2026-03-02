#%%
"""
Compare locomotion bout statistics across conditions and tasks.

This script demonstrates:
1) explicit procedural flow (load -> build_long -> label -> analyze -> plot -> save)
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
from databench.config import CONDITION_COLORS, CONDITION_ORDER
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

            # overall (full-trace) speed and distance stats
            overall_mean_speed_cms = float(np.nanmean(speed_cms))
            dt_arr = np.diff(tt)
            total_distance_m = float(np.nansum(np.abs(speed_cms[1:]) * dt_arr)) / 100.0

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
                        "overall_mean_speed_cms": overall_mean_speed_cms,
                        "total_distance_m": total_distance_m,
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
                    "overall_mean_speed_cms": overall_mean_speed_cms,
                    "total_distance_m": total_distance_m,
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
            "overall_mean_speed_cms",
            "total_distance_m",
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
        "overall_mean_speed_cms",
        "total_distance_m",
        "n_bouts",
        "bout_rate_per_min",
        "mean_bout_duration_s",
        "mean_bout_speed_cms",
        "total_bout_distance_m",
        "running_fraction_time",
    )

    def plot(self, result: AnalysisResult):
        subject_stats = result.data["subject_stats"]
        d = subject_stats.copy()

        metric_labels = {
            "overall_mean_speed_cms": "Mean speed (cm/s)",
            "total_distance_m": "Total distance (m)",
            "n_bouts": "Bout count",
            "bout_rate_per_min": "Bouts / min",
            "mean_bout_duration_s": "Mean bout duration (s)",
            "mean_bout_speed_cms": "Mean bout speed (cm/s)",
            "total_bout_distance_m": "Total bout distance (m)",
            "running_fraction_time": "Running time fraction",
        }

        n_metrics = len(self.metrics)
        conds = list(self.condition_order)
        tasks = list(self.task_order)
        n_tasks = len(tasks)
        x = np.arange(len(conds), dtype=float)

        if n_tasks != 2:
            raise ValueError("This plot layout expects exactly 2 tasks.")

        nrows = int(np.ceil(n_metrics / 2))
        ncols = 4

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(15.5, 11.0),
            sharex=False,
            squeeze=False,
        )

        for metric_idx, metric in enumerate(self.metrics):
            row_idx = metric_idx // 2
            pair_idx = metric_idx % 2
            col0 = pair_idx * 2
            col1 = col0 + 1

            # share y-axis between task panels for the same metric
            axes[row_idx, col1].sharey(axes[row_idx, col0])

            for task_offset, task in enumerate(tasks):
                col_idx = col0 + task_offset
                ax = axes[row_idx, col_idx]
                dt = d[d["Task"] == task]

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
                    patch.set_facecolor(CONDITION_COLORS.get(cond, "#999999"))
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
                        color=CONDITION_COLORS.get(cond, "#999999"),
                        edgecolors="white",
                        linewidths=0.4,
                        alpha=0.9,
                        zorder=3,
                    )

                ax.set_xticks(x)
                ax.set_xticklabels(conds, rotation=25, ha="right")
                ax.tick_params(axis="both", labelsize=9)
                ax.grid(axis="y", alpha=0.25)

                if task_offset == 0:
                    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=10)
                else:
                    ax.tick_params(labelleft=False)

                ax.set_title(task, fontsize=10)

        fig.suptitle("Locomotion comparison across conditions", y=0.99, fontsize=13)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, hspace=0.55, wspace=0.28)

        return fig, axes


#%%
# Procedural workflow

from databench.config import resolve_dataset
DATASET = resolve_dataset()

bench = Bench()
bench.setup(
    DATASET,
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name="locomotion",
    tag="locomotion-condition-task",
)

bouts_feature = bench.get_feature("locomotion_bouts_n")

ses_to_cond = {
    "ses-01": "baseline",
    "ses-02": "saline",
    "ses-03": "ethanol_low",
    "ses-04": "ethanol_high",
}

analysis = LocomotionByConditionTaskAnalysis(
    task_filter=("task-spont", "task-movies"),
    condition_col="Condition",
    bout_events_extractor=LocomotionBoutEventsExtractor.from_feature(bouts_feature),
)

plotter = LocomotionConditionTaskPlotter(
    condition_col=analysis.condition_col,
    task_order=analysis.task_filter,
)

with bench.run("locomotion-condition-task") as run:
    (run
        .build_long(
            sources=[("treadmill", ["speed_mm"])],
            tol=0.25,
            time_column="time_elapsed_s",
            reference_source="treadmill",
        )
        .label_conditions(ses_to_cond))

    result = run.analyze(analysis, df=run.long)
    run.plot(plotter, result, save="locomotion_condition_summary.png")
    run.save_tables(result, prefix="locomotion_compare")
    run.save_run_summary(
        notes="Locomotion bout/stats comparison across conditions and tasks using DataFrame event API.",
    )

# %%
