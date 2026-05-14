"""
Canonical locomotion condition comparison script.

Supports two modes:
    1. Single task (e.g., task-spont): condition comparison + LME outputs.
    2. Multiple tasks (e.g., task-spont, task-movies): paired condition/task plots.

Usage:
    python Scripts/locomotion/locomotion-condition-task-compare.py
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from databench.project import Project
from databench.analysis.locomotion import locomotion_bout_events
from databench.plotting.style import CONDITION_COLORS
from databench.config import resolve_dataset
from databench.types import BoutEventsTable
from databench.plotting import set_theme
from databench.utils import clean_xy

set_theme()

# ─── Helpers ──────────────────────────────────────────────────────────────


def _sem(x: pd.Series) -> float:
    n = int(x.notna().sum())
    return float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan


def detect_bouts_long(
    long: pd.DataFrame, *, speed_scale_to_cms: float = 100.0,
    min_speed_cms: float = 0.5, min_duration_s: float = 1.0, merge_gap_s: float = 0.5,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (subj, ses, task), g in long.groupby(["Subject", "Session", "Task"], sort=False):
        t, speed = clean_xy(
            g["time_elapsed_s"].to_numpy(),
            g["speed_mm"].to_numpy() / speed_scale_to_cms,
        )
        if t.size < 3:
            continue
        epochs: BoutEventsTable = locomotion_bout_events(
            t, speed, min_speed_cms=min_speed_cms,
            min_duration_s=min_duration_s, merge_gap_s=merge_gap_s,
        )
        for _, r in epochs.iterrows():
            rows.append({
                "Subject": subj, "Session": ses, "Task": task,
                "bout_id": int(r["epoch_id"]), "onset_idx": int(r["start_idx"]),
                "offset_idx": int(r["end_idx"]),
                "onset_t": r["start_s"], "offset_t": r["end_s"],
            })
    cols = ["Subject", "Session", "Task", "bout_id", "onset_idx", "offset_idx", "onset_t", "offset_t"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# ─── Analysis ─────────────────────────────────────────────────────────────


def locomotion_condition_task_analysis(
    long: pd.DataFrame, events: pd.DataFrame, *,
    condition_col: str = "Condition", speed_scale_to_cms: float = 100.0,
    min_speed_cms: float = 0.5,
) -> dict[str, pd.DataFrame]:
    key_cols = ["Subject", "Session", "Task"]
    grouped_events = {k: g for k, g in events.groupby(key_cols, sort=False)}

    session_rows: list[dict] = []
    bout_rows: list[dict] = []

    for key, g in long.groupby(key_cols, sort=False):
        subj, ses, task = key
        cond = g[condition_col].iloc[0]
        tt, vv = clean_xy(g["time_elapsed_s"].to_numpy(), g["speed_mm"].to_numpy())
        if tt.size < 3:
            continue

        speed_cms = vv / speed_scale_to_cms
        dt_med = float(np.nanmedian(np.diff(tt))) if tt.size > 1 else 0.0
        dur_s = float(tt[-1] - tt[0] + dt_med)
        dur_min = dur_s / 60.0 if dur_s > 0 else np.nan
        run_frac = float(np.mean(speed_cms >= min_speed_cms))
        mean_spd = float(np.nanmean(speed_cms))
        total_dist_m = float(np.nansum(np.abs(speed_cms[1:]) * np.diff(tt))) / 100.0

        ge = grouped_events.get(key)
        if ge is None or ge.empty:
            session_rows.append({
                "Subject": subj, "Session": ses, "Task": task, condition_col: cond,
                "session_duration_s": dur_s, "session_duration_min": dur_min,
                "running_fraction_time": run_frac, "overall_mean_speed_cms": mean_spd,
                "total_distance_m": total_dist_m, "n_bouts": 0, "bout_rate_per_min": 0.0,
                "mean_bout_duration_s": np.nan, "mean_bout_speed_cms": np.nan,
                "total_bout_distance_m": 0.0,
            })
            continue

        durs, spds, dists = [], [], []
        for _, r in ge.iterrows():
            s, e = int(r["onset_idx"]), int(r["offset_idx"])
            if s < 0 or e < s or e >= tt.size:
                continue
            d = float(tt[e] - tt[s] + dt_med)
            seg = np.abs(speed_cms[s: e + 1])
            ms = float(np.nanmean(seg))
            dm = float(np.nansum(seg[1:] * np.diff(tt[s: e + 1]))) / 100.0 if e > s else 0.0
            durs.append(d)
            spds.append(ms)
            dists.append(dm)
            bout_rows.append({
                "Subject": subj, "Session": ses, "Task": task, condition_col: cond,
                "bout_id": int(r["bout_id"]), "onset_t": float(r["onset_t"]),
                "offset_t": float(r["offset_t"]),
                "bout_duration_s": d, "bout_mean_speed_cms": ms, "bout_distance_m": dm,
            })

        n = len(durs)
        rate = (n / dur_min) if dur_min and np.isfinite(dur_min) else np.nan
        session_rows.append({
            "Subject": subj, "Session": ses, "Task": task, condition_col: cond,
            "session_duration_s": dur_s, "session_duration_min": dur_min,
            "running_fraction_time": run_frac, "overall_mean_speed_cms": mean_spd,
            "total_distance_m": total_dist_m, "n_bouts": n, "bout_rate_per_min": rate,
            "mean_bout_duration_s": float(np.nanmean(durs)) if durs else np.nan,
            "mean_bout_speed_cms": float(np.nanmean(spds)) if spds else np.nan,
            "total_bout_distance_m": float(np.nansum(dists)),
        })

    bout_table = pd.DataFrame(bout_rows)
    session_stats = pd.DataFrame(session_rows)

    stat_cols = [
        "overall_mean_speed_cms", "total_distance_m", "n_bouts", "bout_rate_per_min",
        "mean_bout_duration_s", "mean_bout_speed_cms", "total_bout_distance_m",
        "running_fraction_time",
    ]
    subject_stats = (
        session_stats.groupby(["Subject", condition_col, "Task"], as_index=False)[stat_cols]
        .mean(numeric_only=True)
    ) if not session_stats.empty else pd.DataFrame()

    group_rows: list[pd.DataFrame] = []
    if not subject_stats.empty:
        for m in stat_cols:
            gg = subject_stats.groupby([condition_col, "Task"], as_index=False).agg(
                mean=(m, "mean"), sem=(m, _sem), n_subjects=("Subject", "nunique"),
            )
            gg.insert(0, "metric", m)
            group_rows.append(gg)
    group_stats = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()

    return {
        "events": events, "bout_table": bout_table,
        "session_stats": session_stats, "subject_stats": subject_stats,
        "group_stats": group_stats,
    }


# ─── Plotter ──────────────────────────────────────────────────────────────


def plot_condition_task(
    subject_stats: pd.DataFrame, *,
    condition_col: str = "Condition",
    condition_order: tuple[str, ...] = ("baseline", "saline", "ethanol_low", "ethanol_high"),
    task_order: tuple[str, ...] = ("task-spont", "task-movies"),
) -> plt.Figure:
    metrics = [
        "overall_mean_speed_cms", "total_distance_m", "n_bouts", "bout_rate_per_min",
        "mean_bout_duration_s", "mean_bout_speed_cms", "total_bout_distance_m",
        "running_fraction_time",
    ]
    labels = {
        "overall_mean_speed_cms": "Mean speed (cm/s)", "total_distance_m": "Total distance (m)",
        "n_bouts": "Bout count", "bout_rate_per_min": "Bouts / min",
        "mean_bout_duration_s": "Mean bout duration (s)",
        "mean_bout_speed_cms": "Mean bout speed (cm/s)",
        "total_bout_distance_m": "Bout distance (m)",
        "running_fraction_time": "Running fraction",
    }
    conds = list(condition_order)
    x = np.arange(len(conds), dtype=float)
    nrows = int(np.ceil(len(metrics) / 2))
    ncols = 4  # 2 tasks × 2 metric columns

    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 11.0), squeeze=False)

    for mi, metric in enumerate(metrics):
        row_i = mi // 2
        pair_i = mi % 2
        col0 = pair_i * 2
        axes[row_i, col0 + 1].sharey(axes[row_i, col0])

        for ti, task in enumerate(task_order):
            ax = axes[row_i, col0 + ti]
            dt = subject_stats[subject_stats["Task"] == task]
            data = [dt.loc[dt[condition_col] == c, metric].dropna().to_numpy() for c in conds]

            bp = ax.boxplot(
                data, positions=x, widths=0.62, patch_artist=True, showfliers=False,
                medianprops={"color": "black", "linewidth": 1.2},
            )
            for patch, c in zip(bp["boxes"], conds):
                patch.set_facecolor(CONDITION_COLORS.get(c, "#999"))
                patch.set_alpha(0.45)
            for i, c in enumerate(conds):
                y = data[i]
                if y.size:
                    jitter = (np.random.rand(y.size) - 0.5) * 0.2
                    ax.scatter(x[i] + jitter, y, s=20, color=CONDITION_COLORS.get(c, "#999"),
                               edgecolors="white", linewidths=0.4, alpha=0.9, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(conds, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
            if ti == 0:
                ax.set_ylabel(labels.get(metric, metric))
            else:
                ax.tick_params(labelleft=False)
            ax.set_title(task)

    fig.suptitle("Locomotion comparison across conditions", y=0.99)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, hspace=0.55, wspace=0.28)
    return fig


def fit_condition_lme(
    session_stats: pd.DataFrame,
    *,
    task: str,
    condition_col: str = "Condition",
    metric: str = "bout_rate_per_min",
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Fit condition and dose-trend mixed models for a single task.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame] | None
        (condition coefficients, dose-trend coefficients), or None when
        fitting prerequisites are not met.
    """
    if session_stats.empty:
        return None

    required_conditions = ["saline", "ethanol_low", "ethanol_high", "baseline"]
    df_lme = session_stats[session_stats["Task"] == task].dropna(subset=[metric]).copy()
    if df_lme.empty or df_lme["Subject"].nunique() < 2:
        return None

    df_lme[condition_col] = pd.Categorical(
        df_lme[condition_col],
        categories=required_conditions,
        ordered=False,
    )

    model = smf.mixedlm(
        f"{metric} ~ C({condition_col})",
        data=df_lme,
        groups=df_lme["Subject"],
    )
    lme_fit = model.fit(reml=True)

    print("\n" + "=" * 72)
    print(f"LME MODEL: {metric} ~ Condition + (1 | Subject)")
    print("Reference level: saline")
    print("=" * 72)
    print(lme_fit.summary())

    dose_map = {"saline": 0, "ethanol_low": 1, "ethanol_high": 2}
    df_dose = df_lme[df_lme[condition_col].isin(dose_map.keys())].copy()
    if df_dose.empty:
        return None
    df_dose["dose"] = df_dose[condition_col].map(dose_map).astype(float)

    trend_model = smf.mixedlm(
        f"{metric} ~ dose",
        data=df_dose,
        groups=df_dose["Subject"],
    )
    trend_fit = trend_model.fit(reml=True)

    print("\n" + "=" * 72)
    print("LINEAR DOSE TREND: saline(0) -> low(1) -> high(2)")
    print("=" * 72)
    print(trend_fit.summary())

    coef_df = pd.DataFrame({
        "estimate": lme_fit.fe_params,
        "se": lme_fit.bse_fe,
        "z": lme_fit.tvalues,
        "p": lme_fit.pvalues,
    })
    trend_df = pd.DataFrame({
        "estimate": trend_fit.fe_params,
        "se": trend_fit.bse_fe,
        "z": trend_fit.tvalues,
        "p": trend_fit.pvalues,
    })
    return coef_df, trend_df


# ─── Procedural workflow ──────────────────────────────────────────────────

DATASET = resolve_dataset("etoh")
TASKS = ("task-spont", "task-movies")
OUTPUT_ROOT = "outputs"
RUN_NAME = "locomotion-condition-compare"
TAG = "canonical"
OUTPUT_PREFIX = "locomotion_condition_compare"
SESSION_TO_CONDITION = {
    "ses-01": "baseline",
    "ses-02": "saline",
    "ses-03": "ethanol_low",
    "ses-04": "ethanol_high",
}

proj = Project(
    dataset=DATASET,
    output_root=OUTPUT_ROOT,
    run_name=RUN_NAME,
    tag=TAG,
)

# Combine sessions from both tasks
frames = []
for task in TASKS:
    group = proj.sessions(task=task)
    ad = group.align({"treadmill": ["speed_mm"]}, reference="treadmill", tolerance_s=0.25)
    frames.append(ad.df)

long = pd.concat(frames, ignore_index=True)
long["Condition"] = long["Session"].map(SESSION_TO_CONDITION)
print(f"Built long table: {len(long)} rows, tasks: {long['Task'].unique().tolist()}")

# Detect bouts
bout_events = detect_bouts_long(long, speed_scale_to_cms=100.0, min_speed_cms=0.5)

# Analyze
tables = locomotion_condition_task_analysis(long, bout_events, speed_scale_to_cms=100.0, min_speed_cms=0.5)

# Plot
fig = plot_condition_task(tables["subject_stats"], task_order=TASKS)
proj.io.figure(fig, f"{OUTPUT_PREFIX}_summary.svg")

# Save
proj.io.tables(
    {name: df for name, df in tables.items() if isinstance(df, pd.DataFrame) and not df.empty},
    prefix=f"{OUTPUT_PREFIX}_",
)

if len(TASKS) == 1:
    lme_tables = fit_condition_lme(tables["session_stats"], task=TASKS[0])
    if lme_tables is not None:
        coef_df, trend_df = lme_tables
        proj.io.table(coef_df, f"{OUTPUT_PREFIX}_lme_condition_coefficients.csv", index=True)
        proj.io.table(trend_df, f"{OUTPUT_PREFIX}_lme_dose_trend_coefficients.csv", index=True)

proj.io.report(
    notes=(
        "Locomotion bout/stats comparison across condition labels. "
        f"Tasks: {', '.join(TASKS)}. "
        "When a single task is selected, LME condition and dose-trend outputs are saved."
    ),
)
print("Done.")
