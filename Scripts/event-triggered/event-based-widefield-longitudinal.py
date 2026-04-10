"""
Event-based ETA for longitudinal widefield recordings.

Pools event-triggered averages across all recording days and generates:
    1. Pooled average ETA traces (all days combined)
    2. Per-day heatmaps (session × rel_time)
    3. Longitudinal metric plots (peak response per session)

Usage:
    python Scripts/event-based-widefield-longitudinal.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench.project import Project
from databench.analysis.eta import EtaAnalysis
from databench.analysis.locomotion import locomotion_events
from databench.config import resolve_dataset
from databench.session import SaveableFigure
from databench._utils._logger import get_logger
from databench.plotting import set_theme

set_theme()

_log = get_logger(__name__)

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET = resolve_dataset()
ROI_COLUMNS = ("L_MOp", "R_MOp", "L_MOs", "R_MOs", "L_VISp", "R_VISp")
TASK = "task-widefield"
WINDOW = (-1.0, 1.0)
BASELINE = (-2.0, 0.0)
METRIC_WINDOW = (0.0, 2.0)

# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name="widefield-10day",
    tag="ROI-speed_eta-2s",
)

group = proj.sessions(task=TASK)
_log.info(f"Selected {len(group)} sessions for task={TASK}")

# ─── Detect events ───────────────────────────────────────────────────────

events = locomotion_events(
    group,
    speed_source="treadmill",
    speed_column="speed_mm",
    speed_scale_to_cms=10.0,
    min_speed_cms=0.5,
    min_duration_s=1.0,
    merge_gap_s=0.5,
)
_log.info(f"Detected {len(events)} locomotion events")

# ─── Run ETA ─────────────────────────────────────────────────────────────

eta = EtaAnalysis(
    roi_columns=ROI_COLUMNS,
    window=WINDOW,
    dt=0.02,
    baseline=BASELINE,
    source="mesomap",
    reference_source="mesomap",
    alignment_tolerance_s=0.25,
)

result = eta.run(group, events)

# ─── 1. Pooled average plots ─────────────────────────────────────────────

for etype in result.event_types:
    result.plot(event=etype, rois=list(ROI_COLUMNS)).save(f"eta_{etype}_rois_10day_avg.svg")

# ─── 2. Per-day heatmaps ─────────────────────────────────────────────────

# Compute per-session mean ETA (day-level)
session_means = (
    result.eta_events
    .groupby(["Subject", "Session", "EventType", "ROI", "rel_time"], sort=False)
    .agg(mean=("value", "mean"))
    .reset_index()
)

# Extract session index (ses-01 → 1, etc.) for ordering
session_means = proj.tabler.add_session_number(session_means)
sorted_sessions = proj.tabler.sorted_sessions(session_means)

for etype in result.event_types:
    ncols = min(3, len(ROI_COLUMNS))
    nrows = int(np.ceil(len(ROI_COLUMNS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for i, roi in enumerate(ROI_COLUMNS):
        ax = axes_flat[i]
        sub = session_means[
            (session_means["EventType"] == etype) & (session_means["ROI"] == roi)
        ]
        if sub.empty:
            ax.set_visible(False)
            continue

        # Build matrix: rows = sessions, cols = time bins
        pivot = proj.tabler.pivot_time(
            sub,
            session_col="Session",
            time_col="rel_time",
            value_col="mean",
            aggfunc="mean",
            session_order=sorted_sessions,
        )

        if pivot.empty:
            ax.set_visible(False)
            continue

        im = ax.imshow(
            pivot.values, aspect="auto",
            extent=[pivot.columns.min(), pivot.columns.max(), len(pivot) - 0.5, -0.5],
            cmap="RdBu_r", interpolation="nearest",
        )
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_title(roi)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Longitudinal ETA heatmap — {etype}")
    fig.tight_layout()
    SaveableFigure(fig, proj._context).save(f"eta_{etype}_longitudinal_heatmap.svg")

# ─── 3. Longitudinal metric plot ─────────────────────────────────────────

# Metric: mean response in METRIC_WINDOW per (Subject, Session, ROI, EventType)
eta_ev = result.eta_events.copy()
metric_mask = (eta_ev["rel_time"] >= METRIC_WINDOW[0]) & (eta_ev["rel_time"] <= METRIC_WINDOW[1])
metric_df = (
    eta_ev.loc[metric_mask]
    .groupby(["Subject", "Session", "EventType", "ROI", "event_id"], sort=False)
    .agg(peak=("value", "mean"))
    .reset_index()
    .groupby(["Subject", "Session", "EventType", "ROI"], sort=False)
    .agg(metric=("peak", "mean"))
    .reset_index()
)
metric_df = proj.tabler.add_session_number(metric_df)

for etype in result.event_types:
    ncols = min(3, len(ROI_COLUMNS))
    nrows = int(np.ceil(len(ROI_COLUMNS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for i, roi in enumerate(ROI_COLUMNS):
        ax = axes_flat[i]
        sub = metric_df[(metric_df["EventType"] == etype) & (metric_df["ROI"] == roi)]
        if sub.empty:
            ax.set_visible(False)
            continue

        # Plot per-subject lines + group mean
        for subj, sg in sub.groupby("Subject"):
            sg = sg.sort_values("session_n")
            ax.plot(sg["session_n"], sg["metric"], "o-", alpha=0.4, ms=4, label=subj)

        group_mean = proj.tabler.group_mean_sem(
            sub,
            group_col="session_n",
            value_col="metric",
        )
        ax.errorbar(
            group_mean["session_n"], group_mean["mean"],
            yerr=group_mean["sem"], fmt="s-", color="black", ms=6, lw=2, zorder=5,
        )
        ax.set_title(roi)
        ax.set_xlabel("Session")
        if i % ncols == 0:
            ax.set_ylabel(f"Mean ΔF/F [{METRIC_WINDOW[0]}, {METRIC_WINDOW[1]}]s")
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Longitudinal metric — {etype}")
    fig.tight_layout()
    SaveableFigure(fig, proj._context).save(f"eta_{etype}_longitudinal_metric.svg")

# ─── Save ─────────────────────────────────────────────────────────────────

result.save_tables(prefix="eta_widefield_10day")

stats_dir = proj._context.stats_dir
stats_dir.mkdir(parents=True, exist_ok=True)
metric_df.to_csv(stats_dir / "eta_longitudinal_metric.csv", index=False)

report_path = proj.save_report(
    result,
    notes=(
        "Widefield event-based ETA workflow for a longitudinal 10-day dataset. "
        "Includes pooled 10-day averages and day-wise longitudinal visualizations."
    ),
)

print(f"Done — {len(result.events)} events across {result.events['Subject'].nunique()} subjects.")
print(f"Report: {report_path}")
