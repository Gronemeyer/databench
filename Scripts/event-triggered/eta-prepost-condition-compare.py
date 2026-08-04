"""
Compare ETA pre/post ΔF/F differences for locomotion onset vs offset.

Computes ETA at locomotion events, measures mean pre-window and post-window
signal, and plots the difference as boxplots across conditions.

Usage:
    python Scripts/eta-prepost-condition-compare.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench.project import Project
from databench.analysis.eta import EtaAnalysis
from databench.analysis.locomotion import locomotion_events
from databench.config import resolve_dataset
from databench.types import EventsTable
from databench.utils.logger import get_logger
from databench.plotting import set_theme

set_theme()

_log = get_logger(__name__)

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET = resolve_dataset("etoh")
ROI_COLUMNS = ("L_VISp", "L_VISa", "L_SSp-ll", "L_SSp-bfd")
TASK = "task-spont"
PRE_WINDOW = (-1.0, 0.0)
POST_WINDOW = (0.0, 1.0)
FULL_WINDOW = (-2.0, 3.0)
BASELINE = (-2.0, -1.0)

# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(dataset=DATASET)

# Per-dataset metadata pulled from datasets.toml [datasets.etoh]
SESSION_TO_CONDITION = proj.params["session_map"]
CONDITION_ORDER = tuple(proj.params["condition_order"])
CONDITION_COLORS = proj.params["condition_colors"]

group = proj.sessions(task=TASK)
run = proj.run(name="eta-prepost", tag="vis-primary-secondary")

# ─── Detect events & add conditions ──────────────────────────────────────

events: EventsTable = locomotion_events(group, min_speed_cms=0.5, min_duration_s=1.0, merge_gap_s=0.5)
events["Condition"] = events["Session"].map(SESSION_TO_CONDITION)
_log.info(f"Detected {len(events)} events across {events['Subject'].nunique()} subjects")

# ─── Run ETA ─────────────────────────────────────────────────────────────

eta = EtaAnalysis(
    roi_columns=ROI_COLUMNS,
    window=FULL_WINDOW,
    dt=0.02,
    baseline=BASELINE,
    source="mesomap",
    reference_source="mesomap",
    alignment_tolerance_s=0.25,
)

result = eta.run(group, events)

# ─── Compute pre/post difference per event ───────────────────────────────

pre_df  = result.window_mean(PRE_WINDOW,  name="pre_mean")
post_df = result.window_mean(POST_WINDOW, name="post_mean")
merge_keys = [c for c in pre_df.columns if c != "pre_mean"]
diff_df = pre_df.merge(post_df, on=merge_keys, how="inner")
diff_df["diff"] = diff_df["post_mean"] - diff_df["pre_mean"]

# ─── Plot pre/post diff boxplot ──────────────────────────────────────────

conds = list(CONDITION_ORDER)

for etype in result.event_types:
    fig, axes = plt.subplots(
        1, len(ROI_COLUMNS), figsize=(4 * len(ROI_COLUMNS), 4), sharey=True,
    )
    if len(ROI_COLUMNS) == 1:
        axes = [axes]

    sub = diff_df[diff_df["EventType"] == etype]

    for ax, roi in zip(axes, ROI_COLUMNS):
        roi_data = sub[sub["ROI"] == roi]
        bp_data = [
            roi_data.loc[roi_data["Condition"] == c, "diff"].dropna().values
            for c in conds
        ]
        x = np.arange(len(conds))
        bp = ax.boxplot(
            bp_data, positions=x, widths=0.6, patch_artist=True, showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch, cond in zip(bp["boxes"], conds):
            patch.set_facecolor(CONDITION_COLORS.get(cond, "#999999"))
            patch.set_alpha(0.45)

        # Overlay individual events
        for i, cond in enumerate(conds):
            y = bp_data[i]
            if y.size:
                jitter = (np.random.default_rng(42).random(y.size) - 0.5) * 0.2
                ax.scatter(
                    x[i] + jitter, y, s=14,
                    color=CONDITION_COLORS.get(cond, "#999"),
                    edgecolors="white", linewidths=0.3, alpha=0.8, zorder=3,
                )

        ax.set_title(roi)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=25, ha="right")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        if ax is axes[0]:
            ax.set_ylabel("Post − Pre ΔF/F")

    fig.suptitle(f"ETA Pre/Post Diff — {etype}")
    run.save_figure(fig, f"eta_prepost_diff_{etype}.svg")

# ─── Save ─────────────────────────────────────────────────────────────────

run.save_table(diff_df, "eta_prepost_diff.csv")
for key, df in result.tables.items():
    run.save_table(df, f"eta_prepost_{key}.csv")

# Standard ETA plots
for etype in result.event_types:
    fig = result.plot(event=etype, rois=list(ROI_COLUMNS))
    run.save_figure(fig, f"eta_{etype}_traces.svg")

run.finish(
    notes=(
        f"ETA pre/post difference comparison across conditions for locomotion events. "
        f"Pre window: {PRE_WINDOW}, Post window: {POST_WINDOW}."
    ),
)

print(f"Done — {len(diff_df)} comparisons computed.")
