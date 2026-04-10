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
from databench.session import SaveableFigure
from databench._utils._logger import get_logger
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

SESSION_TO_CONDITION = {
    "ses-01": "baseline",
    "ses-02": "saline",
    "ses-03": "ethanol_low",
    "ses-04": "ethanol_high",
}
CONDITION_ORDER = ("baseline", "saline", "ethanol_low", "ethanol_high")
CONDITION_COLORS = {
    "baseline": "#bbabab",
    "saline": "#4289e6",
    "ethanol_low": "#ffa251",
    "ethanol_high": "#ce1818",
}

# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name="eta-prepost",
    tag="vis-primary-secondary",
)

group = proj.sessions(task=TASK)

# ─── Detect events & add conditions ──────────────────────────────────────

events = locomotion_events(group, min_speed_cms=0.5, min_duration_s=1.0, merge_gap_s=0.5)
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


def _mean_in_window(g: pd.DataFrame, window: tuple[float, float]) -> float:
    mask = (g["rel_time"] >= window[0]) & (g["rel_time"] <= window[1])
    return float(g.loc[mask, "value"].mean())


diff_rows: list[dict] = []
for keys, g in result.eta_events.groupby(
    ["Subject", "Session", "Task", "Condition", "EventType", "event_id", "ROI"],
    sort=False,
):
    subj, ses, task, cond, etype, eid, roi = keys
    pre = _mean_in_window(g, PRE_WINDOW)
    post = _mean_in_window(g, POST_WINDOW)
    diff_rows.append({
        "Subject": subj, "Session": ses, "Task": task,
        "Condition": cond, "EventType": etype, "ROI": roi,
        "pre_mean": pre, "post_mean": post, "diff": post - pre,
    })

diff_df = pd.DataFrame(diff_rows)

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
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"ETA Pre/Post Diff — {etype}")
    fig.tight_layout()
    SaveableFigure(fig, proj._context).save(f"eta_prepost_diff_{etype}.svg")

# ─── Save ─────────────────────────────────────────────────────────────────

stats_dir = proj._context.stats_dir
stats_dir.mkdir(parents=True, exist_ok=True)
diff_df.to_csv(stats_dir / "eta_prepost_diff.csv", index=False)
result.save_tables(prefix="eta_prepost")

# Standard ETA plots
for etype in result.event_types:
    result.plot(event=etype, rois=list(ROI_COLUMNS)).save(f"eta_{etype}_traces.svg")

proj.save_report(
    result,
    notes=(
        f"ETA pre/post difference comparison across conditions for locomotion events. "
        f"Pre window: {PRE_WINDOW}, Post window: {POST_WINDOW}."
    ),
)

print(f"Done — {len(diff_df)} comparisons computed.")
