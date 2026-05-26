"""Pupil baseline-normalization using session-1 quiescent epochs as anchor.

For each animal, derive a single baseline pupil diameter (median across
quiescent samples of ses-01) and express every other session's pupil values
as a dimensionless ratio relative to that baseline.  Values around 1.0
correspond to the naive resting state; values >1.0 / <1.0 indicate dilation
/ constriction relative to the anchor day.

Outputs:
    stats/pupil_baselines_per_subject.csv
    stats/pupil_baseline_normalized_long.csv
    plots/pupil_baseline_trajectories.svg

Usage:
    python Scripts/pupil/pupil-baseline-normalize.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from databench.project import Project
from databench.config import resolve_dataset
from databench.analysis.pupil_baseline import (
    apply_baseline_normalization,
    compute_session1_baselines,
)
from databench.plotting import set_theme
from databench.utils import session_to_int

set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET                  = resolve_dataset("etoh-hfsa")
TASK                     = "task-widefield"
PUPIL_SOURCE             = "pupil"   # aliased to "pupil_dlc" by schema
PUPIL_COLUMN             = "pupil_diameter_mm"
SPEED_SOURCE             = "treadmill"
SPEED_COLUMN             = "speed_mm"
SPEED_SCALE_TO_CMS       = 10.0
QUIESCENCE_SPEED_CMS     = 0.5
MIN_QUIESCENT_DURATION_S = 2.0
BASELINE_SESSION         = "ses-01"
RUN_NAME                 = "pupil-baseline-normalize"
TAG                      = "session1-anchor"

# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(dataset=DATASET).filter(exclude={"session": ["ses-00", "ses-11"]})
group = proj.sessions(task=TASK)
run = proj.run(name=RUN_NAME, tag=TAG)
print(f"Selected {len(group)} sessions for task={TASK!r}")

# ─── Compute per-subject baselines from ses-01 quiescence ────────────────

baselines = compute_session1_baselines(
    group,
    pupil_source=PUPIL_SOURCE,
    pupil_column=PUPIL_COLUMN,
    speed_source=SPEED_SOURCE,
    speed_column=SPEED_COLUMN,
    speed_scale_to_cms=SPEED_SCALE_TO_CMS,
    quiescence_speed_cms=QUIESCENCE_SPEED_CMS,
    min_quiescent_duration_s=MIN_QUIESCENT_DURATION_S,
    baseline_session=BASELINE_SESSION,
)
print(f"Computed baselines for {len(baselines)} subjects:")
print(baselines.to_string(index=False))
run.save_table(baselines, "pupil_baselines_per_subject.csv")

# ─── Apply normalization across every session ─────────────────────────────

long = apply_baseline_normalization(
    group,
    baselines,
    pupil_source=PUPIL_SOURCE,
    pupil_column=PUPIL_COLUMN,
)
print(
    f"Normalized {len(long)} pupil samples across "
    f"{long['Subject'].nunique()} subjects × "
    f"{long['Session'].nunique()} sessions."
)
run.save_table(long, "pupil_baseline_normalized_long.csv")

# ─── Plot per-subject pupil_norm trajectories across sessions ────────────

per_session = (
    long
    .assign(day=lambda d: d["Session"].map(session_to_int))
    .groupby(["Subject", "Session", "day"], as_index=False)["pupil_norm"]
    .mean()
    .sort_values(["Subject", "day"])
)

fig, ax = plt.subplots(figsize=(8.0, 4.5))
subjects = sorted(per_session["Subject"].unique())
palette = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(subjects), 1)))
for color, subj in zip(palette, subjects):
    sub = per_session[per_session["Subject"] == subj]
    ax.plot(
        sub["day"], sub["pupil_norm"],
        "o-", color=color, ms=4, lw=1.2, alpha=0.7, label=subj,
    )

group_mean = (
    per_session
    .groupby("day")["pupil_norm"]
    .agg(["mean", "sem"])
    .reset_index()
)
ax.errorbar(
    group_mean["day"], group_mean["mean"], yerr=group_mean["sem"],
    fmt="s-", color="black", ms=6, lw=2, zorder=5, label="group mean",
)
ax.axhline(1.0, color="gray", ls="--", lw=0.8, zorder=0)
ax.set_xlabel("Session day")
ax.set_ylabel("Pupil (fraction of ses-01 quiescent baseline)")
ax.set_title("Pupil habituation across sessions (session-1 anchor)",
             fontweight="bold")
ax.legend(fontsize=8, ncol=2, frameon=False)
fig.tight_layout()

run.save_figure(fig, "pupil_baseline_trajectories.svg", formats=("png",))

# ─── Report ───────────────────────────────────────────────────────────────

run.finish(
    notes=(
        "Pupil baseline-normalization (session-1 anchor).\n"
        "\n"
        "Methods\n"
        "-------\n"
        f"For each subject, a single pupil baseline was derived from "
        f"{BASELINE_SESSION} as the median raw pupil diameter across all "
        f"samples falling inside quiescent epochs — defined as continuous "
        f"intervals of treadmill speed < {QUIESCENCE_SPEED_CMS} cm/s lasting "
        f"at least {MIN_QUIESCENT_DURATION_S}s. Pupil was merge_asof-aligned "
        f"to the treadmill timebase prior to epoching. The baseline is "
        f"median (not mean) so that blink/dropout artifacts in the eye-track "
        f"do not bias it. All other sessions report pupil_norm = pupil_raw "
        f"/ baseline, a dimensionless ratio. Values around 1.0 correspond to "
        f"the naive resting state on the anchor day.\n"
        "\n"
        "Important caveat: the camera setups are uncalibrated to a physical "
        "reference, so absolute mm diameters are not directly comparable "
        "across animals or cohorts. Only within-subject ratios computed "
        "against this baseline should be used for cross-session inference.\n"
        "\n"
        f"Baselines computed for {len(baselines)} subjects."
    ),
)
print("Done.")
