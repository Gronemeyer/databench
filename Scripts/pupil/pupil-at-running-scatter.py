"""
Pupil diameter at running start vs. running end — scatter plot.

For each locomotion bout, sample the pupil diameter at onset and offset,
then scatter-plot onset vs. offset with an identity line.

Usage:
    python Scripts/pupil-at-running-scatter.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from databench.project import Project
from databench.analysis.locomotion import locomotion_events
from databench.config import resolve_dataset
from databench.types import EventsTable
from databench.plotting import set_theme
from databench.utils import clean_xy

set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET = resolve_dataset('etoh-hfsa')
PUPIL_KEY = "pupil_diameter_mm"
TASK = "task-widefield"

# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    run_name="pupil-at-running",
    tag="scatter",
).filter(exclude={"session": ["ses-00", "ses-11"]})  

group = proj.sessions(task=TASK)

# ─── Detect locomotion events ────────────────────────────────────────────

events: EventsTable = locomotion_events(
    group,
    speed_source="treadmill",
    speed_column="speed_mm",
    speed_scale_to_cms=10.0,
    min_speed_cms=0.5,
    min_duration_s=1.0,
    merge_gap_s=0.5,
    event_types=("onset", "offset"),
)

onset_events = events[events["EventType"] == "onset"].reset_index(drop=True)
offset_events = events[events["EventType"] == "offset"].reset_index(drop=True)
print(f"Detected {len(onset_events)} locomotion bouts across {len(group)} sessions")

# ─── Sample z-scored pupil at bout onset / offset ────────────────────────

pupil_at_start = []
pupil_at_end = []

for sess in group:
    ad = sess.align(
        {"treadmill": ["speed_mm"], "pupil": [PUPIL_KEY]},
        reference="treadmill",
        tolerance_s=0.25,
    )
    df = ad.df
    t_valid, pup_valid = clean_xy(df["time_elapsed_s"].to_numpy(), df[PUPIL_KEY].to_numpy())
    if t_valid.size < 2:
        continue

    # Per-session (subject/session) z-scoring.
    pup_mu = float(np.mean(pup_valid))
    pup_sd = float(np.std(pup_valid))
    if not np.isfinite(pup_sd) or pup_sd <= 0:
        continue
    pup_z = (pup_valid - pup_mu) / pup_sd

    sess_onsets = onset_events[
        (onset_events["Subject"] == sess.subject)
        & (onset_events["Session"] == sess.session)
        & (onset_events["Task"] == sess.task)
    ]["event_time"].to_numpy()

    sess_offsets = offset_events[
        (offset_events["Subject"] == sess.subject)
        & (offset_events["Session"] == sess.session)
        & (offset_events["Task"] == sess.task)
    ]["event_time"].to_numpy()

    n = min(len(sess_onsets), len(sess_offsets))
    for i in range(n):
        i_on = int(np.argmin(np.abs(t_valid - sess_onsets[i])))
        i_off = int(np.argmin(np.abs(t_valid - sess_offsets[i])))
        pupil_at_start.append(pup_z[i_on])
        pupil_at_end.append(pup_z[i_off])

pupil_at_start = np.array(pupil_at_start)
pupil_at_end = np.array(pupil_at_end)
print(f"Paired pupil samples: {len(pupil_at_start)} bouts")

# ─── Scatter plot ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(pupil_at_start, pupil_at_end, s=18, color="k", alpha=0.7, edgecolors="none")

lim = float(np.nanmax(np.abs(np.r_[pupil_at_start, pupil_at_end]))) if len(pupil_at_start) else 1.0
lim = max(1.0, min(4.0, lim * 1.05))
lo, hi = -lim, lim
ax.plot([lo, hi], [lo, hi], "k-", lw=1)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xticks(np.linspace(lo, hi, 5))
ax.set_yticks(np.linspace(lo, hi, 5))
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Pupil z-score at running start")
ax.set_ylabel("Pupil z-score at running end")
ax.set_title("Pupil diameter (z-scored per session)", fontweight="bold")
fig.tight_layout()

proj.io.figure(fig, "pupil_diameter_running_scatter.svg")

proj.io.report(
    notes=(
        f"Pupil z-scored diameter at locomotion bout onset vs. offset scatter. "
        f"{len(pupil_at_start)} bouts sampled."
    ),
)

print(f"Done — {len(pupil_at_start)} paired samples plotted.")
