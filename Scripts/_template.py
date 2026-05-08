"""Template script — copy this file, rename it, and edit the body.

A canonical databench analysis script.  Demonstrates:

  * Project / dataset setup
  * Discovery (``proj.describe()``, ``sess.describe()``)
  * Iterating a SessionGroup
  * Saving tables, figures, and a markdown report
  * Optional debug dump of every local DataFrame

Run::

    python Scripts/_template.py

Then duplicate this file to ``Scripts/<group>/<my-analysis>.py`` and
edit the marked DEVELOPER sections.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench import Project, set_theme
from databench.analysis.locomotion import locomotion_bout_events
from databench.config import resolve_dataset

# DEVELOPER: optional — opt-in schemas for hover-rich locals.
# Delete if you don't want them; they are doc-only.
from databench.types import BoutEventsTable

set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────
# DEVELOPER: UPPER_CASE module-level constants are auto-snapshotted into
# `config/params.json` by `proj.io.report(...)`.

DATASET = resolve_dataset()           # honors DATABENCH_DATASET / default
TASK = "task-spont"
SPEED_SOURCE = "treadmill"
SPEED_COL = "speed_mm"
SPEED_SCALE_TO_CMS = 10.0
MIN_SPEED_CMS = 0.5
MIN_DURATION_S = 1.0
MERGE_GAP_S = 0.5
RUN_NAME = "template"
TAG = ""

# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    run_name=RUN_NAME,
    tag=TAG,
)
proj.describe()                       # prints subjects / sessions / dirs

group = proj.sessions(task=TASK)
group.describe()

# DEVELOPER: uncomment to inspect the first session's sources/signals:
# group[0].describe()

# ─── Per-session computation ──────────────────────────────────────────────

bout_rows: list[dict] = []
for sess in group:
    t = sess.time(SPEED_SOURCE)
    speed_mm = sess.signal(SPEED_SOURCE, SPEED_COL)
    if t is None or speed_mm is None or t.size < 3:
        continue
    speed_cms = speed_mm / SPEED_SCALE_TO_CMS

    epochs: BoutEventsTable = locomotion_bout_events(
        t, speed_cms,
        min_speed_cms=MIN_SPEED_CMS,
        min_duration_s=MIN_DURATION_S,
        merge_gap_s=MERGE_GAP_S,
    )
    # DEVELOPER: epochs.<column> autocompletes (start_s, end_s, ...).
    for _, r in epochs.iterrows():
        bout_rows.append({
            "Subject": sess.subject, "Session": sess.session, "Task": sess.task,
            "start_s": r["start_s"], "end_s": r["end_s"],
            "duration_s": r["duration_s"],
            "mean_speed_cms": r["mean_speed_cms"],
        })

bout_table = pd.DataFrame(bout_rows)
print(f"Detected {len(bout_table)} bouts across {len(group)} sessions.")

# ─── Save ─────────────────────────────────────────────────────────────────

proj.io.table(bout_table, "bouts.csv")

if not bout_table.empty:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(bout_table["duration_s"], bins=30, color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Bout duration (s)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    proj.io.figure(fig, "bout_duration_hist.svg")

# DEVELOPER: dump every local DataFrame for debugging — comment out when
# the script stabilises.
proj.io.dump(locals())

proj.io.report(
    notes=(
        f"Template analysis: detected locomotion bouts across {len(group)} "
        f"sessions of task={TASK}."
    ),
)
print("Done.")
