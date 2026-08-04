"""
Batch feature extraction to CSV.

Exports per-session time-series signals (speed, distance, pupil, ROIs)
to individual CSV files for external analysis or sharing.

Each output is a long-format CSV with columns:
  Subject, Session, Task, time_elapsed_s, <signal>

Usage:
    python Scripts/extract-features-csv.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from databench.project import Project
from databench.config import resolve_dataset
from databench.utils import clean_xy

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET = resolve_dataset()

# Optional filters (set to None to include all)
SUBJECT = None
SESSION = None
TASK = None

# Each output name → list of (source, feature) candidates to try.
# The first match found in a session is used.
EXTRACT_FEATURES = {
    "locomotion_speed_mm": [
        ("treadmill", "speed_mm"),
        ("encoder", "speed"),
    ],
    "locomotion_distance_mm": [
        ("treadmill", "distance_mm"),
        ("encoder", "distance"),
    ],
    "pupil_diameter_mm": [
        ("pupil", "pupil_diameter_mm"),
        ("pupil", "diameter_mm"),
    ],
    # "roi_R_VISp": [("mesomap", "R_VISp")],
}


# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(dataset=DATASET)
group = proj.sessions(subject=SUBJECT, task=TASK)
run = proj.run(name="feature_extraction", tag="ETOH_R01")
print(f"Selected {len(group)} sessions")

# ─── Extract and save ────────────────────────────────────────────────────

for out_name, candidates in EXTRACT_FEATURES.items():
    if not candidates:
        print(f"  Skipping {out_name}: no candidates provided.")
        continue

    rows: list[dict] = []
    for sess in group:
        if SESSION is not None and sess.session != SESSION:
            continue

        # Try each candidate (source, feature) until one works
        source, feature, sig, time = None, None, None, None
        for src, feat in candidates:
            sig = sess.signal(src, feat)
            if sig is not None:
                source, feature = src, feat
                time = sess.time(src)
                break

        if sig is None or time is None:
            continue

        t_v, s_v = clean_xy(time, sig)
        for i in range(t_v.size):
            rows.append({
                "Subject": sess.subject,
                "Session": sess.session,
                "Task": sess.task,
                "time_elapsed_s": float(t_v[i]),
                out_name: float(s_v[i]),
            })

    if not rows:
        cand_str = ", ".join(f"({s}, {f})" for s, f in candidates)
        print(f"  No data found for {out_name}. Tried: {cand_str}")
        continue

    csv_path = run.save_table(pd.DataFrame(rows), f"{out_name}.csv")
    print(f"  Saved {out_name} → {csv_path}  ({len(rows)} samples)")

# ─── Report ───────────────────────────────────────────────────────────────

run.finish(
    notes="Batch feature extraction to CSV for locomotion, pupil, and ROI signals.",
)
print("Done.")
