"""
Blessed workflow — novice-friendly procedural example.

Demonstrates the Project/Session API:
  1. Load and filter data
  2. Compute first-order features per session
  3. Derive second-order features (delta from baseline)
  4. Plot both first- and second-order features
  5. Save session table + report

Discovering your data:
    proj.subjects       → list of subject IDs
    proj.tasks          → list of task names
    proj.all_sessions   → list of session IDs
    resolve_dataset()   → reads datasets.toml for the dataset path
                          (env var DATABENCH_DATASET overrides the default)

Usage:
    python Scripts/scriptings/blessed-workflow.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench.project import Project
from databench.analysis.locomotion import locomotion_bout_events
from databench.config import resolve_dataset
from databench.plotting import set_theme

set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET = resolve_dataset()
BASELINE_SESSION_N = 1      # session_n=1 is the baseline reference
DROP_RULES = [{"subject": "GS27", "session": "ses-02", "task": "task-spont"}]

# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    run_name="sandbox",
    tag="blessed",
).filter(exclude=DROP_RULES)  # Example of dropping specific sessions

# ─── 1. Select sessions (skip specified rows) ────────────────────────────

group = proj.sessions()
print(f"Loaded {len(group)} sessions")


# ─── 2. Compute first-order features per session ─────────────────────────

def compute_session_features(sess) -> dict | None:
    """Extract scalar features from a single session."""
    # Speed (prefer treadmill, fall back to encoder)
    for src in ("treadmill", "encoder"):
        t = sess.time(src)
        spd = sess.signal(src, "speed_mm")
        if t is not None and spd is not None:
            break
    else:
        return None

    valid = np.isfinite(t) & np.isfinite(spd)
    t, spd = t[valid], spd[valid]
    if t.size < 3:
        return None

    speed_cms = np.abs(spd) / 10.0
    speed_mean = float(np.nanmean(speed_cms))

    # Pupil
    pupil = sess.signal("pupil", "pupil_diameter_mm")
    pupil_mean = float(np.nanmean(pupil)) if pupil is not None and np.isfinite(pupil).any() else np.nan

    return {
        "Subject": sess.subject,
        "Session": sess.session,
        "Task": sess.task,
        "speed_mean_cms": speed_mean,
        "pupil_mean_mm": pupil_mean,
    }


rows = [r for sess in group if (r := compute_session_features(sess)) is not None]
session_table = pd.DataFrame(rows)
session_table = proj.tabler.add_session_number(session_table)
print(f"Computed features for {len(session_table)} sessions")


# ─── 3. Second-order feature: delta from baseline ────────────────────────

def delta_from_baseline(
    table: pd.DataFrame, y: str, baseline_session_n: int = 1,
) -> pd.DataFrame:
    """Subtract per-subject baseline session mean from a column."""
    out_col = f"d_{y}"
    bl = (
        table.loc[table["session_n"] == baseline_session_n, ["Subject", y]]
        .groupby("Subject", as_index=False)[y]
        .mean()
        .rename(columns={y: "_bl"})
    )
    table = table.merge(bl, on="Subject", how="left")
    table[out_col] = table[y] - table["_bl"]
    return table.drop(columns=["_bl"])


session_table = delta_from_baseline(session_table, "speed_mean_cms", BASELINE_SESSION_N)
print(f"Added d_speed_mean_cms (delta from session_n={BASELINE_SESSION_N})")


# ─── 4. Plotting ─────────────────────────────────────────────────────────

def feature_boxplot(table: pd.DataFrame, y: str, *, x_label="Session") -> plt.Figure:
    """Simple boxplot of a feature across sessions."""
    sessions = sorted(table["Session"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [table.loc[table["Session"] == s, y].dropna().values for s in sessions]
    bp = ax.boxplot(data, positions=np.arange(len(sessions)),
                    widths=0.55, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4c72b0")
        patch.set_alpha(0.45)
    ax.set_xticks(np.arange(len(sessions)))
    ax.set_xticklabels(sessions, rotation=25, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y)
    ax.set_title(y)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# First-order
fig = feature_boxplot(session_table, "speed_mean_cms")
proj.io.figure(fig, "first_order_speed.png")

# Second-order
fig = feature_boxplot(session_table, "d_speed_mean_cms")
proj.io.figure(fig, "second_order_delta_speed.png")


# ─── 5. Save table + report ──────────────────────────────────────────────

proj.io.table(session_table, "blessed_session_table.csv")

proj.io.report(
    notes="Blessed procedural workflow: first-order features + delta-from-baseline second-order analysis.",
)
print("Done.")
