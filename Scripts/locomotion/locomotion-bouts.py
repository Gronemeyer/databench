"""
Locomotion bout detection and first-order feature statistics.

Iterates all sessions, detects locomotion bouts, computes per-session
scalar features (speed, distance, bout count, etc.), and produces:
  1. Session-level CSV table
  2. Per-feature boxplots
  3. PDF report with per-session speed traces + bout highlights
  4. Bout duration / speed histograms

Usage:
    python Scripts/locomotion/locomotion-bouts.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from databench.project import Project
from databench.analysis.locomotion import locomotion_bout_events
from databench.config import resolve_dataset
from databench.session import SaveableFigure
from databench.plotting import set_theme

set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET = resolve_dataset("hfsa")
TASK = "task-widefield"
SPEED_SOURCE = "treadmill"
SPEED_COL = "speed_mm"
SPEED_SCALE_TO_CMS = 10.0       # speed_mm / 10 -> cm/s
MIN_SPEED_CMS = 0.5
MIN_DURATION_S = 2.0
MERGE_GAP_S = 0.5
EXPORT_SVG = True
OUTPUT_ROOT = "outputs"
RUN_NAME = "locomotion-bouts"
TAG = "canonical"
OUTPUT_PREFIX = "locomotion_bouts"
LEFT_EARLY_DAYS = (1, 5)
LEFT_LATE_DAYS = (6, 10)
RIGHT_EARLY_DAYS = (1, 3)
RIGHT_LATE_DAYS = (7, 10)

# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    output_root=OUTPUT_ROOT,
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name=RUN_NAME,
    tag=TAG,
).filter(exclude={"session": "ses-11"})

group = proj.sessions(task=TASK)
print(f"Selected {len(group)} sessions for task={TASK}")


# ─── Per-session feature extraction ──────────────────────────────────────

session_rows: list[dict] = []
bout_rows: list[dict] = []
# Store per-session raw data for the PDF report
raw_traces: list[dict] = []

for sess in group:
    t = sess.time(SPEED_SOURCE)
    spd_mm = sess.signal(SPEED_SOURCE, SPEED_COL)
    if t is None or spd_mm is None or t.size < 3:
        continue

    valid = np.isfinite(t) & np.isfinite(spd_mm)
    t, spd_mm = t[valid], spd_mm[valid]
    speed_cms = spd_mm / SPEED_SCALE_TO_CMS

    dt_med = float(np.nanmedian(np.diff(t))) if t.size > 1 else 0.0
    dur_s = float(t[-1] - t[0] + dt_med)
    dur_min = dur_s / 60.0 if dur_s > 0 else np.nan
    speed_mean = float(np.nanmean(np.abs(speed_cms)))
    speed_std = float(np.nanstd(np.abs(speed_cms)))
    dist_m = float(np.nansum(np.abs(speed_cms[1:]) * np.diff(t))) / 100.0

    epochs = locomotion_bout_events(
        t, speed_cms,
        min_speed_cms=MIN_SPEED_CMS,
        min_duration_s=MIN_DURATION_S,
        merge_gap_s=MERGE_GAP_S,
    )

    n_bouts = len(epochs)
    bout_rate = (n_bouts / dur_min) if dur_min and np.isfinite(dur_min) else np.nan
    for _, r in epochs.iterrows():
        bout_rows.append({
            "Subject": sess.subject, "Session": sess.session, "Task": sess.task,
            "onset_t": r["start_s"], "offset_t": r["end_s"],
            "duration_s": r["duration_s"], "mean_speed_cms": r["mean_speed_cms"],
            "distance_m": r["distance_m"],
        })

    session_rows.append({
        "Subject": sess.subject, "Session": sess.session, "Task": sess.task,
        "session_duration_s": dur_s, "session_duration_min": dur_min,
        "speed_mean_cms": speed_mean, "speed_std_cms": speed_std,
        "distance_m": dist_m,
        "locomotion_bouts_n": n_bouts, "bout_rate_per_min": bout_rate,
        "locomotion_bout_duration_s": float(epochs["duration_s"].mean()) if n_bouts else np.nan,
        "locomotion_bout_speed_mean_cms": float(epochs["mean_speed_cms"].mean()) if n_bouts else np.nan,
        "locomotion_bout_distance_m": float(epochs["distance_m"].sum()) if n_bouts else 0.0,
    })
    raw_traces.append({
        "subject": sess.subject, "session": sess.session, "task": sess.task,
        "t": t, "speed_cms": speed_cms, "epochs": epochs,
    })

session_table = pd.DataFrame(session_rows)
bout_table = pd.DataFrame(bout_rows)
print(f"Computed features for {len(session_table)} sessions, {len(bout_table)} total bouts")


# ─── Save session table ──────────────────────────────────────────────────

stats_dir = proj._context.stats_dir
stats_dir.mkdir(parents=True, exist_ok=True)
session_table.to_csv(stats_dir / f"{OUTPUT_PREFIX}_session_table.csv", index=False)
bout_table.to_csv(stats_dir / f"{OUTPUT_PREFIX}_bout_table.csv", index=False)


# ─── Per-feature boxplots ────────────────────────────────────────────────

feature_cols = [
    "speed_mean_cms", "speed_std_cms", "distance_m",
    "locomotion_bouts_n", "locomotion_bout_speed_mean_cms",
    "locomotion_bout_distance_m", "locomotion_bout_duration_s",
]
feature_labels = {
    "speed_mean_cms": "Mean speed (cm/s)",
    "speed_std_cms": "Speed std (cm/s)",
    "distance_m": "Distance (m)",
    "locomotion_bouts_n": "Bout count",
    "locomotion_bout_speed_mean_cms": "Bout speed (cm/s)",
    "locomotion_bout_distance_m": "Bout distance (m)",
    "locomotion_bout_duration_s": "Bout duration (s)",
}


def _session_to_day(session_label: str) -> int | None:
    """Parse session labels like 'ses-01' into integer day numbers."""
    if not isinstance(session_label, str):
        return None
    if session_label.startswith("ses-"):
        try:
            return int(session_label.split("-")[-1])
        except ValueError:
            return None
    return None

if not session_table.empty and "Session" in session_table.columns:
    sessions_sorted = sorted(session_table["Session"].unique())
    for col in feature_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        data = [session_table.loc[session_table["Session"] == s, col].dropna().values for s in sessions_sorted]
        bp = ax.boxplot(data, positions=np.arange(len(sessions_sorted)),
                        widths=0.55, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4c72b0")
            patch.set_alpha(0.45)
        ax.set_xticks(np.arange(len(sessions_sorted)))
        ax.set_xticklabels(sessions_sorted, rotation=25, ha="right")
        ax.set_ylabel(feature_labels.get(col, col))
        ax.set_xlabel("Session (days)")
        ax.set_title(col)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        sf = SaveableFigure(fig, proj._context)
        sf.save(f"{OUTPUT_PREFIX}_{col}_boxplot.png")
        if EXPORT_SVG:
            sf.save(f"{OUTPUT_PREFIX}_{col}_boxplot.svg")
        plt.close(fig)


# ─── PDF report with per-session speed traces + bout highlights ──────────

report_dir = proj._context.reports_dir
report_dir.mkdir(parents=True, exist_ok=True)
report_path = report_dir / f"{OUTPUT_PREFIX}_report.pdf"

with PdfPages(report_path) as pdf:
    for tr in raw_traces:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tr["t"], tr["speed_cms"], color="#2ca02c", lw=1.2, label="Speed (cm/s)")
        for _, r in tr["epochs"].iterrows():
            ax.axvspan(r["start_s"], r["end_s"], color="#2ca02c", alpha=0.2)
        title = f"Subject={tr['subject']} | Session={tr['session']} | Task={tr['task']}"
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (cm/s)")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # Bout duration histogram
    if bout_table.size > 0 and "duration_s" in bout_table.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(bout_table["duration_s"].dropna(), bins=30, color="#4c72b0", edgecolor="white")
        ax.set_title("Bout duration distribution")
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Count")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # Bout speed histogram
    if bout_table.size > 0 and "mean_speed_cms" in bout_table.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(bout_table["mean_speed_cms"].dropna(), bins=30, color="#55a868", edgecolor="white")
        ax.set_title("Bout mean speed distribution")
        ax.set_xlabel("Speed (cm/s)")
        ax.set_ylabel("Count")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print(f"PDF report saved to {report_path}")


# ─── Overlay histogram: bout velocity early vs late days ────────────────

if not bout_table.empty and "mean_speed_cms" in bout_table.columns and "Session" in bout_table.columns:
    day_n = bout_table["Session"].map(_session_to_day)
    speed_cms = bout_table["mean_speed_cms"]

    comparisons = [
        (LEFT_EARLY_DAYS, LEFT_LATE_DAYS),
        (RIGHT_EARLY_DAYS, RIGHT_LATE_DAYS),
    ]
    bins = np.arange(0.0, 25.0 + 0.5, 0.5).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), facecolor="#d9d9d9", sharey=True)
    has_data = False

    for ax, (early_days, late_days) in zip(axes, comparisons):
        ax.set_facecolor("#d9d9d9")
        early = speed_cms[(day_n >= early_days[0]) & (day_n <= early_days[1])].dropna()
        late = speed_cms[(day_n >= late_days[0]) & (day_n <= late_days[1])].dropna()

        if not late.empty:
            ax.hist(late, bins=bins, color="#ef3b3b", alpha=0.75, edgecolor="#333333", linewidth=0.5)
            has_data = True
        if not early.empty:
            ax.hist(early, bins=bins, color="#5b2a86", alpha=0.75, edgecolor="#333333", linewidth=0.5)
            has_data = True

        ax.set_xlim(0, 25)
        ax.set_xlabel("Average velocity of a bout (cm/s)")
        ax.text(
            0.98,
            0.92,
            f"Days {early_days[0]}-{early_days[1]}",
            color="#5b2a86",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        ax.text(
            0.98,
            0.83,
            f"Days {late_days[0]}-{late_days[1]}",
            color="#ef3b3b",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Number of bouts (#)")

    if has_data:
        fig.tight_layout()
        sf = SaveableFigure(fig, proj._context)
        sf.save(f"{OUTPUT_PREFIX}_speed_hist_comparison_panels.png")
        if EXPORT_SVG:
            sf.save(f"{OUTPUT_PREFIX}_speed_hist_comparison_panels.svg")
    plt.close(fig)

proj.save_report(
    notes=(
        f"First-order locomotion bout features for ETOH R01 pre-condition dataset.\n"
        f"{len(session_table)} sessions, {len(bout_table)} total bouts detected.\n"
        f"Bout criteria: min_speed={MIN_SPEED_CMS} cm/s, "
        f"min_duration={MIN_DURATION_S}s, merge_gap={MERGE_GAP_S}s."
    ),
)
print("Done.")
