"""Locomotion bout detection and first-order feature statistics.

For each session: detect locomotion bouts, compute scalar features, draw a
speed trace into a per-session PDF page, and accumulate a per-bout table.
A final overlay histogram compares bout speed distributions between
early- and late-day windows.

Usage
-----
    python Scripts/locomotion/locomotion-bouts.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench import Project, resolve_dataset, set_theme
from databench.analysis.locomotion import locomotion_bout_events
from databench.types import BoutEventsTable
from databench.utils import clean_xy

set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET             = resolve_dataset("etoh")
TASK                = "task-widefield"
SPEED_SOURCE        = "treadmill"
SPEED_COL           = "speed_mm"
SPEED_SCALE_TO_CMS  = 10.0
MIN_SPEED_CMS       = 0.5
MIN_DURATION_S      = 2.0
MERGE_GAP_S         = 0.5
FRAME_SOURCE        = "mesomap"   # Camera clock the mask table's frame indices refer to
RUN_NAME            = "locomotion-bouts"
TAG                 = "canonical"
EXPORT_SVG          = True

# (early_days, late_days) panels for the bout-speed histogram comparison.
DAY_COMPARISONS = [
    ((1, 5), (6, 10)),
    ((1, 3), (7, 10)),
]

FEATURE_LABELS = {
    "speed_mean_cms":                 "Mean speed (cm/s)",
    "speed_std_cms":                  "Speed std (cm/s)",
    "distance_m":                     "Distance (m)",
    "locomotion_bouts_n":             "Bout count",
    "locomotion_bout_speed_mean_cms": "Bout speed (cm/s)",
    "locomotion_bout_distance_m":     "Bout distance (m)",
    "locomotion_bout_duration_s":     "Bout duration (s)",
}

# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET, output_root="outputs", run_name=RUN_NAME, tag=TAG,
).filter(exclude={"session": "ses-11"})
group = proj.sessions(task=TASK)
print(f"Selected {len(group)} sessions for task={TASK}")


# ─── Per-session detection (one PDF page per session, two row sinks) ─────

session_rows: list[dict] = []
bout_rows:    list[dict] = []
mask_rows:    list[dict] = []

with proj.io.pdf("locomotion_bouts_report.pdf") as pdf:
    for sess in group:
        time_s   = sess.time(SPEED_SOURCE)
        speed_mm = sess.signal(SPEED_SOURCE, SPEED_COL)
        if time_s is None or speed_mm is None:
            continue

        time_s, speed_mm = clean_xy(time_s, speed_mm)
        if time_s.size < 3:
            continue
        speed_cms = speed_mm / SPEED_SCALE_TO_CMS

        sample_dt_s   = float(np.nanmedian(np.diff(time_s))) if time_s.size > 1 else 0.0
        recording_s   = float(time_s[-1] - time_s[0] + sample_dt_s)
        recording_min = recording_s / 60.0 if recording_s > 0 else np.nan

        bouts: BoutEventsTable = locomotion_bout_events(
            time_s, speed_cms,
            min_speed_cms=MIN_SPEED_CMS,
            min_duration_s=MIN_DURATION_S,
            merge_gap_s=MERGE_GAP_S,
        )

        for _, bout in bouts.iterrows():
            bout_rows.append({
                "Subject": sess.subject, "Session": sess.session, "Task": sess.task,
                "day": sess.day,
                "onset_t":        bout["start_s"],
                "offset_t":       bout["end_s"],
                "duration_s":     bout["duration_s"],
                "mean_speed_cms": bout["mean_speed_cms"],
                "distance_m":     bout["distance_m"],
            })

        # ── Running/quiescent mask rows, indexed by camera frame ─────────
        # Bouts are detected on the treadmill clock, then projected onto the
        # mesofield camera frame clock so downstream scripts can slice the
        # movie directly (see Scripts/oscillations/optic-flow.py).
        # A bout owns the frames fully contained in its time window;
        # quiescence is the complement, taken in frame space.
        frame_t = sess.time(FRAME_SOURCE)
        if frame_t is not None and np.size(frame_t) > 1:
            frame_t  = np.asarray(frame_t, dtype=float)
            n_frames = frame_t.size

            # (onset_frame, offset_frame, mean_speed_cms) per bout; a bout that
            # spans no complete frame is dropped.
            run_frames: list[tuple[int, int, float]] = []
            for _, bout in bouts.iterrows():
                f0 = int(np.searchsorted(frame_t, bout["start_s"], side="left"))
                f1 = int(np.searchsorted(frame_t, bout["end_s"],   side="right")) - 1
                if f0 <= f1 and f0 < n_frames:
                    run_frames.append((f0, min(f1, n_frames - 1),
                                       float(bout["mean_speed_cms"])))

            # Complement of the running frames within [0, n_frames - 1].
            quiet_frames: list[tuple[int, int, float | None]] = []
            cursor = 0
            for f0, f1, _ in run_frames:
                if f0 - 1 > cursor:          # skip degenerate single-frame gaps
                    quiet_frames.append((cursor, f0 - 1, None))
                cursor = f1 + 1
            if cursor < n_frames - 1:
                quiet_frames.append((cursor, n_frames - 1, None))

            # Running rows carry the treadmill-derived speed verbatim;
            # quiescent rows get their speed from the same treadmill trace.
            for state, pairs in (("running", run_frames), ("quiescent", quiet_frames)):
                for f0, f1, speed in pairs:
                    t0, t1 = float(frame_t[f0]), float(frame_t[f1])
                    if speed is None:
                        window = (time_s >= t0) & (time_s <= t1)
                        speed = (float(np.nanmean(np.abs(speed_cms[window])))
                                 if window.any() else np.nan)
                    mask_rows.append({
                        "Subject": sess.subject, "Session": sess.session, "Task": sess.task,
                        "state":          state,
                        "onset_frame":    f0,
                        "offset_frame":   f1,
                        "onset_t":        t0,
                        "offset_t":       t1,
                        "duration_s":     t1 - t0,
                        "mean_speed_cms": float(speed),
                    })

        n_bouts = len(bouts)
        session_rows.append({
            "Subject": sess.subject, "Session": sess.session, "Task": sess.task,
            "day": sess.day,
            "session_duration_s":             recording_s,
            "session_duration_min":           recording_min,
            "speed_mean_cms":                 float(np.nanmean(np.abs(speed_cms))),
            "speed_std_cms":                  float(np.nanstd(np.abs(speed_cms))),
            "distance_m":                     float(np.nansum(np.abs(speed_cms[1:]) * np.diff(time_s))) / 100.0,
            "locomotion_bouts_n":             n_bouts,
            "bout_rate_per_min":              n_bouts / recording_min if recording_min and np.isfinite(recording_min) else np.nan,
            "locomotion_bout_duration_s":     float(bouts["duration_s"].mean())     if n_bouts else np.nan,
            "locomotion_bout_speed_mean_cms": float(bouts["mean_speed_cms"].mean()) if n_bouts else np.nan,
            "locomotion_bout_distance_m":     float(bouts["distance_m"].sum())      if n_bouts else 0.0,
        })

        # Per-session speed trace with bout highlights → PDF page.
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_s, speed_cms, color="#2ca02c", lw=1.2, label="Speed (cm/s)")
        for _, bout in bouts.iterrows():
            ax.axvspan(bout["start_s"], bout["end_s"], color="#2ca02c", alpha=0.2)
        ax.set_title(sess.label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (cm/s)")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # Bout-level summary histograms appended to the same PDF.
    bout_table = pd.DataFrame(bout_rows)
    if not bout_table.empty:
        for column, color, title in [
            ("duration_s",     "#4c72b0", "Bout duration distribution"),
            ("mean_speed_cms", "#55a868", "Bout mean speed distribution"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(bout_table[column].dropna(), bins=30, color=color, edgecolor="white")
            ax.set_title(title)
            ax.set_xlabel(FEATURE_LABELS.get(column, column))
            ax.set_ylabel("Count")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

session_table = pd.DataFrame(session_rows)
mask_table    = pd.DataFrame(mask_rows)
print(f"Computed {len(session_table)} sessions, {len(bout_table)} total bouts, "
      f"{len(mask_table)} mask epochs")


# ─── Save tables ─────────────────────────────────────────────────────────

proj.io.table(session_table, "locomotion_bouts_session_table.csv")
proj.io.table(bout_table,    "locomotion_bouts_bout_table.csv")
proj.io.table(mask_table,    "locomotion_mask_table.csv")


# ─── Per-feature boxplots across sessions ───────────────────────────────

if not session_table.empty:
    sessions_sorted = sorted(session_table["Session"].unique())
    formats         = ("svg",) if EXPORT_SVG else None
    for column, ylabel in FEATURE_LABELS.items():
        if column not in session_table.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        per_session = [session_table.loc[session_table["Session"] == s, column].dropna().values
                       for s in sessions_sorted]
        box = ax.boxplot(per_session, positions=np.arange(len(sessions_sorted)),
                         widths=0.55, patch_artist=True, showfliers=False)
        for patch in box["boxes"]:
            patch.set_facecolor("#4c72b0")
            patch.set_alpha(0.45)
        ax.set_xticks(np.arange(len(sessions_sorted)))
        ax.set_xticklabels(sessions_sorted, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Session (days)")
        ax.set_title(column)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        proj.io.figure(fig, f"locomotion_bouts_{column}_boxplot.png", formats=formats)


# ─── Bout speed: early vs late days, two comparison panels ──────────────

if not bout_table.empty and "day" in bout_table.columns:
    bins    = np.arange(0.0, 25.5, 0.5)
    has_any = False

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), facecolor="#d9d9d9", sharey=True)
    for ax, (early_days, late_days) in zip(axes, DAY_COMPARISONS):
        ax.set_facecolor("#d9d9d9")
        early = bout_table.loc[bout_table["day"].between(*early_days), "mean_speed_cms"].dropna()
        late  = bout_table.loc[bout_table["day"].between(*late_days),  "mean_speed_cms"].dropna()

        if not late.empty:
            ax.hist(late,  bins=bins, color="#ef3b3b", alpha=0.75, edgecolor="#333333", linewidth=0.5)
            has_any = True
        if not early.empty:
            ax.hist(early, bins=bins, color="#5b2a86", alpha=0.75, edgecolor="#333333", linewidth=0.5)
            has_any = True

        ax.set_xlim(0, 25)
        ax.set_xlabel("Average velocity of a bout (cm/s)")
        ax.text(0.98, 0.92, f"Days {early_days[0]}-{early_days[1]}",
                color="#5b2a86", transform=ax.transAxes, ha="right", va="top")
        ax.text(0.98, 0.83, f"Days {late_days[0]}-{late_days[1]}",
                color="#ef3b3b", transform=ax.transAxes, ha="right", va="top")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Number of bouts (#)")

    if has_any:
        fig.tight_layout()
        proj.io.figure(fig, "locomotion_bouts_speed_hist_comparison_panels.png",
                       formats=("svg",) if EXPORT_SVG else None)
    else:
        plt.close(fig)


# ─── Report ──────────────────────────────────────────────────────────────

proj.io.report(
    notes=(
        f"First-order locomotion bout features for ETOH R01 pre-condition dataset.\n"
        f"{len(session_table)} sessions, {len(bout_table)} total bouts detected.\n"
        f"Bout criteria: min_speed={MIN_SPEED_CMS} cm/s, "
        f"min_duration={MIN_DURATION_S}s, merge_gap={MERGE_GAP_S}s."
    ),
)
print("Done.")
