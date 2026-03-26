"""Bout proportion analysis — replicates the 25-day protocol motor skill plots.

Extracts locomotion bouts, classifies them into 4 categories by duration and
velocity, and produces:
  (c) Stacked bar chart of bout proportions per day (single-animal or pooled).
  (h) Line plot with error bars of bout proportions across animals per day.

Bout categories (based on duration × velocity cutoffs):
  - Long & slow : duration >= 5 s AND mean speed < 10 cm/s  (600 cm/min)
  - Long & fast : duration >= 5 s AND mean speed >= 10 cm/s
  - Short & slow: duration <  5 s AND mean speed < 10 cm/s
  - Short & fast: duration <  5 s AND mean speed >= 10 cm/s
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench.config import resolve_dataset
from databench.features.treadmill import (
    locomotion_bout_events,
    _bout_stats,
)
from databench.project import Project
from databench.session import SessionGroup
from databench.utils import clean_xy, session_to_int

# ── Parameters ──────────────────────────────────────────────────────────────

DURATION_CUTOFF_S = 3.0       # short vs long
SPEED_CUTOFF_CMS = 8.0       # slow vs fast  (600 cm/min ≈ 10 cm/s)

# Bout detection thresholds (match LocomotionBoutFeature defaults)
MIN_SPEED_CMS = 0.5
MIN_DURATION_S = 1.0
MERGE_GAP_S = 0.5

# Category display order (bottom → top for stacked bar)
CATEGORIES = ["Long & fast", "Short & fast", "Short & slow", "Long & slow"]

CATEGORY_COLORS = {
    "Long & fast":  "#4a4a4a",   # dark gray
    "Short & fast": "#b0b0b0",   # light gray
    "Short & slow": "#ffbe76",   # light orange
    "Long & slow":  "#e67e22",   # dark orange
}

CATEGORY_MARKERS = {
    "Long & fast":  "^",
    "Short & fast": "s",
    "Short & slow": "v",
    "Long & slow":  "o",
}

CATEGORY_LINE_COLORS = {
    "Long & fast":  "#4a4a4a",
    "Short & fast": "#b0b0b0",
    "Short & slow": "#e67e22",
    "Long & slow":  "#e67e22",
}

# ── Helpers ─────────────────────────────────────────────────────────────────


def classify_bouts(
    t: np.ndarray,
    speed_cms: np.ndarray,
    bouts: list[tuple[int, int]],
    *,
    duration_cutoff_s: float = DURATION_CUTOFF_S,
    speed_cutoff_cms: float = SPEED_CUTOFF_CMS,
) -> pd.DataFrame:
    """Classify each bout into one of 4 categories.

    Returns a DataFrame with columns:
        bout_id, duration_s, mean_speed_cms, category
    """
    mean_speeds, _, durations = _bout_stats(t, speed_cms, bouts)

    records = []
    for i, (dur, spd) in enumerate(zip(durations, mean_speeds)):
        is_long = dur >= duration_cutoff_s
        is_fast = spd >= speed_cutoff_cms
        if is_long and is_fast:
            cat = "Long & fast"
        elif is_long and not is_fast:
            cat = "Long & slow"
        elif not is_long and is_fast:
            cat = "Short & fast"
        else:
            cat = "Short & slow"
        records.append({
            "bout_id": i,
            "duration_s": dur,
            "mean_speed_cms": spd,
            "category": cat,
        })
    return pd.DataFrame(records)


def extract_bout_proportions(sessions: SessionGroup) -> pd.DataFrame:
    """Extract bout proportions per subject per session day.

    Parameters
    ----------
    sessions : SessionGroup
        Group of sessions obtained from ``Project.sessions()``.

    Returns a DataFrame with columns:
        Subject, session_n, category, proportion
    """
    all_rows = []

    for sess in sessions:
        # Try treadmill first, fall back to encoder
        for source in ("treadmill", "encoder"):
            try:
                t_raw = sess.time(source)
                break
            except Exception:
                continue
        else:
            continue

        speed_col = "speed_mm" if source == "treadmill" else "speed"
        try:
            spd_raw = sess.signal(source, speed_col)
        except Exception:
            continue

        t, spd_mm = clean_xy(t_raw, spd_raw)
        speed_cms = spd_mm / 10.0

        if t.size < 10:
            continue

        # Detect bouts
        result = locomotion_bout_events(
            t, speed_cms,
            min_speed_cms=MIN_SPEED_CMS,
            min_duration_s=MIN_DURATION_S,
            merge_gap_s=MERGE_GAP_S,
        )
        bouts, onset_idx, offset_idx, onset_t, offset_t = result

        if not bouts:
            continue

        # Classify
        bout_df = classify_bouts(
            t, speed_cms, bouts,
            duration_cutoff_s=DURATION_CUTOFF_S,
            speed_cutoff_cms=SPEED_CUTOFF_CMS,
        )

        # Compute proportions
        total = len(bout_df)
        counts = bout_df["category"].value_counts()
        day = session_to_int(sess.session)

        for cat in CATEGORIES:
            all_rows.append({
                "Subject": sess.subject,
                "session_n": day,
                "category": cat,
                "proportion": 100.0 * counts.get(cat, 0) / total,
            })

    return pd.DataFrame(all_rows)


# ── Plotting ────────────────────────────────────────────────────────────────


def plot_stacked_bar(
    prop_df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    title: str = "",
) -> tuple[plt.Figure | None, plt.Axes]:
    """Panel (c): stacked bar chart of bout proportions per day.

    If multiple subjects exist, proportions are averaged across subjects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = None

    # Average across subjects per day
    avg = (
        prop_df
        .groupby(["session_n", "category"])["proportion"]
        .mean()
        .reset_index()
    )

    days = sorted(avg["session_n"].unique())
    bottom = np.zeros(len(days))

    for cat in CATEGORIES:
        values = []
        for d in days:
            row = avg[(avg["session_n"] == d) & (avg["category"] == cat)]
            values.append(row["proportion"].values[0] if len(row) else 0.0)
        values = np.array(values)
        ax.bar(
            days, values, bottom=bottom,
            color=CATEGORY_COLORS[cat], label=cat,
            edgecolor="white", linewidth=0.3, width=0.8,
        )
        bottom += values

    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Bouts proportion (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xlim(0.4, max(days) + 0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")

    return fig, ax


def plot_line_with_errorbars(
    prop_df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    title: str = "",
) -> tuple[plt.Figure | None, plt.Axes]:
    """Panel (h): line plot of bout proportions with SEM error bars per day."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = None

    summary = (
        prop_df
        .groupby(["session_n", "category"])["proportion"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    for cat in CATEGORIES:
        sub = summary[summary["category"] == cat].sort_values("session_n")
        ax.errorbar(
            sub["session_n"],
            sub["mean"],
            yerr=sub["sem"],
            label=cat,
            color=CATEGORY_LINE_COLORS[cat],
            marker=CATEGORY_MARKERS[cat],
            markersize=5,
            capsize=3,
            linewidth=1.2,
            markerfacecolor=CATEGORY_LINE_COLORS[cat],
            markeredgecolor=CATEGORY_LINE_COLORS[cat],
        )

    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Bouts proportion (%)", fontsize=11)
    ax.set_ylim(0, 60)
    ax.set_xlim(-0.5, max(prop_df["session_n"]) + 0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")

    return fig, ax


def plot_bout_proportions(
    prop_df: pd.DataFrame,
    save_dir: Path | str | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create both panels side-by-side (c and h)."""
    fig, (ax_c, ax_h) = plt.subplots(1, 2, figsize=(14, 5))

    plot_stacked_bar(prop_df, ax=ax_c, title="c")
    plot_line_with_errorbars(prop_df, ax=ax_h, title="h")

    fig.tight_layout(w_pad=3)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "bout_proportions.png", dpi=300, bbox_inches="tight")
        fig.savefig(save_dir / "bout_proportions.svg", dpi=300, bbox_inches="tight")
        print(f"Saved to {save_dir / 'bout_proportions.png'}")

    return fig, (ax_c, ax_h)


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    DATASET = resolve_dataset("hfsa")

    project = Project(
        DATASET,
        analyst="Jacob Gronemeyer",
        lab="Sipe Lab",
        run_name="bout-proportions",
    ).filter(("STREHAB07", "ses-11"))
    sessions = project.sessions(task="task-widefield")

    # Extract bout proportions from all sessions
    prop_df = extract_bout_proportions(sessions)

    if prop_df.empty:
        print("No bout data extracted — check that treadmill data is present.")
        raise SystemExit(1)

    print(f"Extracted {len(prop_df)} proportion rows from "
          f"{prop_df['Subject'].nunique()} subjects across "
          f"{int(prop_df['session_n'].nunique())} days.")

    # Save proportions table
    out_dir = Path("outputs") / "bout_proportions"
    out_dir.mkdir(parents=True, exist_ok=True)
    prop_df.to_csv(out_dir / "bout_proportions.csv", index=False)

    # Create both plots
    fig, (ax_c, ax_h) = plot_bout_proportions(prop_df, save_dir=out_dir)
    plt.show()
