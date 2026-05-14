"""Plot HFSA raw weight, average changes, and poops trajectories.

This script uses databench output handling and writes:
1) long-format weights/poops table
2) subject-level longitudinal figure (2x2: raw weight, avg weight change, poops, avg poops change)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from databench import Project, resolve_dataset, set_theme

set_theme()

# Parameters
DATASET = resolve_dataset("etoh-hfsa")
RUN_NAME = "weights-poops"
TAG = ""
WEIGHTS_XLSX = Path(
    "/Users/jakegronemeyer/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/1. Projects/STREHAB/HFSA-weights-poops.xlsx"
)
TASK = "task-widefield"
DROP_HAB = False
BASELINE_HAB_MIN = 1
BASELINE_HAB_MAX = 5
PLOT_SESSION_START = 0
PLOT_SESSION_MIN = 1
PLOT_SESSION_MAX = 10


def load_hfsa_weights_poops(
    path: str | Path,
    task: str = TASK,
    drop_hab: bool = DROP_HAB,
) -> pd.DataFrame:
    """Load HFSA weights/poops into a (Subject, Session, Task)-indexed table."""
    table = pd.read_excel(path)

    table["session"] = table["session"].astype(str).str.replace("_", "-", regex=False)
    if drop_hab:
        table = table[~table["session"].str.startswith("hab-")].copy()

    table["date"] = pd.to_datetime(table["date"])
    table = table.rename(columns={"subject": "Subject", "session": "Session"})
    table["Task"] = task
    table = table.set_index(["Subject", "Session", "Task"]).sort_index()
    return table[["date", "weight_g", "poops", "notes"]]


def prepare_longitudinal_table(raw: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Convert the MultiIndex table into long form with session metadata."""
    table = raw.reset_index()
    table = proj.tabler.add_session_number(table, session_col="Session", out_col="session_n")
    table = table[np.isfinite(table["session_n"])].copy()

    table["session_n"] = table["session_n"].astype(int)
    table["session_label"] = table["Session"].astype(str).str.lower()
    table["session_kind"] = np.where(
        table["session_label"].str.startswith("hab-"),
        "hab",
        np.where(table["session_label"].str.startswith("ses-"), "ses", "other"),
    )

    table["weight_g"] = pd.to_numeric(table["weight_g"], errors="coerce")
    table["poops"] = pd.to_numeric(table["poops"], errors="coerce")

    return table.sort_values(["Subject", "session_n"]).reset_index(drop=True)


def build_baseline_table(long_table: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subject baselines over hab-01..hab-05."""
    hab_rows = long_table[
        (long_table["session_kind"] == "hab")
        & (long_table["session_n"] >= BASELINE_HAB_MIN)
        & (long_table["session_n"] <= BASELINE_HAB_MAX)
    ].copy()

    if hab_rows.empty:
        raise ValueError("No habituation rows found for baseline computation.")

    hab_counts = hab_rows.groupby("Subject")["session_n"].nunique()
    missing_subjects = hab_counts[hab_counts < (BASELINE_HAB_MAX - BASELINE_HAB_MIN + 1)]
    if not missing_subjects.empty:
        missing_str = ", ".join(
            f"{subject} ({int(count)}/{BASELINE_HAB_MAX - BASELINE_HAB_MIN + 1} days)"
            for subject, count in missing_subjects.items()
        )
        raise ValueError(
            "Missing habituation baseline days for subject(s): "
            f"{missing_str}. Expected hab-{BASELINE_HAB_MIN:02d}..hab-{BASELINE_HAB_MAX:02d}."
        )

    return (
        hab_rows
        .groupby("Subject", as_index=False)
        .agg(
            weight_baseline_g=("weight_g", "mean"),
            poops_baseline_count=("poops", "mean"),
        )
    )


def build_weight_change_table(
    raw_weight_table: pd.DataFrame,
    baseline_table: pd.DataFrame,
) -> pd.DataFrame:
    """Create raw change table (weight/poops from baseline) across timeline."""
    change_rows = raw_weight_table.merge(
        baseline_table,
        on="Subject",
        how="left",
        validate="many_to_one",
    )

    missing_baseline = change_rows["weight_baseline_g"].isna()
    if missing_baseline.any():
        missing_subjects_no_baseline = sorted(change_rows.loc[missing_baseline, "Subject"].astype(str).unique())
        raise ValueError(f"Missing baseline for subject(s): {missing_subjects_no_baseline}")

    change_rows["weight_change_g"] = change_rows["weight_g"] - change_rows["weight_baseline_g"]
    change_rows["poops_change_count"] = change_rows["poops"] - change_rows["poops_baseline_count"]

    return change_rows.sort_values(["Subject", "timeline_n"]).reset_index(drop=True)


def build_raw_weight_table(long_table: pd.DataFrame) -> pd.DataFrame:
    """Create raw-weight table spanning hab days and sessions 1-10."""
    raw_rows = long_table[
        (
            (long_table["session_kind"] == "hab")
            & (long_table["session_n"] >= BASELINE_HAB_MIN)
            & (long_table["session_n"] <= BASELINE_HAB_MAX)
        )
        |
        (
            (long_table["session_kind"] == "ses")
            & (long_table["session_n"] >= PLOT_SESSION_MIN)
            & (long_table["session_n"] <= PLOT_SESSION_MAX)
        )
    ].copy()

    if raw_rows.empty:
        raise ValueError(
            "No rows available for raw-weight panel across hab and session periods."
        )

    raw_rows["timeline_n"] = np.where(
        raw_rows["session_kind"] == "hab",
        raw_rows["session_n"] - (BASELINE_HAB_MAX + 1),
        raw_rows["session_n"],
    )

    return raw_rows.sort_values(["Subject", "timeline_n"]).reset_index(drop=True)


def build_weight_change_summary(weight_change_table: pd.DataFrame) -> pd.DataFrame:
    """Compute group mean and SEM for weight change (g) across timeline."""
    summary = (
        weight_change_table
        .groupby("timeline_n", as_index=False)
        .agg(
            mean_weight_change_g=("weight_change_g", "mean"),
            sem_weight_change_g=("weight_change_g", "sem"),
        )
        .sort_values("timeline_n")
        .reset_index(drop=True)
    )
    summary["sem_weight_change_g"] = summary["sem_weight_change_g"].fillna(0.0)
    return summary


def build_poop_change_summary(change_table: pd.DataFrame) -> pd.DataFrame:
    """Compute group mean and SEM for poop-count change across timeline."""
    summary = (
        change_table
        .groupby("timeline_n", as_index=False)
        .agg(
            mean_poop_change_count=("poops_change_count", "mean"),
            sem_poop_change_count=("poops_change_count", "sem"),
        )
        .sort_values("timeline_n")
        .reset_index(drop=True)
    )
    summary["sem_poop_change_count"] = summary["sem_poop_change_count"].fillna(0.0)
    return summary


def plot_subject_traces(
    weight_change_table: pd.DataFrame,
    raw_weight_table: pd.DataFrame,
    mean_trace_table: pd.DataFrame,
    mean_poop_trace_table: pd.DataFrame,
) -> Figure:
    """Plot raw weights, average changes, and poops in a 2x2 figure."""
    subject_labels = sorted(raw_weight_table["Subject"].dropna().astype(str).unique())
    palette = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(subject_labels), 1)))
    color_by_subject = dict(zip(subject_labels, palette))

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=False)
    ax_raw, ax_avg = axes[0]
    ax_poops, ax_poop_avg = axes[1]

    weight_subject_series = weight_change_table["Subject"].astype(str)
    raw_subject_series = raw_weight_table["Subject"].astype(str)

    raw_ticks = list(range(-BASELINE_HAB_MAX, 0)) + list(range(PLOT_SESSION_START, PLOT_SESSION_MAX + 1))
    raw_tick_labels = (
        [f"H{hab_day}" for hab_day in range(BASELINE_HAB_MIN, BASELINE_HAB_MAX + 1)]
        + [f"S{session_n}" for session_n in range(PLOT_SESSION_START, PLOT_SESSION_MAX + 1)]
    )

    for subject_label in subject_labels:
        subject_rows = weight_change_table[weight_subject_series == subject_label].sort_values("timeline_n")
        raw_subject_rows = raw_weight_table[raw_subject_series == subject_label].sort_values("timeline_n")

        ax_raw.plot(
            raw_subject_rows["timeline_n"],
            raw_subject_rows["weight_g"],
            "o-",
            lw=1.5,
            ms=4,
            alpha=0.9,
            color=color_by_subject[subject_label],
            label=subject_label,
        )

        ax_poops.plot(
            raw_subject_rows["timeline_n"],
            raw_subject_rows["poops"],
            "o-",
            lw=1.5,
            ms=4,
            alpha=0.9,
            color=color_by_subject[subject_label],
            drawstyle="steps-mid",
        )

    ax_avg.plot(
        mean_trace_table["timeline_n"],
        mean_trace_table["mean_weight_change_g"],
        "o-",
        color="black",
        lw=2.2,
        ms=5,
        label="Mean",
        zorder=3,
    )
    ax_avg.fill_between(
        mean_trace_table["timeline_n"],
        mean_trace_table["mean_weight_change_g"] - mean_trace_table["sem_weight_change_g"],
        mean_trace_table["mean_weight_change_g"] + mean_trace_table["sem_weight_change_g"],
        color="black",
        alpha=0.18,
        linewidth=0,
        zorder=2,
        label="Mean ± SEM",
    )

    ax_poop_avg.plot(
        mean_poop_trace_table["timeline_n"],
        mean_poop_trace_table["mean_poop_change_count"],
        "o-",
        color="black",
        lw=2.2,
        ms=5,
        label="Mean",
        zorder=3,
    )
    ax_poop_avg.fill_between(
        mean_poop_trace_table["timeline_n"],
        mean_poop_trace_table["mean_poop_change_count"] - mean_poop_trace_table["sem_poop_change_count"],
        mean_poop_trace_table["mean_poop_change_count"] + mean_poop_trace_table["sem_poop_change_count"],
        color="black",
        alpha=0.18,
        linewidth=0,
        zorder=2,
        label="Mean ± SEM",
    )

    mean_low = float(
        (mean_trace_table["mean_weight_change_g"] - mean_trace_table["sem_weight_change_g"]).min()
    )
    mean_high = float(
        (mean_trace_table["mean_weight_change_g"] + mean_trace_table["sem_weight_change_g"]).max()
    )
    mean_pad = max((mean_high - mean_low) * 0.1, 0.2)

    poop_low = float(
        (mean_poop_trace_table["mean_poop_change_count"] - mean_poop_trace_table["sem_poop_change_count"]).min()
    )
    poop_high = float(
        (mean_poop_trace_table["mean_poop_change_count"] + mean_poop_trace_table["sem_poop_change_count"]).max()
    )
    poop_pad = max((poop_high - poop_low) * 0.1, 0.4)

    ax_raw.set_title("Raw Weight: Habituation Through Session 10")
    ax_raw.set_xlabel("Session Timeline")
    ax_raw.set_ylabel("Weight (g)")
    ax_raw.set_xticks(raw_ticks)
    ax_raw.set_xticklabels(raw_tick_labels, rotation=0)
    ax_raw.set_xlim(raw_ticks[0] - 0.5, raw_ticks[-1] + 0.5)
    ax_raw.grid(alpha=0.3)
    ax_raw.axvline(0.0, color="black", lw=1.0, alpha=0.4)

    ax_avg.set_title(
        f"Average Weight Change from Hab-{BASELINE_HAB_MIN:02d}..{BASELINE_HAB_MAX:02d} Baseline"
    )
    ax_avg.set_xlabel("Session Timeline")
    ax_avg.set_ylabel("Weight Change (g)")
    ax_avg.set_xticks(raw_ticks)
    ax_avg.set_xticklabels(raw_tick_labels, rotation=0)
    ax_avg.set_xlim(raw_ticks[0] - 0.5, raw_ticks[-1] + 0.5)
    ax_avg.set_ylim(mean_low - mean_pad, mean_high + mean_pad)
    ax_avg.grid(alpha=0.3)
    ax_avg.axhline(0.0, color="black", lw=1.0, alpha=0.6)
    ax_avg.axvline(0.0, color="black", lw=1.0, alpha=0.4)

    ax_poops.set_title("Poops: Habituation Through Session 10")
    ax_poops.set_xlabel("Session Timeline")
    ax_poops.set_ylabel("Count")
    ax_poops.set_xticks(raw_ticks)
    ax_poops.set_xticklabels(raw_tick_labels, rotation=0)
    ax_poops.set_xlim(raw_ticks[0] - 0.5, raw_ticks[-1] + 0.5)
    ax_poops.grid(alpha=0.3)
    ax_poops.axvline(0.0, color="black", lw=1.0, alpha=0.4)
    ax_poops.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax_poop_avg.set_title(
        f"Average Poop Change from Hab-{BASELINE_HAB_MIN:02d}..{BASELINE_HAB_MAX:02d} Baseline"
    )
    ax_poop_avg.set_xlabel("Session Timeline")
    ax_poop_avg.set_ylabel("Poop Change (count)")
    ax_poop_avg.set_xticks(raw_ticks)
    ax_poop_avg.set_xticklabels(raw_tick_labels, rotation=0)
    ax_poop_avg.set_xlim(raw_ticks[0] - 0.5, raw_ticks[-1] + 0.5)
    ax_poop_avg.set_ylim(poop_low - poop_pad, poop_high + poop_pad)
    ax_poop_avg.grid(alpha=0.3)
    ax_poop_avg.axhline(0.0, color="black", lw=1.0, alpha=0.6)
    ax_poop_avg.axvline(0.0, color="black", lw=1.0, alpha=0.4)
    ax_poop_avg.yaxis.set_major_locator(MaxNLocator(integer=True))

    for ax in (ax_raw, ax_avg, ax_poops, ax_poop_avg):
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    handles, labels = ax_raw.get_legend_handles_labels()
    if handles:
        ax_poops.legend(
            handles,
            labels,
            title="Subject",
            loc="upper left",
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            borderaxespad=0.3,
        )
        fig.tight_layout(pad=1.1, w_pad=1.2, h_pad=1.4)
    else:
        fig.tight_layout(pad=1.1, w_pad=1.2, h_pad=1.4)

    return fig


proj = Project(dataset=DATASET, run_name=RUN_NAME, tag=TAG)

weights_poops = load_hfsa_weights_poops(WEIGHTS_XLSX)
longitudinal_all = prepare_longitudinal_table(weights_poops, proj)
raw_weight_longitudinal = build_raw_weight_table(longitudinal_all)
baseline_by_subject = build_baseline_table(longitudinal_all)
longitudinal = build_weight_change_table(raw_weight_longitudinal, baseline_by_subject)
mean_trace = build_weight_change_summary(longitudinal)
mean_poop_trace = build_poop_change_summary(longitudinal)

observed_sessions = sorted(
    int(session_n)
    for session_n in raw_weight_longitudinal.loc[
        raw_weight_longitudinal["session_kind"] == "ses", "session_n"
    ].unique()
)
missing_sessions = [
    session_n
    for session_n in range(PLOT_SESSION_MIN, PLOT_SESSION_MAX + 1)
    if session_n not in observed_sessions
]

proj.io.table(longitudinal, "weights_change_longitudinal.csv")
proj.io.table(baseline_by_subject, "weights_hab_baseline.csv")
proj.io.table(raw_weight_longitudinal, "weights_raw_hab_to_ses10.csv")
proj.io.table(mean_trace, "weights_change_mean_sem.csv")
proj.io.table(mean_poop_trace, "poops_change_mean_sem.csv")

figure = plot_subject_traces(
    longitudinal,
    raw_weight_longitudinal,
    mean_trace,
    mean_poop_trace,
)
plot_path = proj.io.figure(
    figure,
    "weights_poops_subject_traces.svg",
    formats=("png",),
)

report_path = proj.io.report(
    notes=(
        "Per-subject habituation baseline computed as mean weight over hab-01..hab-05. "
        "Figure panels include raw weight trajectories (hab through ses-10), "
        "group mean ± SEM weight-change trace, subject-level poops trajectories, "
        "and group mean ± SEM poop-change trace across the same timeline."
    )
)

print(f"Rows: {len(longitudinal)}")
print(f"Subjects: {longitudinal['Subject'].nunique()}")
print(f"Baseline rows: {len(baseline_by_subject)}")
print(f"Observed sessions: {observed_sessions}")
if missing_sessions:
    print(f"Missing sessions in sheet (ses days): {missing_sessions}")
print(f"Figure: {plot_path}")
print(f"Report: {report_path}")
print(f"Run dir: {proj.io.run_dir}")
