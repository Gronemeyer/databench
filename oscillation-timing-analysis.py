"""
Oscillation timing analysis: onset latency and bout-continuation prediction.

Determines whether oscillations emerge at a characteristic latency (state-entry
model) vs. predict bout continuation (shared arousal model).

Metrics:
    1. Oscillation onset latency — time from quiescence start to first
       oscillation onset, stratified by bout duration terciles.
    2. Early oscillation as predictor — among bouts reaching 10s, does
       oscillation in the first 10s predict continuation to 30s?
    3. Onset latency vs bout duration scatter — does latency scale with
       bout length (diagonal) or stay constant (horizontal band)?

Outputs:
    - 3-panel figure (latency histogram, predictor bar, scatter)
    - Summary statistics in markdown report

Usage:
    python Scripts/scriptings/oscillation-timing-analysis.py
"""
from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from databench import (
    Project,
    OscillationDetector,
    locomotion_bouts,
    quiescent_bouts,
    set_theme,
)
from databench.config import resolve_dataset

set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET = resolve_dataset("etoh-hfsa")
TASK = "task-widefield"
ROI_SOURCE = "mesomap"
ROI_NAME = "L_VISp"

# Oscillation detector settings (match quiescent-oscillation-probability)
FS = 50.0
BAND = (2.0, 4.0)
ORDER = 4
THRESHOLD = 0.02
THRESHOLD_K = 4.0
MIN_DURATION_OSC = 2.0
MERGE_GAP_OSC = 0.0

# Locomotion bout settings
MIN_SPEED_CMS = 0.5
MIN_LOCO_DURATION_S = 2.0
MERGE_LOCO_GAP_S = 0.5

# Quiescent bout minimum duration
MIN_QUIESCENT_S = 1.0

# Thresholds for "early oscillation predicts continuation" analysis
EARLY_WINDOW_S = 10.0  # oscillation must occur within first 10s
CONTINUATION_THRESHOLD_S = 30.0  # bout must reach 30s to count as "continued"


# ─── Helpers ──────────────────────────────────────────────────────────────

def first_oscillation_onset(
    q_start_s: float,
    q_end_s: float,
    osc_events: pd.DataFrame,
) -> float | None:
    """Return the onset time (s) of the first oscillation event overlapping the bout.

    Returns None if no oscillation occurs during the bout.
    """
    if osc_events.empty:
        return None
    # Oscillation events that overlap this quiescent bout
    overlapping = osc_events[
        (osc_events["end_s"] > q_start_s) & (osc_events["start_s"] < q_end_s)
    ]
    if overlapping.empty:
        return None
    # Earliest onset (clamp to bout start if oscillation started before the bout)
    earliest_start = overlapping["start_s"].min()
    return max(earliest_start, q_start_s)


# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    run_name="oscillation-timing-analysis",
    tag=f"{ROI_NAME}-{TASK}",
)

group = proj.sessions(task=TASK)
print(f"Selected {len(group)} sessions for task={TASK!r}")
print(f"  Subjects: {group.subjects}")

detector = OscillationDetector(
    source=ROI_SOURCE,
    signal=ROI_NAME,
    fs=FS,
    band_hz=BAND,
    filter_order=ORDER,
    threshold=THRESHOLD,
    threshold_k=THRESHOLD_K,
    min_duration_s=MIN_DURATION_OSC,
    merge_gap_s=MERGE_GAP_OSC,
)

# ─── Collect per-bout data ───────────────────────────────────────────────

rows: list[dict] = []

for sess in group:
    try:
        speed_t = sess.time("treadmill", "time_elapsed_s")
        speed_v = sess.signal("treadmill", "speed_mm")
    except Exception as e:
        print(f"  SKIP {sess.subject}/{sess.session} (no treadmill): {e}")
        continue

    speed_cms = speed_v / 10.0
    loco = locomotion_bouts(
        speed_t, speed_cms,
        min_speed_cms=MIN_SPEED_CMS,
        min_duration_s=MIN_LOCO_DURATION_S,
        merge_gap_s=MERGE_LOCO_GAP_S,
    )
    quiet = quiescent_bouts(speed_t, loco, min_duration_s=MIN_QUIESCENT_S)

    try:
        osc_result = detector.run(sess)
    except Exception as e:
        print(f"  SKIP {sess.subject}/{sess.session} (oscillation detection): {e}")
        continue

    osc_events = osc_result.events

    n_with_osc = 0
    for q_s, q_e in quiet:
        q_start_s = float(speed_t[q_s])
        q_end_s = float(speed_t[q_e])
        q_dur = q_end_s - q_start_s

        onset_t = first_oscillation_onset(q_start_s, q_end_s, osc_events)
        has_osc = onset_t is not None
        latency = (onset_t - q_start_s) if has_osc else None
        n_with_osc += int(has_osc)

        rows.append({
            "Subject": sess.subject,
            "Session": sess.session,
            "q_start_s": round(q_start_s, 3),
            "q_end_s": round(q_end_s, 3),
            "q_duration_s": round(q_dur, 3),
            "has_oscillation": has_osc,
            "osc_onset_s": round(onset_t, 3) if onset_t is not None else None,
            "osc_latency_s": round(latency, 3) if latency is not None else None,
        })

    print(
        f"  {sess.subject}/{sess.session}: "
        f"{len(quiet)} quiescent bouts, {n_with_osc} with oscillation"
    )

# ─── Build DataFrame ─────────────────────────────────────────────────────

df = pd.DataFrame(rows)
subjects = sorted(df["Subject"].unique())
n_subjects = len(subjects)
COLORS = plt.cm.tab10(np.linspace(0, 1, max(n_subjects, 1)))
subject_colors = {s: COLORS[i] for i, s in enumerate(subjects)}

osc_df = df[df["has_oscillation"]].copy()

print(f"\nTotal quiescent bouts: {len(df)}")
print(f"  With oscillation: {len(osc_df)}")
print(f"  Median onset latency: {osc_df['osc_latency_s'].median():.1f}s")

# ─── Tercile labels for bout duration ────────────────────────────────────

tercile_edges = osc_df["q_duration_s"].quantile([0, 1/3, 2/3, 1]).values
tercile_labels = [
    f"short (<{tercile_edges[1]:.0f}s)",
    f"medium ({tercile_edges[1]:.0f}–{tercile_edges[2]:.0f}s)",
    f"long (>{tercile_edges[2]:.0f}s)",
]
osc_df["dur_tercile"] = pd.cut(
    osc_df["q_duration_s"],
    bins=tercile_edges,
    labels=tercile_labels,
    include_lowest=True,
)

# ─── Early-oscillation prediction analysis ───────────────────────────────
# Among bouts that reach EARLY_WINDOW_S: did oscillation occur in first 10s?
# Then: P(bout reaches CONTINUATION_THRESHOLD_S | early osc) vs P(... | no early osc)

long_enough = df[df["q_duration_s"] >= EARLY_WINDOW_S].copy()
long_enough["early_osc"] = (
    long_enough["has_oscillation"]
    & long_enough["osc_latency_s"].notna()
    & (long_enough["osc_latency_s"] <= EARLY_WINDOW_S)
)
long_enough["continued"] = long_enough["q_duration_s"] >= CONTINUATION_THRESHOLD_S

n_early_osc = long_enough["early_osc"].sum()
n_no_early_osc = (~long_enough["early_osc"]).sum()

if n_early_osc > 0:
    p_continue_given_osc = long_enough.loc[long_enough["early_osc"], "continued"].mean()
else:
    p_continue_given_osc = np.nan

if n_no_early_osc > 0:
    p_continue_given_no_osc = long_enough.loc[~long_enough["early_osc"], "continued"].mean()
else:
    p_continue_given_no_osc = np.nan

# Fisher's exact test on the 2x2 table
ct = pd.crosstab(long_enough["early_osc"], long_enough["continued"])
if ct.shape == (2, 2):
    fisher_or, fisher_p = sp_stats.fisher_exact(ct)
else:
    fisher_or, fisher_p = np.nan, np.nan

print(f"\n── Early oscillation as predictor ──")
print(f"  Bouts reaching {EARLY_WINDOW_S}s: {len(long_enough)}")
print(f"  With early oscillation (≤{EARLY_WINDOW_S}s): {n_early_osc}")
print(f"  Without early oscillation: {n_no_early_osc}")
print(f"  P(continue to {CONTINUATION_THRESHOLD_S}s | early osc): {p_continue_given_osc:.3f}")
print(f"  P(continue to {CONTINUATION_THRESHOLD_S}s | no early osc): {p_continue_given_no_osc:.3f}")
print(f"  Fisher's exact OR: {fisher_or:.3f}, p = {fisher_p:.2e}")

# Latency summary
latency_med = osc_df["osc_latency_s"].median()
latency_q1 = osc_df["osc_latency_s"].quantile(0.25)
latency_q3 = osc_df["osc_latency_s"].quantile(0.75)
latency_iqr = latency_q3 - latency_q1

print(f"\n── Oscillation onset latency ──")
print(f"  Median: {latency_med:.1f}s  IQR: [{latency_q1:.1f}, {latency_q3:.1f}]s")

# Spearman correlation: latency vs bout duration
rho, rho_p = sp_stats.spearmanr(osc_df["q_duration_s"], osc_df["osc_latency_s"])
print(f"  Spearman ρ(bout_dur, latency): {rho:.3f}, p = {rho_p:.2e}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE: 3-panel plot
# ═════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ── Panel A: Onset latency histograms by bout-duration tercile ───────────

ax = axes[0]
tercile_colors = ["#4C72B0", "#DD8452", "#55A868"]
for i, label in enumerate(tercile_labels):
    subset = osc_df[osc_df["dur_tercile"] == label]["osc_latency_s"]
    if subset.empty:
        continue
    ax.hist(
        subset, bins=20, alpha=0.55, color=tercile_colors[i],
        label=f"{label} (n={len(subset)})", edgecolor="white", lw=0.5,
    )
ax.axvline(latency_med, color="black", ls="--", lw=1.2, label=f"median = {latency_med:.1f}s")
ax.set_xlabel("Oscillation onset latency (s)")
ax.set_ylabel("Count")
ax.set_title("A. Onset latency by bout-duration tercile")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)

# ── Panel B: P(continuation) conditional on early oscillation ────────────

ax = axes[1]
bar_labels = [f"Early osc\n(n={n_early_osc})", f"No early osc\n(n={n_no_early_osc})"]
bar_vals = [p_continue_given_osc, p_continue_given_no_osc]
bar_colors = ["#DD8452", "#4C72B0"]
bars = ax.bar(bar_labels, bar_vals, color=bar_colors, width=0.5, edgecolor="white")

for bar, val in zip(bars, bar_vals):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

fisher_label = f"Fisher OR = {fisher_or:.2f}, p = {fisher_p:.2e}" if not np.isnan(fisher_or) else "insufficient data"
ax.set_ylabel(f"P(bout ≥ {CONTINUATION_THRESHOLD_S:.0f}s)")
ax.set_title(
    f"B. Early oscillation predicts continuation?\n"
    f"(bouts ≥ {EARLY_WINDOW_S:.0f}s, {fisher_label})",
)
ax.set_ylim(0, min(max(bar_vals) + 0.15, 1.05) if bar_vals else 1.0)
ax.grid(axis="y", alpha=0.3)

# ── Panel C: Scatter — bout duration vs onset latency ────────────────────

ax = axes[2]
for subj in subjects:
    sub = osc_df[osc_df["Subject"] == subj]
    ax.scatter(
        sub["q_duration_s"], sub["osc_latency_s"],
        s=18, alpha=0.5, color=subject_colors[subj], label=subj, edgecolors="none",
    )

# Reference lines: identity (diagonal) and median latency (horizontal)
max_dur = osc_df["q_duration_s"].max()
ax.plot([0, max_dur], [0, max_dur], "k:", lw=0.8, alpha=0.4, label="y = x (diagonal)")
ax.axhline(latency_med, color="black", ls="--", lw=1, alpha=0.6,
           label=f"median latency = {latency_med:.1f}s")

ax.set_xlabel("Quiescent bout duration (s)")
ax.set_ylabel("Oscillation onset latency (s)")
ax.set_title(
    f"C. Latency vs bout duration\n"
    f"ρ = {rho:.3f}, p = {rho_p:.2e}",
)
ax.legend(frameon=False, markerscale=1.5)
ax.grid(alpha=0.3)

fig.suptitle(
    f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK} | N = {n_subjects} animals",
    y=1.02,
)
fig.tight_layout()
proj.io.figure(fig, "oscillation_timing_analysis.svg")

# ═════════════════════════════════════════════════════════════════════════
# Save tables
# ═════════════════════════════════════════════════════════════════════════

proj.io.table(df, "quiescent_bouts_timing.csv")

# ═════════════════════════════════════════════════════════════════════════
# Markdown report
# ═════════════════════════════════════════════════════════════════════════

# Per-tercile latency summary
tercile_stats_lines = []
for label in tercile_labels:
    subset = osc_df[osc_df["dur_tercile"] == label]["osc_latency_s"]
    if subset.empty:
        tercile_stats_lines.append(f"| {label} | – | – | 0 |")
        continue
    med = subset.median()
    q1 = subset.quantile(0.25)
    q3 = subset.quantile(0.75)
    tercile_stats_lines.append(f"| {label} | {med:.1f} | [{q1:.1f}, {q3:.1f}] | {len(subset)} |")
tercile_table = "\n".join(tercile_stats_lines)

report_path = proj.io.report(
    notes=(
        f"## Oscillation timing analysis\n\n"
        f"**Dataset:** ETOH-HFSA | **Task:** {TASK}\n\n"
        f"**ROI:** {ROI_NAME} | **Band:** {BAND[0]}–{BAND[1]} Hz\n\n"
        f"**Oscillation:** threshold={THRESHOLD} (ΔF/F envelope), "
        f"min_duration={MIN_DURATION_OSC}s\n\n"
        f"**Min quiescent bout:** {MIN_QUIESCENT_S}s\n\n"
        f"---\n\n"
        f"### 1. Oscillation onset latency\n\n"
        f"Median latency: **{latency_med:.1f}s** "
        f"(IQR: [{latency_q1:.1f}, {latency_q3:.1f}]s)\n\n"
        f"| Bout-duration tercile | Median latency (s) | IQR | N bouts |\n"
        f"|----------------------|--------------------:|----:|--------:|\n"
        f"{tercile_table}\n\n"
        f"Spearman ρ(bout duration, onset latency) = {rho:.3f} (p = {rho_p:.2e})\n\n"
        f"---\n\n"
        f"### 2. Early oscillation as predictor of bout continuation\n\n"
        f"Among bouts ≥ {EARLY_WINDOW_S:.0f}s (n = {len(long_enough)}):\n\n"
        f"| Condition | P(bout ≥ {CONTINUATION_THRESHOLD_S:.0f}s) | N |\n"
        f"|-----------|---:|---:|\n"
        f"| Early oscillation (≤{EARLY_WINDOW_S:.0f}s) | {p_continue_given_osc:.3f} | {n_early_osc} |\n"
        f"| No early oscillation | {p_continue_given_no_osc:.3f} | {n_no_early_osc} |\n\n"
        f"Fisher's exact test: OR = {fisher_or:.2f}, p = {fisher_p:.2e}\n\n"
        f"---\n\n"
        f"### Interpretation guide\n\n"
        f"- **State-entry model** predicts: latency peaks at ~8–15s regardless "
        f"of bout duration (horizontal band in scatter, similar tercile medians).\n"
        f"- **Shared arousal model** predicts: latency correlates negatively with "
        f"bout duration (drowsier → oscillate earlier & stay still longer), and "
        f"early oscillation predicts continuation (higher conditional probability).\n\n"
        f"**Animals:** {', '.join(subjects)}\n"
    ),
)
print(f"\nReport: {report_path}")
print(f"Outputs saved to: {proj.io.run_dir}")
