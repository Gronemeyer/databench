"""
Quiescent bout duration vs oscillation probability.

For each session (ETOH-HFSA, task-widefield), detects locomotion bouts,
inverts them to get quiescent (non-locomotion) periods, detects oscillation
events, then asks: does the probability of observing an oscillation event
increase with quiescent bout duration?

Statistical approach:
    - GEE logistic model (clustered by animal) for population-averaged
      inference on the duration effect, avoiding pseudoreplication.
    - Per-animal logistic fits to verify within-animal consistency.
    - Bout durations truncated at data-driven percentile to avoid
      sparse-data artefacts in the tail.
    - Naive pooled correlation reported for reference only, clearly
      labelled as descriptive.

Outputs:
    1. Per-animal logistic fits (consistency check) + forest plot
    2. GEE model curve with 95% CI + data density histogram
    3. Per-animal binned P(oscillation) curves (descriptive)
    4. Scatter: duration vs oscillation fraction, by animal
    5. Combined quiescent bout table (CSV)
    6. Markdown report with GEE table

Usage:
    python Scripts/scriptings/quiescent-oscillation-probability.py
"""
from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.special import expit  # logistic function
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

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

# Oscillation detector settings (match descriptive-stats script)
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

# Quiescent bout minimum duration (seconds) to include
MIN_QUIESCENT_S = 1.0

# Truncate bouts beyond this percentile for modelling (sparse tail)
DURATION_TRUNCATE_PCTL = 95

# Duration bins for descriptive plots (seconds)
DURATION_BIN_WIDTH = 2  # seconds


# ─── Helper: compute oscillation overlap for a quiescent bout ────────────

def oscillation_overlap(
    q_start_s: float,
    q_end_s: float,
    osc_events: pd.DataFrame,
) -> Tuple[float, bool]:
    """Return (fraction of quiescent bout spent in oscillation, any_overlap)."""
    q_dur = q_end_s - q_start_s
    if q_dur <= 0 or osc_events.empty:
        return 0.0, False
    overlap_s = 0.0
    for _, ev in osc_events.iterrows():
        ov_start = max(q_start_s, ev["start_s"])
        ov_end = min(q_end_s, ev["end_s"])
        if ov_end > ov_start:
            overlap_s += ov_end - ov_start
    return overlap_s / q_dur, overlap_s > 0


# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    run_name="quiescent-oscillation-probability",
    tag=f"{ROI_NAME}-{TASK}",
)

group = proj.sessions(task=TASK)
print(f"Selected {len(group)} sessions for task={TASK!r}")
print(f"  Subjects: {group.subjects}")
print(f"  Sessions: {group.session_labels}")

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

# ─── Process each session ────────────────────────────────────────────────

all_quiet_rows: list[dict] = []

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

    for q_s, q_e in quiet:
        q_start_s = float(speed_t[q_s])
        q_end_s = float(speed_t[q_e])
        q_dur = q_end_s - q_start_s
        frac, has_osc = oscillation_overlap(q_start_s, q_end_s, osc_events)

        all_quiet_rows.append({
            "Subject": sess.subject,
            "Session": sess.session,
            "Task": sess.task,
            "q_start_s": round(q_start_s, 3),
            "q_end_s": round(q_end_s, 3),
            "q_duration_s": round(q_dur, 3),
            "has_oscillation": has_osc,
            "osc_fraction": round(frac, 5),
        })

    n_quiet = len(quiet)
    n_with_osc = sum(1 for r in all_quiet_rows[-n_quiet:] if r["has_oscillation"])
    print(
        f"  {sess.subject} / {sess.session}: "
        f"{len(loco)} loco bouts, {n_quiet} quiescent bouts, "
        f"{n_with_osc}/{n_quiet} with oscillation"
    )

# ─── Combine data ────────────────────────────────────────────────────────

qdf = pd.DataFrame(all_quiet_rows)
qdf["session_n"] = qdf["Session"].str.extract(r"(\d+)", expand=False).astype(float)
qdf["has_osc_int"] = qdf["has_oscillation"].astype(int)
qdf["log_duration"] = np.log(qdf["q_duration_s"].clip(lower=0.01))

subjects = sorted(qdf["Subject"].unique())
n_subjects = len(subjects)
COLORS = plt.cm.tab10(np.linspace(0, 1, max(n_subjects, 1)))
subject_colors = {subj: COLORS[i] for i, subj in enumerate(subjects)}

# Truncation threshold (drop sparse tail)
dur_cutoff = qdf["q_duration_s"].quantile(DURATION_TRUNCATE_PCTL / 100)
qdf_model = qdf[qdf["q_duration_s"] <= dur_cutoff].copy()

print(f"\nTotal quiescent bouts: {len(qdf)}")
print(f"  With oscillation: {qdf['has_oscillation'].sum()}")
print(f"  Overall P(osc): {qdf['has_oscillation'].mean():.3f}")
print(f"  Duration truncation at p{DURATION_TRUNCATE_PCTL}: {dur_cutoff:.1f}s "
      f"({len(qdf_model)}/{len(qdf)} bouts retained)")
print(f"  N animals: {n_subjects}")

# Duration bins (for descriptive plots only — not for inference)
DURATION_BINS = np.arange(0, dur_cutoff + DURATION_BIN_WIDTH, DURATION_BIN_WIDTH)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 1: GEE logistic regression — population-averaged effect
#   has_osc ~ log(duration), clustered by Subject
#   Exchangeable correlation, robust (sandwich) SE
# ═══════════════════════════════════════════════════════════════════════════

print("\n── GEE logistic regression (clustered by animal) ──")

# Subject must be numeric for GEE groups
subj_map = {s: i for i, s in enumerate(subjects)}
qdf_model["subj_id"] = qdf_model["Subject"].map(subj_map)
qdf_model = qdf_model.sort_values("subj_id")  # GEE requires sorted groups

gee_model = GEE(
    endog=qdf_model["has_osc_int"],
    exog=sm.add_constant(qdf_model["log_duration"]),
    groups=qdf_model["subj_id"],
    family=Binomial(),
    cov_struct=Exchangeable(),
)
gee_result = gee_model.fit()

print(gee_result.summary())

gee_intercept = gee_result.params["const"]
gee_slope = gee_result.params["log_duration"]
gee_slope_se = gee_result.bse["log_duration"]
gee_slope_p = gee_result.pvalues["log_duration"]
gee_slope_ci = gee_result.conf_int().loc["log_duration"]

print(f"\n  Fixed effect of log(duration): β = {gee_slope:.4f}")
print(f"  Robust SE: {gee_slope_se:.4f}")
print(f"  95% CI: [{gee_slope_ci[0]:.4f}, {gee_slope_ci[1]:.4f}]")
print(f"  p = {gee_slope_p:.2e}")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2: Per-animal logistic regressions (consistency check)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Per-animal logistic regressions ──")

animal_fits: dict[str, dict] = {}
for subj in subjects:
    sub = qdf_model[qdf_model["Subject"] == subj]
    if sub["has_osc_int"].nunique() < 2 or len(sub) < 10:
        print(f"  {subj}: insufficient data (n={len(sub)}, skipping)")
        continue
    X = sm.add_constant(sub["log_duration"])
    y = sub["has_osc_int"]
    try:
        fit = sm.Logit(y, X).fit(disp=0)
        animal_fits[subj] = {
            "intercept": fit.params["const"],
            "slope": fit.params["log_duration"],
            "slope_se": fit.bse["log_duration"],
            "slope_p": fit.pvalues["log_duration"],
            "slope_ci_lo": fit.conf_int().loc["log_duration", 0],
            "slope_ci_hi": fit.conf_int().loc["log_duration", 1],
            "n": len(sub),
            "pseudo_r2": fit.prsquared,
        }
        print(f"  {subj}: β = {fit.params['log_duration']:.4f} "
              f"(p = {fit.pvalues['log_duration']:.2e}, n = {len(sub)})")
    except Exception as e:
        print(f"  {subj}: fit failed ({e})")

# Check consistency: are all per-animal slopes in the same direction?
slopes = [v["slope"] for v in animal_fits.values()]
slopes_same_sign = all(s > 0 for s in slopes) or all(s < 0 for s in slopes)
print(f"\n  All slopes same sign? {slopes_same_sign}")
print(f"  Slopes: {[round(s, 4) for s in slopes]}")

# ═══════════════════════════════════════════════════════════════════════════
# Descriptive (naive) correlations — reported for reference, labelled
# ═══════════════════════════════════════════════════════════════════════════

if len(qdf_model) > 5:
    r_pb, p_pb = sp_stats.pointbiserialr(
        qdf_model["has_osc_int"], qdf_model["q_duration_s"]
    )
    r_sp, p_sp = sp_stats.spearmanr(
        qdf_model["q_duration_s"], qdf_model["osc_fraction"]
    )
else:
    r_pb, p_pb, r_sp, p_sp = np.nan, np.nan, np.nan, np.nan

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1: Per-animal logistic fits — consistency check + forest plot
# ═══════════════════════════════════════════════════════════════════════════

fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))

# 1a: Individual logistic curves per animal
ax = axes1[0]
dur_plot = np.linspace(qdf_model["q_duration_s"].min(), dur_cutoff, 200)
log_dur_plot = np.log(np.clip(dur_plot, 0.01, None))

for subj in subjects:
    if subj not in animal_fits:
        continue
    fit = animal_fits[subj]
    p_curve = expit(fit["intercept"] + fit["slope"] * log_dur_plot)
    ax.plot(dur_plot, p_curve, lw=2, color=subject_colors[subj], label=subj)

# GEE population-averaged curve
gee_curve = expit(gee_intercept + gee_slope * log_dur_plot)
ax.plot(dur_plot, gee_curve, lw=2.5, color="black", ls="--", label="GEE (population avg)")

ax.set_xlabel("Quiescent bout duration (s)")
ax.set_ylabel("P(oscillation event)")
ax.set_title("Per-animal logistic fits vs GEE")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# 1b: Forest plot of per-animal slopes
ax = axes1[1]
fit_subjects = sorted(animal_fits.keys())
y_pos = np.arange(len(fit_subjects))
for i, subj in enumerate(fit_subjects):
    f = animal_fits[subj]
    ax.errorbar(
        f["slope"], i,
        xerr=[[f["slope"] - f["slope_ci_lo"]], [f["slope_ci_hi"] - f["slope"]]],
        fmt="o", color=subject_colors[subj], ms=8, capsize=5, lw=1.5,
    )
ax.axvline(gee_slope, color="black", ls="--", lw=1.5, label=f"GEE β = {gee_slope:.3f}")
ax.fill_betweenx(
    [-0.5, len(fit_subjects) - 0.5], gee_slope_ci[0], gee_slope_ci[1],
    color="gray", alpha=0.15, label="GEE 95% CI",
)
ax.axvline(0, color="red", ls=":", lw=1, alpha=0.6, label="β = 0 (no effect)")
ax.set_yticks(y_pos)
ax.set_yticklabels(fit_subjects)
ax.set_xlabel("log(duration) slope (β)")
ax.set_title("Forest plot: per-animal slopes")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="x", alpha=0.3)

fig1.suptitle(
    f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK} | N = {n_subjects} animals",
    y=1.01,
)
fig1.tight_layout()
proj.io.figure(fig1, "per_animal_logistic_fits.svg")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2: GEE model curve with 95% CI + data density histogram
# ═══════════════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [2, 1]})

# 2a: GEE logistic with CI (delta-method)
ax = axes2[0]
X_pred = sm.add_constant(pd.DataFrame({"log_duration": log_dur_plot}))
lin_pred = X_pred @ gee_result.params
se_pred = np.sqrt(np.sum((X_pred @ gee_result.cov_params()) * X_pred, axis=1))
ci_lo_logit = lin_pred - 1.96 * se_pred
ci_hi_logit = lin_pred + 1.96 * se_pred

ax.fill_between(dur_plot, expit(ci_lo_logit), expit(ci_hi_logit),
                color="#1f77b4", alpha=0.15, label="GEE 95% CI")
ax.plot(dur_plot, expit(lin_pred), lw=2.5, color="#1f77b4", label="GEE model")

# Overlay per-animal binned proportions as points
qdf_model["dur_bin"] = pd.cut(qdf_model["q_duration_s"], bins=DURATION_BINS)
for subj in subjects:
    sub = qdf_model[qdf_model["Subject"] == subj]
    sb = sub.groupby("dur_bin", observed=True).agg(
        n=("has_osc_int", "count"),
        p=("has_osc_int", "mean"),
    ).reset_index()
    sb["bin_center"] = sb["dur_bin"].apply(lambda x: x.mid)
    sb = sb[sb["n"] >= 3]
    ax.scatter(
        sb["bin_center"], sb["p"],
        s=sb["n"] * 3, alpha=0.5, color=subject_colors[subj], label=subj,
        edgecolors="white", lw=0.5,
    )

ax.set_ylabel("P(oscillation event)")
ax.set_title(
    f"GEE logistic: P(oscillation) ~ log(duration), clustered by animal\n"
    f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | N = {n_subjects} animals | "
    f"β = {gee_slope:.3f} [{gee_slope_ci[0]:.3f}, {gee_slope_ci[1]:.3f}], "
    f"p = {gee_slope_p:.2e}",
)
ax.legend(frameon=False, ncol=2)
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# 2b: Bout count histogram (data density)
ax = axes2[1]
pooled_binned = qdf_model.groupby("dur_bin", observed=True)["has_osc_int"].count().reset_index()
pooled_binned.columns = ["dur_bin", "n_bouts"]
pooled_binned["bin_center"] = pooled_binned["dur_bin"].apply(lambda x: x.mid)
ax.bar(
    pooled_binned["bin_center"], pooled_binned["n_bouts"],
    width=DURATION_BIN_WIDTH, color="#1f77b4", alpha=0.4, edgecolor="white",
)
ax.axvline(dur_cutoff, color="red", ls="--", lw=1, alpha=0.6,
           label=f"p{DURATION_TRUNCATE_PCTL} truncation ({dur_cutoff:.0f}s)")
ax.set_xlabel("Quiescent bout duration (s)")
ax.set_ylabel("N bouts")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)

fig2.tight_layout()
proj.io.figure(fig2, "gee_model_curve.svg")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3: Per-animal binned P(oscillation) + pooled (descriptive only)
# ═══════════════════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(figsize=(8, 5))

pooled = qdf_model.groupby("dur_bin", observed=True).agg(
    n_bouts=("has_osc_int", "count"),
    p_osc=("has_osc_int", "mean"),
).reset_index()
pooled["bin_center"] = pooled["dur_bin"].apply(lambda x: x.mid)

for subj in subjects:
    sub = qdf_model[qdf_model["Subject"] == subj]
    sb = sub.groupby("dur_bin", observed=True).agg(
        n=("has_osc_int", "count"),
        p=("has_osc_int", "mean"),
    ).reset_index()
    sb["bin_center"] = sb["dur_bin"].apply(lambda x: x.mid)
    sb = sb[sb["n"] >= 3]
    ax3.plot(
        sb["bin_center"], sb["p"],
        "o-", color=subject_colors[subj], ms=4, lw=1.5, alpha=0.7, label=subj,
    )

ax3.plot(
    pooled["bin_center"], pooled["p_osc"],
    "s-", color="black", ms=6, lw=2, zorder=5, label="pooled",
)

ax3.set_xlabel("Quiescent bout duration (s)")
ax3.set_ylabel("P(oscillation event)")
ax3.set_title(
    f"Binned P(oscillation) per animal (descriptive)\n"
    f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK} | "
    f"bouts truncated at {dur_cutoff:.0f}s",
)
ax3.legend(frameon=False)
ax3.grid(alpha=0.3)
ax3.set_ylim(-0.05, 1.05)
fig3.tight_layout()
proj.io.figure(fig3, "per_animal_binned_probability.svg")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4: Scatter — duration vs oscillation fraction, by animal
# ═══════════════════════════════════════════════════════════════════════════

fig4, ax4 = plt.subplots(figsize=(8, 5))

for subj in subjects:
    sub = qdf_model[qdf_model["Subject"] == subj]
    ax4.scatter(
        sub["q_duration_s"], sub["osc_fraction"],
        s=15, alpha=0.4, color=subject_colors[subj], label=subj, edgecolors="none",
    )

ax4.set_xlabel("Quiescent bout duration (s)")
ax4.set_ylabel("Fraction of bout in oscillation")
ax4.set_title(
    f"Quiescent bout duration vs oscillation fraction\n"
    f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK} | "
    f"bouts ≤ {dur_cutoff:.0f}s",
)
ax4.legend(frameon=False, markerscale=2)
ax4.grid(alpha=0.3)
fig4.tight_layout()
proj.io.figure(fig4, "scatter_duration_vs_osc_fraction.svg")

# ═══════════════════════════════════════════════════════════════════════════
# Per-animal summary table
# ═══════════════════════════════════════════════════════════════════════════

animal_summary = qdf.groupby("Subject").agg(
    n_quiet_bouts=("q_duration_s", "count"),
    mean_quiet_dur_s=("q_duration_s", "mean"),
    median_quiet_dur_s=("q_duration_s", "median"),
    p_oscillation=("has_oscillation", "mean"),
    mean_osc_fraction=("osc_fraction", "mean"),
).round(4).reset_index()

for subj in subjects:
    if subj in animal_fits:
        f = animal_fits[subj]
        animal_summary.loc[animal_summary["Subject"] == subj, "logistic_slope"] = round(f["slope"], 4)
        animal_summary.loc[animal_summary["Subject"] == subj, "logistic_slope_p"] = f["slope_p"]
    else:
        animal_summary.loc[animal_summary["Subject"] == subj, "logistic_slope"] = np.nan
        animal_summary.loc[animal_summary["Subject"] == subj, "logistic_slope_p"] = np.nan

print("\n── Per-animal summary ──")
print(animal_summary.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# Save tables
# ═══════════════════════════════════════════════════════════════════════════

proj.io.table(qdf, "quiescent_bouts_all.csv")
proj.io.table(animal_summary, "per_animal_summary.csv")

# GEE summary as text
proj.io.text(
    "gee_logistic_summary.txt",
    str(gee_result.summary())
    + f"\n\nExchangeable correlation estimate: {gee_result.cov_struct.summary()}\n",
)

# Per-animal fits table
if animal_fits:
    fits_df = pd.DataFrame(animal_fits).T.reset_index().rename(columns={"index": "Subject"})
    proj.io.table(fits_df, "per_animal_logistic_fits.csv")

# ═══════════════════════════════════════════════════════════════════════════
# Markdown report
# ═══════════════════════════════════════════════════════════════════════════

slopes_str = ", ".join(
    f"{s}: {animal_fits[s]['slope']:.3f}" for s in sorted(animal_fits)
)

report_path = proj.io.report(
    notes=(
        f"## Quiescent bout duration vs oscillation probability\n\n"
        f"**Dataset:** ETOH-HFSA (10 sessions) | **Task:** {TASK}\n\n"
        f"**ROI:** {ROI_NAME} | **Band:** {BAND[0]}–{BAND[1]} Hz\n\n"
        f"**Locomotion:** min_speed={MIN_SPEED_CMS} cm/s, "
        f"min_duration={MIN_LOCO_DURATION_S}s, merge_gap={MERGE_LOCO_GAP_S}s\n\n"
        f"**Oscillation:** threshold={THRESHOLD} (ΔF/F envelope), "
        f"min_duration={MIN_DURATION_OSC}s, merge_gap={MERGE_GAP_OSC}s\n\n"
        f"**Min quiescent bout:** {MIN_QUIESCENT_S}s | "
        f"**Duration truncation:** p{DURATION_TRUNCATE_PCTL} = {dur_cutoff:.1f}s\n\n"
        f"---\n\n"
        f"### GEE logistic regression\n\n"
        f"Model: `has_oscillation ~ log(bout_duration)`, "
        f"clustered by animal (exchangeable correlation, robust SE)\n\n"
        f"| Parameter | Estimate | Robust SE | 95% CI | p |\n"
        f"|-----------|----------|-----------|--------|---|\n"
        f"| Intercept | {gee_intercept:.4f} | {gee_result.bse['const']:.4f} | "
        f"[{gee_result.conf_int().loc['const', 0]:.4f}, "
        f"{gee_result.conf_int().loc['const', 1]:.4f}] | "
        f"{gee_result.pvalues['const']:.2e} |\n"
        f"| log(duration) | {gee_slope:.4f} | {gee_slope_se:.4f} | "
        f"[{gee_slope_ci[0]:.4f}, {gee_slope_ci[1]:.4f}] | "
        f"{gee_slope_p:.2e} |\n\n"
        f"**N animals:** {n_subjects} | "
        f"**N bouts (modelled):** {len(qdf_model)} / {len(qdf)} total\n\n"
        f"### Per-animal consistency\n\n"
        f"All per-animal slopes same sign: **{slopes_same_sign}**\n\n"
        f"Per-animal log(duration) slopes: {slopes_str}\n\n"
        f"### Descriptive correlations (naive — treats bouts as independent)\n\n"
        f"- Point-biserial r: {r_pb:.4f} (p = {p_pb:.2e}) — N bouts, not N animals\n"
        f"- Spearman ρ: {r_sp:.4f} (p = {p_sp:.2e}) — N bouts, not N animals\n\n"
        f"*These p-values are anticonservative due to pseudoreplication "
        f"(~{len(qdf_model)} bouts from {n_subjects} animals). "
        f"Use GEE results for inference.*\n\n"
        f"**Animals:** {', '.join(subjects)}\n"
    ),
)
print(f"\nReport: {report_path}")
print(f"Outputs saved to: {proj.io.run_dir}")
