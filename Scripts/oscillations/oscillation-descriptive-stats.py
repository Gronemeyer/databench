"""
Oscillation event descriptive statistics — cross-animal comparison.

Runs oscillation detection across all sessions in the 10-session ETOH-HFSA
dataset (task-widefield) and compares event characteristics between animals.

Outputs:
    1. Oscillation event counts per session, per animal (line plot)
    2. Duration distributions per animal (histogram + KDE overlay)
    3. Peak envelope distributions per animal (histogram + KDE overlay)
    4. Box plots of event characteristics by animal
    5. Longitudinal burst power (mean peak envelope) across sessions
    6. Longitudinal peak frequency across sessions
    7. Spatial distribution of events across ROIs over sessions
    8. Summary statistics table (CSV)
    9. Combined events table (CSV)
   10. Markdown report

Usage:
    python Scripts/scriptings/oscillation-descriptive-stats.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import welch

from databench import Project, OscillationDetector
from databench.analysis._signal.bandpass import bandpass_envelope
from databench.config import resolve_dataset
from databench.session import SaveableFigure
from databench.plotting import set_theme

set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET = resolve_dataset("etoh-hfsa")
TASK = "task-widefield"
ROI_SOURCE = "mesomap"
ROI_NAME = "L_VISp"

# Additional ROIs for spatial distribution analysis
ALL_ROIS = ("L_VISp", "R_VISp", "L_SSp-ll", "R_SSp-ll")

FS = 50.0
BAND = (2.0, 4.0)
ORDER = 4
THRESHOLD = 0.02          # adaptive (median + k * robust_std)
THRESHOLD_K = 4.0
MIN_DURATION = 2.0
MERGE_GAP = 0.0

# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name="oscillation-descriptive-stats",
    tag=f"{ROI_NAME}-{TASK}",
)

group = proj.sessions(task=TASK)
print(f"Selected {len(group)} sessions for task={TASK!r}")
print(f"  Subjects: {group.subjects}")
print(f"  Sessions: {group.session_labels}")

# ─── Configure detector ──────────────────────────────────────────────────

detector = OscillationDetector(
    source=ROI_SOURCE,
    signal=ROI_NAME,
    fs=FS,
    band_hz=BAND,
    filter_order=ORDER,
    threshold=THRESHOLD,
    threshold_k=THRESHOLD_K,
    min_duration_s=MIN_DURATION,
    merge_gap_s=MERGE_GAP,
)

# ─── Helper: peak frequency within a burst ──────────────────────────────

def burst_peak_frequency(filtered: np.ndarray, s: int, e: int, fs: float) -> float:
    """Estimate peak oscillation frequency within a burst via Welch PSD."""
    segment = filtered[s : e + 1]
    if len(segment) < int(fs * 0.5):  # need at least 0.5s for reasonable estimate
        return np.nan
    nperseg = min(len(segment), int(fs * 2))
    freqs, psd = welch(segment, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return float(freqs[np.argmax(psd)])


# ─── Run detection across all sessions (primary ROI) ─────────────────────

all_events: list[pd.DataFrame] = []
session_summaries: list[dict] = []

for sess in group:
    try:
        result = detector.run(sess)
    except Exception as e:
        print(f"  SKIP {sess.subject}/{sess.session}: {e}")
        continue

    n_bursts = len(result.bursts)
    total_dur = float(result.events["duration_s"].sum()) if not result.events.empty else 0.0
    rec_dur = float(result.time[-1] - result.time[0]) if len(result.time) > 1 else 0.0

    # Compute per-burst peak frequency
    peak_freqs = [
        burst_peak_frequency(result.filtered_signal, s, e, FS)
        for s, e in result.bursts
    ]
    mean_peak_freq = float(np.nanmean(peak_freqs)) if peak_freqs else np.nan
    mean_peak_env = float(result.events["peak_env"].mean()) if not result.events.empty else np.nan
    mean_duration = float(result.events["duration_s"].mean()) if not result.events.empty else np.nan

    print(
        f"  {sess.subject} / {sess.session}: "
        f"{n_bursts} bursts, total {total_dur:.1f}s / {rec_dur:.0f}s recording, "
        f"mean freq {mean_peak_freq:.2f} Hz"
    )

    # Tag events with session info + per-burst frequency
    if not result.events.empty:
        ev = result.events.copy()
        ev.insert(0, "Subject", sess.subject)
        ev.insert(1, "Session", sess.session)
        ev.insert(2, "Task", sess.task)
        ev["peak_freq_hz"] = peak_freqs
        all_events.append(ev)

    session_summaries.append({
        "Subject": sess.subject,
        "Session": sess.session,
        "Task": sess.task,
        "n_bursts": n_bursts,
        "total_burst_duration_s": round(total_dur, 3),
        "recording_duration_s": round(rec_dur, 1),
        "burst_rate_per_min": round(n_bursts / (rec_dur / 60), 3) if rec_dur > 0 else 0,
        "fraction_in_burst": round(total_dur / rec_dur, 4) if rec_dur > 0 else 0,
        "threshold": round(result.threshold_value, 6),
        "mean_peak_env": round(mean_peak_env, 6),
        "mean_peak_freq_hz": round(mean_peak_freq, 4) if not np.isnan(mean_peak_freq) else np.nan,
        "mean_duration_s": round(mean_duration, 4) if not np.isnan(mean_duration) else np.nan,
    })

# ─── Combine events ──────────────────────────────────────────────────────

events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
summary_df = pd.DataFrame(session_summaries)

# Extract session number for ordering
summary_df["session_n"] = (
    summary_df["Session"].str.extract(r"(\d+)", expand=False).astype(float)
)
if not events_df.empty:
    events_df["session_n"] = (
        events_df["Session"].str.extract(r"(\d+)", expand=False).astype(float)
    )

subjects = sorted(summary_df["Subject"].unique())
n_subjects = len(subjects)
COLORS = plt.cm.tab10(np.linspace(0, 1, max(n_subjects, 1)))
subject_colors = {subj: COLORS[i] for i, subj in enumerate(subjects)}

print(f"\nTotal events across all sessions: {len(events_df)}")
print(f"Animals: {subjects}")

# ─── Run detection across ALL ROIs for spatial distribution ──────────────

print("\n── Spatial distribution: running detection across all ROIs ──")
spatial_rows: list[dict] = []

for sess in group:
    rec_dur = 0.0
    for roi in ALL_ROIS:
        roi_detector = OscillationDetector(
            source=ROI_SOURCE,
            signal=roi,
            fs=FS,
            band_hz=BAND,
            filter_order=ORDER,
            threshold=THRESHOLD,
            threshold_k=THRESHOLD_K,
            min_duration_s=MIN_DURATION,
            merge_gap_s=MERGE_GAP,
        )
        try:
            roi_result = roi_detector.run(sess)
        except Exception:
            continue
        n = len(roi_result.bursts)
        if rec_dur == 0.0 and len(roi_result.time) > 1:
            rec_dur = float(roi_result.time[-1] - roi_result.time[0])
        total_dur = float(roi_result.events["duration_s"].sum()) if not roi_result.events.empty else 0.0
        mean_env = float(roi_result.events["peak_env"].mean()) if not roi_result.events.empty else np.nan
        spatial_rows.append({
            "Subject": sess.subject,
            "Session": sess.session,
            "ROI": roi,
            "n_bursts": n,
            "burst_rate_per_min": round(n / (rec_dur / 60), 3) if rec_dur > 0 else 0,
            "total_burst_duration_s": round(total_dur, 3),
            "mean_peak_env": round(mean_env, 6) if not np.isnan(mean_env) else np.nan,
        })
    print(f"  {sess.subject} / {sess.session}: done")

spatial_df = pd.DataFrame(spatial_rows)
spatial_df["session_n"] = spatial_df["Session"].str.extract(r"(\d+)", expand=False).astype(float)

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1: Event count per session, per animal (longitudinal line plot)
# ═══════════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(8, 4.5))

for subj in subjects:
    sub = summary_df[summary_df["Subject"] == subj].sort_values("session_n")
    ax1.plot(
        sub["session_n"], sub["n_bursts"],
        "o-", color=subject_colors[subj], ms=5, lw=1.5, alpha=0.7, label=subj,
    )

# Group mean ± SEM
grp = summary_df.groupby("session_n")["n_bursts"].agg(["mean", "sem"]).reset_index()
ax1.errorbar(
    grp["session_n"], grp["mean"], yerr=grp["sem"],
    fmt="s-", color="black", ms=6, lw=2, zorder=5, label="group mean",
)

ax1.set_xlabel("Session")
ax1.set_ylabel("Number of oscillation events")
ax1.set_title(
    f"Oscillation event count across sessions\n"
    f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK}",
)
ax1.legend(loc="best", frameon=False)
ax1.grid(alpha=0.3)
fig1.tight_layout()
SaveableFigure(fig1, proj._context).save("event_count_per_session.svg")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2: Burst rate per minute (normalised by recording length)
# ═══════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(8, 4.5))

for subj in subjects:
    sub = summary_df[summary_df["Subject"] == subj].sort_values("session_n")
    ax2.plot(
        sub["session_n"], sub["burst_rate_per_min"],
        "o-", color=subject_colors[subj], ms=5, lw=1.5, alpha=0.7, label=subj,
    )

grp_rate = summary_df.groupby("session_n")["burst_rate_per_min"].agg(["mean", "sem"]).reset_index()
ax2.errorbar(
    grp_rate["session_n"], grp_rate["mean"], yerr=grp_rate["sem"],
    fmt="s-", color="black", ms=6, lw=2, zorder=5, label="group mean",
)

ax2.set_xlabel("Session")
ax2.set_ylabel("Burst rate (events / min)")
ax2.set_title(
    f"Oscillation burst rate across sessions\n"
    f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK}",
)
ax2.legend(loc="best", frameon=False)
ax2.grid(alpha=0.3)
fig2.tight_layout()
SaveableFigure(fig2, proj._context).save("burst_rate_per_session.svg")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3: Duration distribution — overlaid histograms per animal
# ═══════════════════════════════════════════════════════════════════════════

if not events_df.empty:
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))

    # 3a: Overlaid histograms
    ax = axes3[0]
    bins = np.linspace(0, events_df["duration_s"].quantile(0.99), 30)
    for subj in subjects:
        dur = events_df.loc[events_df["Subject"] == subj, "duration_s"]
        ax.hist(
            dur, bins=bins, alpha=0.4, color=subject_colors[subj],
            label=f"{subj} (n={len(dur)})", density=True, edgecolor="none",
        )
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Density")
    ax.set_title("Event duration distribution by animal")
    ax.legend(frameon=False)

    # 3b: KDE overlay
    ax = axes3[1]
    dur_range = np.linspace(0, events_df["duration_s"].quantile(0.99), 200)
    for subj in subjects:
        dur = events_df.loc[events_df["Subject"] == subj, "duration_s"]
        if len(dur) > 2:
            kde = sp_stats.gaussian_kde(dur, bw_method=0.3)
            ax.plot(dur_range, kde(dur_range), lw=1.8, color=subject_colors[subj], label=subj)
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Density")
    ax.set_title("Event duration KDE by animal")
    ax.legend(frameon=False)

    fig3.suptitle(
        f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK}",
        y=1.02,
    )
    fig3.tight_layout()
    SaveableFigure(fig3, proj._context).save("duration_distribution.svg")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4: Peak envelope distribution — overlaid histograms per animal
# ═══════════════════════════════════════════════════════════════════════════

if not events_df.empty:
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4.5))

    # 4a: Overlaid histograms
    ax = axes4[0]
    bins = np.linspace(0, events_df["peak_env"].quantile(0.99), 30)
    for subj in subjects:
        pe = events_df.loc[events_df["Subject"] == subj, "peak_env"]
        ax.hist(
            pe, bins=bins, alpha=0.4, color=subject_colors[subj],
            label=f"{subj} (n={len(pe)})", density=True, edgecolor="none",
        )
    ax.set_xlabel("Peak envelope amplitude")
    ax.set_ylabel("Density")
    ax.set_title("Peak envelope distribution by animal")
    ax.legend(frameon=False)

    # 4b: KDE overlay
    ax = axes4[1]
    pe_range = np.linspace(0, events_df["peak_env"].quantile(0.99), 200)
    for subj in subjects:
        pe = events_df.loc[events_df["Subject"] == subj, "peak_env"]
        if len(pe) > 2:
            kde = sp_stats.gaussian_kde(pe, bw_method=0.3)
            ax.plot(pe_range, kde(pe_range), lw=1.8, color=subject_colors[subj], label=subj)
    ax.set_xlabel("Peak envelope amplitude")
    ax.set_ylabel("Density")
    ax.set_title("Peak envelope KDE by animal")
    ax.legend(frameon=False)

    fig4.suptitle(
        f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK}",
        y=1.02,
    )
    fig4.tight_layout()
    SaveableFigure(fig4, proj._context).save("peak_envelope_distribution.svg")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5: Box plots — duration and peak envelope by animal
# ═══════════════════════════════════════════════════════════════════════════

if not events_df.empty:
    fig5, axes5 = plt.subplots(1, 2, figsize=(12, 5))

    # 5a: Duration box plot
    ax = axes5[0]
    dur_data = [events_df.loc[events_df["Subject"] == s, "duration_s"].values for s in subjects]
    bp = ax.boxplot(
        dur_data, labels=subjects, patch_artist=True, showfliers=False,
        medianprops=dict(color="black", lw=1.5),
    )
    for patch, subj in zip(bp["boxes"], subjects):
        patch.set_facecolor(subject_colors[subj])
        patch.set_alpha(0.5)
    # Overlay individual points (jittered)
    for i, subj in enumerate(subjects):
        vals = events_df.loc[events_df["Subject"] == subj, "duration_s"].values
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.full_like(vals, i + 1) + jitter, vals,
            s=8, alpha=0.3, color=subject_colors[subj], zorder=3,
        )
    ax.set_ylabel("Duration (s)")
    ax.set_title("Event duration by animal")
    ax.grid(axis="y", alpha=0.3)

    # 5b: Peak envelope box plot
    ax = axes5[1]
    pe_data = [events_df.loc[events_df["Subject"] == s, "peak_env"].values for s in subjects]
    bp = ax.boxplot(
        pe_data, labels=subjects, patch_artist=True, showfliers=False,
        medianprops=dict(color="black", lw=1.5),
    )
    for patch, subj in zip(bp["boxes"], subjects):
        patch.set_facecolor(subject_colors[subj])
        patch.set_alpha(0.5)
    for i, subj in enumerate(subjects):
        vals = events_df.loc[events_df["Subject"] == subj, "peak_env"].values
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.full_like(vals, i + 1) + jitter, vals,
            s=8, alpha=0.3, color=subject_colors[subj], zorder=3,
        )
    ax.set_ylabel("Peak envelope amplitude")
    ax.set_title("Peak envelope by animal")
    ax.grid(axis="y", alpha=0.3)

    fig5.suptitle(
        f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK}",
        y=1.01,
    )
    fig5.tight_layout()
    SaveableFigure(fig5, proj._context).save("event_characteristics_boxplots.svg")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 6: Fraction of recording spent in oscillation bursts
# ═══════════════════════════════════════════════════════════════════════════

fig6, ax6 = plt.subplots(figsize=(8, 4.5))

for subj in subjects:
    sub = summary_df[summary_df["Subject"] == subj].sort_values("session_n")
    ax6.plot(
        sub["session_n"], sub["fraction_in_burst"] * 100,
        "o-", color=subject_colors[subj], ms=5, lw=1.5, alpha=0.7, label=subj,
    )

grp_frac = summary_df.groupby("session_n")["fraction_in_burst"].agg(["mean", "sem"]).reset_index()
ax6.errorbar(
    grp_frac["session_n"], grp_frac["mean"] * 100, yerr=grp_frac["sem"] * 100,
    fmt="s-", color="black", ms=6, lw=2, zorder=5, label="group mean",
)

ax6.set_xlabel("Session")
ax6.set_ylabel("Time in oscillation bursts (%)")
ax6.set_title(
    f"Fraction of recording in oscillation bursts\n"
    f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK}",
)
ax6.legend(loc="best", frameon=False)
ax6.grid(alpha=0.3)
fig6.tight_layout()
SaveableFigure(fig6, proj._context).save("fraction_in_burst_per_session.svg")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 7: Longitudinal burst power (mean peak envelope) across sessions
# ═══════════════════════════════════════════════════════════════════════════

fig7, ax7 = plt.subplots(figsize=(8, 4.5))

for subj in subjects:
    sub = summary_df[summary_df["Subject"] == subj].sort_values("session_n")
    ax7.plot(
        sub["session_n"], sub["mean_peak_env"],
        "o-", color=subject_colors[subj], ms=5, lw=1.5, alpha=0.7, label=subj,
    )

grp_pow = summary_df.groupby("session_n")["mean_peak_env"].agg(["mean", "sem"]).reset_index()
ax7.errorbar(
    grp_pow["session_n"], grp_pow["mean"], yerr=grp_pow["sem"],
    fmt="s-", color="black", ms=6, lw=2, zorder=5, label="group mean",
)

ax7.set_xlabel("Session")
ax7.set_ylabel("Mean peak envelope (ΔF/F)")
ax7.set_title(
    f"Longitudinal burst power\n"
    f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK}",
)
ax7.legend(loc="best", frameon=False)
ax7.grid(alpha=0.3)
fig7.tight_layout()
SaveableFigure(fig7, proj._context).save("burst_power_longitudinal.svg")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 7b: Longitudinal mean burst duration across sessions
# ═══════════════════════════════════════════════════════════════════════════

fig7b, ax7b = plt.subplots(figsize=(8, 4.5))

for subj in subjects:
    sub = summary_df[summary_df["Subject"] == subj].sort_values("session_n")
    ax7b.plot(
        sub["session_n"], sub["mean_duration_s"],
        "o-", color=subject_colors[subj], ms=5, lw=1.5, alpha=0.7, label=subj,
    )

grp_dur = summary_df.groupby("session_n")["mean_duration_s"].agg(["mean", "sem"]).reset_index()
ax7b.errorbar(
    grp_dur["session_n"], grp_dur["mean"], yerr=grp_dur["sem"],
    fmt="s-", color="black", ms=6, lw=2, zorder=5, label="group mean",
)

ax7b.set_xlabel("Session")
ax7b.set_ylabel("Mean burst duration (s)")
ax7b.set_title(
    f"Longitudinal mean burst duration\n"
    f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK}",
)
ax7b.legend(loc="best", frameon=False)
ax7b.grid(alpha=0.3)
fig7b.tight_layout()
SaveableFigure(fig7b, proj._context).save("burst_duration_longitudinal.svg")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 8: Longitudinal peak frequency across sessions
# ═══════════════════════════════════════════════════════════════════════════

fig8, ax8 = plt.subplots(figsize=(8, 4.5))

for subj in subjects:
    sub = summary_df[summary_df["Subject"] == subj].sort_values("session_n")
    ax8.plot(
        sub["session_n"], sub["mean_peak_freq_hz"],
        "o-", color=subject_colors[subj], ms=5, lw=1.5, alpha=0.7, label=subj,
    )

grp_freq = summary_df.groupby("session_n")["mean_peak_freq_hz"].agg(["mean", "sem"]).reset_index()
ax8.errorbar(
    grp_freq["session_n"], grp_freq["mean"], yerr=grp_freq["sem"],
    fmt="s-", color="black", ms=6, lw=2, zorder=5, label="group mean",
)

ax8.axhspan(BAND[0], BAND[1], color="gray", alpha=0.1, label=f"bandpass {BAND[0]}–{BAND[1]} Hz")
ax8.set_xlabel("Session")
ax8.set_ylabel("Mean peak frequency (Hz)")
ax8.set_title(
    f"Longitudinal peak oscillation frequency\n"
    f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK}",
)
ax8.legend(loc="best", frameon=False)
ax8.grid(alpha=0.3)
fig8.tight_layout()
SaveableFigure(fig8, proj._context).save("peak_frequency_longitudinal.svg")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 9: Spatial distribution — event count by ROI across sessions
# ═══════════════════════════════════════════════════════════════════════════

if not spatial_df.empty:
    roi_colors = dict(zip(ALL_ROIS, plt.cm.Set2(np.linspace(0, 1, len(ALL_ROIS)))))

    # 9a: Mean burst rate across sessions, grouped by ROI (all animals pooled)
    fig9a, ax9a = plt.subplots(figsize=(9, 5))
    for roi in ALL_ROIS:
        sub = spatial_df[spatial_df["ROI"] == roi].copy()
        grp = sub.groupby("session_n")["burst_rate_per_min"].agg(["mean", "sem"]).reset_index()
        ax9a.errorbar(
            grp["session_n"], grp["mean"], yerr=grp["sem"],
            fmt="o-", color=roi_colors[roi], ms=5, lw=1.5, label=roi, capsize=3,
        )
    ax9a.set_xlabel("Session")
    ax9a.set_ylabel("Burst rate (events / min)")
    ax9a.set_title(
        f"Spatial distribution of oscillation events across sessions\n"
        f"{BAND[0]}–{BAND[1]} Hz | {TASK} | group mean ± SEM",
    )
    ax9a.legend(frameon=False, ncol=2)
    ax9a.grid(alpha=0.3)
    fig9a.tight_layout()
    SaveableFigure(fig9a, proj._context).save("spatial_burst_rate_longitudinal.svg")

    # 9b: Mean burst power by ROI across sessions
    fig9b, ax9b = plt.subplots(figsize=(9, 5))
    for roi in ALL_ROIS:
        sub = spatial_df[spatial_df["ROI"] == roi].copy()
        grp = sub.groupby("session_n")["mean_peak_env"].agg(["mean", "sem"]).reset_index()
        ax9b.errorbar(
            grp["session_n"], grp["mean"], yerr=grp["sem"],
            fmt="o-", color=roi_colors[roi], ms=5, lw=1.5, label=roi, capsize=3,
        )
    ax9b.set_xlabel("Session")
    ax9b.set_ylabel("Mean peak envelope (ΔF/F)")
    ax9b.set_title(
        f"Spatial distribution of burst power across sessions\n"
        f"{BAND[0]}–{BAND[1]} Hz | {TASK} | group mean ± SEM",
    )
    ax9b.legend(frameon=False, ncol=2)
    ax9b.grid(alpha=0.3)
    fig9b.tight_layout()
    SaveableFigure(fig9b, proj._context).save("spatial_burst_power_longitudinal.svg")

    # 9c: Per-animal spatial heatmap (ROI × Session, values = burst rate)
    for subj in subjects:
        sub = spatial_df[spatial_df["Subject"] == subj]
        pivot = sub.pivot_table(
            index="ROI", columns="session_n", values="burst_rate_per_min", aggfunc="mean",
        )
        pivot = pivot.reindex([r for r in ALL_ROIS if r in pivot.index])
        if pivot.empty:
            continue
        fig_h, ax_h = plt.subplots(figsize=(10, 3.5))
        im = ax_h.imshow(
            pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest",
        )
        ax_h.set_xticks(range(pivot.shape[1]))
        ax_h.set_xticklabels([f"ses-{int(c):02d}" for c in pivot.columns], rotation=45)
        ax_h.set_yticks(range(pivot.shape[0]))
        ax_h.set_yticklabels(pivot.index)
        ax_h.set_xlabel("Session")
        ax_h.set_title(f"{subj} — burst rate by ROI across sessions")
        fig_h.colorbar(im, ax=ax_h, label="events / min", fraction=0.03, pad=0.04)
        fig_h.tight_layout()
        SaveableFigure(fig_h, proj._context).save(f"spatial_heatmap_{subj}.svg")

# ═══════════════════════════════════════════════════════════════════════════
# Summary statistics table
# ═══════════════════════════════════════════════════════════════════════════

if not events_df.empty:
    desc_stats = (
        events_df.groupby("Subject")
        .agg(
            n_events=("duration_s", "count"),
            duration_mean=("duration_s", "mean"),
            duration_median=("duration_s", "median"),
            duration_std=("duration_s", "std"),
            duration_min=("duration_s", "min"),
            duration_max=("duration_s", "max"),
            peak_env_mean=("peak_env", "mean"),
            peak_env_median=("peak_env", "median"),
            peak_env_std=("peak_env", "std"),
        )
        .round(4)
        .reset_index()
    )
    print("\n── Descriptive Statistics by Animal ──")
    print(desc_stats.to_string(index=False))
else:
    desc_stats = pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
# Save tables
# ═══════════════════════════════════════════════════════════════════════════

stats_dir = proj._context.stats_dir
stats_dir.mkdir(parents=True, exist_ok=True)

events_df.to_csv(stats_dir / "oscillation_events_all.csv", index=False)
summary_df.to_csv(stats_dir / "session_summary.csv", index=False)
if not desc_stats.empty:
    desc_stats.to_csv(stats_dir / "descriptive_stats_by_animal.csv", index=False)
if not spatial_df.empty:
    spatial_df.to_csv(stats_dir / "spatial_distribution.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# Markdown report
# ═══════════════════════════════════════════════════════════════════════════

report_path = proj.save_report(
    notes=(
        f"## Oscillation descriptive statistics\n\n"
        f"**Dataset:** ETOH-HFSA (10 sessions) | **Task:** {TASK}\n\n"
        f"**Primary ROI:** {ROI_NAME} | **Band:** {BAND[0]}–{BAND[1]} Hz\n\n"
        f"**Detector settings:** threshold={THRESHOLD}, "
        f"threshold_k={THRESHOLD_K}, min_duration={MIN_DURATION}s, "
        f"merge_gap={MERGE_GAP}s\n\n"
        f"**Animals:** {', '.join(subjects)}\n\n"
        f"**Total events detected:** {len(events_df)}\n\n"
        f"### Analyses\n\n"
        f"1. Event counts and burst rates across sessions\n"
        f"2. Duration and peak envelope distributions by animal\n"
        f"3. Longitudinal burst power (mean peak envelope)\n"
        f"3b. Longitudinal mean burst duration\n"
        f"4. Longitudinal peak oscillation frequency\n"
        f"5. Spatial distribution across {len(ALL_ROIS)} ROIs ({', '.join(ALL_ROIS)})\n"
        f"6. Per-animal spatial heatmaps (ROI × session)\n"
    ),
)
print(f"Report: {report_path}")
print(f"\nOutputs saved to: {proj._context.run_dir}")
