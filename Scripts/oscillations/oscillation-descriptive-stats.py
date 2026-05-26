"""Oscillation event descriptive statistics — cross-animal comparison.

Detects oscillation bursts in one primary ROI, then re-runs detection across
a wider ROI set for spatial comparison.  All longitudinal-by-subject plots
go through ``plot_metric_by_session``; the per-session feature loop goes
through ``group.to_frame(...)``.

Usage
-----
    python Scripts/oscillations/oscillation-descriptive-stats.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import welch

from databench import Project, resolve_dataset
from databench.analysis.oscillation import OscillationDetector
from databench.plotting import plot_metric_by_session, set_theme

set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET     = resolve_dataset("etoh-hfsa")
TASK        = "task-widefield"
ROI_SOURCE  = "mesomap"
ROI_NAME    = "L_VISp"
ALL_ROIS    = ("L_VISp", "R_VISp", "L_SSp-ll", "R_SSp-ll")

FS          = 50.0
BAND        = (2.0, 4.0)
ORDER       = 4
THRESHOLD   = 0.02
THRESHOLD_K = 4.0
MIN_DUR_S   = 2.0
MERGE_GAP_S = 0.0


# ─── Helpers ─────────────────────────────────────────────────────────────

def burst_peak_frequency(filtered: np.ndarray, start: int, end: int, fs: float) -> float:
    """Peak frequency of a burst segment via Welch PSD."""
    segment = filtered[start : end + 1]
    if len(segment) < int(fs * 0.5):
        return np.nan
    nperseg = min(len(segment), int(fs * 2))
    freqs, psd = welch(segment, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return float(freqs[np.argmax(psd)])


def make_detector(roi_name: str) -> OscillationDetector:
    return OscillationDetector(
        source=ROI_SOURCE, signal=roi_name, fs=FS, band_hz=BAND,
        filter_order=ORDER, threshold=THRESHOLD, threshold_k=THRESHOLD_K,
        min_duration_s=MIN_DUR_S, merge_gap_s=MERGE_GAP_S,
    )


# ─── Project setup ───────────────────────────────────────────────────────

proj = Project(dataset=DATASET)
group = proj.sessions(task=TASK)
run = proj.run(name="oscillation-descriptive-stats", tag=f"{ROI_NAME}-{TASK}")
print(f"Selected {len(group)} sessions for task={TASK!r}; subjects={group.subjects}")


# ─── Per-session detection (primary ROI) ─────────────────────────────────

detector = make_detector(ROI_NAME)
event_frames: list[pd.DataFrame] = []

def session_summary(sess):
    try:
        result = detector.run(sess)
    except Exception as err:
        print(f"  SKIP {sess.subject}/{sess.session}: {err}")
        return None

    n_bursts        = len(result.bursts)
    burst_durations = result.events["duration_s"] if not result.events.empty else pd.Series(dtype=float)
    peak_envs       = result.events["peak_env"]   if not result.events.empty else pd.Series(dtype=float)
    total_burst_s   = float(burst_durations.sum())
    recording_s     = float(result.time[-1] - result.time[0]) if len(result.time) > 1 else 0.0

    peak_freqs     = [burst_peak_frequency(result.filtered_signal, s, e, FS) for s, e in result.bursts]
    mean_peak_freq = float(np.nanmean(peak_freqs)) if peak_freqs else np.nan

    if not result.events.empty:
        events = result.events.copy()
        events.insert(0, "Subject", sess.subject)
        events.insert(1, "Session", sess.session)
        events.insert(2, "Task",    sess.task)
        events["peak_freq_hz"] = peak_freqs
        events["day"]          = sess.day
        event_frames.append(events)

    print(
        f"  {sess.subject}/{sess.session}: {n_bursts} bursts, "
        f"{total_burst_s:.1f}s/{recording_s:.0f}s, peak {mean_peak_freq:.2f} Hz"
    )
    yield {
        "day":                    sess.day,
        "n_bursts":               n_bursts,
        "total_burst_duration_s": total_burst_s,
        "recording_duration_s":   recording_s,
        "burst_rate_per_min":     n_bursts / (recording_s / 60.0) if recording_s > 0 else 0.0,
        "fraction_in_burst":      total_burst_s / recording_s if recording_s > 0 else 0.0,
        "threshold":              float(result.threshold_value),
        "mean_peak_env":          float(peak_envs.mean()) if not peak_envs.empty else np.nan,
        "mean_peak_freq_hz":      mean_peak_freq,
        "mean_duration_s":        float(burst_durations.mean()) if not burst_durations.empty else np.nan,
    }

summary_df = group.to_frame(session_summary)
events_df  = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
subjects   = sorted(summary_df["Subject"].unique())

palette        = plt.cm.tab10(np.linspace(0, 1, max(len(subjects), 1)))
subject_colors = dict(zip(subjects, palette))

print(f"\nTotal events: {len(events_df)} across {subjects}")


# ─── Spatial detection (all ROIs) ────────────────────────────────────────

print("\n── Spatial distribution: detection across all ROIs ──")

def per_roi_rows(sess):
    recording_s = 0.0
    rows: list[dict] = []
    for roi_name in ALL_ROIS:
        try:
            result = make_detector(roi_name).run(sess)
        except Exception:
            continue
        if recording_s == 0.0 and len(result.time) > 1:
            recording_s = float(result.time[-1] - result.time[0])
        n_bursts      = len(result.bursts)
        total_burst_s = float(result.events["duration_s"].sum()) if not result.events.empty else 0.0
        mean_peak_env = float(result.events["peak_env"].mean()) if not result.events.empty else np.nan
        rows.append({
            "ROI":                    roi_name,
            "day":                    sess.day,
            "n_bursts":               n_bursts,
            "burst_rate_per_min":     n_bursts / (recording_s / 60.0) if recording_s > 0 else 0.0,
            "total_burst_duration_s": total_burst_s,
            "mean_peak_env":          mean_peak_env,
        })
    print(f"  {sess.subject}/{sess.session}: done")
    return rows

spatial_df = group.to_frame(per_roi_rows)


# ─── Longitudinal-by-subject plots (one call per metric) ─────────────────

LONGITUDINAL_PANELS = [
    ("n_bursts",           "Number of oscillation events", "Event count",                  "event_count_per_session.svg"),
    ("burst_rate_per_min", "Burst rate (events / min)",    "Burst rate",                   "burst_rate_per_session.svg"),
    ("fraction_in_burst",  "Time in oscillation bursts",   "Fraction in burst",            "fraction_in_burst_per_session.svg"),
    ("mean_peak_env",      "Mean peak envelope (ΔF/F)",    "Longitudinal burst power",     "burst_power_longitudinal.svg"),
    ("mean_duration_s",    "Mean burst duration (s)",      "Longitudinal burst duration",  "burst_duration_longitudinal.svg"),
    ("mean_peak_freq_hz",  "Mean peak frequency (Hz)",     "Longitudinal peak frequency",  "peak_frequency_longitudinal.svg"),
]
for metric, ylabel, title, fname in LONGITUDINAL_PANELS:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_metric_by_session(summary_df, x="day", y=metric, ax=ax, subject_colors=subject_colors)
    if metric == "mean_peak_freq_hz":
        ax.axhspan(BAND[0], BAND[1], color="gray", alpha=0.1, label=f"bandpass {BAND[0]}–{BAND[1]} Hz")
    ax.set_xlabel("Session (day)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK}")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    run.save_figure(fig, fname)


# ─── Per-animal distributions (histogram + KDE) ──────────────────────────

def hist_kde(metric: str, xlabel: str, fname: str) -> None:
    if events_df.empty:
        return
    upper = events_df[metric].quantile(0.99)
    bins  = np.linspace(0, upper, 30)
    xs    = np.linspace(0, upper, 200)

    fig, (ax_hist, ax_kde) = plt.subplots(1, 2, figsize=(12, 4.5))
    for subj in subjects:
        vals = events_df.loc[events_df["Subject"] == subj, metric]
        ax_hist.hist(vals, bins=bins, alpha=0.4, density=True, edgecolor="none",
                     color=subject_colors[subj], label=f"{subj} (n={len(vals)})")
        if len(vals) > 2:
            kde = sp_stats.gaussian_kde(vals, bw_method=0.3)
            ax_kde.plot(xs, kde(xs), lw=1.8, color=subject_colors[subj], label=subj)
    for ax, kind in ((ax_hist, "histogram"), (ax_kde, "KDE")):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"{xlabel} — {kind}")
        ax.legend(frameon=False)
    fig.suptitle(f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK}", y=1.02)
    fig.tight_layout()
    run.save_figure(fig, fname)

hist_kde("duration_s", "Duration (s)",            "duration_distribution.svg")
hist_kde("peak_env",   "Peak envelope amplitude", "peak_envelope_distribution.svg")


# ─── Per-animal box + scatter ────────────────────────────────────────────

def boxplot_with_jitter(metric: str, ylabel: str, ax: plt.Axes) -> None:
    data = [events_df.loc[events_df["Subject"] == s, metric].values for s in subjects]
    box  = ax.boxplot(
        data, labels=subjects, patch_artist=True, showfliers=False,
        medianprops=dict(color="black", lw=1.5),
    )
    for patch, subj in zip(box["boxes"], subjects):
        patch.set_facecolor(subject_colors[subj])
        patch.set_alpha(0.5)
    rng = np.random.default_rng(42)
    for i, subj in enumerate(subjects, start=1):
        vals   = events_df.loc[events_df["Subject"] == subj, metric].values
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals,
                   s=8, alpha=0.3, color=subject_colors[subj], zorder=3)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

if not events_df.empty:
    fig, (ax_dur, ax_pe) = plt.subplots(1, 2, figsize=(12, 5))
    boxplot_with_jitter("duration_s", "Duration (s)",            ax_dur)
    boxplot_with_jitter("peak_env",   "Peak envelope amplitude", ax_pe)
    ax_dur.set_title("Event duration by animal")
    ax_pe.set_title("Peak envelope by animal")
    fig.suptitle(f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK}", y=1.01)
    fig.tight_layout()
    run.save_figure(fig, "event_characteristics_boxplots.svg")


# ─── Spatial: by-ROI longitudinal + per-animal heatmaps ──────────────────

if not spatial_df.empty:
    roi_palette = plt.cm.Set2(np.linspace(0, 1, len(ALL_ROIS)))
    roi_colors  = dict(zip(ALL_ROIS, roi_palette))

    SPATIAL_PANELS = [
        ("burst_rate_per_min", "Burst rate (events / min)", "Spatial distribution of oscillation events", "spatial_burst_rate_longitudinal.svg"),
        ("mean_peak_env",      "Mean peak envelope (ΔF/F)", "Spatial distribution of burst power",        "spatial_burst_power_longitudinal.svg"),
    ]
    for metric, ylabel, title, fname in SPATIAL_PANELS:
        fig, ax = plt.subplots(figsize=(9, 5))
        for roi_name in ALL_ROIS:
            roi_rows = spatial_df[spatial_df["ROI"] == roi_name]
            grouped  = roi_rows.groupby("day")[metric].agg(["mean", "sem"]).reset_index()
            ax.errorbar(grouped["day"], grouped["mean"], yerr=grouped["sem"],
                        fmt="o-", color=roi_colors[roi_name], ms=5, lw=1.5,
                        label=roi_name, capsize=3)
        ax.set_xlabel("Session (day)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{BAND[0]}–{BAND[1]} Hz | {TASK} | group mean ± SEM")
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        run.save_figure(fig, fname)

    for subj in subjects:
        subj_rows = spatial_df[spatial_df["Subject"] == subj]
        pivot = subj_rows.pivot_table(
            index="ROI", columns="day", values="burst_rate_per_min", aggfunc="mean",
        ).reindex([r for r in ALL_ROIS if r in subj_rows["ROI"].unique()])
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 3.5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([f"day {int(c)}" for c in pivot.columns], rotation=45)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Session")
        ax.set_title(f"{subj} — burst rate by ROI across sessions")
        fig.colorbar(im, ax=ax, label="events / min", fraction=0.03, pad=0.04)
        fig.tight_layout()
        run.save_figure(fig, f"spatial_heatmap_{subj}.svg")


# ─── Tables ──────────────────────────────────────────────────────────────

descriptive_stats = (
    events_df.groupby("Subject").agg(
        n_events       =("duration_s", "count"),
        duration_mean  =("duration_s", "mean"),
        duration_median=("duration_s", "median"),
        duration_std   =("duration_s", "std"),
        duration_min   =("duration_s", "min"),
        duration_max   =("duration_s", "max"),
        peak_env_mean  =("peak_env",   "mean"),
        peak_env_median=("peak_env",   "median"),
        peak_env_std   =("peak_env",   "std"),
    ).reset_index()
    if not events_df.empty else pd.DataFrame()
)

run.save_table(events_df,  "oscillation_events_all.csv")
run.save_table(summary_df, "session_summary.csv")
if not descriptive_stats.empty:
    run.save_table(descriptive_stats, "descriptive_stats_by_animal.csv")
if not spatial_df.empty:
    run.save_table(spatial_df, "spatial_distribution.csv")


# ─── Report ──────────────────────────────────────────────────────────────

report_path = run.finish(
    notes=(
        f"## Oscillation descriptive statistics\n\n"
        f"**Dataset:** ETOH-HFSA | **Task:** {TASK}\n\n"
        f"**Primary ROI:** {ROI_NAME} | **Band:** {BAND[0]}–{BAND[1]} Hz\n\n"
        f"**Detector:** threshold={THRESHOLD}, threshold_k={THRESHOLD_K}, "
        f"min_duration={MIN_DUR_S}s, merge_gap={MERGE_GAP_S}s\n\n"
        f"**Animals:** {', '.join(subjects)}\n\n"
        f"**Total events:** {len(events_df)}\n\n"
        f"**Spatial ROIs:** {', '.join(ALL_ROIS)}\n"
    ),
)
print(f"Report: {report_path}")
print(f"Outputs: {run.dir}")
