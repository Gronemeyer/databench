
#!/usr/bin/env python3
"""
Hysteresis-based event detection for arbitrary signals.

Detects transient events using Savitzky–Golay smoothing, hysteresis
thresholding, gap-filling, and minimum-duration filtering.

Supports multiple signal sources. Comment/uncomment entries in the
SIGNALS list below to choose which signals to analyze.

Two stages per signal:
  1. Single-session test: first session, diagnostic plot.
  2. Group analysis: all sessions, PDF report + summary stats.

Usage:
    DATABENCH_DATASET=hfsa python event_analysis.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from databench.project import Project
from databench.signal.epoching import make_events
from databench.config import resolve_dataset
from databench.plotting import set_theme
set_theme()
# ─── Dataset ──────────────────────────────────────────────────────────────

DATASET = resolve_dataset("hfsa")

# ─── Signal definitions ──────────────────────────────────────────────────
# Comment/uncomment entries to select which signals to analyze.


@dataclass
class SignalSpec:
    """Configuration for one signal to run event detection on."""
    source: str             # data source name in dataset
    signal: str             # signal column name within the source
    label: str              # human-readable label for plots / filenames
    ylabel: str             # y-axis label
    use_dff: bool           # True → compute ΔF/F; False → use raw values
    color_event: str        # event shading color
    color_mask: str         # mask fill color
    palette: tuple[str, ...]  # bar chart palette


SIGNALS: list[SignalSpec] = [
    # ── Mesofield mean ──
    SignalSpec(
        source="meso_mean",
        signal="Mean",
        label="mesofield",
        ylabel="ΔF/F",
        # meso_mean/Mean in current HFSA datasets is already ΔF/F.
        # Do not apply a second ΔF/F normalization.
        use_dff=False,
        color_event="#B3D9FF",
        color_mask="#457B9D",
        palette=("#457B9D", "#1D3557", "#A8DADC", "#2A9D8F",
                 "#264653", "#E9C46A", "#F4A261", "#E76F51"),
    ),
    # ── Pupil diameter ──
    SignalSpec(
        source="pupil",
        signal="pupil_diameter_mm",
        label="pupil",
        ylabel="Pupil diameter (mm)",
        use_dff=False,
        color_event="#E8D5F5",
        color_mask="#7B2D8E",
        palette=("#7B2D8E", "#4A0E5C", "#C39BD3", "#884EA0",
                 "#6C3483", "#D2B4DE", "#A569BD", "#8E44AD"),
    ),
    # ── Add more signals here ──
    # SignalSpec(
    #     source="treadmill",
    #     signal="speed_mm",
    #     label="treadmill",
    #     ylabel="Speed (mm/s)",
    #     use_dff=False,
    #     color_event="#D4EDDA",
    #     color_mask="#28A745",
    #     palette=("#28A745", "#155724", "#71D88A", "#218838",
    #              "#1E7E34", "#A3D9A5", "#6FCF97", "#27AE60"),
    # ),
]

TIME_COLUMN = "time_elapsed_s"

# ─── Detection parameters ────────────────────────────────────────────────

# Gaussian pre-filter
GAUSSIAN_SIGMA = 50.0  # σ in samples; set to 0 to skip

# Savitzky–Golay smoothing
SG_WINDOW = 11
SG_POLYORDER = 3

# Hysteresis thresholds (percentiles of the smoothed trace)
HIGH_PERCENTILE = 80.0
LOW_PERCENTILE = 60.0

# Event filtering
MAX_GAP_S = 0.5      # fill gaps shorter than this (seconds)
MIN_DURATION_S = 0.5  # discard events shorter than this (seconds)

# ─── Event detection functions ────────────────────────────────────────────


def smooth_trace(trace: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """Apply Savitzky–Golay filter; handles NaN/Inf and short traces."""
    clean = trace.copy()
    bad = ~np.isfinite(clean)
    if bad.all():
        return clean
    if bad.any():
        good_idx = np.where(~bad)[0]
        clean[bad] = np.interp(np.where(bad)[0], good_idx, clean[good_idx])

    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(clean):
        window_length = len(clean) if len(clean) % 2 == 1 else len(clean) - 1
    if window_length < polyorder + 2:
        return clean
    return savgol_filter(clean, window_length, polyorder)


def compute_hysteresis_thresholds(
    trace: np.ndarray, high_percentile: float, low_percentile: float
) -> tuple[float, float]:
    """Percentile-based high/low thresholds for hysteresis gating."""
    return float(np.percentile(trace, high_percentile)), float(np.percentile(trace, low_percentile))


def apply_hysteresis_mask(trace: np.ndarray, high_th: float, low_th: float) -> np.ndarray:
    """Boolean mask: enters event when trace ≥ high_th, exits when ≤ low_th."""
    mask = np.zeros(len(trace), dtype=bool)
    in_event = False
    start = 0
    for i, v in enumerate(trace):
        if not in_event and v >= high_th:
            in_event = True
            start = i
        elif in_event and v <= low_th:
            mask[start:i] = True
            in_event = False
    if in_event:
        mask[start:] = True
    return mask


def fill_short_gaps(mask: np.ndarray, t_s: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Close gaps in the boolean mask shorter than max_gap_s."""
    if len(t_s) < 2:
        return mask
    dt = np.median(np.diff(t_s))
    if not np.isfinite(dt) or dt <= 0:
        return mask
    max_gap_frames = int(max_gap_s / dt)
    inv = ~mask
    edges = np.diff(inv.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    filled = mask.copy()
    for s, e in zip(starts, ends):
        if (e - s) <= max_gap_frames:
            filled[s:e] = True
    return filled


def extract_events(
    mask: np.ndarray, t_s: np.ndarray, min_duration_s: float
) -> tuple[list[tuple[int, int]], list[float]]:
    """Extract (start_idx, end_idx) pairs and durations from the final mask."""
    edges = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    events: list[tuple[int, int]] = []
    durations_s: list[float] = []
    for s, e in zip(starts, ends):
        dur = t_s[min(e, len(t_s) - 1)] - t_s[s]
        if dur >= min_duration_s:
            events.append((s, e))
            durations_s.append(float(dur))
    return events, durations_s


def detect_events(
    trace: np.ndarray,
    t_s: np.ndarray,
    *,
    gauss_sigma: float = GAUSSIAN_SIGMA,
    sg_window: int = SG_WINDOW,
    sg_poly: int = SG_POLYORDER,
    high_pct: float = HIGH_PERCENTILE,
    low_pct: float = LOW_PERCENTILE,
    max_gap_s: float = MAX_GAP_S,
    min_dur_s: float = MIN_DURATION_S,
) -> dict:
    """Full pipeline: gaussian → SG smooth → threshold → hysteresis → gap-fill → extract."""
    if gauss_sigma > 0:
        trace = gaussian_filter1d(trace, sigma=gauss_sigma)
    smoothed = smooth_trace(trace, sg_window, sg_poly)
    high_th, low_th = compute_hysteresis_thresholds(smoothed, high_pct, low_pct)
    raw_mask = apply_hysteresis_mask(smoothed, high_th, low_th)
    filled_mask = fill_short_gaps(raw_mask, t_s, max_gap_s)
    events, durations_s = extract_events(filled_mask, t_s, min_dur_s)
    return {
        "smoothed": smoothed,
        "high_th": high_th,
        "low_th": low_th,
        "mask": filled_mask,
        "events": events,
        "durations_s": durations_s,
    }


# ─── Preprocessing ────────────────────────────────────────────────────────


def preprocess(raw: np.ndarray, use_dff: bool) -> np.ndarray:
    """Optionally compute ΔF/F (5th-percentile baseline)."""
    if not use_dff:
        return raw
    f0 = np.percentile(raw, 5)
    if f0 <= 0:
        raise ValueError(
            "Requested dF/F normalization but 5th-percentile baseline F0 <= 0. "
            "This usually means the signal is already baseline-normalized (e.g., "
            "already ΔF/F). Disable dF/F for this signal."
        )
    return (raw - f0) / f0


# ─── Plotting helpers ────────────────────────────────────────────────────


SAMPLE_DURATION_S = 400  # seconds to show in single-session comparison


def _plot_panel(axes_pair, t_s, det, title, spec):
    """Draw smoothed trace + event mask into a (trace_ax, mask_ax) pair."""
    ax1, ax2 = axes_pair
    # clip to sample window
    mask_t = t_s <= t_s[0] + SAMPLE_DURATION_S
    t_clip = t_s[mask_t]
    sm_clip = det["smoothed"][mask_t]
    ev_mask_clip = det["mask"][mask_t]

    for s, e in det["events"]:
        e_idx = min(e, len(t_s) - 1)
        if t_s[s] > t_clip[-1]:
            continue
        ax1.axvspan(
            t_s[s], min(t_s[e_idx], t_clip[-1]),
            color=spec.color_event, alpha=0.45, lw=0,
        )

    ax1.plot(t_clip, sm_clip, color="#1A1A2E", linewidth=1.0, label="smoothed")
    ax1.axhline(det["high_th"], color="#E63946", ls="--", lw=0.9, alpha=0.85,
                label=f"high thr. ({HIGH_PERCENTILE:.0f}th %ile)")
    ax1.axhline(det["low_th"], color="#F4A261", ls="--", lw=0.9, alpha=0.85,
                label=f"low thr. ({LOW_PERCENTILE:.0f}th %ile)")
    ax1.set_title(title, fontsize=11, fontweight="bold", pad=5)
    ax1.tick_params(labelsize=10)

    ax2.fill_between(t_clip, ev_mask_clip.astype(float), step="mid",
                     color=spec.color_mask, alpha=0.6, lw=0)
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["off", "on"], fontsize=10)
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.tick_params(labelsize=10)


def plot_single_session_comparison(
    t_s_a: np.ndarray, det_a: dict, label_a: str,
    t_s_b: np.ndarray, det_b: dict, label_b: str,
    spec: SignalSpec,
    suptitle: str = "",
) -> plt.Figure:
    """Side-by-side comparison of two sessions (first SAMPLE_DURATION_S s)."""
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 4.0),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08, "wspace": 0.02},
        facecolor="white", layout="constrained",
    )
    # axes[row, col]: top row = traces, bottom row = event masks
    _plot_panel((axes[0, 0], axes[1, 0]), t_s_a, det_a, label_a, spec)
    _plot_panel((axes[0, 1], axes[1, 1]), t_s_b, det_b, label_b, spec)

    # sync y-axis of right trace panel to session 1 (left) scale
    axes[0, 1].set_ylim(axes[0, 0].get_ylim())

    # remove redundant y-axis ticks/labels on right panels
    axes[0, 1].set_yticklabels([])
    axes[0, 1].tick_params(axis="y", length=0)
    axes[1, 1].set_yticklabels([])
    axes[1, 1].tick_params(axis="y", length=0)

    # shared labels on left column only
    axes[0, 0].set_ylabel(spec.ylabel, fontsize=11)
    axes[1, 0].set_ylabel("Events", fontsize=11)
    # legend outside plot area to avoid overlap
    axes[0, 0].legend(
        loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3,
        fontsize=8, framealpha=0.9, edgecolor="0.8", borderaxespad=0,
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    fig.align_ylabels(axes[:, 0])
    return fig


def plot_session_page(
    t_s: np.ndarray, det: dict, title: str, spec: SignalSpec,
) -> plt.Figure:
    """Single-session page for the PDF report."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 3.2), sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.06},
        facecolor="white", layout="constrained",
    )
    _plot_panel((ax1, ax2), t_s, det, title, spec)
    ax1.set_ylabel(spec.ylabel, fontsize=11)
    ax2.set_ylabel("Events", fontsize=11)
    ax1.legend(
        loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3,
        fontsize=8, framealpha=0.9, edgecolor="0.8", borderaxespad=0,
    )
    fig.align_ylabels([ax1, ax2])
    return fig


def plot_group_summary(summary_df: pd.DataFrame, spec: SignalSpec) -> plt.Figure:
    """Bar chart of event counts and mean durations per subject (grant-ready)."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), facecolor="white")

    palette = list(spec.palette)
    counts = summary_df.groupby("Subject")["n_events"].sum().sort_index()
    colors = [palette[i % len(palette)] for i in range(len(counts))]

    bars1 = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", lw=0.5)
    axes[0].set_ylabel("Total events", fontsize=11)
    axes[0].set_title("Event count per subject", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45, labelsize=10)
    axes[0].tick_params(axis="y", labelsize=10)
    axes[0].bar_label(bars1, fontsize=9, padding=2)

    mean_dur = summary_df.groupby("Subject")["mean_duration_s"].mean().sort_index()
    bars2 = axes[1].bar(mean_dur.index, mean_dur.values, color=colors, edgecolor="white", lw=0.5)
    axes[1].set_ylabel("Mean duration (s)", fontsize=11)
    axes[1].set_title("Mean event duration per subject", fontsize=12, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45, labelsize=10)
    axes[1].tick_params(axis="y", labelsize=10)
    axes[1].bar_label(bars2, fmt="%.1f", fontsize=9, padding=2)

    fig.suptitle(f"{spec.label} event detection — group summary",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ─── Analysis runner ─────────────────────────────────────────────────────


def run_signal(spec: SignalSpec, proj: Project, run, all_sessions) -> None:
    """Run single-session test + group analysis for one signal spec."""
    source = spec.source
    signal = spec.signal
    tag = spec.label

    print(f"\n{'═' * 60}")
    print(f"  Signal: {source}/{signal}  ({tag})")
    print(f"{'═' * 60}")

    # ── Discover actual signal name ──
    first_sess = all_sessions[0]
    available_sources = first_sess.sources
    if source not in available_sources:
        print(f"  ✗ Source {source!r} not found. Available: {available_sources}")
        return

    available_signals = first_sess.signals(source)
    if signal not in available_signals:
        if available_signals:
            signal = available_signals[0]
            print(f"  Signal {spec.signal!r} not found → using fallback: {signal}")
        else:
            print(f"  ✗ No signals found for source {source!r}")
            return

    # ── Stage 1: Single-session comparison (session 1 vs session 10) ──
    print(f"\n── Single-session comparison ({tag}) ──")
    sess_a = all_sessions[0]
    sess_b = all_sessions[min(9, len(all_sessions) - 1)]

    def _load_and_detect(sess):
        t = sess.time(source, column=TIME_COLUMN)[1:]
        raw = sess.signal(source, signal)[1:]
        tr = preprocess(raw, spec.use_dff)
        return t, tr, detect_events(tr, t)

    t_a, trace_a, det_a = _load_and_detect(sess_a)
    t_b, trace_b, det_b = _load_and_detect(sess_b)

    for label, det in [(sess_a.label, det_a), (sess_b.label, det_b)]:
        n = len(det["events"])
        durs = det["durations_s"]
        print(f"  {label} → {n} events")
        if durs:
            print(f"    Durations (s): min={min(durs):.2f}, max={max(durs):.2f}, mean={np.mean(durs):.2f}")

    fig_test = plot_single_session_comparison(
        t_a, det_a, f"Session 1 — {sess_a.label}",
        t_b, det_b, f"Session {min(10, len(all_sessions))} — {sess_b.label}",
        spec=spec,
        suptitle=f"{source}/{signal} — first {SAMPLE_DURATION_S}s comparison",
    )
    run.save_figure(fig_test, f"{tag}_single_session_test.png", dpi=200)

    # ── Stage 2: All sessions ──
    print(f"\n── Group analysis ({tag}, {len(all_sessions)} sessions) ──")

    summary_rows: list[dict] = []
    all_events: list[pd.DataFrame] = []

    with run.pdf(f"{tag}_event_detection.pdf") as pdf:
        for sess in all_sessions:
            try:
                t_s = sess.time(source, column=TIME_COLUMN)[1:]
                raw_sig = sess.signal(source, signal)[1:]
            except Exception as exc:
                print(f"  SKIP {sess.label}: {exc}")
                continue

            if len(raw_sig) < 2:
                print(f"  SKIP {sess.label}: too few samples ({len(raw_sig)})")
                continue

            trace = preprocess(raw_sig, spec.use_dff)
            det = detect_events(trace, t_s)
            n_ev = len(det["events"])
            durs = det["durations_s"]

            print(f"  {sess.label}  → {n_ev} events")

            summary_rows.append({
                "Subject": sess.subject,
                "Session": sess.session,
                "Task": sess.task,
                "signal": f"{source}/{signal}",
                "n_events": n_ev,
                "mean_duration_s": float(np.mean(durs)) if durs else 0.0,
                "total_event_time_s": float(np.sum(durs)),
                "high_th": det["high_th"],
                "low_th": det["low_th"],
            })

            if det["events"]:
                onsets = np.array([t_s[s] for s, _ in det["events"]])
                offsets = np.array([t_s[min(e, len(t_s) - 1)] for _, e in det["events"]])
                ev = make_events(
                    {"onset": onsets, "offset": offsets},
                    subject=sess.subject,
                    session=sess.session,
                    task=sess.task,
                )
                all_events.append(ev)

            fig = plot_session_page(
                t_s, det,
                title=f"{sess.label}\n{source}/{signal} — {n_ev} events",
                spec=spec,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("  PDF report saved.")

    # ── Save tables ──
    summary_df = pd.DataFrame(summary_rows)
    run.save_table(summary_df, f"{tag}_event_summary.csv")

    if all_events:
        events_df = pd.concat(all_events, ignore_index=True)
    else:
        events_df = pd.DataFrame(columns=["Subject", "Session", "Task", "EventType", "event_time"])

    run.save_table(events_df, f"{tag}_events.csv")

    n_total = len(events_df) // 2
    n_subjects = events_df["Subject"].nunique() if not events_df.empty else 0
    print(f"  Total events: {n_total} across {n_subjects} subjects")

    # ── Group summary plot ──
    if not summary_df.empty and summary_df["n_events"].sum() > 0:
        fig_summary = plot_group_summary(summary_df, spec)
        run.save_figure(fig_summary, f"{tag}_group_summary.png", dpi=200)

    return n_total, n_subjects


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

proj = Project(dataset=DATASET, analyst="databench")
all_sessions = proj.sessions()
run = proj.run(name="event-detection", tag="hfsa")
print(f"Dataset: {DATASET.name} — {len(all_sessions)} sessions")

results: dict[str, tuple] = {}

for spec in SIGNALS:
    result = run_signal(spec, proj, run, all_sessions)
    if result is not None:
        results[spec.label] = result

# ─── Combined report ─────────────────────────────────────────────────────

notes_lines = [
    "Hysteresis-based event detection.",
    f"Savitzky–Golay: window={SG_WINDOW}, poly={SG_POLYORDER}",
    f"Thresholds: high={HIGH_PERCENTILE}%ile, low={LOW_PERCENTILE}%ile",
    f"Gap fill: {MAX_GAP_S}s, min duration: {MIN_DURATION_S}s",
    f"Sessions: {len(all_sessions)}",
    "",
    "Signals analyzed:",
]
for spec in SIGNALS:
    r = results.get(spec.label)
    if r:
        notes_lines.append(f"  • {spec.source}/{spec.signal} ({spec.label}): "
                           f"{r[0]} events, {r[1]} subjects")
    else:
        notes_lines.append(f"  • {spec.source}/{spec.signal} ({spec.label}): skipped")

run.finish(notes="\n".join(notes_lines))

print("\nDone.")
