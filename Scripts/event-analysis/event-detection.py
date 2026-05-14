#!/usr/bin/env python3
"""
Hysteresis-based event detection for arbitrary signals.

Detects transient events using Savitzky–Golay smoothing, hysteresis
thresholding, gap-filling, and minimum-duration filtering.

Includes:
  - Artifact rejection: flags physiologically implausible transients
    via adaptive derivative thresholding (MAD-based), interpolates across them.
  - Baseline detrending: rolling-median subtraction to remove slow drift
    before thresholding.

Dual-trace measurement strategy for longitudinal comparisons:
  Event *boundaries* are defined on the detrended trace (removes slow
  drift that would shift thresholds).  Event *amplitudes*
  (peak_raw, peak_z, mean_raw, mean_z) are measured on the non-detrended
  cleaned trace, preserving absolute scale for within-animal longitudinal
  and between-animal hierarchical comparisons.  Detrended-space metrics
  are retained as *_detrended columns for detection diagnostics.

Supports multiple signal sources. Comment/uncomment entries in the
SIGNALS list below to choose which signals to analyze.

Two stages per signal:
  1. Single-session test: first session, diagnostic plot.
  2. Group analysis: all sessions, PDF report + summary stats.

Usage:
    DATABENCH_DATASET=hfsa python event-detection.py
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from databench.project import Project
from databench.signal.epoching import make_events
from databench.config import resolve_dataset
from databench.plotting import set_theme, style_axes, get_theme

set_theme()

# ─── Dataset ──────────────────────────────────────────────────────────────

DATASET = resolve_dataset("hfsa")

# ─── Signal definitions ──────────────────────────────────────────────────


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

    # ── Gaussian pre-smoothing ──
    gauss_sigma: float = 75        # Gaussian σ in samples (~1.5 s at 50 Hz)

    # ── Artifact rejection ──
    artifact_k: float = 8.0        # MAD multiplier for derivative outlier detection
    artifact_pad: int = 3          # frames to expand around each flagged artifact

    # ── Detrending ──
    detrend: bool = False          # whether to subtract rolling baseline
    detrend_window_s: float = 120.0  # rolling window in seconds
    detrend_quantile: float = 0.5  # 0.5 = median; lower (e.g. 0.1) tracks the floor

    # ── Normalization ──
    normalize: str = "none"        # "none", "zscore", or "dff"

    # ── Noise-floor thresholds ──
    noise_high_k: float = 4.0     # MAD multiplier for event onset
    noise_low_k: float = 2.5      # MAD multiplier for event offset


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
        # Mesofield ΔF/F drifts over sessions; detrend with shorter window
        detrend=True,
        detrend_window_s=60.0,
        detrend_quantile=0.1,
        artifact_k=8.0,
        noise_high_k=4.0,
        noise_low_k=2.5,
    ),
    # ── Pupil diameter ──
    SignalSpec(
        source="pupil",
        signal="pupil_diameter_mm",
        label="pupil",
        ylabel="Pupil (z-score)",
        use_dff=False,
        color_event="#E8D5F5",
        color_mask="#7B2D8E",
        palette=("#7B2D8E", "#4A0E5C", "#C39BD3", "#884EA0",
                 "#6C3483", "#D2B4DE", "#A569BD", "#8E44AD"),
        # Pupil drifts substantially — detrend by default
        detrend=True,
        gauss_sigma=50,
        detrend_window_s=120.0,
        artifact_k=5.0,      # pupil blinks are sharp; be more aggressive
        artifact_pad=5,
        normalize="zscore",
        noise_high_k=3.0,
        noise_low_k=2.0,
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
    #     detrend=True,
    #     detrend_window_s=60.0,
    # ),
]

TIME_COLUMN = "time_elapsed_s"

# ─── Detection parameters ────────────────────────────────────────────────

# Savitzky–Golay smoothing
SG_WINDOW = 11
SG_POLYORDER = 3

# Hysteresis thresholds (noise-floor MAD multipliers)
NOISE_MAD_HIGH_K = 4.0   # MAD multiplier for event onset
NOISE_MAD_LOW_K = 2.5    # MAD multiplier for event offset

# Event filtering
MAX_GAP_S = 1      # fill gaps shorter than this (seconds)
MIN_DURATION_S = 2  # discard events shorter than this (seconds)


# ─── Artifact rejection ──────────────────────────────────────────────────


def reject_artifacts(
    trace: np.ndarray,
    t_s: np.ndarray,
    k: float = 8.0,
    pad: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect and interpolate across physiologically implausible transients.

    Uses the median absolute deviation (MAD) of the frame-to-frame
    derivative to set an adaptive threshold.  Frames where |Δx| > k·MAD
    are flagged, expanded by ``pad`` frames on each side, and linearly
    interpolated over.

    Why MAD instead of std?  The outliers we want to catch (blinks,
    tracking dropouts) are themselves heavy-tailed — they inflate σ,
    making a std-based cutoff too permissive.  MAD ignores them.

    Parameters
    ----------
    trace : array
        Raw signal values.
    t_s : array
        Timestamps (same length as trace).
    k : float
        Multiplier on the MAD-derived scale (σ̂ = 1.4826 · MAD).
        Higher = more permissive.  6–8 is a reasonable starting range
        for pupil; 8–10 for smoother signals like widefield ΔF/F.
    pad : int
        Frames to expand each flagged region on both sides.
        Catches the ramps into/out of an artifact that individually
        may not exceed the derivative threshold.

    Returns
    -------
    cleaned : array
        Trace with artifacts replaced by linear interpolation.
    artifact_mask : boolean array
        True where artifacts were detected (before interpolation).
    """
    if len(trace) < 3:
        return trace.copy(), np.zeros(len(trace), dtype=bool)

    # ── Frame-to-frame derivative ──
    dx = np.diff(trace, prepend=trace[0])

    # ── Adaptive threshold via MAD ──
    med_dx = np.median(dx)
    mad = np.median(np.abs(dx - med_dx))

    # 1.4826 is the consistency constant: MAD → σ under normality.
    # Not that the derivative is necessarily Gaussian, but it gives
    # a principled scale factor without assuming it.
    sigma_est = 1.4826 * mad if mad > 0 else np.std(dx)

    threshold = k * sigma_est
    if threshold <= 0:
        return trace.copy(), np.zeros(len(trace), dtype=bool)

    # ── Flag outlier frames ──
    flagged = np.abs(dx - med_dx) > threshold

    # ── Expand flagged regions by `pad` frames ──
    if pad > 0 and flagged.any():
        expanded = flagged.copy()
        for shift in range(1, pad + 1):
            expanded[shift:] |= flagged[:-shift]
            expanded[:-shift] |= flagged[shift:]
        flagged = expanded

    artifact_mask = flagged.copy()

    # ── Interpolate across flagged regions ──
    cleaned = trace.copy()
    if flagged.any():
        good = np.where(~flagged)[0]
        if len(good) >= 2:
            bad = np.where(flagged)[0]
            cleaned[bad] = np.interp(bad, good, cleaned[good])
        # If nearly everything is flagged, fall back to original —
        # better noisy data than a flat interpolated line.

    return cleaned, artifact_mask


# ─── Detrending ───────────────────────────────────────────────────────────


def detrend_rolling_baseline(
    trace: np.ndarray,
    t_s: np.ndarray,
    window_s: float = 120.0,
    quantile: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove slow drift via rolling-quantile subtraction.

    Parameters
    ----------
    trace : array
        Signal to detrend (ideally already artifact-cleaned).
    t_s : array
        Timestamps (used to compute window size in frames).
    window_s : float
        Window duration in seconds.  Should be several times longer
        than the longest event you expect.
    quantile : float
        Quantile for the rolling window.  0.5 = median (tracks the
        center).  Lower values (e.g. 0.1) track the floor of the
        signal, preventing transient peaks from inflating the
        baseline and producing artificial negative dips.

    Returns
    -------
    detrended : array
        trace − baseline.
    baseline : array
        The rolling-quantile estimate (for plotting / diagnostics).
    """
    if len(trace) < 3:
        return trace.copy(), np.zeros_like(trace)

    dt = np.median(np.diff(t_s))
    if not np.isfinite(dt) or dt <= 0:
        return trace.copy(), np.zeros_like(trace)

    window_frames = int(window_s / dt)
    if window_frames % 2 == 0:
        window_frames += 1
    window_frames = max(window_frames, 3)

    # pandas rolling median wraps a C-level implementation —
    # fast enough for traces up to ~10⁶ samples.
    baseline = pd.Series(trace).rolling(
        window=window_frames, center=True, min_periods=1
    ).quantile(quantile).values

    detrended = trace - baseline

    return detrended, baseline


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
    trace: np.ndarray, high_k: float, low_k: float
) -> tuple[float, float]:
    """Noise-floor MAD thresholds for hysteresis gating.

    Selects the lower half of the trace (values ≤ median) as a proxy
    for quiet periods, computes the MAD of that subset, scales to σ̂
    via the 1.4826 consistency constant, and returns thresholds at
    median_lower + k × σ̂.  This anchors detection to the noise floor
    rather than the signal distribution, so thresholds stay stable as
    long as noise is stable — regardless of biological signal amplitude.
    """
    trace_median = np.median(trace)
    lower_half = trace[trace <= trace_median]
    if len(lower_half) < 2:
        lower_half = trace
    lower_median = float(np.median(lower_half))
    mad = float(np.median(np.abs(lower_half - lower_median)))
    noise_sigma = 1.4826 * mad if mad > 0 else float(np.std(lower_half))
    high_th = lower_median + high_k * noise_sigma
    low_th = lower_median + low_k * noise_sigma
    return high_th, low_th


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
    sg_window: int = SG_WINDOW,
    sg_poly: int = SG_POLYORDER,
    noise_high_k: float = NOISE_MAD_HIGH_K,
    noise_low_k: float = NOISE_MAD_LOW_K,
    max_gap_s: float = MAX_GAP_S,
    min_dur_s: float = MIN_DURATION_S,
    # ── Gaussian pre-smoothing ──
    gauss_sigma: float = 75,
    # ── Artifact rejection ──
    artifact_k: float = 8.0,
    artifact_pad: int = 3,
    # ── Detrending ──
    do_detrend: bool = False,
    detrend_window_s: float = 120.0,
    detrend_quantile: float = 0.5,
) -> dict:
    """
    Full pipeline:
        artifact reject → detrend → smooth → threshold →
        hysteresis → gap-fill → extract.

    Returns a dict with all intermediate results for diagnostics:
      artifact_mask   – boolean, True at rejected frames
      cleaned         – trace after artifact interpolation
      baseline        – rolling-median baseline (zeros if detrend=False)
      detrended       – trace after baseline subtraction
      working_trace   – the trace that was actually thresholded
      smoothed, high_th, low_th, mask, events, durations_s – as before
    """
    # ── 1. Artifact rejection ──
    cleaned, artifact_mask = reject_artifacts(
        trace, t_s, k=artifact_k, pad=artifact_pad
    )

    # ── 2. Detrending ──
    if do_detrend:
        detrended, baseline = detrend_rolling_baseline(
            cleaned, t_s, window_s=detrend_window_s, quantile=detrend_quantile
        )
    else:
        detrended = cleaned
        baseline = np.zeros_like(cleaned)

    # ── 3. Gaussian smoothing ──
    # Apply to both the detrended trace (for detection) and the cleaned
    # trace (for amplitude measurement) so smoothing is consistent.
    if gauss_sigma > 0:
        detrended = gaussian_filter1d(detrended, sigma=gauss_sigma)
        cleaned_smooth = gaussian_filter1d(cleaned, sigma=gauss_sigma)
    else:
        cleaned_smooth = cleaned.copy()

    # ── 4. Smooth → threshold → hysteresis → filter ──
    smoothed = smooth_trace(detrended, sg_window, sg_poly)
    high_th, low_th = compute_hysteresis_thresholds(smoothed, noise_high_k, noise_low_k)
    raw_mask = apply_hysteresis_mask(smoothed, high_th, low_th)
    filled_mask = fill_short_gaps(raw_mask, t_s, max_gap_s)
    events, durations_s = extract_events(filled_mask, t_s, min_dur_s)

    # ── 5. Peak detection: timestamp of maximum within each event (smoothed trace) ──
    peak_times_s: list[float] = []
    for event_start, event_end in events:
        e_idx = min(event_end, len(smoothed) - 1)
        segment = smoothed[event_start:e_idx]
        if len(segment) > 0:
            peak_offset = int(np.argmax(segment))
            peak_times_s.append(float(t_s[event_start + peak_offset]))
        else:
            peak_times_s.append(float(t_s[event_start]))

    return {
        # ── Diagnostic intermediates ──
        "artifact_mask": artifact_mask,
        "cleaned": cleaned,
        "cleaned_smooth": cleaned_smooth,  # non-detrended, Gaussian-smoothed
        "baseline": baseline,
        "detrended": detrended,
        "working_trace": detrended,
        # ── Detection outputs ──
        "smoothed": smoothed,
        "high_th": high_th,
        "low_th": low_th,
        "noise_high_k": noise_high_k,
        "noise_low_k": noise_low_k,
        "mask": filled_mask,
        "events": events,
        "durations_s": durations_s,
        "peak_times_s": peak_times_s,
    }


# ─── Per-event metric extraction (z-scored + raw) ────────────────────────


def _zscore_trace(trace: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Session-level z-score.  Returns (z_trace, mu, sigma)."""
    mu = float(np.mean(trace))
    sigma = float(np.std(trace))
    if sigma == 0:
        return np.zeros_like(trace), mu, sigma
    return (trace - mu) / sigma, mu, sigma


def extract_event_metrics(
    det: dict,
    t_s: np.ndarray,
    subject: str,
    session: str,
    task: str,
    signal_label: str,
    session_index: int = 0,
) -> list[dict]:
    """
    Measure each event on two traces and emit a flat dict per event.

    Primary amplitudes (peak_raw, peak_z, mean_raw, mean_z) are measured
    on ``cleaned_smooth`` — artifact-interpolated + Gaussian-smoothed but
    NOT baseline-subtracted.  This preserves absolute scale for cross-
    session and cross-animal hierarchical modelling.

    Detrended-space metrics (*_detrended) are measured on ``working_trace``
    (the trace used for event boundary detection) for diagnostics.

    Session-level covariates (baseline mean, recording duration,
    within-session fractional time) are included per-row so downstream
    models can control for nuisance variance without destructive
    preprocessing.
    """
    measurement = det["cleaned_smooth"]   # non-detrended, smoothed
    working = det["working_trace"]        # detrended (detection space)
    events = det["events"]

    # ── Z-score both traces at session level ──
    z_meas, mu_meas, sigma_meas = _zscore_trace(measurement)
    z_det, mu_det, sigma_det = _zscore_trace(working)

    # ── Session-level covariates ──
    baseline_mean = float(np.mean(det["baseline"]))
    rec_dur = float(t_s[-1] - t_s[0]) if len(t_s) > 1 else 0.0

    rows: list[dict] = []
    for (s, e), dur, peak_t in zip(events, det["durations_s"], det["peak_times_s"]):
        e_idx = min(e, len(measurement) - 1)
        seg = measurement[s:e_idx]
        seg_z = z_meas[s:e_idx]
        seg_det = working[s:e_idx]
        seg_z_det = z_det[s:e_idx]

        if len(seg) == 0:
            continue

        onset_t = float(t_s[s])
        rows.append({
            "Subject": subject,
            "Session": session,
            "Task": task,
            "signal": signal_label,
            "session_index": session_index,
            "onset_s": onset_t,
            "offset_s": float(t_s[e_idx]),
            "duration_s": dur,
            "peak_time_s": peak_t,
            "time_to_peak_s": peak_t - onset_t,
            # ── Primary: non-detrended (for cross-session comparison) ──
            "peak_raw": float(np.max(seg)),
            "mean_raw": float(np.mean(seg)),
            "peak_z": float(np.max(seg_z)),
            "mean_z": float(np.mean(seg_z)),
            # ── Detrended (detection-space diagnostics) ──
            "peak_raw_detrended": float(np.max(seg_det)),
            "mean_raw_detrended": float(np.mean(seg_det)),
            "peak_z_detrended": float(np.max(seg_z_det)),
            "mean_z_detrended": float(np.mean(seg_z_det)),
            # ── Session-level covariates for hierarchical model ──
            "session_mu": mu_meas,
            "session_sigma": sigma_meas,
            "session_mu_detrended": mu_det,
            "session_sigma_detrended": sigma_det,
            "session_baseline_mean": baseline_mean,
            "recording_duration_s": rec_dur,
            "event_time_frac": (onset_t - t_s[0]) / rec_dur if rec_dur > 0 else 0.0,
        })

    return rows


# ─── Preprocessing ────────────────────────────────────────────────────────


def preprocess(raw: np.ndarray, use_dff: bool, normalize: str = "none") -> np.ndarray:
    """Normalize signal: 'dff' for ΔF/F, 'zscore' for z-score, 'none' for raw."""
    # Legacy use_dff flag overrides normalize
    if use_dff:
        normalize = "dff"
    if normalize == "dff":
        f0 = np.percentile(raw, 5)
        if f0 <= 0:
            raise ValueError(
                "Requested dF/F normalization but 5th-percentile baseline F0 <= 0. "
                "This usually means the signal is already baseline-normalized (e.g., "
                "already ΔF/F). Disable dF/F for this signal."
            )
        return (raw - f0) / f0
    elif normalize == "zscore":
        mu = np.nanmean(raw)
        sd = np.nanstd(raw)
        if sd > 0:
            return (raw - mu) / sd
        return raw - mu
    return raw


# ─── Plotting helpers ────────────────────────────────────────────────────


def _style_axis(ax: plt.Axes) -> None:
    style_axes(ax)


def plot_single_session(
    t_s: np.ndarray,
    raw: np.ndarray,
    det: dict,
    title: str,
    spec: SignalSpec,
) -> plt.Figure:
    """
    Diagnostic plot.

    When artifact rejection or detrending is active, a top panel shows
    the original trace with artifact highlights and the rolling-median
    baseline overlay.  The middle panel shows the working (detrended)
    trace with smoothed line, thresholds, and event shading.  Bottom
    panel is the binary event mask.

    When neither feature is active, falls back to the original
    two-panel layout.
    """
    has_artifacts = det["artifact_mask"].any()
    has_detrend = spec.detrend
    show_top = has_artifacts or has_detrend

    if show_top:
        fig, (ax0, ax1, ax2) = plt.subplots(
            3, 1, figsize=(12, 5.5), sharex=True,
            gridspec_kw={"height_ratios": [2.5, 4, 1], "hspace": 0.08},
            layout="constrained",
        )
    else:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 4.0), sharex=True,
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
            layout="constrained",
        )
        ax0 = None

    # ── Top panel: original signal + artifacts + baseline ──
    if ax0 is not None:
        ax0.plot(t_s, raw, color="#AAAAAA", linewidth=0.4,
                 label=f"raw {spec.label}", rasterized=True)
        ax0.plot(t_s, det["cleaned"], color=get_theme().fg, linewidth=0.6,
                 label="artifact-cleaned")

        if has_detrend:
            ax0.plot(t_s, det["baseline"], color=get_theme().colors[1], linewidth=1.4,
                     alpha=0.85, label="rolling-median baseline")

        if has_artifacts:
            art = det["artifact_mask"]
            ylo, yhi = raw[np.isfinite(raw)].min(), raw[np.isfinite(raw)].max()
            ax0.fill_between(t_s, ylo, yhi, where=art,
                             color=get_theme().colors[3], alpha=0.25, lw=0, label="artifacts")
            n_art_events = (np.diff(art.astype(int), prepend=0) == 1).sum()
            ax0.set_title(
                f"Original signal — {n_art_events} artifact region(s) rejected",
                fontsize=9, fontstyle="italic", loc="left", pad=4,
            )

        ax0.set_ylabel(spec.ylabel)
        ax0.legend(loc="upper right", fontsize=7, framealpha=0.9, edgecolor="0.8")
        _style_axis(ax0)

    # ── Middle panel: working trace + smoothed + thresholds + events ──
    for s, e in det["events"]:
        e_idx = min(e, len(t_s) - 1)
        ax1.axvspan(t_s[s], t_s[e_idx], color=spec.color_event, alpha=0.45, lw=0)

    working = det["working_trace"]
    ylabel_mid = f"{spec.ylabel} (detrended)" if has_detrend else spec.ylabel

    # Ghost: pre-smoothed trace (detrended but no Gaussian/SG)
    ghost = det["cleaned"] - det["baseline"]
    ax1.plot(t_s, ghost, color="#AAAAAA", linewidth=0.35,
             label="raw signal", rasterized=True)
    ax1.plot(t_s, det["smoothed"], color=get_theme().fg, linewidth=1.0, label="smoothed")
    ax1.axhline(det["high_th"], color=get_theme().colors[3], ls="--", lw=1.0, alpha=0.85,
                label=f"high threshold ({det['noise_high_k']}× MAD)")
    ax1.axhline(det["low_th"], color=get_theme().colors[1], ls="--", lw=1.0, alpha=0.85,
                label=f"low threshold ({det['noise_low_k']}× MAD)")

    # Threshold value labels
    th_gap = abs(det["high_th"] - det["low_th"])
    y_range = np.ptp(working[np.isfinite(working)])
    low_va = "top" if (th_gap / max(y_range, 1e-9)) < 0.06 else "center"
    ax1.text(t_s[-1], det["high_th"], f' {det["high_th"]:.3f}',
             fontsize=6, va="center", ha="left",
             color=get_theme().colors[3], clip_on=False)
    ax1.text(t_s[-1], det["low_th"], f' {det["low_th"]:.3f}',
             fontsize=6, va=low_va, ha="left",
             color=get_theme().colors[1], clip_on=False)

    # Baseline zero reference
    if has_detrend:
        ax1.axhline(0, color=get_theme().colors[1], linewidth=0.5,
                    alpha=0.35, ls=":")

    # ── Peak markers ──
    if det["peak_times_s"]:
        peak_values = np.interp(det["peak_times_s"], t_s, det["smoothed"])
        ax1.scatter(
            det["peak_times_s"], peak_values,
            marker="v", color=get_theme().colors[3], s=28, zorder=5,
            label="peaks", edgecolors="white", linewidths=0.5,
        )

    # ── Duration brackets along bottom ──
    ylims = ax1.get_ylim()
    bracket_y = ylims[0] + (ylims[1] - ylims[0]) * 0.015
    for s, e in det["events"]:
        e_idx = min(e, len(t_s) - 1)
        ax1.plot([t_s[s], t_s[e_idx]], [bracket_y, bracket_y],
                 color=spec.color_mask, linewidth=1.5,
                 solid_capstyle="butt", alpha=0.7, zorder=4)

    # ── Events summary annotation ──
    n_events = len(det["events"])
    durs = det["durations_s"]
    mean_dur = f"{np.mean(durs):.1f}" if durs else "\u2014"
    ax1.text(1.0, 1.02,
             f"{n_events} events  |  x\u0304 dur {mean_dur} s",
             transform=ax1.transAxes, fontsize=7, va="bottom",
             ha="right", color=spec.color_mask, fontweight="bold")

    ax1.set_ylabel(ylabel_mid)
    ax1.set_title(title)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9, edgecolor="0.8")
    _style_axis(ax1)

    # ── Bottom panel: event mask ──
    ax2.fill_between(t_s, det["mask"].astype(float), step="mid",
                     color=spec.color_mask, alpha=0.6, lw=0)
    ax2.set_ylabel("Events")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["off", "on"], fontsize=8)
    _style_axis(ax2)

    all_axes = [ax for ax in [ax0, ax1, ax2] if ax is not None]
    fig.align_ylabels(all_axes)

    # ── Parameter footer ──
    param_parts = [
        f"artifact k={spec.artifact_k} pad={spec.artifact_pad}",
        f"gauss \u03c3={spec.gauss_sigma}",
    ]
    if has_detrend:
        param_parts.append(
            f"detrend {spec.detrend_window_s:.0f}s q={spec.detrend_quantile}"
        )
    param_parts.extend([
        f"SG({SG_WINDOW},{SG_POLYORDER})",
        f"high={spec.noise_high_k}\u00d7 low={spec.noise_low_k}\u00d7 MAD",
        f"gap<{MAX_GAP_S}s  min dur\u2265{MIN_DURATION_S}s",
    ])
    fig.text(
        0.5, -0.02, "   \u2502   ".join(param_parts),
        ha="center", va="top", fontsize=6, family="monospace",
        color="0.45",
    )

    return fig


def plot_session_comparison(
    session_data: list[dict],
    spec: SignalSpec,
    session_labels: list[str],
) -> plt.Figure:
    """
    Compact side-by-side of two sessions for grant/publication figures.

    Single overlaid axis per column showing the raw signal as a ghost,
    the smoothed detection trace, MAD thresholds, event shading, peaks,
    and duration brackets.  A legend and parameter footer label the
    visual elements.
    """
    n_cols = len(session_data)
    has_detrend = spec.detrend

    fig, axes = plt.subplots(
        1, n_cols, figsize=(7.0, 2.2), sharex="col", sharey="row",
        gridspec_kw={"wspace": 0.08},
        layout="constrained",
    )
    if n_cols == 1:
        axes = [axes]

    legend_handles = []

    for col, sd in enumerate(session_data):
        t_s = sd["t_s"]
        det = sd["det"]
        ax = axes[col]

        working = det["working_trace"]

        # ── Ghost: pre-smoothed trace (detrended but no Gaussian/SG) ──
        # det["cleaned"] is artifact-interpolated; subtract baseline to
        # put it on the same y-scale as the smoothed detection trace.
        ghost = det["cleaned"] - det["baseline"]
        working = det["working_trace"]
        h_raw, = ax.plot(t_s, ghost, color="#AAAAAA", linewidth=0.3,
                         alpha=0.5, rasterized=True)

        # ── Artifact regions ──
        if det["artifact_mask"].any():
            art = det["artifact_mask"]
            finite_w = working[np.isfinite(working)]
            ylo, yhi = finite_w.min(), finite_w.max()
            margin = (yhi - ylo) * 0.03
            ax.fill_between(
                t_s, ylo - margin, yhi + margin, where=art,
                color="#E76F51", alpha=0.12, lw=0,
            )

        # ── Baseline zero ──
        if has_detrend:
            ax.axhline(0, color=get_theme().colors[1], linewidth=0.5,
                       alpha=0.35, ls=":")

        # ── Event spans ──
        for s, e in det["events"]:
            e_idx = min(e, len(t_s) - 1)
            ax.axvspan(t_s[s], t_s[e_idx],
                       color=spec.color_event, alpha=0.35, lw=0)

        # ── Smoothed detection trace ──
        h_smooth, = ax.plot(t_s, det["smoothed"], color=get_theme().fg,
                            linewidth=0.7, zorder=3)

        # ── MAD thresholds ──
        h_high = ax.axhline(det["high_th"], color=get_theme().colors[3],
                            ls="--", lw=0.6, alpha=0.7, zorder=2)
        h_low = ax.axhline(det["low_th"], color=get_theme().colors[1],
                           ls="--", lw=0.6, alpha=0.7, zorder=2)

        # Threshold labels — offset low label if thresholds are close
        th_gap = abs(det["high_th"] - det["low_th"])
        y_range = np.ptp(working[np.isfinite(working)])
        low_va = "top" if (th_gap / max(y_range, 1e-9)) < 0.06 else "center"
        ax.text(t_s[-1], det["high_th"], f' {det["high_th"]:.3f}',
                fontsize=4.5, va="center", ha="left",
                color=get_theme().colors[3], clip_on=False)
        ax.text(t_s[-1], det["low_th"], f' {det["low_th"]:.3f}',
                fontsize=4.5, va=low_va, ha="left",
                color=get_theme().colors[1], clip_on=False)

        # ── Peak markers ──
        if det["peak_times_s"]:
            peak_values = np.interp(det["peak_times_s"], t_s,
                                    det["smoothed"])
            ax.scatter(
                det["peak_times_s"], peak_values, marker="v",
                color=get_theme().colors[3], s=12, zorder=5,
                edgecolors="white", linewidths=0.25,
            )

        # ── Duration brackets along bottom ──
        ylims = ax.get_ylim()
        bracket_y = ylims[0] + (ylims[1] - ylims[0]) * 0.015
        for s, e in det["events"]:
            e_idx = min(e, len(t_s) - 1)
            ax.plot([t_s[s], t_s[e_idx]], [bracket_y, bracket_y],
                    color=spec.color_mask, linewidth=1.2,
                    solid_capstyle="butt", alpha=0.7, zorder=4)

        # ── Title (left) + events label (right, same height) ──
        n_events = len(det["events"])
        durs = det["durations_s"]
        mean_dur = f"{np.mean(durs):.1f}" if durs else "—"

        ax.text(0.0, 1.02, session_labels[col],
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="bottom", ha="left")
        ax.text(1.0, 1.02,
                f"{n_events} events  |  x\u0304 dur {mean_dur} s",
                transform=ax.transAxes, fontsize=6, va="bottom",
                ha="right", color=spec.color_mask, fontweight="bold")
        ax.set_title("")  # clear default title

        if col == 0:
            ylabel = (f"{spec.ylabel} (detrended)" if has_detrend
                      else spec.ylabel)
            ax.set_ylabel(ylabel, fontsize=7)
            legend_handles = [h_raw, h_smooth, h_high, h_low]

        ax.set_xlabel("Time (s)", fontsize=7)
        _style_axis(ax)

    # ── Legend (shared, anchored top-center) ──
    if legend_handles:
        fig.legend(
            legend_handles,
            ["raw signal", "smoothed", "high threshold", "low threshold"],
            loc="upper center", ncol=4, fontsize=5.5,
            frameon=True, framealpha=0.9, edgecolor="0.8",
            handlelength=1.8, columnspacing=1.2,
            bbox_to_anchor=(0.5, 1.06),
        )

    # ── Parameter footer ──
    param_parts = [
        f"artifact k={spec.artifact_k} pad={spec.artifact_pad}",
        f"gauss \u03c3={spec.gauss_sigma}",
    ]
    if has_detrend:
        param_parts.append(
            f"detrend {spec.detrend_window_s:.0f}s q={spec.detrend_quantile}"
        )
    param_parts.extend([
        f"SG({SG_WINDOW},{SG_POLYORDER})",
        f"high={spec.noise_high_k}\u00d7 low={spec.noise_low_k}\u00d7 MAD",
        f"gap<{MAX_GAP_S}s  min dur\u2265{MIN_DURATION_S}s",
    ])
    fig.text(
        0.5, -0.01, "   \u2502   ".join(param_parts),
        ha="center", va="top", fontsize=5.5, family="monospace",
        color="0.45",
    )

    return fig


def plot_group_summary(summary_df: pd.DataFrame, spec: SignalSpec) -> plt.Figure:
    """Bar chart of event counts and mean durations per subject."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    palette = list(spec.palette)
    counts = summary_df.groupby("Subject")["n_events"].sum().sort_index()
    colors = [palette[i % len(palette)] for i in range(len(counts))]

    bars1 = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", lw=0.5)
    axes[0].set_ylabel("Total events")
    axes[0].set_title("Event count per subject")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].bar_label(bars1, fontsize=7, padding=2)
    _style_axis(axes[0])

    mean_dur = summary_df.groupby("Subject")["mean_duration_s"].mean().sort_index()
    bars2 = axes[1].bar(mean_dur.index, mean_dur.values, color=colors, edgecolor="white", lw=0.5)
    axes[1].set_ylabel("Mean duration (s)")
    axes[1].set_title("Mean event duration per subject")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].bar_label(bars2, fmt="%.1f", fontsize=7, padding=2)
    _style_axis(axes[1])

    fig.suptitle(f"{spec.label} event detection — group summary", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_cross_animal_comparison(
    metrics_df: pd.DataFrame, spec: SignalSpec
) -> plt.Figure:
    """
    Cross-animal comparison using z-scored amplitudes and raw durations.

    Three panels:
      1. Peak z-score per subject (box + strip)
      2. Mean z-score per subject (box + strip)
      3. Duration per subject (box + strip, raw seconds)

    Box plots show median + IQR per subject; overlaid points show
    individual events so you can see the spread and any outlier
    structure.
    """
    subjects = sorted(metrics_df["Subject"].unique())
    n_sub = len(subjects)
    palette = list(spec.palette)
    colors = [palette[i % len(palette)] for i in range(n_sub)]
    sub_to_color = dict(zip(subjects, colors))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    panels = [
        ("peak_z", "Peak amplitude (z)", "Peak z-score per subject"),
        ("mean_z", "Mean amplitude (z)", "Mean z-score per subject"),
        ("duration_s", "Duration (s)", "Event duration per subject"),
    ]

    for ax, (col, ylabel, title) in zip(axes, panels):
        # ── Collect data per subject for boxplot ──
        data_per_sub = [
            metrics_df.loc[metrics_df["Subject"] == s, col].dropna().values
            for s in subjects
        ]

        bp = ax.boxplot(
            data_per_sub,
            positions=range(n_sub),
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="white", linewidth=1.5),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
            patch.set_edgecolor(c)

        # ── Overlaid strip of individual events ──
        for i, s in enumerate(subjects):
            vals = metrics_df.loc[metrics_df["Subject"] == s, col].dropna().values
            if len(vals) == 0:
                continue
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter, vals,
                color=sub_to_color[s], alpha=0.35, s=8, edgecolor="none",
                rasterized=True,
            )

        ax.set_xticks(range(n_sub))
        ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _style_axis(ax)

    fig.suptitle(
        f"{spec.label} — cross-animal event comparison (z-scored amplitudes)",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ─── Analysis runner ─────────────────────────────────────────────────────


def run_signal(spec: SignalSpec, proj: Project, all_sessions) -> None:
    """Run single-session test + group analysis for one signal spec."""
    source = spec.source
    signal = spec.signal
    tag = spec.label

    print(f"\n{'═' * 60}")
    print(f"  Signal: {source}/{signal}  ({tag})")
    print(f"{'═' * 60}")

    # ── Discover actual signal name ──
    first_sess = all_sessions[0]
    available_sources = first_sess._available_sources()
    if source not in available_sources:
        print(f"  ✗ Source {source!r} not found. Available: {available_sources}")
        return

    available_signals = first_sess._available_signals(source)
    if signal not in available_signals:
        if available_signals:
            signal = available_signals[0]
            print(f"  Signal {spec.signal!r} not found → using fallback: {signal}")
        else:
            print(f"  ✗ No signals found for source {source!r}")
            return

    # ── Stage 1: Single-session test ──
    print(f"\n── Single-session test ({tag}) ──")
    test_sess = first_sess

    ad = test_sess.align(
        {source: [signal]},
        reference=source,
        tolerance_s=0.25,
        time_column=TIME_COLUMN,
    )
    df_aligned = ad.df.iloc[1:]  # skip first frame (often an alignment artifact)
    t_s = df_aligned[TIME_COLUMN].to_numpy()
    raw_signal = df_aligned[signal].to_numpy()
    trace = preprocess(raw_signal, spec.use_dff, normalize=spec.normalize)

    det = detect_events(
        trace, t_s,
        gauss_sigma=spec.gauss_sigma,
        artifact_k=spec.artifact_k,
        artifact_pad=spec.artifact_pad,
        do_detrend=spec.detrend,
        detrend_window_s=spec.detrend_window_s,
        detrend_quantile=spec.detrend_quantile,
        noise_high_k=spec.noise_high_k,
        noise_low_k=spec.noise_low_k,
    )
    n = len(det["events"])
    durs = det["durations_s"]
    n_artifacts = (np.diff(det["artifact_mask"].astype(int), prepend=0) == 1).sum()
    print(f"  {test_sess.label}")
    print(f"  {source}/{signal} → {len(raw_signal)} samples, {n} events, "
          f"{n_artifacts} artifact regions")
    if durs:
        print(f"  Durations (s): min={min(durs):.2f}, max={max(durs):.2f}, "
              f"mean={np.mean(durs):.2f}")

    stats_dir = proj.stats_dir
    plots_dir = proj.plots_dir

    fig_test = plot_single_session(
        t_s, trace, det,
        title=f"Single-session test — {test_sess.label}\n{source}/{signal}",
        spec=spec,
    )
    fig_test.savefig(plots_dir / f"{tag}_single_session_test.png",
                     dpi=200, bbox_inches="tight")
    plt.close(fig_test)

    # ── Stage 2: All sessions ──
    print(f"\n── Group analysis ({tag}, {len(all_sessions)} sessions) ──")

    summary_rows: list[dict] = []
    all_events: list[pd.DataFrame] = []
    all_metric_rows: list[dict] = []
    comparison_sessions = ("ses-01", "ses-10")
    comparison_data_by_subject: dict[str, dict[str, dict | None]] = {}

    report_pdf = proj.reports_dir / f"{tag}_event_detection.pdf"
    report_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(report_pdf) as pdf:
        for sess in all_sessions:
            try:
                ad = sess.align(
                    {source: [signal]},
                    reference=source,
                    tolerance_s=0.25,
                    time_column=TIME_COLUMN,
                )
                df_aligned = ad.df.iloc[1:]  # skip first frame
                t_s = df_aligned[TIME_COLUMN].to_numpy()
                raw_sig = df_aligned[signal].to_numpy()
            except Exception as exc:
                print(f"  SKIP {sess.label}: {exc}")
                continue

            if len(raw_sig) < 2:
                print(f"  SKIP {sess.label}: too few samples ({len(raw_sig)})")
                continue

            trace = preprocess(raw_sig, spec.use_dff, normalize=spec.normalize)
            det = detect_events(
                trace, t_s,
                gauss_sigma=spec.gauss_sigma,
                artifact_k=spec.artifact_k,
                artifact_pad=spec.artifact_pad,
                do_detrend=spec.detrend,
                detrend_window_s=spec.detrend_window_s,
                detrend_quantile=spec.detrend_quantile,
                noise_high_k=spec.noise_high_k,
                noise_low_k=spec.noise_low_k,
            )
            n_ev = len(det["events"])
            durs = det["durations_s"]
            n_art = (np.diff(det["artifact_mask"].astype(int),
                             prepend=0) == 1).sum()

            print(f"  {sess.label}  → {n_ev} events, {n_art} artifacts")

            # ── Stash data for per-subject ses-01 / ses-10 comparison figures ──
            if sess.session in comparison_sessions:
                subject_comparison = comparison_data_by_subject.setdefault(
                    sess.subject,
                    {"ses-01": None, "ses-10": None},
                )
                if subject_comparison[sess.session] is None:
                    subject_comparison[sess.session] = {
                        "t_s": t_s,
                        "det": det,
                        "trace": trace,
                        "label": sess.label,
                    }

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
                "n_artifacts": n_art,
            })

            # ── Per-event metrics (non-detrended + detrended) ──
            if det["events"]:
                # Derive numeric session index for longitudinal ordering
                sess_num = int(sess.session.replace("ses-", "")) if sess.session.startswith("ses-") else 0
                metric_rows = extract_event_metrics(
                    det, t_s,
                    subject=sess.subject,
                    session=sess.session,
                    task=sess.task,
                    signal_label=f"{source}/{signal}",
                    session_index=sess_num,
                )
                all_metric_rows.extend(metric_rows)

                onsets = np.array([t_s[s] for s, _ in det["events"]])
                offsets = np.array([t_s[min(e, len(t_s) - 1)]
                                    for _, e in det["events"]])
                ev = make_events(
                    {"onset": onsets, "offset": offsets},
                    subject=sess.subject,
                    session=sess.session,
                    task=sess.task,
                )
                all_events.append(ev)

            fig = plot_single_session(
                t_s, trace, det,
                title=f"{sess.label}\n{source}/{signal} — {n_ev} events",
                spec=spec,
            )
            pdf.savefig(fig, dpi=120, bbox_inches="tight")
            plt.close(fig)
            del fig

    print(f"  PDF report: {report_pdf}")

    # ── Save tables ──
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(stats_dir / f"{tag}_event_summary.csv", index=False)

    if all_events:
        events_df = pd.concat(all_events, ignore_index=True)
    else:
        events_df = pd.DataFrame(
            columns=["Subject", "Session", "Task", "EventType", "event_time"]
        )

    events_df.to_csv(stats_dir / f"{tag}_events.csv", index=False)

    # ── Per-event metrics table (z-scored + raw amplitudes) ──
    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_df.to_csv(stats_dir / f"{tag}_event_metrics.csv", index=False)

    n_total = len(events_df) // 2
    n_subjects = events_df["Subject"].nunique() if not events_df.empty else 0
    n_metrics = len(metrics_df)
    print(f"  Total events: {n_total} across {n_subjects} subjects")
    print(f"  Event metrics table: {n_metrics} rows → {tag}_event_metrics.csv")
    if not metrics_df.empty:
        print(f"    peak_z (non-detrended):  mean={metrics_df['peak_z'].mean():.2f}, "
              f"std={metrics_df['peak_z'].std():.2f}")
        print(f"    mean_z (non-detrended):  mean={metrics_df['mean_z'].mean():.2f}, "
              f"std={metrics_df['mean_z'].std():.2f}")

    # ── Group summary plot ──
    if not summary_df.empty and summary_df["n_events"].sum() > 0:
        fig_summary = plot_group_summary(summary_df, spec)
        fig_summary.savefig(plots_dir / f"{tag}_group_summary.png",
                            dpi=200, bbox_inches="tight")
        plt.close(fig_summary)

    # ── Cross-animal comparison plot (z-scored amplitudes) ──
    if not metrics_df.empty and metrics_df["Subject"].nunique() > 1:
        fig_comp = plot_cross_animal_comparison(metrics_df, spec)
        fig_comp.savefig(plots_dir / f"{tag}_cross_animal_comparison.png",
                         dpi=200, bbox_inches="tight")
        plt.close(fig_comp)

    # ── Per-subject comparison: ses-01 vs ses-10 side-by-side ──
    n_comparison_plots = 0
    for subject in sorted(comparison_data_by_subject):
        subject_comparison = comparison_data_by_subject[subject]
        if any(subject_comparison[s] is None for s in comparison_sessions):
            continue

        session_data = [
            subject_comparison["ses-01"],
            subject_comparison["ses-10"],
        ]
        fig_pub = plot_session_comparison(
            session_data,
            spec,
            session_labels=[
                f"{subject} | ses-01",
                f"{subject} | ses-10",
            ],
        )
        out_name = f"{tag}_session_comparison_{subject}.png"
        fig_pub.savefig(plots_dir / out_name, dpi=300, bbox_inches="tight")
        plt.close(fig_pub)
        print(f"  Session comparison figure: {out_name}")
        n_comparison_plots += 1

    if n_comparison_plots == 0:
        print("  Session comparison figure: skipped (no subjects had both ses-01 and ses-10)")

    return n_total, n_subjects


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hysteresis-based event detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Full dataset (all subjects, all sessions)
  python event-detection.py

  # Single subject, all sessions
  python event-detection.py --subject STREHAB02

  # Single subject + session
  python event-detection.py --subject STREHAB02 --session ses-01

  # Explicit task (default: task-widefield)
  python event-detection.py --subject STREHAB02 --session ses-01 --task task-spont
""",
    )
    p.add_argument("--subject", "-s", default=None,
                   help="Filter to a single subject (e.g. STREHAB02)")
    p.add_argument("--session", "-n", default=None,
                   help="Filter to a single session (e.g. ses-01)")
    p.add_argument("--task", "-t", default=None,
                   help="Task filter (default: task-widefield when --subject is set)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

args = parse_args()

proj = Project(
    dataset=DATASET,
    analyst="databench",
    run_name="event-detection",
    tag="hfsa",
).filter(drop_rows=[
    {"Session": "ses-11"},
    {"Session": "ses-00"},
    {"Subject": "STREHAB14", "Session": "ses-01"},
    {"Subject": "STREHAB07", "Session": "ses-08"},
])
# ── Apply CLI filters ──
if args.subject:
    task = args.task or "task-widefield"
    proj = proj.filter(Subject=args.subject, Task=task)
    if args.session:
        proj = proj.filter(Session=args.session)
elif args.task:
    proj = proj.filter(Task=args.task)

all_sessions = proj.sessions()
print(f"Dataset: {DATASET.name} — {len(all_sessions)} sessions")
if args.subject or args.session:
    print(f"  Filtered: subject={args.subject or 'all'}, "
          f"session={args.session or 'all'}, task={args.task or 'task-widefield'}")

results: dict[str, tuple] = {}

for spec in SIGNALS:
    result = run_signal(spec, proj, all_sessions)
    if result is not None:
        results[spec.label] = result

# ─── Combined report ─────────────────────────────────────────────────────

notes_lines = [
    "# tl;dr",
    "",
    "Transient events (mesofield calcium surges, pupil dilations) were detected",
    "using hysteresis thresholding on detrended traces, then peak amplitudes",
    "were measured on the *non-detrended* signal to preserve absolute scale",
    "for longitudinal and cross-animal comparisons in hierarchical models.",
    "",
    "# Methods — Event Detection",
    "",
    "## Preprocessing",
    "",
    "Each signal was first cleaned by an adaptive artifact-rejection step:",
    "the first temporal derivative was computed and frames exceeding",
    "k × MAD (median absolute deviation) were flagged as artifacts, expanded",
    "by a pad of ±N frames, and replaced via linear interpolation from the",
    "flanking clean values.  This removes blink artifacts (pupil) and",
    "motion-induced transients (mesofield) without distorting event shape.",
    "",
    "Mesofield input in this dataset is already ΔF/F and was used directly",
    "(no additional ΔF/F transform).  Pupil diameter was z-scored at the session level.",
    "",
    "## Baseline Detrending (for detection only)",
    "",
    "Slow drift was removed by subtracting a rolling-quantile baseline",
    "(window and quantile set per signal; see table below).  This detrended",
    "trace was used only for event boundary detection — amplitude metrics",
    "were measured on the non-detrended trace (see Dual-trace strategy).",
    "",
    "## Smoothing",
    "",
    "Two smoothing stages were applied:",
    "  1. Gaussian smoothing (σ per-signal, applied to both detrended and",
    "     non-detrended traces for consistency).",
    "  2. Savitzky–Golay filter (window={}, polyorder={}) on the detrended".format(SG_WINDOW, SG_POLYORDER),
    "     trace to produce the final detection trace.",
    "",
    "## Event Detection",
    "",
    "Events were detected via hysteresis thresholding on the smoothed",
    "detrended trace.  Thresholds were set relative to the noise floor",
    "rather than the full signal distribution.  The lower half of the",
    "smoothed trace (values ≤ median) was taken as a proxy for quiet",
    "periods; the MAD of this subset was scaled by 1.4826 to estimate σ̂,",
    "and thresholds were placed at median_quiet + k × σ̂:",
    f"  - High threshold: {NOISE_MAD_HIGH_K}× MAD above noise floor (event onset)",
    f"  - Low threshold:  {NOISE_MAD_LOW_K}× MAD above noise floor (event offset)",
    f"  - Short gaps (<{MAX_GAP_S} s) between events were filled",
    f"  - Events shorter than {MIN_DURATION_S} s were discarded",
    "",
    "Peak timing was defined as the timestamp of the maximum of the",
    "smoothed trace within each event window.",
    "",
    "## Dual-Trace Amplitude Measurement",
    "",
    "This is the key methodological choice for longitudinal analysis.",
    "Event boundaries (onset, offset) were defined on the detrended +",
    "smoothed trace, where slow baseline drift has been removed so that",
    "noise-floor-based thresholds are stable within a session.",
    "",
    "However, amplitude metrics (peak_raw, mean_raw, peak_z, mean_z) were",
    "measured on the *non-detrended* cleaned trace (artifact-interpolated +",
    "Gaussian-smoothed).  This preserves absolute signal scale so that:",
    "  - Within-animal session-to-session changes are not masked by",
    "    per-session baseline subtraction",
    "  - Between-animal comparisons remain meaningful",
    "  - Hierarchical mixed-effects models can partition variance into",
    "    animal-level and session-level components without destructive",
    "    normalization",
    "",
    "Z-scored metrics (peak_z, mean_z) use the session-level mean and SD",
    "of the non-detrended trace.  Detrended-space metrics are stored as",
    "*_detrended columns for diagnostic purposes.",
    "",
    "## Session-Level Covariates",
    "",
    "Each event row includes covariates for downstream modelling:",
    "  - session_baseline_mean: mean of the rolling-quantile baseline,",
    "    capturing gross session-to-session signal level shifts",
    "  - recording_duration_s: total recording length",
    "  - event_time_frac: fractional position of the event within the",
    "    recording (0 = start, 1 = end), for within-session trend control",
    "  - session_index: ordinal session number for longitudinal ordering",
    "",
    "## Exclusions",
    "",
    "Sessions excluded prior to analysis:",
    "  - ses-11, ses-00 (all subjects): non-standard protocol",
    "  - STREHAB14/ses-01: insufficient signal quality",
    "  - STREHAB07/ses-08: hardware failure",
    "",
    "## Signal-Specific Parameters",
    "",
]

for spec in SIGNALS:
    r = results.get(spec.label)
    detrend_note = (
        f"detrend_window={spec.detrend_window_s}s, "
        f"quantile={spec.detrend_quantile}"
    ) if spec.detrend else "detrend=off"
    notes_lines.extend([
        f"### {spec.label} ({spec.source}/{spec.signal})",
        f"  - Normalization: {spec.normalize}"
        + (", ΔF/F" if spec.use_dff else ""),
        f"  - Gaussian σ: {spec.gauss_sigma} samples",
        f"  - Artifact rejection: k={spec.artifact_k}, pad={spec.artifact_pad}",
        f"  - Baseline: {detrend_note}",
        f"  - Thresholds: noise_high_k={spec.noise_high_k}, noise_low_k={spec.noise_low_k}",
    ])
    if r:
        notes_lines.append(
            f"  - Result: {r[0]} events across {r[1]} subjects"
        )
    else:
        notes_lines.append("  - Result: skipped or no data")
    notes_lines.append("")

notes_lines.extend([
    f"Total sessions analyzed: {len(all_sessions)}",
])

proj.io.report(notes="\n".join(notes_lines))

print("\nDone.")