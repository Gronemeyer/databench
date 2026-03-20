"""Oscillation plotting helpers — used by OscillationResult, not called by users."""
from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from databench._plotting.trace_config import get_style
from databench._plotting.traces import (
    prepare_trace_styled,
    plot_trace_styled,
    time_mask,
)


# ── Oscillation-specific utilities ─────────────────────────────────────────

def shade_bursts(
    ax: plt.Axes,
    t: np.ndarray,
    bursts: list[Tuple[int, int]],
    t_lo: float | None = None,
    t_hi: float | None = None,
) -> None:
    """Add translucent orange spans for each detected burst."""
    for s, e in bursts:
        ts, te = t[s], t[e]
        if t_lo is not None and te < t_lo:
            continue
        if t_hi is not None and ts > t_hi:
            continue
        ax.axvspan(ts, te, color="#ff7f0e", alpha=0.12)


# ── Overview Plot ──────────────────────────────────────────────────────────

def plot_oscillation_overview(
    *,
    time: np.ndarray,
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    envelope: np.ndarray,
    threshold_value: float,
    bursts: list[Tuple[int, int]],
    signal_name: str,
    band_hz: Tuple[float, float],
    subject: str,
    session: str,
    task: str,
    aligned_time: np.ndarray | None = None,
    pupil: np.ndarray | None = None,
    speed: np.ndarray | None = None,
    speed_time: np.ndarray | None = None,
    speed_values: np.ndarray | None = None,
    smooth_pupil_s: float = 0.5,
    window: Tuple[float, float] | None = None,
) -> plt.Figure:
    """Create a multi-panel oscillation overview figure.

    Panels: raw signal, bandpassed+envelope, optional pupil, optional speed.
    All with burst shading.
    """
    est_fs = band_hz[0] * 12.5  # rough estimate for smoothing

    # ── Prepare pupil via TraceStyle ──────────────────────────────────
    if pupil is not None and len(pupil) > 0:
        pupil_style = get_style("pupil", smooth_savgol_s=smooth_pupil_s)
        pupil = prepare_trace_styled(
            aligned_time if aligned_time is not None else time,
            None,  # raw method — already on aligned timebase
            pupil,
            pupil_style,
            est_fs=est_fs,
        )

    # ── Prepare speed via TraceStyle ──────────────────────────────────
    if speed_time is not None and speed_values is not None and len(speed_values) > 0:
        # Preferred path: raw treadmill remapped via config-driven style
        tread_style = get_style("treadmill")
        speed = prepare_trace_styled(
            aligned_time if aligned_time is not None else time,
            speed_time,
            speed_values,
            tread_style,
        )

    has_pupil = pupil is not None and len(pupil) > 0
    has_speed = speed is not None and len(speed) > 0
    n_panels = 2 + int(has_pupil) + int(has_speed)

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(12.8, 2.8 * n_panels),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    t = time
    _, sl = time_mask(t, window)
    if sl is None:
        return fig
    ts = t[sl]
    xlim = (ts[0], ts[-1])

    band_label = f"{band_hz[0]}–{band_hz[1]} Hz"
    label = f"Subject={subject} | Session={session} | Task={task}"

    # Panel 1: raw ROI trace + burst shading
    ax = axes[0]
    ax.plot(ts, raw_signal[sl], color="#1f77b4", lw=1.4, alpha=0.9)
    shade_bursts(ax, t, bursts, *xlim)
    ax.set_ylabel("ΔF/F", fontsize=10)
    ax.set_title(
        f"{label} | {signal_name} | {band_label}\n"
        f"{len(bursts)} bursts detected",
        fontsize=10,
    )
    ax.tick_params(labelsize=8)

    # Panel 2: bandpassed + envelope + threshold
    ax = axes[1]
    ax.plot(ts, filtered_signal[sl], color="#2ca02c", lw=0.8, alpha=0.8, label="bandpassed")
    ax.plot(ts, envelope[sl], color="#d62728", lw=1.2, alpha=0.9, label="envelope")
    ax.axhline(
        threshold_value, color="#d62728", ls="--", lw=0.9, alpha=0.6,
        label=f"threshold ({threshold_value:.4f})",
    )
    shade_bursts(ax, t, bursts, *xlim)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.tick_params(labelsize=9)

    panel_idx = 2

    # Panel 3: pupil
    if has_pupil and aligned_time is not None:
        ax = axes[panel_idx]
        pupil_style = get_style("pupil")
        plot_trace_styled(ax, aligned_time, pupil, pupil_style, window=xlim)
        shade_bursts(ax, t, bursts, *xlim)
        panel_idx += 1

    # Panel 4: speed
    if has_speed and aligned_time is not None:
        ax = axes[panel_idx]
        tread_style = get_style("treadmill")
        plot_trace_styled(ax, aligned_time, speed, tread_style, window=xlim)
        shade_bursts(ax, t, bursts, *xlim)

    for a in axes:
        a.set_xlim(xlim)
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.tight_layout()
    return fig


# ── Burst Detail Plot ──────────────────────────────────────────────────────

def plot_oscillation_burst(
    *,
    time: np.ndarray,
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    envelope: np.ndarray,
    threshold_value: float,
    bursts: list[Tuple[int, int]],
    signal_name: str,
    band_hz: Tuple[float, float],
    subject: str,
    session: str,
    task: str,
    burst_idx: int,
    fs: float,
    pad_s: float = 5.0,
    aligned_time: np.ndarray | None = None,
    pupil: np.ndarray | None = None,
    speed: np.ndarray | None = None,
    speed_time: np.ndarray | None = None,
    speed_values: np.ndarray | None = None,
    smooth_pupil_s: float = 0.5,
) -> plt.Figure:
    """Plot a zoomed window around a single burst."""
    s, e = bursts[burst_idx]
    t_start = time[s] - pad_s
    t_end = time[e] + pad_s
    window = (t_start, t_end)

    fig = plot_oscillation_overview(
        time=time,
        raw_signal=raw_signal,
        filtered_signal=filtered_signal,
        envelope=envelope,
        threshold_value=threshold_value,
        bursts=bursts,
        signal_name=signal_name,
        band_hz=band_hz,
        subject=subject,
        session=session,
        task=task,
        aligned_time=aligned_time,
        pupil=pupil,
        speed=speed,
        speed_time=speed_time,
        speed_values=speed_values,
        smooth_pupil_s=smooth_pupil_s,
        window=window,
    )
    burst_dur = (e - s + 1) / fs
    peak_env = float(envelope[s : e + 1].max())
    fig.suptitle(
        f"Burst #{burst_idx + 1}\n"
        f"{time[s]:.1f}–{time[e]:.1f} s  (dur={burst_dur:.1f}s, peak_env={peak_env:.4f})",
        fontsize=10,
        y=1.03,
    )
    return fig
