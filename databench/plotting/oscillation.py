"""Oscillation detection plotting helpers.

Provides:
  * ``shade_bursts`` — translucent burst overlays on an axes.
  * ``plot_oscillation_overview`` — multi-panel oscillation overview figure.
  * ``plot_oscillation_burst`` — zoomed window around a single burst.
"""
from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from databench.utils.logger import get_logger
from databench.signal.preproc import smooth_savgol
from databench.utils import time_mask
from databench.plotting.traces import dense_lw, prepare_trace_styled, plot_trace_styled
from databench.plotting import get_theme, style_axes
from databench.plotting.style import get_style

_log = get_logger(__name__)


# ── Oscillation-specific plotting (moved from analysis.oscillation) ────────

def shade_bursts(
    ax: plt.Axes,
    t: np.ndarray,
    bursts: list[Tuple[int, int]],
    t_lo: float | None = None,
    t_hi: float | None = None,
    *,
    color: str | None = None,
    alpha: float = 0.18,
) -> None:
    """Add translucent spans for each detected burst."""
    if color is None:
        color = get_theme().colors[4]  # accent/burst colour from theme
    for s, e in bursts:
        ts, te = t[s], t[e]
        if t_lo is not None and te < t_lo:
            continue
        if t_hi is not None and ts > t_hi:
            continue
        ax.axvspan(ts, te, color=color, alpha=alpha, lw=0)


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
    smooth_pupil_s: float = 0.5,
    smooth_speed_s: float = 0.2,
    window: Tuple[float, float] | None = None,
) -> plt.Figure:
    """Create a multi-panel oscillation overview figure.

    Panels: raw signal, bandpassed+envelope, optional pupil, optional speed.
    All with burst shading.
    """
    est_fs = band_hz[0] * 12.5  # rough estimate for smoothing

    # Prepare pupil via TraceStyle (already aligned by session.align)
    if pupil is not None and len(pupil) > 0:
        pupil_style = get_style("pupil", smooth_savgol_s=smooth_pupil_s)
        pupil = prepare_trace_styled(
            aligned_time if aligned_time is not None else time,
            None,
            pupil,
            pupil_style,
            est_fs=est_fs,
        )

    # Prepare speed via TraceStyle (already aligned by session.align)
    if speed is not None and len(speed) > 0:
        speed = np.where(np.isnan(speed), 0.0, speed)
        tread_style = get_style("treadmill", method="raw", smooth_savgol_s=smooth_speed_s)
        speed = prepare_trace_styled(
            aligned_time if aligned_time is not None else time,
            None,
            speed,
            tread_style,
            est_fs=est_fs,
        )

    has_pupil = pupil is not None and len(pupil) > 0
    has_speed = speed is not None and len(speed) > 0
    n_panels = 2 + int(has_pupil) + int(has_speed)

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(7.5, 1.8 * n_panels),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    t = time
    _, sl = time_mask(t, window)
    if sl is None:
        return fig
    ts = t[sl]
    xlim = (ts[0], ts[-1])
    n_vis = len(ts)
    fig_w = fig.get_size_inches()[0]

    band_label = f"{band_hz[0]}\u2013{band_hz[1]} Hz"
    roi_style = get_style("roi")
    filt_style = get_style("filtered")
    env_style = get_style("envelope")

    # Panel 1: raw ROI trace + burst shading
    ax = axes[0]
    ax.plot(ts, raw_signal[sl], color=roi_style.color,
            lw=dense_lw(n_vis, fig_w, roi_style.lw), alpha=roi_style.alpha)
    shade_bursts(ax, t, bursts, *xlim)
    ax.set_ylabel("\u0394F/F")
    ax.set_title(
        f"{subject} | {session} | {task}  \u2014  {signal_name}  {band_label}"
        f"  ({len(bursts)} bursts)",
        fontweight="medium", pad=6,
    )
    style_axes(ax)

    # Panel 2: bandpassed + envelope + threshold
    ax = axes[1]
    ax.plot(ts, filtered_signal[sl], color=filt_style.color,
            lw=dense_lw(n_vis, fig_w, filt_style.lw),
            alpha=filt_style.alpha, label="bandpassed")
    ax.plot(ts, envelope[sl], color=env_style.color,
            lw=dense_lw(n_vis, fig_w, env_style.lw),
            alpha=env_style.alpha, label="envelope")
    ax.axhline(
        threshold_value, color=env_style.color, ls="--", lw=0.7, alpha=0.5,
        label=f"threshold ({threshold_value:.4f})",
    )
    shade_bursts(ax, t, bursts, *xlim)
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", frameon=False, ncol=3, handlelength=1.5)
    style_axes(ax)

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
        ax.set_ylim(bottom=0)

    for a in axes:
        a.set_xlim(xlim)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(h_pad=0.4)
    return fig


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
    smooth_pupil_s: float = 0.5,
    smooth_speed_s: float = 0.2,
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
        smooth_pupil_s=smooth_pupil_s,
        smooth_speed_s=smooth_speed_s,
        window=window,
    )
    burst_dur = (e - s + 1) / fs
    peak_env = float(envelope[s : e + 1].max())
    fig.suptitle(
        f"Burst #{burst_idx + 1}  \u2014  "
        f"{time[s]:.1f}\u2013{time[e]:.1f} s  "
        f"(dur = {burst_dur:.1f} s, peak envelope = {peak_env:.4f})",
        fontweight="medium",
        y=1.02,
    )
    return fig


# ─── Overview Plotter ─────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Tuple as _Tuple

from databench.plotting.base import Plotter


@dataclass(frozen=True)
class OscillationOverviewPlotter(Plotter):
    """Reusable Plotter for oscillation-detection overviews.

    Parameters
    ----------
    pupil, speed : str, optional
        Column names in the aligned auxiliary data.
    smooth_pupil_s, smooth_speed_s : float
        Smoothing windows in seconds.
    window : (float, float), optional
        Time window to zoom into; ``None`` shows full session.
    aligned : AlignedData, optional
        Pre-aligned auxiliary data (pupil/speed traces).  Pass via
        ``OscillationResult.overview_plotter(aligned=...)`` rather than
        constructing the plotter directly.
    """

    name: str = "oscillation_overview"
    pupil: Optional[str] = None
    speed: Optional[str] = None
    smooth_pupil_s: float = 0.5
    smooth_speed_s: float = 0.2
    window: Optional[_Tuple[float, float]] = None
    aligned: object = field(default=None, repr=False, compare=False)

    def plot(self, result):
        return result.plot_overview(
            aligned=self.aligned,
            pupil=self.pupil,
            speed=self.speed,
            smooth_pupil_s=self.smooth_pupil_s,
            smooth_speed_s=self.smooth_speed_s,
            window=self.window,
        )

    def recipe(self, result=None) -> dict:
        rec = super().recipe(result)
        # Drop the non-serialisable AlignedData reference.
        rec["config"].pop("aligned", None)
        if result is not None:
            rec["result"] = {
                "subject": getattr(result, "subject", None),
                "session": getattr(result, "session", None),
                "task": getattr(result, "task", None),
                "signal": getattr(result, "signal_name", None),
                "band_hz": list(getattr(result, "band_hz", ()) or ()),
                "fs": getattr(result, "fs", None),
                "threshold": getattr(result, "threshold_value", None),
                "n_bursts": len(getattr(result, "bursts", []) or []),
            }
        return rec


@dataclass(frozen=True)
class OscillationBurstPlotter(Plotter):
    """Reusable Plotter that renders zoomed burst windows."""

    name: str = "oscillation_bursts"
    max_examples: int = 12
    pad_s: float = 5.0
    fs: Optional[float] = None
    pupil: Optional[str] = None
    speed: Optional[str] = None
    smooth_pupil_s: float = 0.5
    smooth_speed_s: float = 0.2
    aligned: object = field(default=None, repr=False, compare=False)

    def plot(self, result):
        return result.plot_bursts(
            aligned=self.aligned,
            max_examples=self.max_examples,
            pad_s=self.pad_s,
            fs=self.fs,
            pupil=self.pupil,
            speed=self.speed,
            smooth_pupil_s=self.smooth_pupil_s,
            smooth_speed_s=self.smooth_speed_s,
        )

    def recipe(self, result=None) -> dict:
        rec = super().recipe(result)
        rec["config"].pop("aligned", None)
        return rec

