"""Oscillation detection plotting helpers and registered plotters.

Provides:
  * ``smooth_savgol`` — Savitzky-Golay smoothing utility.
  * ``shade_bursts`` — translucent burst overlays on an axes.
  * ``time_mask`` — boolean mask + index slice for a time window.
  * ``OscillationOverviewPlotter`` — full-session (or windowed) 4-panel overview.
  * ``OscillationBurstDetailPlotter`` — zoomed window around one burst.
  * ``OscillationReportPagePlotter`` — PDF-ready page (raw + envelope + optional pupil).
  * ``OscillationEtaGroupPlotter`` — group mean ± SEM ETA by ROI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from databench.analysis import AnalysisResult
from databench.analysis.oscillation_detector import OscillationResult
from databench.plotting.base import Plotter
from databench.registry import register_plotter
from databench._utils._logger import get_logger

_LOG = get_logger("databench.plotting.oscillation")


# ─── Utility helpers ──────────────────────────────────────────────────────

def smooth_savgol(signal: np.ndarray, window_s: float, fs: float, polyorder: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay smoothing. Returns original if window is too short."""
    if signal is None or len(signal) == 0:
        return signal
    wl = int(round(window_s * fs))
    if wl % 2 == 0:
        wl += 1
    wl = max(3, wl)
    if len(signal) > wl:
        return savgol_filter(signal, window_length=wl, polyorder=polyorder)
    return signal


def shade_bursts(ax, t: np.ndarray, bursts: list, t_lo: float | None = None, t_hi: float | None = None) -> None:
    """Add translucent orange spans for each detected burst."""
    for s, e in bursts:
        ts, te = t[s], t[e]
        if t_lo is not None and te < t_lo:
            continue
        if t_hi is not None and ts > t_hi:
            continue
        ax.axvspan(ts, te, color="#ff7f0e", alpha=0.12)


def time_mask(t: np.ndarray, window: Tuple[float, float] | None):
    """Boolean mask + index slice for a time window. Returns (None, None) if empty."""
    if t is None or len(t) == 0:
        return None, None
    if window is None:
        return np.ones(len(t), dtype=bool), slice(None)
    m = (t >= window[0]) & (t <= window[1])
    idx = np.where(m)[0]
    if len(idx) == 0:
        return None, None
    return m, slice(idx[0], idx[-1] + 1)


# ─── Overview Plotter ─────────────────────────────────────────────────────

@register_plotter
@dataclass(frozen=True)
class OscillationOverviewPlotter(Plotter):
    """Full-session (or windowed) overview: ROI, Hilbert, pupil, speed.

    Accepts an ``AnalysisResult`` whose ``data`` is an ``OscillationResult``
    and whose ``context`` is the *long* DataFrame (aligned by ``build_long``).

    Alternatively, call ``plot_overview()`` directly with explicit args.
    """

    name: str = "oscillation_overview"
    pupil_col: str = "pupil_diameter_mm"
    speed_col: str = "speed_mm"
    smooth_pupil_s: float = 0.5
    smooth_speed_s: float = 0.2
    band_label: str = ""
    cfg_summary: str = ""
    window: Tuple[float, float] | None = None

    def plot(self, result: AnalysisResult):
        """Plot from AnalysisResult (data=OscillationResult, context=long df)."""
        osc: OscillationResult = result.data
        long: pd.DataFrame = result.context
        return self.plot_overview(osc, long, window=self.window)

    def plot_overview(
        self,
        osc: OscillationResult,
        long: pd.DataFrame,
        window: Tuple[float, float] | None = None,
    ):
        t_aligned = long["time_elapsed_s"].to_numpy()
        pupil_raw = long[self.pupil_col].to_numpy() if self.pupil_col in long.columns else None
        speed_raw = long[self.speed_col].to_numpy() if self.speed_col in long.columns else None

        pupil = smooth_savgol(pupil_raw, self.smooth_pupil_s, osc.band[0] * 12.5) if pupil_raw is not None else None
        speed = smooth_savgol(speed_raw, self.smooth_speed_s, osc.band[0] * 12.5) if speed_raw is not None else None

        has_pupil = pupil is not None and len(pupil) > 0
        has_speed = speed is not None and len(speed) > 0
        n_panels = 2 + int(has_pupil) + int(has_speed)

        per_panel_h = 2.8
        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(12.8, per_panel_h * n_panels),
            sharex=True,
            gridspec_kw={"height_ratios": [1] * n_panels},
        )
        axes = np.atleast_1d(axes)

        t = osc.t
        _, sl = time_mask(t, window)
        ts = t[sl]
        xlim = (ts[0], ts[-1])

        band_label = self.band_label or f"{osc.band[0]}–{osc.band[1]} Hz"

        # Panel 1: raw ROI trace + burst shading
        ax = axes[0]
        ax.plot(ts, osc.x[sl], color="#1f77b4", lw=1.4, alpha=0.9)
        shade_bursts(ax, t, osc.bursts, *xlim)
        ax.set_ylabel("ΔF/F", fontsize=10)
        ax.set_title(
            f"{osc.context.label()} | {osc.context.signal_key} | {band_label}\n"
            f"{len(osc.bursts)} bursts{' (' + self.cfg_summary + ')' if self.cfg_summary else ''}",
            fontsize=10,
        )
        ax.tick_params(labelsize=8)

        # Panel 2: bandpassed + envelope + threshold
        ax = axes[1]
        ax.plot(ts, osc.xf[sl], color="#2ca02c", lw=0.8, alpha=0.8, label="bandpassed")
        ax.plot(ts, osc.env[sl], color="#d62728", lw=1.2, alpha=0.9, label="envelope")
        ax.axhline(osc.thr, color="#d62728", ls="--", lw=0.9, alpha=0.6,
                    label=f"threshold ({osc.thr:.4f})")
        shade_bursts(ax, t, osc.bursts, *xlim)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.01, 0.98), ncol=1, frameon=False)
        ax.tick_params(labelsize=9)

        panel_idx = 2

        # Panel 3: pupil
        if has_pupil:
            ax = axes[panel_idx]
            pm, _ = time_mask(t_aligned, (xlim[0], xlim[1]))
            if pm is not None:
                ax.plot(t_aligned[pm], pupil[pm], color="#EF553B", lw=1.6, alpha=0.9)
            shade_bursts(ax, t, osc.bursts, *xlim)
            ax.set_ylabel("Pupil (mm)", fontsize=9)
            ax.tick_params(labelsize=9)
            panel_idx += 1

        # Panel 4: locomotion speed
        if has_speed:
            ax = axes[panel_idx]
            sm, _ = time_mask(t_aligned, (xlim[0], xlim[1]))
            if sm is not None:
                ax.plot(t_aligned[sm], speed[sm], color="#00CC96", lw=1.6, alpha=0.9)
            shade_bursts(ax, t, osc.bursts, *xlim)
            ax.set_ylabel("Speed (mm)", fontsize=9)
            ax.tick_params(labelsize=9)

        for a in axes:
            a.set_xlim(xlim)
        axes[-1].set_xlabel("Time (s)", fontsize=9)
        fig.tight_layout()
        return fig, axes


# ─── Burst Detail Plotter ────────────────────────────────────────────────

@register_plotter
@dataclass(frozen=True)
class OscillationBurstDetailPlotter(Plotter):
    """Zoomed window around a single burst, delegates to OscillationOverviewPlotter.

    Accepts an ``AnalysisResult`` whose ``data`` is an ``OscillationResult``
    and whose ``context`` is the *long* DataFrame.
    """

    name: str = "oscillation_burst_detail"
    overview: OscillationOverviewPlotter = field(default_factory=lambda: OscillationOverviewPlotter())
    pad_s: float = 5.0
    burst_idx: int = 0
    fs: float = 50.0

    def plot(self, result: AnalysisResult):
        """Plot from AnalysisResult (data=OscillationResult, context=long df)."""
        osc: OscillationResult = result.data
        long: pd.DataFrame = result.context
        return self.plot_burst(osc, long, self.burst_idx, self.fs)

    def plot_burst(
        self,
        osc: OscillationResult,
        long: pd.DataFrame,
        burst_idx: int,
        fs: float = 50.0,
    ):
        s, e = osc.bursts[burst_idx]
        t_start = osc.t[s] - self.pad_s
        t_end = osc.t[e] + self.pad_s
        window = (t_start, t_end)

        fig, axes = self.overview.plot_overview(osc, long, window=window)
        burst_dur = (e - s + 1) / fs
        peak_env = float(osc.env[s:e + 1].max())
        fig.suptitle(
            f"Burst #{burst_idx + 1} — {osc.context.label()}\n"
            f"{osc.t[s]:.1f}–{osc.t[e]:.1f} s  (dur={burst_dur:.1f}s, peak_env={peak_env:.4f})",
            fontsize=10, y=1.03,
        )
        return fig, axes


# ─── Report Page Plotter (for PDF) ───────────────────────────────────────

@register_plotter
@dataclass(frozen=True)
class OscillationReportPagePlotter(Plotter):
    """One PDF page: raw signal, bandpassed + envelope + threshold, optional pupil.

    Accepts either:
    - An ``AnalysisResult`` with ``data=OscillationResult`` and optional
      ``context={"pupil": array, "pupil_time": array}``
    - Direct call via ``plot_page()`` with explicit args.

    Reads pupil directly from the raw row (own time base) — no resampling.
    """

    name: str = "oscillation_report_page"
    pupil_source: str = "pupil"
    pupil_key: str = "pupil_diameter_mm"
    band_label: str = ""
    cfg_summary: str = ""
    time_window: Tuple[float, float] | None = None

    def plot(self, result: AnalysisResult):
        """Plot from AnalysisResult (data=OscillationResult)."""
        osc: OscillationResult = result.data
        ctx = result.context or {}
        pupil = ctx.get("pupil") if isinstance(ctx, dict) else None
        pupil_time = ctx.get("pupil_time") if isinstance(ctx, dict) else None
        return self.plot_page(osc, pupil=pupil, pupil_time=pupil_time, time_window=self.time_window)

    def plot_page(
        self,
        osc: OscillationResult,
        pupil: np.ndarray | None = None,
        pupil_time: np.ndarray | None = None,
        time_window: Tuple[float, float] | None = None,
    ):
        t, x, xf, env, thr = osc.t, osc.x, osc.xf, osc.env, osc.thr
        ctx = osc.context
        band_label = self.band_label or f"{osc.band[0]}–{osc.band[1]} Hz"

        if time_window is not None:
            mask = (t >= time_window[0]) & (t <= time_window[1])
        else:
            mask = np.ones(len(t), dtype=bool)

        has_pupil = (
            pupil is not None
            and pupil_time is not None
            and len(pupil) > 0
            and len(pupil_time) > 0
        )
        n_panels = 3 if has_pupil else 2
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.2 * n_panels))

        tm = t[mask]

        # Panel 1: raw signal + burst shading
        ax0 = axes[0]
        ax0.plot(tm, x[mask], color="#1f77b4", lw=0.6, alpha=0.8)
        for s, e in osc.bursts:
            if t[s] > tm[-1] or t[e] < tm[0]:
                continue
            ax0.axvspan(t[s], t[e], color="#ff7f0e", alpha=0.15)
        ax0.set_ylabel(f"{ctx.signal_key} (raw)", fontsize=9)
        ax0.set_title(
            f"{ctx.label()} | {ctx.signal_key} | {band_label}\n"
            f"{len(osc.bursts)} bursts detected{' (' + self.cfg_summary + ')' if self.cfg_summary else ''}",
            fontsize=10,
        )
        ax0.tick_params(axis="both", labelsize=8)

        # Panel 2: band-passed + envelope + threshold
        ax1 = axes[1]
        ax1.plot(tm, xf[mask], color="#2ca02c", lw=0.5, alpha=0.7, label="bandpassed")
        ax1.plot(tm, env[mask], color="#d62728", lw=0.8, label="envelope")
        ax1.axhline(thr, color="#d62728", ls="--", lw=0.8, alpha=0.6, label=f"threshold ({thr:.4f})")
        for s, e in osc.bursts:
            if t[s] > tm[-1] or t[e] < tm[0]:
                continue
            ax1.axvspan(t[s], t[e], color="#ff7f0e", alpha=0.15)
        ax1.set_ylabel("Amplitude", fontsize=9)
        ax1.legend(fontsize=7, loc="upper right", frameon=False)
        ax1.tick_params(axis="both", labelsize=8)

        # Panel 3 (optional): pupil trace on its own time axis
        if has_pupil:
            ax2 = axes[2]
            if time_window is not None:
                pmask = (pupil_time >= time_window[0]) & (pupil_time <= time_window[1])
            else:
                pmask = np.ones(len(pupil_time), dtype=bool)
            ax2.plot(pupil_time[pmask], pupil[pmask], color="#9467bd", lw=0.6, alpha=0.8)
            for s, e in osc.bursts:
                if t[s] > tm[-1] or t[e] < tm[0]:
                    continue
                ax2.axvspan(t[s], t[e], color="#ff7f0e", alpha=0.15)
            ax2.set_ylabel("Pupil diameter (mm)", fontsize=9)
            ax2.tick_params(axis="both", labelsize=8)

        # Sync x-limits across all panels
        xlim = (tm[0], tm[-1])
        for ax in axes:
            ax.set_xlim(xlim)
        axes[-1].set_xlabel("Time (s)", fontsize=9)
        fig.tight_layout()
        return fig, axes


# ─── ETA Group Plotter ───────────────────────────────────────────────────

@register_plotter
@dataclass(frozen=True)
class OscillationEtaGroupPlotter(Plotter):
    """Group mean ± SEM ETA for each ROI at oscillation burst onset or offset.

    Accepts an ``AnalysisResult`` (auto-supplied by ``bench.plot()``).
    Extracts ``eta_group`` from ``result.data["eta_group"]``, matching the
    ``EtaConditionPlotter`` contract.
    """

    name: str = "oscillation_eta_group"
    roi_cols: tuple[str, ...] = ()
    task: str = "task-widefield"
    event_type: str = "onset"
    band_label: str = ""
    baseline_label: str = ""
    detect_roi: str = ""
    colors: dict[str, str] = field(default_factory=dict)
    ylabels: dict[str, str] = field(default_factory=dict)

    def plot(self, result: AnalysisResult):
        eta_group = result.data["eta_group"]
        event_type = self.event_type
        rois = list(self.roi_cols)
        d = eta_group.query(
            "Task == @self.task and EventType == @event_type and ROI in @rois"
        ).copy()

        ncols = len(rois)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 3.5), sharex=True)
        axes = np.atleast_1d(axes).ravel()

        default_colors = {rois[0]: "#1f77b4"} if rois else {}
        if len(rois) > 1:
            default_colors[rois[-1]] = "#9467bd"
        colors = {**default_colors, **self.colors}

        default_ylabels = {r: "Value (baselined)" for r in rois}
        ylabels = {**default_ylabels, **self.ylabels}

        for ax, roi in zip(axes, rois):
            g = d[d["ROI"] == roi].sort_values("rel_time")
            c = colors.get(roi, "#1f77b4")
            if not g.empty:
                ax.plot(g["rel_time"], g["mean"], color=c, lw=1.8)
                ax.fill_between(
                    g["rel_time"],
                    g["mean"] - g["sem"],
                    g["mean"] + g["sem"],
                    alpha=0.2, color=c,
                )
            ax.axvline(0, color="k", lw=1)
            ax.axhline(0, color="k", lw=0.5, alpha=0.5)
            ax.set_title(roi, fontsize=11)
            ax.set_xlabel(f"Time relative to burst {self.event_type} (s)", fontsize=9)
            ax.set_ylabel(ylabels.get(roi, "Value (baselined)"), fontsize=9)
            ax.tick_params(axis="both", labelsize=8)

        band_label = self.band_label or ""
        detect_roi = self.detect_roi or (rois[0] if rois else "")
        fig.suptitle(
            f"Oscillation-triggered ETA — {band_label} burst {self.event_type}\n"
            f"detect: {detect_roi} | comparison: {', '.join(rois)}\n"
            f"{self.baseline_label} | mean ± SEM across sessions",
            y=1.05, fontsize=11,
        )
        fig.tight_layout()
        return fig, axes
