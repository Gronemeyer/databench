"""Minimal plotting style helpers for publication-ready figures.

This module intentionally keeps styling explicit and conservative.
The public API remains stable for existing scripts.
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.ndimage import gaussian_filter1d
from typing import Optional, List, cast


# ════════════════════════════════════════════════════════
#  PALETTE
# ════════════════════════════════════════════════════════

_LIGHT = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#7f7f7f",
]

_DARK = [
    "#8fbce6",
    "#f2a4a4",
    "#9ed89e",
    "#f2c28b",
    "#c5b0d5",
    "#c0c0c0",
]

_THEMES = {
    "light": dict(
        cycle      = _LIGHT,
        bg         = "#FFFFFF",
        surface    = "#FFFFFF",
        fg         = "#111111",
        tick       = "#333333",
        shade      = "#E6E6E6",
        lbl_bg     = "#FFFFFF",
        bar_hi     = "#4D4D4D",
        bar_lo     = "#C7C7C7",
        dot_color  = "#111111",
        dot_alpha  = 0.35,
    ),
    "dark": dict(
        cycle      = _DARK,
        bg         = "#1A1A1A",
        surface    = "#1A1A1A",
        fg         = "#E6E6E6",
        tick       = "#B8B8B8",
        shade      = "#2A2A2A",
        lbl_bg     = "#1A1A1A",
        bar_hi     = "#CFCFCF",
        bar_lo     = "#5A5A5A",
        dot_color  = "#E6E6E6",
        dot_alpha  = 0.45,
    ),
}


# ════════════════════════════════════════════════════════
#  THEME
# ════════════════════════════════════════════════════════

class Theme:
    """Color theme with apply() to push rcParams globally."""

    def __init__(self, mode: str = "light"):
        if mode not in ("light", "dark"):
            raise ValueError(f"mode must be 'light' or 'dark', got {mode!r}")
        self.mode = mode
        self.p    = _THEMES[mode]

    @property
    def colors(self) -> List[str]:
        return cast(List[str], self.p["cycle"])

    @property
    def bg(self) -> str:
        return cast(str, self.p["bg"])

    @property
    def surface(self) -> str:
        return cast(str, self.p["surface"])

    @property
    def fg(self) -> str:
        return cast(str, self.p["fg"])

    @property
    def shade(self) -> str:
        return cast(str, self.p["shade"])

    @property
    def lbl_bg(self) -> str:
        return cast(str, self.p["lbl_bg"])

    @property
    def bar_hi(self) -> str:
        return cast(str, self.p["bar_hi"])

    @property
    def bar_lo(self) -> str:
        return cast(str, self.p["bar_lo"])

    @property
    def dot_color(self) -> str:
        return cast(str, self.p["dot_color"])

    @property
    def dot_alpha(self) -> float:
        return cast(float, self.p["dot_alpha"])

    def apply(self) -> "Theme":
        """Push this theme's colors and sizing into mpl.rcParams."""
        p = self.p
        mpl.rcParams.update({
            "figure.facecolor":      p["bg"],
            "figure.dpi":            150,
            "axes.facecolor":        p["surface"],
            "axes.edgecolor":        p["tick"],
            "axes.linewidth":        0.8,
            "axes.spines.top":       False,
            "axes.spines.right":     False,
            "axes.labelcolor":       p["fg"],
            "axes.labelsize":        8,
            "axes.labelpad":         4,
            "axes.titlesize":        9,
            "axes.titleweight":      "normal",
            "axes.titlepad":         6,
            "axes.prop_cycle":       cycler(color=cast(List[str], p["cycle"])),
            "axes.axisbelow":        True,
            "xtick.color":           p["tick"],
            "ytick.color":           p["tick"],
            "xtick.labelsize":       7,
            "ytick.labelsize":       7,
            "xtick.major.size":      3.0,
            "ytick.major.size":      3.0,
            "xtick.minor.size":      1.8,
            "ytick.minor.size":      1.8,
            "xtick.major.width":     0.8,
            "ytick.major.width":     0.8,
            "xtick.direction":       "out",
            "ytick.direction":       "out",
            "xtick.major.pad":       3,
            "ytick.major.pad":       3,
            "lines.linewidth":       1.2,
            "lines.solid_capstyle":  "butt",
            "patch.linewidth":       0.6,
            "patch.edgecolor":       "none",
            "legend.frameon":        False,
            "legend.fontsize":       7,
            "legend.handlelength":   1.2,
            "legend.handletextpad":  0.4,
            "legend.labelspacing":   0.3,
            "legend.borderaxespad":  0.4,
            "axes.grid":             False,
            "font.family":           "DejaVu Sans",
            "text.color":            p["fg"],
            "savefig.dpi":           300,
            "savefig.bbox":          "tight",
            "savefig.facecolor":     p["bg"],
            "svg.fonttype":          "none",
            "pdf.fonttype":          42,
            "ps.fonttype":           42,
        })
        return self


# ════════════════════════════════════════════════════════
#  AXIS HELPERS
# ════════════════════════════════════════════════════════

def clean_ax(ax, offset: int = 0):
    """Remove top/right spines and optionally offset remaining spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if offset:
        ax.spines["left"].set_position(("outward", offset))
        ax.spines["bottom"].set_position(("outward", offset))


def strip_ax(ax):
    """Remove all spines and ticks (e.g. for waterfall / scalebar panels)."""
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def anchor_ticks(ax, n_x: int = 3, n_y: int = 3):
    """LinearLocator from xlim[0]→xlim[-1]. First tick = spine origin."""
    ax.xaxis.set_major_locator(LinearLocator(numticks=n_x))
    ax.yaxis.set_major_locator(LinearLocator(numticks=n_y))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.4g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4g"))


def panel_label(ax, letter: str, theme: Theme,
                x: float = -0.08, y: float = 1.04):
    """Panel label in axes-fraction coordinates."""
    ax.text(x, y, letter,
            transform=ax.transAxes,
            fontsize=9, fontstyle="normal", fontweight="bold",
            color=theme.fg, va="bottom", ha="right", clip_on=False)


def baseline(ax, theme: Theme, alpha: float = 0.18):
    """Subtle y=0 reference line."""
    ax.axhline(0, color=theme.fg, lw=0.4, alpha=alpha, zorder=0)


def shade_epoch(ax, t0: float, t1: float,
                theme: Optional[Theme] = None,
                color: Optional[str] = None,
                alpha: float = 0.18):
    """Translucent vertical span (e.g. stimulus epoch)."""
    c = color or (theme.shade if theme else "#E6E6E6")
    ax.axvspan(t0, t1, color=c, alpha=alpha, linewidth=0, zorder=0)


def plot_mean_sem(ax, x, mean, sem,
                  color: Optional[str] = None, label: Optional[str] = None,
                  alpha_fill: float = 0.13,
                  smooth_sigma: Optional[float] = None,
                  lw: Optional[float] = None,
                  theme: Optional[Theme] = None):
    """Line + shaded SEM band."""
    if color is None:
        color = theme.colors[0] if theme else _LIGHT[0]
    if lw is None:
        lw = float(mpl.rcParams.get("lines.linewidth", 1.65))
    if smooth_sigma is not None:
        mean = gaussian_filter1d(mean, float(smooth_sigma))
        sem  = gaussian_filter1d(sem,  float(smooth_sigma))
    ax.fill_between(x, mean - sem, mean + sem,
                    color=color, alpha=alpha_fill, linewidth=0, zorder=2)
    return ax.plot(x, mean, color=color, label=label, lw=lw, zorder=3)[0]


def styled_legend(ax, loc: str = "best", theme: Optional[Theme] = None):
    """Legend styled to match the active theme."""
    leg = ax.legend(loc=loc, frameon=False,
                    handlelength=1.1, handletextpad=0.4,
                    labelspacing=0.3, borderaxespad=0.5)
    if theme:
        for txt in leg.get_texts():
            txt.set_color(theme.fg)
    return leg


def label_traces(ax, labels: Optional[List[str]] = None,
                 x_offset_frac: float = 0.012,
                 fontsize: int = 8,
                 theme: Optional[Theme] = None):
    """
    Inline end-of-trace labels, placed just past the trace endpoint.
    When endpoints are within *min_sep* of each other in y, labels are
    nudged apart.  For truly converging traces use ``styled_legend()``.
    """
    lines = [l for l in ax.get_lines()
             if not l.get_label().startswith("_") and len(l.get_xdata()) > 0]
    if labels is None:
        labels = [l.get_label() for l in lines]

    xlim  = ax.get_xlim()
    ylim  = ax.get_ylim()
    dx    = (xlim[1] - xlim[0]) * x_offset_frac
    min_sep = (ylim[1] - ylim[0]) * (fontsize / 72) * 1.6

    ys = []
    for line in lines:
        yd = np.asarray(line.get_ydata())
        ys.append(float(yd[-1]) if len(yd) > 0 else 0.0)

    order  = sorted(range(len(ys)), key=lambda i: ys[i], reverse=True)
    nudged = list(ys)
    for k in range(1, len(order)):
        prev, cur = order[k-1], order[k]
        if nudged[prev] - nudged[cur] < min_sep:
            nudged[cur] = nudged[prev] - min_sep

    ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.15)

    for i, (line, label) in enumerate(zip(lines, labels)):
        xd = np.asarray(line.get_xdata())
        if len(xd) == 0:
            continue
        ax.text(xd[-1] + dx, nudged[i], label,
                color=line.get_color(), fontsize=fontsize,
                va="center", ha="left", clip_on=False)


def _pad_ax(ax, x, y, pad: float = 0.06):
    """Add proportional padding so edge points aren't clipped."""
    xspan = float(np.ptp(x))
    yspan = float(np.ptp(y))
    ax.set_xlim(float(x.min()) - xspan * pad, float(x.max()) + xspan * pad)
    ax.set_ylim(float(y.min()) - yspan * pad, float(y.max()) + yspan * pad)


def sig_bracket(ax, x1: float, x2: float, y_base: float,
                text: str = "*", h_frac: float = 0.04,
                theme: Optional[Theme] = None):
    """Statistical significance bracket between two x positions."""
    fg = theme.fg if theme else "#1A1924"
    yr = ax.get_ylim()
    h  = (yr[1] - yr[0]) * h_frac
    y  = y_base + (yr[1] - yr[0]) * 0.018
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y],
            lw=0.8, color=fg,
            solid_capstyle="butt", solid_joinstyle="miter", clip_on=False)
    ax.text((x1+x2)/2, y+h, text,
            ha="center", va="bottom", fontsize=9, color=fg)


# ════════════════════════════════════════════════════════
#  SCALE BAR — direct data-unit arithmetic
# ════════════════════════════════════════════════════════

def draw_scalebar(ax, x_corner: float, y_corner: float,
                  scale_x: float, scale_y: float,
                  label_x: Optional[str] = None,
                  label_y: Optional[str] = None,
                  lw: float = 1.2,
                  theme: Optional[Theme] = None):
    """
    L-shaped scale bar with corner at *(x_corner, y_corner)* in data coords.
    Horizontal arm extends LEFT by *scale_x*; vertical arm extends UP by *scale_y*.
    """
    fg = theme.fg if theme else "#1A1924"
    x0 = x_corner - scale_x

    ax.plot([x0, x_corner], [y_corner, y_corner],
            color=fg, lw=lw, solid_capstyle="butt",
            clip_on=False, zorder=10)
    ax.plot([x0, x0], [y_corner, y_corner + scale_y],
            color=fg, lw=lw, solid_capstyle="butt",
            clip_on=False, zorder=10)

    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    x_span = xr - xl
    y_span = yt - yb

    if label_x:
        ax.text((x0 + x_corner) / 2, y_corner - y_span * 0.022,
                label_x, ha="center", va="top",
                fontsize=6.5, color=fg, clip_on=False)
    if label_y:
        ax.text(x0 - x_span * 0.01, y_corner + scale_y / 2,
                label_y, ha="right", va="center",
                fontsize=6.5, color=fg, clip_on=False,
                linespacing=1.35)


# ════════════════════════════════════════════════════════
#  RAW TIMESERIES — waterfall
# ════════════════════════════════════════════════════════

def plot_timeseries(ax, data,
                    dt: float = 1.0, t_start: float = 0.0,
                    offset_factor: float = 4.0,
                    labels: Optional[List[str]] = None,
                    colors: Optional[List[str]] = None,
                    lw: float = 0.8, alpha: float = 0.85,
                    smooth_sigma: Optional[float] = None,
                    scale_x: Optional[float] = None,
                    scale_y: Optional[float] = None,
                    label_x: Optional[str] = None,
                    label_y: Optional[str] = None,
                    theme: Optional[Theme] = None):
    """Waterfall (stacked offset) timeseries with automatic scale bar."""
    data   = np.atleast_2d(np.asarray(data, dtype=float))
    n_ch, n_t = data.shape
    t      = np.arange(n_t) * dt + t_start
    colors = colors or (theme.colors if theme else _LIGHT)
    labels = labels or [f"ch {i+1}" for i in range(n_ch)]

    sig     = float(np.median(np.std(data, axis=1)))
    offsets = np.arange(n_ch - 1, -1, -1, dtype=float) * sig * offset_factor

    for i in range(n_ch):
        tr  = data[i].copy()
        if smooth_sigma:
            tr = gaussian_filter1d(tr, float(smooth_sigma))
        col = colors[i % len(colors)]
        ax.plot(t, tr + offsets[i],
                color=col, lw=lw, alpha=alpha, zorder=2)
        ax.text(
            t[0] - (t[-1] - t[0]) * 0.011, offsets[i],
            labels[i],
            ha="right", va="center", fontsize=7,
            color=col, clip_on=False,
        )

    scale_y_ = scale_y if scale_y is not None else sig
    scale_x_ = scale_x if scale_x is not None else max((t[-1]-t[0])*0.08, dt*12)
    lx = label_x or f"{scale_x_:.4g} s"
    ly = label_y or f"{scale_y_:.4g}\nΔF/F"

    gap      = sig * 5.0
    y_corner = float(offsets[-1]) - gap - scale_y_
    y_top    = float(offsets[0])  + sig * 2.0
    y_bottom = y_corner - sig * 2.5
    x_corner = t[-1] - (t[-1] - t[0]) * 0.02

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(y_bottom, y_top)

    strip_ax(ax)

    draw_scalebar(ax, x_corner=x_corner, y_corner=y_corner,
                  scale_x=scale_x_, scale_y=scale_y_,
                  label_x=lx, label_y=ly, theme=theme)
    return offsets


# ════════════════════════════════════════════════════════
#  DEMO  — 3 rows, 8 panels
# ════════════════════════════════════════════════════════

def make_demo(mode: str = "light"):
    """Build an 8-panel demonstration figure for *cold_field_v5*."""
    from matplotlib.gridspec import GridSpec

    theme = Theme(mode).apply()
    p     = theme.p
    rng   = np.random.default_rng(42)

    fig = plt.figure(figsize=(14, 11))
    gs  = GridSpec(3, 1, figure=fig,
                   height_ratios=[1.05, 1.1, 1.1],
                   hspace=0.62,
                   left=0.09, right=0.96,
                   top=0.94, bottom=0.07)

    gs0 = gs[0].subgridspec(1, 2, width_ratios=[1.85, 1.0], wspace=0.42)
    gs1 = gs[1].subgridspec(1, 3, width_ratios=[1.1, 1.0, 0.95], wspace=0.50)
    gs2 = gs[2].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.0],  wspace=0.50)

    ax_ts    = fig.add_subplot(gs0[0])   # a — raw timeseries
    ax_eta   = fig.add_subplot(gs0[1])   # b — event-triggered avg
    ax_ml    = fig.add_subplot(gs1[0])   # c — multi-condition traces
    ax_bar   = fig.add_subplot(gs1[1])   # d — bar + scatter
    ax_heat  = fig.add_subplot(gs1[2])   # e — trial heatmap
    ax_tune  = fig.add_subplot(gs2[0])   # f — tuning curve (polar-like)
    ax_long  = fig.add_subplot(gs2[1])   # g — longitudinal / session
    ax_scat  = fig.add_subplot(gs2[2])   # h — scatter + regression

    fig.text(0.09, 0.968,
             "cold_field  v0.5" + ("  ·  dark" if mode == "dark" else ""),
             fontsize=7.5, color=p["tick"], va="top", ha="left")

    # ── A — RAW TIMESERIES ────────────────────────────
    n_ch, n_t, dt = 5, 1800, 1/30
    drift = 0.006 * np.cumsum(rng.normal(0, 0.01, (n_ch, n_t)), axis=1)
    evs   = np.zeros((n_ch, n_t))
    for t_ev in [95, 320, 570, 840, 1110, 1390, 1620]:
        amp = rng.uniform(0.25, 0.9, n_ch)
        for ch in range(n_ch):
            tail = np.arange(n_t - t_ev)
            evs[ch, t_ev:] += amp[ch] * np.exp(-tail * dt / 1.3)
    ts_data = (drift + evs * 0.42 + rng.normal(0, 0.032, (n_ch, n_t))) * 0.48

    shade_epoch(ax_ts, 320*dt, 375*dt, theme=theme, alpha=0.28)
    shade_epoch(ax_ts, 840*dt, 895*dt, theme=theme, alpha=0.28)

    plot_timeseries(ax_ts, ts_data, dt=dt,
                    labels=["V1","LM","PM","RSC","M1"],
                    scale_x=2.0, scale_y=0.2,
                    label_x="2 s", label_y="0.2 ΔF/F",
                    theme=theme)
    ax_ts.set_title("widefield dF/F  —  5 ROIs", pad=7)
    panel_label(ax_ts, "a", theme)

    # ── B — EVENT-TRIGGERED AVG ± SEM ─────────────────
    t_e = np.linspace(-1.5, 4.5, 280)
    shade_epoch(ax_eta, 0, 2, theme=theme, alpha=0.22)
    ax_eta.axvline(0, color=p["fg"], lw=0.65, ls="--", alpha=0.32)
    baseline(ax_eta, theme)

    for label, amp, peak, col in [
        ("V1",  0.85, 1.0, theme.colors[0]),
        ("LM",  1.25, 1.6, theme.colors[2]),
        ("RSC", 0.50, 2.2, theme.colors[4]),
    ]:
        k  = amp * np.exp(-((t_e - peak)**2) / (2*0.55**2))
        tr = k[None,:] + rng.normal(0, 0.08, (30, len(t_e)))
        m, s = tr.mean(0), tr.std(0) / np.sqrt(30)
        plot_mean_sem(ax_eta, t_e, m, s, color=col,
                      label=label, smooth_sigma=5, theme=theme)

    styled_legend(ax_eta, loc="upper right", theme=theme)
    ax_eta.set_xlabel("time from event (s)")
    ax_eta.set_ylabel("ΔF/F")
    ax_eta.set_title("event-triggered avg  ±SEM")
    clean_ax(ax_eta)
    ax_eta.set_xlim(t_e[0], t_e[-1])
    anchor_ticks(ax_eta, n_x=4, n_y=3)
    panel_label(ax_eta, "b", theme)

    # ── C — MULTI-CONDITION TRACES ────────────────────
    t_ml = np.linspace(0, 5*np.pi, 340)
    conds = [("visual",1.0,0.0),("opto",0.65,1.2),
             ("combined",1.2,0.5),("ctrl",0.35,2.1)]
    for i, (label, amp, ph) in enumerate(conds):
        raw   = amp * np.sin(t_ml + ph) + rng.normal(0, 0.06, len(t_ml))
        trace = gaussian_filter1d(raw, sigma=9)
        ax_ml.plot(t_ml, trace, color=theme.colors[i], label=label)

    label_traces(ax_ml, [c[0] for c in conds], theme=theme)
    baseline(ax_ml, theme)
    ax_ml.set_xlabel("time (s)")
    ax_ml.set_ylabel("ΔF/F")
    ax_ml.set_title("multi-condition traces")
    clean_ax(ax_ml)
    ax_ml.set_xlim(t_ml[0], t_ml[-1])
    anchor_ticks(ax_ml, n_x=4, n_y=3)
    panel_label(ax_ml, "c", theme)

    # ── D — BAR + SCATTER + SIG BRACKET ───────────────
    groups  = ["pre", "post", "wash"]
    means_b = [0.38, 1.12, 0.41]
    sems_b  = [0.055, 0.085, 0.06]
    n_pts   = 14
    bar_cols = [theme.bar_lo, theme.bar_hi, theme.bar_lo]

    for i, (m, s, c) in enumerate(zip(means_b, sems_b, bar_cols)):
        ax_bar.bar(i, m, width=0.50, color=c, zorder=2, alpha=0.92)
        ax_bar.errorbar(i, m, yerr=s, fmt="none",
                        color=p["fg"], capsize=2.5, capthick=0.8,
                        lw=0.9, zorder=3)
        pts = rng.normal(m, s*1.5, n_pts)
        jit = rng.uniform(-0.12, 0.12, n_pts)
        ax_bar.scatter(i + jit, pts,
                       color=theme.dot_color, s=14,
                       alpha=theme.dot_alpha,
                       zorder=4, linewidths=0)

    ax_bar.set_xticks([0, 1, 2])
    ax_bar.set_xticklabels(groups)
    ax_bar.set_ylabel("peak ΔF/F")
    ax_bar.set_title("group comparison")
    ax_bar.set_xlim(-0.55, 2.55)
    ax_bar.set_ylim(0, 1.45)
    ax_bar.yaxis.set_major_locator(LinearLocator(numticks=3))
    ax_bar.yaxis.set_major_formatter(FormatStrFormatter("%.4g"))
    clean_ax(ax_bar)
    top = max(means_b[0]+sems_b[0], means_b[1]+sems_b[1]) + 0.04
    sig_bracket(ax_bar, 0, 1, top, text="**", theme=theme)
    panel_label(ax_bar, "d", theme)

    # ── E — TRIAL HEATMAP ─────────────────────────────
    n_tr = 45
    t_h  = np.linspace(-1, 4.5, 220)
    k_h  = np.exp(-((t_h - 1.1)**2) / (2*0.5**2))
    mat  = k_h[None,:] + rng.normal(0, 0.2, (n_tr, len(t_h)))
    cmap = "inferno" if mode == "dark" else "YlOrRd"
    im   = ax_heat.imshow(mat, aspect="auto",
                           extent=(t_h[0], t_h[-1], 0.0, float(n_tr)),
                           origin="lower", cmap=cmap,
                           vmin=-0.3, vmax=1.4)
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.76, pad=0.03, aspect=20)
    cbar.set_label("ΔF/F", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)
    ax_heat.axvline(0, color="white", lw=0.7, ls="--", alpha=0.45)
    ax_heat.set_xlabel("time (s)")
    ax_heat.set_ylabel("trial")
    ax_heat.set_title("trial-by-trial")
    clean_ax(ax_heat)
    ax_heat.set_xlim(t_h[0], t_h[-1])
    ax_heat.set_ylim(0, n_tr)
    anchor_ticks(ax_heat, n_x=4, n_y=3)
    panel_label(ax_heat, "e", theme)

    # ── F — TUNING CURVE ──────────────────────────────
    orients   = np.linspace(0, 360, 9)[:-1]
    tuning    = 0.85 * np.exp(-0.5 * ((orients - 90) / 45)**2) + 0.08
    sem_tune  = rng.uniform(0.04, 0.09, len(orients))
    orients_w = np.append(orients, 360)
    tuning_w  = np.append(tuning, tuning[0])
    sem_w     = np.append(sem_tune, sem_tune[0])

    ax_tune.fill_between(orients_w, tuning_w - sem_w, tuning_w + sem_w,
                          color=theme.colors[0], alpha=0.13, linewidth=0)
    ax_tune.plot(orients_w, tuning_w,
                  color=theme.colors[0], lw=1.65)
    ax_tune.errorbar(orients, tuning, yerr=sem_tune,
                      fmt="o", color=theme.colors[0],
                      ms=4, lw=0.9, capsize=2.5, capthick=0.8, zorder=5)

    ax_tune.set_xlim(-18, 378)
    ax_tune.set_ylim(0, (tuning_w + sem_w).max() * 1.15)
    ax_tune.set_xticks([0, 90, 180, 270, 360])
    ax_tune.xaxis.set_major_formatter(FormatStrFormatter("%g°"))
    ax_tune.yaxis.set_major_locator(LinearLocator(numticks=3))
    ax_tune.yaxis.set_major_formatter(FormatStrFormatter("%.4g"))
    ax_tune.set_xlabel("orientation (°)")
    ax_tune.set_ylabel("ΔF/F")
    ax_tune.set_title("orientation tuning")
    clean_ax(ax_tune)
    panel_label(ax_tune, "f", theme)

    # ── G — LONGITUDINAL / SESSION PLOT ───────────────
    n_sessions = 8
    n_mice     = 6
    sessions   = np.arange(1, n_sessions + 1)
    true_curve = 0.22 + 0.55 * (1 - np.exp(-sessions / 3.2))
    mouse_data = (true_curve[None, :] +
                  rng.normal(0, 0.06, (n_mice, n_sessions)) +
                  rng.normal(0, 0.03, (n_mice, 1)))

    mean_s = mouse_data.mean(0)
    sem_s  = mouse_data.std(0) / np.sqrt(n_mice)

    for i in range(n_mice):
        ax_long.plot(sessions, mouse_data[i],
                     color=theme.colors[0], lw=0.7,
                     alpha=0.28, zorder=2)

    plot_mean_sem(ax_long, sessions, mean_s, sem_s,
                  color=theme.colors[0], alpha_fill=0.16, lw=1.8)
    ax_long.plot(sessions, mean_s, "o",
                  color=theme.colors[0], ms=4.5, zorder=5)

    ax_long.set_xlim(0.4, n_sessions + 0.6)
    ax_long.xaxis.set_major_locator(LinearLocator(numticks=n_sessions))
    ax_long.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax_long.yaxis.set_major_locator(LinearLocator(numticks=3))
    ax_long.yaxis.set_major_formatter(FormatStrFormatter("%.4g"))
    ax_long.set_xlabel("session")
    ax_long.set_ylabel("ΔF/F")
    ax_long.set_title("longitudinal — individual mice")
    clean_ax(ax_long)
    panel_label(ax_long, "g", theme)

    # ── H — SCATTER + REGRESSION LINE ─────────────────
    x_sc  = rng.uniform(0.1, 1.5, 40)
    noise = rng.normal(0, 0.15, 40)
    y_sc  = 0.55 * x_sc + 0.1 + noise

    ax_scat.scatter(x_sc, y_sc,
                    color=theme.colors[0], s=22,
                    alpha=0.65, linewidths=0, zorder=3)

    m_r, b_r = np.polyfit(x_sc, y_sc, 1)
    x_fit    = np.array([x_sc.min(), x_sc.max()])
    ax_scat.plot(x_fit, m_r * x_fit + b_r,
                  color=theme.colors[1], lw=1.5,
                  alpha=0.88, zorder=4)

    ss_res = np.sum((y_sc - (m_r * x_sc + b_r))**2)
    ss_tot = np.sum((y_sc - y_sc.mean())**2)
    r2     = 1 - ss_res / ss_tot
    ax_scat.text(0.96, 0.08,
                  f"$r^2$ = {r2:.2f}",
                  transform=ax_scat.transAxes,
                  ha="right", va="bottom",
                  fontsize=7.5, color=p["fg"])

    ax_scat.set_xlabel("V1 ΔF/F")
    ax_scat.set_ylabel("LM ΔF/F")
    ax_scat.set_title("inter-area correlation")
    _pad_ax(ax_scat, x_sc, y_sc, pad=0.07)
    clean_ax(ax_scat)
    anchor_ticks(ax_scat, n_x=3, n_y=3)
    panel_label(ax_scat, "h", theme)

    return fig


if __name__ == "__main__":
    for mode in ("light", "dark"):
        fig = make_demo(mode)
        out = f"cold_field_v05_{mode}.png"
        fig.savefig(out, dpi=200)
        print(f"saved -> {out}")
        plt.close(fig)
