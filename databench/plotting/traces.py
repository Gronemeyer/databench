"""Trace processing & rendering configuration layer.

This module is the canonical place for trace-conditioning and trace-plotting
helpers shared across analysis-specific plotting modules (oscillation, ETA,
overview, etc.).  Every function operates on plain NumPy arrays.

Public API
----------
prepare_trace_styled   Condition a trace via a :class:`TraceStyle` recipe.
plot_trace_styled      Draw a conditioned trace with TraceStyle visual params.
plot_trace             Draw a single trace on an Axes with standard styling.
dense_lw               Density-aware linewidth scaling.
TRACE_COLORS           Theme-derived colour dict for auxiliary signals.
"""
from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from databench.analysis._signal.preproc import (  # canonical source
    smooth_savgol,
    smooth_dense,
    remove_outliers_iqr,
)

from databench.analysis._signal.remap import remap_to_timebase
from databench._utils import time_mask


# ── Density-aware line width ────────────────────────────────────────────────

def dense_lw(
    n_points: int,
    fig_width_in: float,
    base_lw: float = 1.4,
    *,
    dpi: float = 100.0,
) -> float:
    """Scale *base_lw* down when the trace is denser than ~1 pt per pixel.

    Parameters
    ----------
    n_points : int
        Number of data points in the visible window.
    fig_width_in : float
        Figure width in inches.
    base_lw : float
        Desired linewidth at low density (≤ 1 pt/px).
    dpi : float
        Screen / render DPI.

    Returns
    -------
    float
        Adjusted linewidth, clamped to ``[0.15, base_lw]``.
    """
    px = fig_width_in * dpi
    if px <= 0 or n_points <= 0:
        return base_lw
    pts_per_px = n_points / px
    if pts_per_px <= 1.0:
        return base_lw
    # Logarithmic taper: halve lw every ~8× density increase
    scaled = base_lw / (1.0 + 0.35 * np.log2(pts_per_px))
    return float(np.clip(scaled, 0.15, base_lw))


# ── Trace plotting ─────────────────────────────────────────────────────────

def _trace_colors() -> dict[str, str]:
    """Build auxiliary-signal colours from the active theme."""
    from databench.plotting import get_theme
    c = get_theme().colors
    return {
        "pupil": c[1],        # sienna
        "speed": c[2],        # viridian
        "locomotion": c[2],   # viridian
        "dff": c[0],          # lapis
    }


class _LazyTraceColors(dict):
    """Dict that populates from the active theme on first access."""
    _initialised: bool = False

    def _ensure(self):
        if not self._initialised:
            self.update(_trace_colors())
            self._initialised = True

    def __getitem__(self, key):
        self._ensure()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure()
        return super().__contains__(key)

    def get(self, key, default=None):
        self._ensure()
        return super().get(key, default)


TRACE_COLORS: dict[str, str] = _LazyTraceColors()


def plot_trace(
    ax: plt.Axes,
    time: np.ndarray,
    values: np.ndarray,
    *,
    window: Tuple[float, float] | None = None,
    color: str | None = None,
    label: str | None = None,
    ylabel: str | None = None,
    lw: float = 1.4,
    alpha: float = 0.9,
) -> plt.Axes:
    """Draw a single trace on *ax*, clipping to an optional time window.

    Parameters
    ----------
    ax : matplotlib Axes
    time : array
        Time vector.
    values : array
        Signal values, same length as *time*.
    window : (lo, hi), optional
        Restrict the visible portion.
    color, label, ylabel : str, optional
        Styling.  *color* defaults to matplotlib's cycle.
    lw, alpha : float
        Line width and opacity.

    Returns
    -------
    ax : plt.Axes
        The same Axes (for chaining).
    """
    mask, _ = time_mask(time, window)
    if mask is None:
        return ax
    kwargs: dict = {"lw": lw, "alpha": alpha}
    if color is not None:
        kwargs["color"] = color
    if label is not None:
        kwargs["label"] = label
    ax.plot(time[mask], values[mask], **kwargs)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    from databench.plotting import style_axes
    style_axes(ax)
    return ax


def prepare_trace_styled(
    reference_time: np.ndarray,
    source_time: np.ndarray | None,
    source_values: np.ndarray,
    style: "TraceStyle",
    *,
    est_fs: float | None = None,
) -> np.ndarray:
    """Condition a trace for plotting according to a :class:`TraceStyle`.

    Performs remap → gap-fill → optional smoothing, all driven by *style*.

    Parameters
    ----------
    reference_time : array
        Target timebase for the plot.
    source_time : array or None
        Raw source timestamps.  Required when ``style.method != "raw"``.
    source_values : array
        Raw signal values.
    style : TraceStyle
        Rendering recipe (from :data:`~databench.plotting.trace_config.TRACE_STYLES`).
    est_fs : float, optional
        Estimated sampling rate (used for ``smooth_savgol_s`` conversion).

    Returns
    -------
    np.ndarray
        Conditioned values on *reference_time*.
    """
    from databench.plotting.style import TraceStyle  # noqa: F811

    values = np.asarray(source_values, dtype=float)

    # 1. Remap onto reference timebase
    if style.method in ("step_previous", "linear"):
        if source_time is None:
            raise ValueError(
                f"source_time is required for method={style.method!r}"
            )
        values = remap_to_timebase(
            reference_time,
            source_time,
            values,
            method=style.method,
            gap_threshold_s=style.gap_threshold_s,
            fill_value=style.fill_value,
        )
    # method == "raw": values stay as-is

    # 2. Optional outlier removal
    if style.outlier_iqr_k is not None:
        values, _ = remove_outliers_iqr(values, k=style.outlier_iqr_k)

    # 3. Optional SG smoothing (seconds-based)
    if style.smooth_savgol_s > 0 and est_fs is not None and est_fs > 0:
        values = smooth_savgol(values, style.smooth_savgol_s, est_fs)

    # 4. Optional median + SG smoothing (sample-based)
    if style.smooth:
        values = smooth_dense(
            values,
            median_size=style.median_size,
            window=style.savgol_window,
            polyorder=style.savgol_polyorder,
        )

    return values


def plot_trace_styled(
    ax: plt.Axes,
    time: np.ndarray,
    values: np.ndarray,
    style: "TraceStyle",
    *,
    window: Tuple[float, float] | None = None,
    label: str | None = None,
) -> plt.Axes:
    """Draw a trace on *ax* using visual parameters from a :class:`TraceStyle`.

    Parameters
    ----------
    ax : matplotlib Axes
    time, values : array
        Timebase and signal (same length, already conditioned).
    style : TraceStyle
        Visual config (color, lw, alpha, drawstyle, ylabel).
    window : (lo, hi), optional
        Restrict the visible portion.
    label : str, optional
        Legend label.

    Returns
    -------
    ax
    """
    mask, _ = time_mask(time, window)
    if mask is None:
        return ax
    kwargs: dict = {"lw": style.lw, "alpha": style.alpha, "color": style.color}
    if style.drawstyle is not None:
        kwargs["drawstyle"] = style.drawstyle
    if label is not None:
        kwargs["label"] = label
    ax.plot(time[mask], values[mask], **kwargs)
    if style.ylabel:
        ax.set_ylabel(style.ylabel)

    from databench.plotting import style_axes
    style_axes(ax)
    return ax
