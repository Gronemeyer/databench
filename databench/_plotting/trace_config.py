"""Centralised trace-rendering configuration.

Every named trace (treadmill speed, pupil diameter, ROI ΔF/F, …) has a
single ``TraceStyle`` that describes *how* it should be processed and drawn.
Analysis-specific plotting modules look up styles by name via
:func:`get_style`, so adding or tweaking a trace type is a one-line change
in ``TRACE_STYLES`` rather than scattered ``if``-branches.

Public API
----------
TraceStyle      Frozen dataclass — one rendering recipe.
TRACE_STYLES    Dict mapping canonical names → TraceStyle instances.
get_style       Lookup helper (returns a copy so callers can override).
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TraceStyle:
    """Complete rendering recipe for one trace type.

    Processing parameters
    ---------------------
    method : {"step_previous", "linear", "raw"}
        How to remap sparse samples onto a reference timebase.
        ``"raw"`` means the trace is already on the correct timebase.
    gap_threshold_s : float or None
        Max distance (seconds) to nearest valid sample before a point
        is treated as a recording gap.  ``None`` disables gap-filling.
    fill_value : float
        Value to assign inside gaps (e.g. 0 for speed).
    smooth : bool
        Whether to apply median + Savitzky-Golay smoothing after remap.
    smooth_savgol_s : float
        Savitzky-Golay smoothing window in seconds (used only when an
        estimated fs is available in the plotting context).
    median_size : int
        Kernel size for the median pre-filter in :func:`smooth_dense`.
    savgol_window : int
        Savitzky-Golay window length (samples, must be odd).
    savgol_polyorder : int
        Savitzky-Golay polynomial order.
    outlier_iqr_k : float or None
        IQR multiplier for outlier removal.  ``None`` skips it.

    Visual parameters
    -----------------
    color : str
        Hex colour for the line.
    lw : float
        Line width.
    alpha : float
        Line opacity.
    drawstyle : str or None
        Matplotlib drawstyle (e.g. ``"steps-post"``).  ``None`` uses
        the default linear join.
    ylabel : str
        Default y-axis label.
    """

    # ── Processing ─────────────────────────────────────────────────────
    method: Literal["step_previous", "linear", "raw"] = "raw"
    gap_threshold_s: float | None = None
    fill_value: float = 0.0
    smooth: bool = False
    smooth_savgol_s: float = 0.0
    median_size: int = 3
    savgol_window: int = 5
    savgol_polyorder: int = 2
    outlier_iqr_k: float | None = None

    # ── Visual ─────────────────────────────────────────────────────────
    color: str = "#1f77b4"
    lw: float = 1.4
    alpha: float = 0.9
    drawstyle: str | None = None
    ylabel: str = ""


# ── Default styles ─────────────────────────────────────────────────────────
# Keep parameter values exactly as they were in the proven working code.

TRACE_STYLES: dict[str, TraceStyle] = {
    # Treadmill speed — previous-sample hold, no smoothing, no gap-fill.
    # Zeros are real timeout values (no movement detected), NOT missing data.
    "treadmill": TraceStyle(
        method="step_previous",
        gap_threshold_s=None,
        fill_value=0.0,
        smooth=False,
        color="#00CC96",
        lw=1.6,
        alpha=0.9,
        drawstyle="steps-post",
        ylabel="Speed (mm/s)",
    ),
    # Pupil diameter — already on the aligned timebase, light SG smoothing.
    "pupil": TraceStyle(
        method="raw",
        smooth=False,
        smooth_savgol_s=0.5,
        color="#EF553B",
        lw=1.6,
        alpha=0.9,
        ylabel="Pupil (mm)",
    ),
    # ROI ΔF/F signal — no extra processing, standard blue.
    "roi": TraceStyle(
        method="raw",
        smooth=False,
        color="#1f77b4",
        lw=1.4,
        alpha=0.9,
        ylabel="ΔF/F",
    ),
    # Bandpass-filtered signal — thin green.
    "filtered": TraceStyle(
        method="raw",
        smooth=False,
        color="#2ca02c",
        lw=0.8,
        alpha=0.8,
        ylabel="Amplitude",
    ),
    # Hilbert envelope — red.
    "envelope": TraceStyle(
        method="raw",
        smooth=False,
        color="#d62728",
        lw=1.2,
        alpha=0.9,
        ylabel="Amplitude",
    ),
}


def get_style(name: str, **overrides) -> TraceStyle:
    """Look up a trace style by name, optionally overriding fields.

    Parameters
    ----------
    name : str
        Key in :data:`TRACE_STYLES`.
    **overrides
        Any ``TraceStyle`` field to override for this call.

    Returns
    -------
    TraceStyle

    Raises
    ------
    KeyError
        If *name* is not a registered style.
    """
    if name not in TRACE_STYLES:
        raise KeyError(
            f"Unknown trace style {name!r}. "
            f"Available: {sorted(TRACE_STYLES)}"
        )
    style = TRACE_STYLES[name]
    if not overrides:
        return style
    # Build a new instance with overridden fields
    return TraceStyle(**{**{f.name: getattr(style, f.name) for f in style.__dataclass_fields__.values()}, **overrides})
