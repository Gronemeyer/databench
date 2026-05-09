"""Plotting utilities for databench analyses.

The active visual theme is controlled by :func:`set_theme` /
:func:`get_theme`.  All plotting modules in this package read the theme
lazily so that ``import databench.plotting`` alone never triggers heavy
rcParams mutations — call ``set_theme()`` once at the top of a script.

Quick start
-----------
>>> from databench.plotting import set_theme, new_figure, style_axes
>>> set_theme()                           # light mode, cold_field_v5
>>> fig, ax = new_figure()                # themed figure + axes
>>> ax.plot(x, y)
>>> style_axes(ax)                        # clean spines / ticks
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from databench.plotting.style.cold_field_v5 import Theme, clean_ax

# ── Module-level theme state ───────────────────────────────────────────────

_active_theme: Optional[Theme] = None


def set_theme(mode: str = "light", *, style: str = "cold_field_v5") -> Theme:
    """Activate a plotting theme globally and return it.

    Parameters
    ----------
    mode : ``"light"`` or ``"dark"``
    style : registered style name (default ``"cold_field_v5"``).
    """
    from databench.plotting.style import use
    global _active_theme
    _active_theme = use(style, mode=mode)
    return _active_theme


def get_theme() -> Theme:
    """Return the active theme, initialising to *light* on first call."""
    global _active_theme
    if _active_theme is None:
        _active_theme = Theme("light").apply()
    return _active_theme


# ── Axes helpers ───────────────────────────────────────────────────────────

def style_axes(ax: plt.Axes, *, fontsize: float | None = None) -> None:
    """Apply default spine and tick styling to *ax*.

    Removes top/right spines, offsets the remaining spines, and applies
    tick parameters from the active theme.  All sizes come from rcParams
    set by :meth:`Theme.apply` unless *fontsize* is explicitly overridden.
    """
    clean_ax(ax)
    kw: dict = dict(
        direction="out", top=False, right=False,
    )
    if fontsize is not None:
        kw["labelsize"] = fontsize
    ax.tick_params(**kw)


def style_figure(fig: plt.Figure) -> None:
    """Apply theme-consistent layout to a completed figure.

    Calls ``fig.tight_layout()`` with standard padding that matches the
    theme's spacing conventions.  Call this **after** all plotting is done
    and just before saving.
    """
    fig.tight_layout()


# ── Figure factory ─────────────────────────────────────────────────────────

def new_figure(
    nrows: int = 1,
    ncols: int = 1,
    *,
    figsize: tuple[float, float] | None = None,
    width: float = 7.0,
    height_per_row: float = 3.2,
    sharex: bool | str = False,
    sharey: bool | str = False,
    squeeze: bool = True,
    gridspec_kw: dict | None = None,
    **subplot_kw,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """Create a themed figure and axes.

    The theme is auto-applied on first call if not already active.  Figure
    size defaults to ``(width, height_per_row * nrows)`` but can be
    overridden directly with *figsize*.

    Returns ``(fig, ax)`` when *squeeze* is True and there is one axes,
    otherwise ``(fig, axes_array)``.
    """
    get_theme()  # ensure theme is active
    if figsize is None:
        figsize = (width, height_per_row * nrows)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        gridspec_kw=gridspec_kw,
        **subplot_kw,
    )
    return fig, axes


# ── Per-subject longitudinal line + group mean/SEM ────────────────────────

def plot_metric_by_session(
    table,
    *,
    x: str,
    y: str,
    ax: plt.Axes,
    subject_col: str = "Subject",
    subject_colors: dict | None = None,
    group_color: str = "black",
    group_label: str = "group mean",
) -> plt.Axes:
    """Per-subject lines + group mean ± SEM errorbar on a single axes.

    *table* must have a subject column (default ``"Subject"``), an x
    column (e.g. ``"day"`` or ``"session_n"``), and a numeric y column.

    Returns the axes for further customisation (titles, labels, legend).
    """
    subjects = sorted(table[subject_col].dropna().unique())
    if subject_colors is None:
        palette = plt.cm.tab10(np.linspace(0, 1, max(len(subjects), 1)))
        subject_colors = dict(zip(subjects, palette))

    for subj in subjects:
        sub = table[table[subject_col] == subj].sort_values(x)
        ax.plot(
            sub[x], sub[y],
            "o-", color=subject_colors[subj], ms=5, lw=1.5, alpha=0.7, label=subj,
        )

    grp = table.groupby(x)[y].agg(["mean", "sem"]).reset_index()
    ax.errorbar(
        grp[x], grp["mean"], yerr=grp["sem"],
        fmt="s-", color=group_color, ms=6, lw=2, zorder=5, label=group_label,
    )
    return ax