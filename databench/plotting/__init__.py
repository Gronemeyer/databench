"""Plotting utilities for databench analyses.

The active visual theme is controlled by :func:`set_theme` /
:func:`get_theme`.  All plotting modules in this package read the theme
lazily so that ``import databench.plotting`` alone never triggers heavy
rcParams mutations — call ``set_theme()`` once at the top of a script.

What's in this module
---------------------
* :func:`set_theme` / :func:`get_theme` — global visual theme.
* :func:`new_figure` — themed figure factory.
* :func:`style_axes` / :func:`style_figure` — spine and layout polish.
* :func:`font_size` — point size for a named text role (``"annotation"``,
  ``"panel_label"``, …), for the text a script draws itself.
* :func:`plot_metric_by_session` — per-subject lines + group mean ± SEM
  for any longitudinal ``(Subject, x, y)`` table.
* :func:`quickplot` — one-line "just plot this signal" for a single
  session, dispatched on the dataset schema's role.
* :func:`quickplot_group` — same dispatch, but across a SessionGroup
  (uses :func:`plot_metric_by_session` for scalar-per-session).

Submodules (``databench.plotting.<name>``) contain analysis-specific
plotters: ``oscillation``, ``traces``, ``treadmill``, ``overview``,
``mesomap``.  The default theme is ``gsipe_v1``.

Quick start
-----------
>>> from databench.plotting import set_theme, new_figure, style_axes
>>> set_theme()                           # light mode, gsipe_v1
>>> fig, ax = new_figure()                # themed figure + axes
>>> ax.plot(x, y)
>>> style_axes(ax)                        # clean spines / ticks

One-line plot from a schema-aware project::

    proj = Project("hfsa")
    fig = quickplot(proj.first_session(), source="pupil")
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench.plotting.style.gsipe_v1 import (
    Theme,
    clean_ax,
    FONT_SIZES,
    font_size,
)

# ── Module-level theme state ───────────────────────────────────────────────

_active_theme: Optional[Theme] = None


def set_theme(mode: str = "light", *, style: str = "gsipe_v1") -> Theme:
    """Activate a plotting theme globally and return it.

    Parameters
    ----------
    mode : ``"light"`` or ``"dark"``
    style : registered style name (default ``"gsipe_v1"``).
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


# ── Schema-aware "just plot it" defaults ──────────────────────────────────


def _pick_default_column(sess, source: str) -> str:
    """Pick a sensible default column for *source*.

    Prefers a schema-declared ``timeseries`` column for that source;
    falls back to the first introspected signal.
    """
    schema = getattr(sess, "schema", None)
    if schema is not None:
        for name, spec in schema.columns_for(source):
            if spec.role == "timeseries":
                return name
    sigs = sess.signals(source)
    if not sigs:
        raise ValueError(
            f"Source {source!r} has no plottable columns in this session."
        )
    return sigs[0]


def quickplot(
    sess,
    source: str,
    column: str | None = None,
    *,
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Figure:
    """One-line plot of a single session's signal, dispatched on schema role.

    Looks up ``role`` in the project schema:

    * ``"timeseries"`` (or unknown) → trace plot, unit-labelled Y axis.
    * ``"scalar"``                  → single-value text annotation.
    * ``"event"``                   → vertical raster of event times.

    Falls back to a trace plot when no schema is configured.  The user
    can still grab the Figure and customise; this is the "give me
    *something* in one line" entry point, not a finished publication
    figure.
    """
    get_theme()
    if column is None:
        column = _pick_default_column(sess, source)

    schema = getattr(sess, "schema", None)
    role = schema.role_of(source, column) if schema is not None else ""
    unit = schema.unit_of(source, column) if schema is not None else ""
    canonical = schema.resolve_source(source) if schema is not None else source
    title = label or f"{sess.label} — {canonical}/{column}"
    ylabel = f"{column} ({unit})" if unit else column

    if ax is None:
        fig, ax = new_figure()
    else:
        fig = ax.figure

    if role == "event":
        events = np.asarray(sess.signal(source, column), dtype=float).ravel()
        for t in events:
            ax.axvline(t, color="black", lw=0.5, alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_yticks([])
        ax.set_title(title)
    elif role == "scalar":
        value = np.asarray(sess.signal(source, column)).ravel()
        ax.text(0.5, 0.5, f"{column} = {value[0]:.3g} {unit}".strip(),
                transform=ax.transAxes, ha="center", va="center", fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title)
    else:
        # timeseries / unknown
        try:
            t = sess.time(source)
        except Exception:
            t = np.arange(sess.signal(source, column).size)
            ax.set_xlabel("Sample")
        else:
            ax.set_xlabel("Time (s)")
        y = sess.signal(source, column)
        m = min(len(t), len(y))
        ax.plot(t[:m], y[:m], lw=1.0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    style_axes(ax)
    return fig


def quickplot_group(
    group,
    source: str,
    column: str | None = None,
    *,
    reducer: str = "mean",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """One-line longitudinal plot for a SessionGroup, schema-dispatched.

    Reduces the per-session signal to a single number (``mean`` by
    default), then renders per-subject lines + group mean ± SEM via
    :func:`plot_metric_by_session`.  Designed for the "habituation
    across days" case.
    """
    get_theme()
    sample = next(iter(group), None)
    if sample is None:
        raise ValueError("Empty SessionGroup")

    if column is None:
        column = _pick_default_column(sample, source)

    schema = getattr(sample, "schema", None)
    unit = schema.unit_of(source, column) if schema is not None else ""
    canonical = schema.resolve_source(source) if schema is not None else source

    rows: list[dict] = []
    from databench.utils import session_to_int
    for sess in group:
        try:
            y = sess.signal(source, column)
        except Exception:
            continue
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue
        if reducer == "mean":
            value = float(np.mean(y))
        elif reducer == "median":
            value = float(np.median(y))
        elif reducer == "max":
            value = float(np.max(y))
        else:
            raise ValueError(f"Unknown reducer {reducer!r}")
        rows.append({
            "Subject": sess.subject,
            "Session": sess.session,
            "day": session_to_int(sess.session),
            "value": value,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(
            f"No usable data for {canonical}/{column} across the group."
        )

    if ax is None:
        fig, ax = new_figure(width=7.5, height_per_row=3.6)
    else:
        fig = ax.figure

    plot_metric_by_session(table, x="day", y="value", ax=ax)
    ylabel = f"{column} ({unit})" if unit else column
    ax.set_xlabel("Session day")
    ax.set_ylabel(f"{reducer} {ylabel}")
    ax.set_title(f"{canonical}/{column} across sessions")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    style_axes(ax)
    return fig