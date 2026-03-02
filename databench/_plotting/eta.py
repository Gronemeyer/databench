"""ETA plotting helpers — used by EtaResult, not called by users."""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_eta_by_condition(
    group_means: pd.DataFrame,
    *,
    event: str,
    rois: list[str],
    conditions: list[str] | None = None,
    condition_colors: Dict[str, str] | None = None,
    baseline_s: tuple[float, float] | None = None,
    ncols: int = 2,
    task: str | None = None,
) -> plt.Figure:
    """Plot group mean ± SEM ETA traces: one subplot per ROI, lines per condition.

    Parameters
    ----------
    group_means : DataFrame
        Must have columns: ``Condition``, ``EventType``, ``ROI``, ``rel_time``,
        ``mean``, ``sem``.  Optionally ``Task``.
    event : str
        Event type to plot (e.g. ``"onset"``).
    rois : list of str
        ROIs to plot (one subplot each).
    conditions : list of str, optional
        Condition ordering.  If None, uses unique values from data.
    condition_colors : dict, optional
        Mapping from condition name to color string.
    baseline_s : tuple, optional
        Baseline window for label.
    """
    d = group_means.query("EventType == @event and ROI in @rois").copy()
    if task is not None and "Task" in d.columns:
        d = d.query("Task == @task")

    if conditions is None:
        conditions = sorted(d["Condition"].unique())
    if condition_colors is None:
        default_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        condition_colors = {c: default_palette[i % len(default_palette)] for i, c in enumerate(conditions)}

    n = len(rois)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols / 2, 3 * nrows),
        sharex=True, sharey="row",
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, roi in zip(axes, rois):
        dd = d[d["ROI"] == roi]
        for cond in conditions:
            g = dd[dd["Condition"] == cond].sort_values("rel_time")
            if g.empty:
                continue
            color = condition_colors.get(cond, "#333333")
            ax.plot(g["rel_time"], g["mean"], label=cond, color=color)
            ax.fill_between(
                g["rel_time"],
                g["mean"] - g["sem"],
                g["mean"] + g["sem"],
                alpha=0.2, color=color,
            )
        ax.axvline(0, color="k", lw=1)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_title(roi, fontsize=10)
        ax.tick_params(axis="both", labelsize=9)

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.92),
            ncol=min(4, len(labels)), frameon=False,
        )

    baseline_label = (
        "no baseline subtraction"
        if baseline_s is None
        else f"baseline-subtracted [{baseline_s[0]:g}, {baseline_s[1]:g}] s"
    )
    fig.supxlabel(f"Time relative to {event} (s)", y=0.02)
    fig.supylabel("Group mean ± SEM (subject-averaged)", x=0.03)
    fig.suptitle(
        f"ETA across ROIs — {event}\n{baseline_label}",
        y=0.98,
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.09, top=0.84, hspace=0.28, wspace=0.35)
    return fig
