from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis import longitudinal_summary
from ..features import FeatureFn
from .base import Plotter

_COLORS = {"primary": "#1f77b4", "secondary": "#9467bd", "accent": "#2ca02c"}

PLOTTERS = {
    "longitudinal": "plot_feature_longitudinal",
    "boxplot": "plot_feature_boxplot",
}


def plot_mean_sem(stats, x: str, y_label: str, title: str, color: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = None
    ax.plot(stats[x], stats["mean"], color=color, lw=2, marker="o", label=y_label)
    ax.fill_between(stats[x], stats["mean"] - stats["sem"], stats["mean"] + stats["sem"],
                    color=color, alpha=0.2, label="± SEM")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    return fig, ax


def plot_boxplot_mean_sem(
    wide,
    y: str,
    x: str = "session_n",
    title: Optional[str] = None,
    color: Optional[str] = None,
    y_label: Optional[str] = None,
    x_label: Optional[str] = None,
    jitter: float = 0.08,
    connect_subjects: bool = True,
    subject_alpha: float = 0.25,
    ax=None,
):
    data = wide[[x, y]].dropna()
    if data.empty:
        raise ValueError(f"No data to plot for {y!r}.")

    if "Subject" in wide.columns:
        data = data.copy()
        data["Subject"] = wide.loc[data.index, "Subject"].values
    elif isinstance(wide.index, pd.MultiIndex) and "Subject" in wide.index.names:
        data = data.copy()
        data["Subject"] = data.index.get_level_values("Subject")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = None

    color = color or _COLORS["primary"]
    title = title or f"{y} across sessions"

    grouped = data.groupby(x)[y]
    x_vals = np.array(sorted(grouped.groups.keys()), dtype=float)
    y_vals = [grouped.get_group(v).to_numpy() for v in x_vals]

    bp = ax.boxplot(
        y_vals,
        positions=x_vals,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#444", "linewidth": 1.2},
        boxprops={"edgecolor": color, "linewidth": 1.2},
        whiskerprops={"color": color},
        capprops={"color": color},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.15)

    rng = np.random.default_rng(0)
    for x0, vals in zip(x_vals, y_vals):
        if vals.size == 0:
            continue
        jitter_x = x0 + rng.uniform(-jitter, jitter, size=vals.size)
        ax.scatter(jitter_x, vals, s=20, alpha=0.7, color=color, edgecolors="none")

    if connect_subjects and "Subject" in data.columns:
        sub = data[["Subject", x, y]].dropna()
        if "Subject" in sub.index.names:
            sub = sub.reset_index(drop=True)
        for _, sub_df in sub.groupby("Subject"):
            sub_df = sub_df.sort_values(x)
            ax.plot(
                sub_df[x].to_numpy(),
                sub_df[y].to_numpy(),
                color="#555",
                alpha=subject_alpha,
                lw=1.0,
            )

    means = np.array([np.nanmean(v) for v in y_vals], dtype=float)
    sems = np.array(
        [np.nanstd(v, ddof=1) / np.sqrt(np.sum(np.isfinite(v))) if np.sum(np.isfinite(v)) > 1 else 0.0 for v in y_vals],
        dtype=float,
    )
    ax.errorbar(x_vals, means, yerr=sems, fmt="o", color="black", lw=1.4, ms=4, label="Mean ± SEM")

    ax.set_title(title)
    ax.set_xlabel(x_label or x)
    ax.set_ylabel(y_label or y)
    ax.set_xticks(x_vals)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False)
    return fig, ax


def plot_feature_longitudinal(
    wide,
    y: str,
    x: str = "session_n",
    title: Optional[str] = None,
    color: Optional[str] = None,
    y_label: Optional[str] = None,
    x_label: Optional[str] = None,
):
    _, stats = longitudinal_summary(wide, y=y, x=x)
    color = color or _COLORS["primary"]
    title = title or f"{y} across sessions"
    fig, ax = plot_mean_sem(stats, x=x, y_label=y_label or y, title=title, color=color)
    if x_label:
        ax.set_xlabel(x_label)
    return fig, ax


def plot_feature_boxplot(
    wide,
    y: str,
    x: str = "session_n",
    title: Optional[str] = None,
    color: Optional[str] = None,
    y_label: Optional[str] = None,
    x_label: Optional[str] = None,
):
    return plot_boxplot_mean_sem(
        wide,
        y=y,
        x=x,
        title=title,
        color=color,
        y_label=y_label,
        x_label=x_label,
    )


def plot_two_panel_longitudinal(
    wide,
    y1: str,
    y2: str,
    x: str = "session_n",
    titles: Tuple[str, str] = ("", ""),
    colors: Tuple[str, str] = (None, None),
    y_labels: Tuple[Optional[str], Optional[str]] = (None, None),
    x_label: Optional[str] = None,
):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    t1 = titles[0] or f"{y1} across sessions"
    t2 = titles[1] or f"{y2} across sessions"
    c1 = colors[0] or _COLORS["accent"]
    c2 = colors[1] or _COLORS["secondary"]
    _, stats1 = longitudinal_summary(wide, y=y1, x=x)
    _, stats2 = longitudinal_summary(wide, y=y2, x=x)
    plot_mean_sem(stats1, x, y_labels[0] or y1, t1, c1, ax=axes[0])
    plot_mean_sem(stats2, x, y_labels[1] or y2, t2, c2, ax=axes[1])
    if x_label:
        axes[0].set_xlabel(x_label)
        axes[1].set_xlabel(x_label)
    fig.tight_layout()
    return fig, axes


def _resolve_plotter(name: str):
    func_name = PLOTTERS.get(name)
    if not func_name:
        raise KeyError(f"Unknown plotter: {name!r}")
    func = globals().get(func_name)
    if func is None:
        raise KeyError(f"Plotter not found: {func_name!r}")
    return func


def plot_feature(
    wide,
    feature: FeatureFn,
    x: str = "session_n",
    color: Optional[str] = None,
    x_label: Optional[str] = None,
):
    """Route plotting based on the feature's plotter hint and label."""
    plotter = _resolve_plotter(feature.plotter)
    use_color = color or feature.color
    return plotter(
        wide,
        y=feature.name,
        x=x,
        title=f"{feature.label} across sessions",
        color=use_color,
        y_label=feature.label,
        x_label=x_label,
    )


@dataclass(frozen=True)
class FeaturePlotter(Plotter):
    name: str = "feature"

    def plot(self, wide, feature: FeatureFn, **kwargs):
        return plot_feature(wide, feature, **kwargs)


@dataclass(frozen=True)
class LongitudinalPlotter(Plotter):
    name: str = "longitudinal"

    def plot(self, wide, y: str, **kwargs):
        return plot_feature_longitudinal(wide, y=y, **kwargs)
