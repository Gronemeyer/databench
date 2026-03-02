from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench.analysis.longitudinal import longitudinal_summary
from databench.features.base import FeatureFn
from databench.plotting.base import Plotter
from databench.registry import register_plotter
from databench._utils._logger import get_logger

_COLORS = {"primary": "#1f77b4", "secondary": "#9467bd", "accent": "#2ca02c"}
_LOGGER = get_logger("databench.plotting")


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
    _LOGGER.debug(
        f"Boxplot: wide_shape={wide.shape} columns={list(wide.columns)[:12]} index_names={list(wide.index.names)}"
    )
    _LOGGER.debug(f"Boxplot: x={x} y={y}")
    data = wide[[x, y]].dropna()
    data = data.copy()
    data["Subject"] = data.index.get_level_values("Subject")
    fig, ax = plt.subplots(figsize=(8, 4))

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
        jitter_x = x0 + rng.uniform(-jitter, jitter, size=vals.size)
        ax.scatter(jitter_x, vals, s=20, alpha=0.7, color=color, edgecolors="none")

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
    ax.set_xlabel(x_label or x)
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
    axes[0].set_xlabel(x_label or x)
    axes[1].set_xlabel(x_label or x)
    fig.tight_layout()
    return fig, axes


PLOTTERS: dict[str, Callable[..., tuple]] = {
    "longitudinal": plot_feature_longitudinal,
    "boxplot": plot_feature_boxplot,
}


def _resolve_plotter(name: str):
    return PLOTTERS[name]


def plot_feature(
    wide,
    feature: Union[FeatureFn, "DerivedColumnSpec", Mapping[str, object], str],
    x: str = "session_n",
    color: Optional[str] = None,
    x_label: Optional[str] = None,
    **kwargs,
):
    """Route plotting based on the feature's plotter hint and label."""
    spec = _coerce_plot_spec(feature)
    plotter = _resolve_plotter(spec.plotter)
    use_color = color or spec.color
    return plotter(
        wide,
        y=spec.name,
        x=x,
        title=f"{spec.label} across sessions",
        color=use_color,
        y_label=spec.label,
        x_label=x_label,
        **kwargs,
    )


@register_plotter
@dataclass(frozen=True)
class FeaturePlotter(Plotter):
    """Plot a feature from a wide table (session_table).

    Set ``feature`` at construction time — accepts a FeatureFn,
    DerivedColumnSpec, dict, or string.
    """
    name: str = "feature"
    feature: Union[FeatureFn, DerivedColumnSpec, Mapping[str, object], str, None] = None
    x: str = "session_n"
    color: Optional[str] = None
    x_label: Optional[str] = None

    def plot(self, result):
        """Plot feature from result.data (a wide DataFrame)."""
        wide = result.data if hasattr(result, 'data') else result
        if self.feature is None:
            raise ValueError("FeaturePlotter requires a feature set at construction.")
        return plot_feature(wide, self.feature, x=self.x, color=self.color, x_label=self.x_label)


@dataclass(frozen=True)
class DerivedColumnSpec:
    name: str
    label: str
    color: Optional[str] = None
    plotter: str = "longitudinal"


def _coerce_plot_spec(feature: Union[FeatureFn, DerivedColumnSpec, Mapping[str, object], str]):
    if isinstance(feature, FeatureFn):
        return feature
    if isinstance(feature, DerivedColumnSpec):
        return feature
    if isinstance(feature, str):
        return DerivedColumnSpec(name=feature, label=feature)
    if isinstance(feature, Mapping):
        name = str(feature["name"])
        label = str(feature.get("label", name))
        color = feature.get("color")
        plotter = str(feature.get("plotter", "longitudinal"))
        return DerivedColumnSpec(name=name, label=label, color=color, plotter=plotter)
    return feature


@register_plotter
@dataclass(frozen=True)
class LongitudinalPlotter(Plotter):
    """Plot a longitudinal summary (mean ± SEM over sessions).

    Set ``y`` at construction time.
    """
    name: str = "longitudinal"
    y: str = ""
    x: str = "session_n"
    title: Optional[str] = None
    color: Optional[str] = None
    y_label: Optional[str] = None
    x_label: Optional[str] = None

    def plot(self, result):
        """Plot from result.data (a wide DataFrame)."""
        wide = result.data if hasattr(result, 'data') else result
        return plot_feature_longitudinal(
            wide, y=self.y, x=self.x,
            title=self.title, color=self.color,
            y_label=self.y_label, x_label=self.x_label,
        )
