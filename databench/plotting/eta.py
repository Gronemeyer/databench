"""ETA (event-triggered average) plotting classes.

All plotters accept an :class:`~databench.analysis.base.AnalysisResult` and
return ``(fig, axes)``.

* :class:`EtaConditionPlotter` — group mean ± SEM lines by condition (one subplot per ROI).
* :class:`EtaSubjectPlotter` — per-subject condition lines (one subplot per ROI × Subject).
* :class:`EtaAllDaysAveragePlotter` — pooled average across all sessions/days.
* :class:`EtaLongitudinalHeatmapPlotter` — day × time heatmap of group ETA.
* :class:`EtaLongitudinalMetricPlotter` — scalar metric over days.
* :class:`EtaPrePostDiffBoxplot` — boxplot of pre/post scalar differences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench.analysis.base import AnalysisResult
from databench.plotting import style_axes, get_theme
from databench.plotting.style import CONDITION_COLORS, CONDITION_ORDER
from databench.plotting.base import Plotter


# ---------------------------------------------------------------------------
# Condition-based ETA (group mean ± SEM)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EtaConditionPlotter(Plotter):
    """Plot group mean ± SEM ETA traces, one subplot per ROI, lines by condition."""

    name: str = "eta_condition"
    rois: tuple[str, ...] = ()
    task: str = "task-movies"
    event_type: str = "onset"
    baseline: tuple[float, float] | None = None
    ncols: int = 2
    condition_colors: Dict[str, str] = field(default_factory=lambda: dict(CONDITION_COLORS))
    condition_order: tuple[str, ...] = tuple(CONDITION_ORDER)

    def plot(self, result: AnalysisResult):
        rois = list(self.rois)
        eta_group = result.data["eta_group"]
        d = eta_group.query("Task == @self.task and EventType == @self.event_type and ROI in @rois").copy()

        n = len(rois)
        nrows = int(np.ceil(n / self.ncols))
        fig, axes = plt.subplots(
            nrows, self.ncols,
            figsize=(4.0 * self.ncols, 3.2 * nrows + 1.0),
            sharex=True, sharey="row",
        )
        axes = np.atleast_1d(axes).ravel()

        for ax, roi in zip(axes, rois):
            dd = d[d["ROI"] == roi]
            for cond in self.condition_order:
                g = dd[dd["Condition"] == cond]
                if g.empty:
                    continue
                color = self.condition_colors.get(cond)
                ax.plot(g["rel_time"], g["mean"], label=cond, color=color)
                ax.fill_between(
                    g["rel_time"],
                    g["mean"] - g["sem"],
                    g["mean"] + g["sem"],
                    alpha=0.2, color=color,
                )
            ax.axvline(0, color=get_theme().fg, lw=0.6, alpha=0.6)
            ax.axhline(0, color=get_theme().fg, lw=0.4, alpha=0.4)
            ax.set_title(roi)
            style_axes(ax)

        for ax in axes[n:]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels,
                loc="upper center", bbox_to_anchor=(0.5, 0.95),
                ncol=min(6, len(labels)), frameon=False,
            )

        baseline_label = (
            "no baseline subtraction"
            if self.baseline is None
            else f"baseline-subtracted [{self.baseline[0]:g}, {self.baseline[1]:g}] s"
        )
        fig.supxlabel(f"Time relative to run {self.event_type} (s)", y=0.01)
        fig.supylabel("Group mean ± SEM (subject-averaged)", x=0.02)
        fig.suptitle(
            f"ETA across ROIs — {self.task} — run {self.event_type}\n{baseline_label}",
        )
        fig.tight_layout(rect=[0.04, 0.03, 1.0, 0.88])
        if handles:
            fig.subplots_adjust(top=0.82)
        return fig, axes


# ---------------------------------------------------------------------------
# Per-subject ETA lines
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EtaSubjectPlotter(Plotter):
    """Plot per-subject ETA traces, one subplot per (ROI × Subject), lines by condition."""

    name: str = "eta_subject"
    rois: tuple[str, ...] = ()
    task: str = "task-spont"
    event_type: str = "onset"
    baseline: tuple[float, float] | None = None
    ncols: int = 2
    condition_colors: Dict[str, str] = field(default_factory=lambda: dict(CONDITION_COLORS))
    condition_order: tuple[str, ...] = tuple(CONDITION_ORDER)

    def plot(self, result: AnalysisResult):
        rois = list(self.rois)
        eta_subj = result.data["eta_subj"]
        d = eta_subj.query("Task == @self.task and EventType == @self.event_type and ROI in @rois").copy()

        subjects = d["Subject"].dropna().unique().tolist()
        n = len(rois) * len(subjects)
        nrows = int(np.ceil(n / self.ncols))
        fig, axes = plt.subplots(
            nrows, self.ncols,
            figsize=(4.0 * self.ncols, 3.2 * nrows + 1.0),
            sharex=True, sharey="row",
        )
        axes = np.atleast_1d(axes).ravel()

        panel_specs = [(roi, subj) for roi in rois for subj in subjects]
        for ax, (roi, subj) in zip(axes, panel_specs):
            dd = d[(d["ROI"] == roi) & (d["Subject"] == subj)]
            for cond in self.condition_order:
                g = dd[dd["Condition"] == cond]
                if g.empty:
                    continue
                color = self.condition_colors.get(cond)
                ax.plot(g["rel_time"], g["value"], label=cond, color=color)
            ax.axvline(0, color=get_theme().fg, lw=0.6, alpha=0.6)
            ax.axhline(0, color=get_theme().fg, lw=0.4, alpha=0.4)
            ax.set_title(f"{subj} | {roi}")
            style_axes(ax)

        for ax in axes[n:]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels,
                loc="upper center", bbox_to_anchor=(0.5, 0.95),
                ncol=min(6, len(labels)), frameon=False,
            )

        baseline_label = (
            "no baseline subtraction"
            if self.baseline is None
            else f"baseline-subtracted [{self.baseline[0]:g}, {self.baseline[1]:g}] s"
        )
        fig.supxlabel(f"Time relative to run {self.event_type} (s)", y=0.01)
        fig.supylabel("Subject mean (event-averaged)", x=0.02)
        fig.suptitle(
            f"ETA by subject — {self.task} — run {self.event_type}\n{baseline_label}",
        )
        fig.tight_layout(rect=[0.04, 0.03, 1.0, 0.88])
        if handles:
            fig.subplots_adjust(top=0.82)
        return fig, axes


# ---------------------------------------------------------------------------
# Longitudinal: pooled across all days
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EtaAllDaysAveragePlotter(Plotter):
    """Plot ETA pooled across all days, one subplot per ROI."""

    name: str = "eta_all_days_avg"
    rois: tuple[str, ...] = ()
    task: str = "task-widefield"
    event_type: str = "onset"
    baseline: tuple[float, float] | None = None
    ncols: int = 2

    def plot(self, result: AnalysisResult):
        rois = list(self.rois)
        eta_group = result.data["eta_group"]
        d = eta_group.query("Task == @self.task and EventType == @self.event_type and ROI in @rois").copy()

        n = len(rois)
        nrows = int(np.ceil(n / self.ncols))
        fig, axes = plt.subplots(
            nrows, self.ncols,
            figsize=(4.0 * self.ncols, 3.2 * nrows + 1.0),
            sharex=True, sharey="row",
        )
        axes = np.atleast_1d(axes).ravel()

        line_color = get_theme().colors[0]
        for ax, roi in zip(axes, rois):
            g = d[d["ROI"] == roi].sort_values("rel_time")
            if not g.empty:
                ax.plot(g["rel_time"], g["mean"], color=line_color, lw=1.8)
                ax.fill_between(
                    g["rel_time"],
                    g["mean"] - g["sem"],
                    g["mean"] + g["sem"],
                    alpha=0.2, color=line_color,
                )
            ax.axvline(0, color=get_theme().fg, lw=0.6, alpha=0.6)
            ax.axhline(0, color=get_theme().fg, lw=0.4, alpha=0.4)
            ax.set_title(roi)
            style_axes(ax)

        for ax in axes[n:]:
            ax.axis("off")

        baseline_label = (
            "no baseline subtraction"
            if self.baseline is None
            else f"baseline-subtracted [{self.baseline[0]:g}, {self.baseline[1]:g}] s"
        )
        fig.supxlabel(f"Time relative to run {self.event_type} (s)", y=0.01)
        fig.supylabel("Group mean \u00b1 SEM (pooled across days)", x=0.02)
        fig.suptitle(
            f"ETA across ROIs \u2014 {self.task} \u2014 run {self.event_type}\n{baseline_label}",
        )
        fig.tight_layout(rect=[0.04, 0.03, 1.0, 0.90])
        return fig, axes


# ---------------------------------------------------------------------------
# Longitudinal: heatmap
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EtaLongitudinalHeatmapPlotter(Plotter):
    """Day × time heatmap of group ETA, one subplot per ROI."""

    name: str = "eta_longitudinal_heatmap"
    rois: tuple[str, ...] = ()
    task: str = "task-widefield"
    event_type: str = "onset"
    ncols: int = 2
    cmap: str = "RdBu_r"
    vmin: float | None = None
    vmax: float | None = None

    def plot(self, result: AnalysisResult):
        rois = list(self.rois)
        eta_long = result.data["eta_longitudinal"]
        d = eta_long.query("Task == @self.task and EventType == @self.event_type and ROI in @rois").copy()

        n = len(rois)
        nrows = int(np.ceil(n / self.ncols))
        fig, axes = plt.subplots(
            nrows, self.ncols,
            figsize=(7 * self.ncols / 2, 3 * nrows),
            sharex=True, sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()

        image = None
        used_axes = []
        for ax, roi in zip(axes, rois):
            dd = d[d["ROI"] == roi]
            if dd.empty:
                ax.axis("off")
                continue

            days = np.array(sorted(dd["day_n"].unique()), dtype=float)
            rel_times = np.array(sorted(dd["rel_time"].unique()), dtype=float)
            mat = (
                dd.pivot(index="day_n", columns="rel_time", values="mean")
                .reindex(index=days, columns=rel_times)
                .to_numpy(dtype=float)
            )

            image = ax.imshow(
                mat,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=[rel_times.min(), rel_times.max(), days.min() - 0.5, days.max() + 0.5],
                cmap=self.cmap,
                vmin=self.vmin,
                vmax=self.vmax,
            )
            ax.axvline(0, color=get_theme().fg, lw=0.6, alpha=0.6)
            ax.set_yticks(days)
            ax.set_title(roi)
            style_axes(ax)
            used_axes.append(ax)

        for ax in axes[n:]:
            ax.axis("off")

        fig.supxlabel(f"Time relative to run {self.event_type} (s)", y=0.03)
        fig.supylabel("Day / session number", x=0.03)
        fig.suptitle(f"Longitudinal ETA heatmap — {self.task} — run {self.event_type}", y=0.98)
        if image is not None and used_axes:
            fig.colorbar(image, ax=used_axes, shrink=0.9, label="Mean dF/F")
        fig.subplots_adjust(left=0.11, right=0.93, bottom=0.11, top=0.88, hspace=0.30, wspace=0.30)
        return fig, axes


# ---------------------------------------------------------------------------
# Longitudinal: metric over days
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EtaLongitudinalMetricPlotter(Plotter):
    """Scalar metric over days, one subplot per ROI."""

    name: str = "eta_longitudinal_metric"
    rois: tuple[str, ...] = ()
    task: str = "task-widefield"
    event_type: str = "onset"
    metric_window: tuple[float, float] = (0.0, 1.0)
    ncols: int = 2

    def plot(self, result: AnalysisResult):
        rois = list(self.rois)
        metric_group = result.data["eta_day_metric_group"]
        d = metric_group.query("Task == @self.task and EventType == @self.event_type and ROI in @rois").copy()

        n = len(rois)
        nrows = int(np.ceil(n / self.ncols))
        fig, axes = plt.subplots(
            nrows, self.ncols,
            figsize=(7 * self.ncols / 2, 3 * nrows),
            sharex=True, sharey="row",
        )
        axes = np.atleast_1d(axes).ravel()

        line_color = get_theme().colors[2]
        for ax, roi in zip(axes, rois):
            g = d[d["ROI"] == roi].sort_values("day_n")
            if not g.empty:
                ax.plot(g["day_n"], g["mean"], marker="o", color=line_color, lw=1.8)
                ax.fill_between(
                    g["day_n"],
                    g["mean"] - g["sem"],
                    g["mean"] + g["sem"],
                    alpha=0.2, color=line_color,
                )
            ax.axhline(0, color=get_theme().fg, lw=0.4, alpha=0.4)
            ax.set_title(roi)
            style_axes(ax)

        for ax in axes[n:]:
            ax.axis("off")

        fig.supxlabel("Day / session number", y=0.03)
        fig.supylabel("Mean ETA value (subject-averaged)", x=0.03)
        fig.suptitle(
            f"Longitudinal ETA summary — {self.task} — run {self.event_type}\n"
            f"metric window [{self.metric_window[0]:g}, {self.metric_window[1]:g}] s",
            y=0.98,
        )
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.11, top=0.85, hspace=0.30, wspace=0.30)
        return fig, axes


# ---------------------------------------------------------------------------
# Pre/post difference boxplot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EtaPrePostDiffBoxplot(Plotter):
    """Boxplot of per-subject pre/post dF/F differences with jittered dots."""

    name: str = "eta_prepost_diff_boxplot"
    rois: tuple[str, ...] = ("L_MOp", "R_MOp", "L_MOs", "R_MOs")
    task: str = "task-spont"
    condition_col: str = "Condition"
    condition_order: tuple[str, ...] = ("baseline", "saline", "ethanol_low", "ethanol_high")
    event_order: tuple[str, ...] = ("onset", "offset")
    condition_colors: Dict[str, str] = field(default_factory=lambda: dict(CONDITION_COLORS))

    def plot(self, result: AnalysisResult):
        d = result.data["subject_diff"].copy()
        d = d.loc[d["Task"] == self.task]

        rois = list(self.rois)
        conds = list(self.condition_order)
        events = list(self.event_order)

        nrows = len(events)
        ncols = len(rois)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.1 * ncols, 2.9 * nrows),
            sharex=False, sharey="row",
        )
        axes = np.atleast_2d(axes)
        x = np.arange(len(conds), dtype=float)

        for i, event_type in enumerate(events):
            for j, roi in enumerate(rois):
                ax = axes[i, j]
                dd = d[(d["EventType"] == event_type) & (d["ROI"] == roi)]

                data_by_cond = []
                for cond in conds:
                    y = dd.loc[dd[self.condition_col] == cond, "diff_mean"].dropna().to_numpy()
                    data_by_cond.append(y)

                bp = ax.boxplot(
                    data_by_cond,
                    positions=x,
                    widths=0.62,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": get_theme().fg, "linewidth": 1.2},
                    whiskerprops={"color": get_theme().fg, "linewidth": 1.0},
                    capprops={"color": get_theme().fg, "linewidth": 1.0},
                    boxprops={"linewidth": 1.0, "edgecolor": get_theme().fg},
                )

                for patch, cond in zip(bp["boxes"], conds):
                    patch.set_facecolor(self.condition_colors.get(cond, get_theme().p["tick"]))
                    patch.set_alpha(0.45)

                for k, cond in enumerate(conds):
                    y = data_by_cond[k]
                    if y.size == 0:
                        continue
                    jitter = (np.random.rand(y.size) - 0.5) * 0.20
                    ax.scatter(
                        np.full(y.size, x[k], dtype=float) + jitter,
                        y,
                        s=20,
                        color=self.condition_colors.get(cond, get_theme().p["tick"]),
                        edgecolors=get_theme().surface,
                        linewidths=0.4,
                        alpha=0.9,
                        zorder=3,
                    )

                ax.axhline(0, color=get_theme().fg, lw=0.4, alpha=0.4)
                ax.set_xticks(x)
                ax.set_xticklabels(conds, rotation=25, ha="right")
                style_axes(ax)
                ax.grid(axis="y", alpha=0.25)

                if i == 0:
                    ax.set_title(roi)
                if j == 0:
                    ax.set_ylabel(event_type)

        fig.suptitle(
            "ETA pre/post difference by condition\n"
            "onset: [0,1] - [-1,0), offset: [-1,0) - [0,1]",
            y=0.99,
        )
        fig.supxlabel("Condition", y=0.04)
        fig.supylabel("Mean dF/F difference", x=0.02)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.14, hspace=0.35, wspace=0.18)
        return fig, axes
