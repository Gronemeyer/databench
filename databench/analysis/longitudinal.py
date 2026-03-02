from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import pandas as pd

from databench.analysis.base import StatFn, Analysis, AnalysisResult
from databench.registry import register_analysis


@dataclass(frozen=True)
class MeanSEM(StatFn):
    name: str = "mean_sem"
    label: str = "Mean ± SEM"

    def __call__(self, wide, y: str, x: str = "session_n"):
        _, stats = longitudinal_summary(wide, y=y, x=x)
        return stats


def longitudinal_summary(
    wide: pd.DataFrame,
    y: Optional[str] = None,
    ycols: Optional[Iterable[str]] = None,
    x: str = "session_n",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Optionally add deltas for ycols, then compute summary for y.

    Returns a tuple: (wide_with_deltas, summary_or_none)
    """
    out = wide.copy()
    if ycols:
        cols = list(ycols)
        baseline = out.groupby("Subject")[cols].transform("first")
        out = pd.concat(
            [
                out,
                (out[cols] - baseline).add_prefix("d_"),
                (out[cols] / baseline - 1.0).add_prefix("pct_"),
            ],
            axis=1,
        )

    if y is None:
        return out, None
    g = out[[x, y]].dropna().groupby(x)[y]
    stats = g.agg(mean="mean", sem="sem", n="count").reset_index().sort_values(x)
    return out, stats


@register_analysis
@dataclass(frozen=True)
class LongitudinalAnalysis(Analysis):
    name: str = "longitudinal_summary"
    y: Optional[str] = None
    ycols: Optional[tuple[str, ...]] = None
    x: str = "session_n"

    def run(
        self,
        wide: pd.DataFrame,
    ) -> AnalysisResult:
        out, stats = longitudinal_summary(wide, y=self.y, ycols=self.ycols, x=self.x)
        meta = {"x": self.x, "y": self.y, "ycols": list(self.ycols) if self.ycols else None}
        return AnalysisResult(name=self.name, data=out, table=stats, meta=meta)

    def plot(self, result: AnalysisResult):
        x = result.meta.get("x", "session_n")
        y = result.meta.get("y")
        title = f"{y} across sessions"
        color = "#1f77b4"
        y_label = y
        from databench.plotting.core import plot_mean_sem

        return plot_mean_sem(result.table, x=x, y_label=y_label, title=title, color=color)
