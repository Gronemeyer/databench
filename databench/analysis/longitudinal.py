from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import pandas as pd

from .base import StatFn, Analysis, AnalysisResult


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
        cols = [c for c in ycols if c in out.columns]
        if cols:
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
    if y not in out.columns:
        raise KeyError(f"{y!r} not in wide.columns")
    g = out[[x, y]].dropna().groupby(x)[y]
    stats = g.agg(mean="mean", sem="sem", n="count").reset_index().sort_values(x)
    return out, stats


@dataclass(frozen=True)
class LongitudinalAnalysis(Analysis):
    name: str = "longitudinal_summary"

    def run(
        self,
        wide: pd.DataFrame,
        y: Optional[str] = None,
        ycols: Optional[Iterable[str]] = None,
        x: str = "session_n",
    ) -> AnalysisResult:
        out, stats = longitudinal_summary(wide, y=y, ycols=ycols, x=x)
        meta = {"x": x, "y": y, "ycols": list(ycols) if ycols else None}
        return AnalysisResult(name=self.name, data=out, table=stats, meta=meta)

    def plot(self, result: AnalysisResult, **kwargs):
        if result.table is None:
            return None
        x = kwargs.pop("x", result.meta.get("x", "session_n"))
        y = kwargs.pop("y", result.meta.get("y"))
        if y is None:
            raise ValueError("Provide y for plotting longitudinal summary.")
        title = kwargs.pop("title", f"{y} across sessions")
        color = kwargs.pop("color", "#1f77b4")
        y_label = kwargs.pop("y_label", y)
        from ..plotting.core import plot_mean_sem

        return plot_mean_sem(result.table, x=x, y_label=y_label, title=title, color=color, **kwargs)
