"""Longitudinal summaries — per-subject delta + group mean/SEM."""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import pandas as pd


def longitudinal_summary(
    wide: pd.DataFrame,
    y: Optional[str] = None,
    ycols: Optional[Iterable[str]] = None,
    x: str = "session_n",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Optionally add per-subject deltas for ``ycols``, then mean/SEM for ``y``.

    Returns
    -------
    (wide_with_deltas, summary_or_none)
        ``summary`` is ``None`` when ``y`` is not provided.  Otherwise it
        has columns ``x``, ``mean``, ``sem``, ``n``.
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
