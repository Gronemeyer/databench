from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import pandas as pd

from databench._utils._logger import log_this_fr


@dataclass(frozen=True)
class AnalysisResult:
    """Container returned by ``Analysis.run()``.

    Attributes
    ----------
    name : str
        Identifier matching the originating analysis.
    data : Any
        Primary payload — typically ``dict[str, DataFrame]``.
    table : Any
        Optional summary table.
    schema : dict
        Documents keys present in ``data`` and their types, e.g.
        ``{"eta_group": "DataFrame", "events": "DataFrame"}``.
    """

    name: str
    data: Any = None
    table: Any = None
    figs: list = field(default_factory=list)
    files: list = field(default_factory=list)
    context: Any = None
    meta: dict = field(default_factory=dict)
    schema: dict = field(default_factory=dict)


def _warn_missing_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    origin: str,
) -> list[str]:
    """Emit a warning for every column in *required* that is absent from *df*."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.warn(
            f"[{origin}] Missing columns: {', '.join(missing)}",
            stacklevel=3,
        )
    return missing


@dataclass(frozen=True)
class Analysis:
    """Base class for analyses.

    Subclasses override ``run(df) -> AnalysisResult``.
    All parameters live on the frozen dataclass — no **kwargs.
    Declare ``required_columns`` for auto-validation.
    """

    name: str
    required_columns: tuple[str, ...] = ()

    @log_this_fr
    def run(self, df: pd.DataFrame) -> AnalysisResult:  # pragma: no cover - interface
        raise NotImplementedError

    @log_this_fr
    def plot(self, result: AnalysisResult):  # pragma: no cover - optional
        return None

    @log_this_fr
    def save(self, result: AnalysisResult) -> list:  # pragma: no cover - optional
        return []

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Warn about missing required columns; returns the list of missing names."""
        return _warn_missing_columns(df, self.required_columns, self.name)


@dataclass(frozen=True)
class StatFn:
    name: str
    label: str
    plotter: str = "longitudinal"
    color: Optional[str] = None

    def __call__(self, *args, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class AxisFn:
    name: str
    label: str