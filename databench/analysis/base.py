from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import pandas as pd

from databench._utils._logger import get_logger, log_run, log_this_fr

_log = get_logger(__name__)


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
class Analysis(ABC):
    """Base class for analyses.

    Subclasses **must** override ``run(df) -> AnalysisResult``.
    All parameters live on the frozen dataclass — no **kwargs.
    Declare ``required_columns`` for auto-validation.
    """

    name: str
    required_columns: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            original = cls.__dict__["run"]
            if not getattr(original, "__log_run_wrapped__", False):
                cls.run = log_run(original)

    @abstractmethod
    def run(self, df: pd.DataFrame) -> AnalysisResult:
        ...

    def plot(self, result: AnalysisResult):
        return None

    def save(self, result: AnalysisResult) -> list:
        return []

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Warn about missing required columns; returns the list of missing names."""
        return _warn_missing_columns(df, self.required_columns, self.name)


@dataclass(frozen=True)
class StatFn(ABC):
    name: str
    label: str
    plotter: str = "longitudinal"
    color: Optional[str] = None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...


@dataclass(frozen=True)
class FeatureFn(ABC):
    """Base class for per-row feature extractors.

    Subclasses **must** override ``_run_impl(row) -> float``.
    """

    name: str
    label: str
    unit: Optional[str] = None
    plotter: str = "longitudinal"
    color: Optional[str] = None
    source: Optional[str] = None
    required_columns: tuple[str, ...] = ()

    @log_this_fr
    def run(self, row, debug: bool = False, context: str | None = None):
        if debug:
            _log.debug(f"[{self.name}] start | {context}")
        # Warn on missing source columns when row has a MultiIndex
        if self.required_columns and isinstance(row.index, pd.MultiIndex):
            present = set(row.index)
            for col in self.required_columns:
                if col not in present:
                    warnings.warn(
                        f"[{self.name}] Row missing expected column: {col}",
                        stacklevel=2,
                    )
        val = self._run_impl(row)
        if debug:
            _log.debug(f"[{self.name}] value={val} | {context}")
        return val

    @abstractmethod
    def _run_impl(self, row) -> float:
        ...

    def __call__(self, row) -> float:
        return self._run_impl(row)