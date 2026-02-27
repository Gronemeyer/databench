from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from databench._utils._logger import log_this_fr


@dataclass(frozen=True)
class FeatureFn:
    """Base class for per-row feature extractors.

    Subclasses override ``_run_impl(row) -> float``.
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
            print(f"[{self.name}] start | {context}")
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
            print(f"[{self.name}] value={val} | {context}")
        return val

    def _run_impl(self, row) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, row) -> float:
        return self._run_impl(row)
