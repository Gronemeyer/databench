from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from databench._utils._logger import log_this_fr


@dataclass(frozen=True)
class FeatureFn:
    name: str
    label: str
    unit: Optional[str] = None
    plotter: str = "longitudinal"
    color: Optional[str] = None
    source: Optional[str] = None

    @log_this_fr
    def run(self, row, debug: bool = False, context: str | None = None):
        if debug:
            print(f"[{self.name}] start | {context}")
        val = self._run_impl(row)
        if debug:
            print(f"[{self.name}] value={val} | {context}")
        return val

    def _run_impl(self, row) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, row) -> float:
        return self._run_impl(row)
