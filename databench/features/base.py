from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FeatureFn:
    name: str
    label: str
    unit: Optional[str] = None
    plotter: str = "longitudinal"
    color: Optional[str] = None
    source: Optional[str] = None

    def run(self, row, debug: bool = False, context: str | None = None):
        if debug and context:
            print(f"[{self.name}] start | {context}")
        val = self._run_impl(row)
        if debug and context:
            print(f"[{self.name}] value={val} | {context}")
        return val

    def _run_impl(self, row) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, row) -> float:
        return self._run_impl(row)
