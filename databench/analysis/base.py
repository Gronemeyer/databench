from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class AnalysisResult:
    name: str
    data: Any = None
    table: Any = None
    figs: list = field(default_factory=list)
    files: list = field(default_factory=list)
    context: Any = None
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Analysis:
    name: str

    def run(self, *args, **kwargs) -> AnalysisResult:  # pragma: no cover - interface
        raise NotImplementedError

    def plot(self, result: AnalysisResult, **kwargs):  # pragma: no cover - optional
        return None

    def save(self, result: AnalysisResult, **kwargs) -> list:  # pragma: no cover - optional
        return []


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


@dataclass(frozen=True)
class AnalysisFn:
    name: str

    def run(self, row, debug: bool = False, context: str | None = None, **kwargs):
        if debug and context:
            print(f"[{self.name}] start | {context}")
        out = self._run_impl(row, debug=debug, context=context, **kwargs)
        if debug and context:
            print(f"[{self.name}] done | {context}")
        return out

    def _run_impl(self, row, debug: bool = False, context: str | None = None, **kwargs):  # pragma: no cover
        raise NotImplementedError