from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

from databench._utils._logger import log_this_fr


@dataclass(frozen=True)
class Plotter:
    """Base class for plotters.

    Subclasses override ``plot(result: AnalysisResult) -> (fig, axes)``.
    All configuration lives on the frozen dataclass — no **kwargs.
    """

    name: str

    @log_this_fr
    def plot(self, result) -> Any:  # pragma: no cover - interface
        raise NotImplementedError
