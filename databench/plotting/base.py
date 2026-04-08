from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Plotter(ABC):
    """Base class for plotters.

    Subclasses **must** override ``plot(result: AnalysisResult) -> (fig, axes)``.
    All configuration lives on the frozen dataclass — no **kwargs.

    The theme is auto-applied when ``plot()`` is called so every figure
    inherits the active style without any manual ``set_theme()`` call.
    """

    name: str

    def __call__(self, result) -> Any:
        """Apply the theme and delegate to :meth:`plot`."""
        from databench.plotting import get_theme
        get_theme()
        return self.plot(result)

    @abstractmethod
    def plot(self, result) -> Any:
        ...
