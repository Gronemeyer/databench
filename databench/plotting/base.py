from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


@dataclass(frozen=True)
class Plotter(ABC):
    """Base class for plotters.

    Subclasses **must** override ``plot(result)`` returning a matplotlib
    ``Figure`` (or list of figures).  All configuration lives on the
    frozen dataclass — no ``**kwargs``.

    The active theme is auto-applied when the plotter is *called* so
    every figure inherits the lab style.

    Recipe sidecars
    ---------------
    Each plotter exposes :meth:`recipe`, which returns a JSON-serialisable
    dict capturing the plotter class and its frozen config.  Persist it
    alongside the saved figure (e.g. ``run.save_json(plotter.recipe(result),
    "<name>.recipe.json")``) for full plotting reproducibility.
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

    def recipe(self, result=None) -> dict:
        """Return a JSON-serialisable dict describing this plotter's config.

        Subclasses may override to add per-result fields (e.g. event names,
        ROI lists).  The default emits ``{"plotter": <class>, "config": …}``.
        """
        config = asdict(self) if is_dataclass(self) else {"name": self.name}
        return {
            "plotter": f"{type(self).__module__}.{type(self).__name__}",
            "config": config,
        }
