from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

from databench._utils._logger import log_this_fr


@dataclass(frozen=True)
class Plotter:
    """Base class for plotters.

    Subclasses should override ``plot(self, result, **kwargs)``.
    Return type is typically ``(fig, axes)`` or ``dict[str, (fig, axes)]``.
    """

    name: str

    @log_this_fr
    def plot(self, result: Any, **kwargs) -> Any:  # pragma: no cover - interface
        raise NotImplementedError
