from __future__ import annotations

from dataclasses import dataclass

from databench._utils._logger import log_this_fr


@dataclass(frozen=True)
class Plotter:
    name: str

    @log_this_fr
    def plot(self, *args, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError
