from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plotter:
    name: str

    def plot(self, *args, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError
