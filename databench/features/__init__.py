"""Holds reusable feature processors grouped by their data source.

FeatureFn is a frozen dataclass so that each feature definition stays immutable and easy to compare.
Its generated initializer, repr, and type checks keep the feature metadata consistent and explicit.

Each FeatureFn can be called like a function (the __call__ method) to compute its value from a row, 
so all features share a simple, uniform interface.

Attributes:
	name: Identifier used in code and logs.
	label: Human-friendly name for charts or reports.
	unit: Optional measurement unit for the feature output.
	plotter: Visualization hint; defaults to "longitudinal".
	color: Optional color hint for plotting.
	source: Optional description of the data provider that generated the feature.
"""
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


from .treadmill import (
    MeanSpeedCMS,
    StdSpeedCMS,
    TotalDistanceM,
    LocomotionBoutFeature,
    LocomotionBoutsCount,
    LocomotionBoutSpeedMeanCMS,
    LocomotionBoutDistanceM,
    LocomotionBoutDurationS,
)
from .pupil import MeanPupilMM, StdPupilMM
from .meso import MeanMeso, StdMeso

__all__ = [
    "FeatureFn",
    "MeanSpeedCMS",
    "StdSpeedCMS",
    "TotalDistanceM",
    "LocomotionBoutFeature",
    "LocomotionBoutsCount",
    "LocomotionBoutSpeedMeanCMS",
    "LocomotionBoutDistanceM",
    "LocomotionBoutDurationS",
    "MeanPupilMM",
    "StdPupilMM",
    "MeanMeso",
    "StdMeso",
]
