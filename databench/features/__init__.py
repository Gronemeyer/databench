"""Holds reusable feature processors grouped by their data source."""
from __future__ import annotations

from databench.features.base import FeatureFn
from databench.features.treadmill import (
    MeanSpeedCMS,
    StdSpeedCMS,
    TotalDistanceM,
    LocomotionBoutFeature,
    LocomotionBoutsCount,
    LocomotionBoutSpeedMeanCMS,
    LocomotionBoutDistanceM,
    LocomotionBoutDurationS,
)
from databench.features.pupil import MeanPupilMM, StdPupilMM
from databench.features.meso import MeanMeso, StdMeso

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
