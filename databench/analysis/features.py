from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from databench.analysis.base import FeatureFn
from databench._utils import as_1d, get_first

MESO_SOURCE_COLOR = "#1f77b4"
PUPIL_SOURCE_COLOR = "#9467bd"


@dataclass(frozen=True)
class MeanMeso(FeatureFn):
    name: str = "meso_mean"
    label: str = "Meso mean (a.u.)"
    color: str = MESO_SOURCE_COLOR
    source: str = "meso"

    def _run_impl(self, row) -> float:
        meso = as_1d(
            get_first(
                row,
                [("meso", "meso_tiff"), ("meso", "meso_mean"), ("Analysis", "meso_dff")],
            )
        )
        return float(np.nanmean(meso))


@dataclass(frozen=True)
class StdMeso(FeatureFn):
    name: str = "meso_std"
    label: str = "Meso SD (a.u.)"
    color: str = MESO_SOURCE_COLOR
    source: str = "meso"

    def _run_impl(self, row) -> float:
        meso = as_1d(
            get_first(
                row,
                [("meso", "meso_tiff"), ("meso", "meso_mean"), ("Analysis", "meso_dff")],
            )
        )
        return float(np.nanstd(meso))


@dataclass(frozen=True)
class MeanPupilMM(FeatureFn):
    name: str = "pupil_mean_mm"
    label: str = "Pupil diameter (mm)"
    color: str = PUPIL_SOURCE_COLOR
    source: str = "pupil"

    def _run_impl(self, row) -> float:
        pupil = as_1d(get_first(row, [("pupil", "pupil_diameter_mm"), ("pupil", "diameter_mm")]))
        return float(np.nanmean(pupil))


@dataclass(frozen=True)
class StdPupilMM(FeatureFn):
    name: str = "pupil_std_mm"
    label: str = "Pupil SD (mm)"
    color: str = PUPIL_SOURCE_COLOR
    source: str = "pupil"

    def _run_impl(self, row) -> float:
        pupil = as_1d(get_first(row, [("pupil", "pupil_diameter_mm"), ("pupil", "diameter_mm")]))
        return float(np.nanstd(pupil))
