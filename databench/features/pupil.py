from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import FeatureFn
from ..utils import as_1d, get_first

SOURCE_COLOR = "#9467bd"


@dataclass(frozen=True)
class MeanPupilMM(FeatureFn):
    name: str = "pupil_mean_mm"
    label: str = "Pupil diameter (mm)"
    color: str = SOURCE_COLOR
    source: str = "pupil"

    def _run_impl(self, row) -> float:
        pup = as_1d(get_first(row, [("pupil", "pupil_diameter_mm"), ("pupil", "diameter_mm")]))
        if pup is None or pup.size == 0:
            return np.nan
        return float(np.nanmean(pup))


@dataclass(frozen=True)
class StdPupilMM(FeatureFn):
    name: str = "pupil_std_mm"
    label: str = "Pupil SD (mm)"
    color: str = SOURCE_COLOR
    source: str = "pupil"

    def _run_impl(self, row) -> float:
        pup = as_1d(get_first(row, [("pupil", "pupil_diameter_mm"), ("pupil", "diameter_mm")]))
        if pup is None or pup.size == 0:
            return np.nan
        return float(np.nanstd(pup))
