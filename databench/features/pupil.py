from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from databench.features.base import FeatureFn
from databench.registry import register_feature
from databench.utils import as_1d, get_first

SOURCE_COLOR = "#9467bd"


@register_feature
@dataclass(frozen=True)
class MeanPupilMM(FeatureFn):
    name: str = "pupil_mean_mm"
    label: str = "Pupil diameter (mm)"
    color: str = SOURCE_COLOR
    source: str = "pupil"

    def _run_impl(self, row) -> float:
        pup = as_1d(get_first(row, [("pupil", "pupil_diameter_mm"), ("pupil", "diameter_mm")]))
        return float(np.nanmean(pup))


@register_feature
@dataclass(frozen=True)
class StdPupilMM(FeatureFn):
    name: str = "pupil_std_mm"
    label: str = "Pupil SD (mm)"
    color: str = SOURCE_COLOR
    source: str = "pupil"

    def _run_impl(self, row) -> float:
        pup = as_1d(get_first(row, [("pupil", "pupil_diameter_mm"), ("pupil", "diameter_mm")]))
        return float(np.nanstd(pup))
