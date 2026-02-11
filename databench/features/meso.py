from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import FeatureFn
from ..utils import as_1d, get_first

SOURCE_COLOR = "#1f77b4"


@dataclass(frozen=True)
class MeanMeso(FeatureFn):
    name: str = "meso_mean"
    label: str = "Meso mean (a.u.)"
    color: str = SOURCE_COLOR
    source: str = "meso"

    def _run_impl(self, row) -> float:
        meso = as_1d(get_first(row, [("meso", "meso_tiff"), ("meso", "meso_mean"), ("Analysis", "meso_dff")]))
        if meso is None or meso.size == 0:
            return np.nan
        return float(np.nanmean(meso))


@dataclass(frozen=True)
class StdMeso(FeatureFn):
    name: str = "meso_std"
    label: str = "Meso SD (a.u.)"
    color: str = SOURCE_COLOR
    source: str = "meso"

    def _run_impl(self, row) -> float:
        meso = as_1d(get_first(row, [("meso", "meso_tiff"), ("meso", "meso_mean"), ("Analysis", "meso_dff")]))
        if meso is None or meso.size == 0:
            return np.nan
        return float(np.nanstd(meso))
