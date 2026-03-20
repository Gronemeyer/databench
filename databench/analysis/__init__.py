"""Analysis/stats module organized like features (minimal classes + core funcs)."""
from __future__ import annotations

from databench.analysis.oscillation import (
    OscillationDetector,
    OscillationResult as OscillationResult,
)
from databench.analysis.eta import (
    EtaAnalysis,
    EtaResult,
)


__all__ = [
    "OscillationDetector",
    "OscillationResult",
    "EtaAnalysis",
    "EtaResult",
]
