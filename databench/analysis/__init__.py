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
from databench.analysis.locomotion import (
    locomotion_bout_events,
    locomotion_events,
)


__all__ = [
    "OscillationDetector",
    "OscillationResult",
    "EtaAnalysis",
    "EtaResult",
    "locomotion_bout_events",
    "locomotion_events",
]
