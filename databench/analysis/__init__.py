"""Analysis/stats module organized like features (minimal classes + core funcs)."""
from __future__ import annotations

# ── New public API (v2) ────────────────────────────────────────────────────
from databench.analysis.oscillation import (
    OscillationDetector,
    OscillationResult as OscillationResult,
)
from databench.analysis.eta import (
    EtaAnalysis,
    EtaResult,
)

# ── Legacy imports (preserved for existing scripts) ───────────────────────
from databench.analysis.base import StatFn, AxisFn, AnalysisFn, Analysis, AnalysisResult, _warn_missing_columns
from databench.analysis.eta_old import (
    eta_baselined,
    EtaByConditionAnalysis,
    EtaLongitudinalAnalysis,
    EtaPrePostDiffAnalysis,
    _safe_sem,
    _session_to_day,
)
from databench.analysis.longitudinal import MeanSEM, longitudinal_summary, LongitudinalAnalysis
from databench.analysis.mesomap_hilbert import run_mesomap_hilbert, export_hilbert_envelopes, MesomapHilbertAnalysis
from databench.analysis.oscillation_detector import (
    OscillationContext,
    OscillationResult as _OscillationResultLegacy,
    analyze_oscillation_row,
    save_oscillation_bursts,
    save_oscillation_plot,
    OscillationDetectorAnalysis,
)


__all__ = [
    # New API
    "OscillationDetector",
    "OscillationResult",
    "EtaAnalysis",
    "EtaResult",
    # Legacy — base
    "StatFn",
    "AxisFn",
    "Analysis",
    "AnalysisResult",
    "_warn_missing_columns",
    "AnalysisFn",
    # Legacy — ETA
    "eta_baselined",
    "EtaByConditionAnalysis",
    "EtaLongitudinalAnalysis",
    "EtaPrePostDiffAnalysis",
    # Legacy — Longitudinal
    "MeanSEM",
    "LongitudinalAnalysis",
    "longitudinal_summary",
    # Legacy — Mesomap
    "run_mesomap_hilbert",
    "export_hilbert_envelopes",
    "MesomapHilbertAnalysis",
    # Legacy — Oscillation
    "OscillationContext",
    "OscillationDetectorAnalysis",
    "analyze_oscillation_row",
    "save_oscillation_bursts",
    "save_oscillation_plot",
]
