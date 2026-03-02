"""Analysis/stats module organized like features (minimal classes + core funcs)."""
from __future__ import annotations

from databench.analysis.base import StatFn, AxisFn, AnalysisFn, Analysis, AnalysisResult, _warn_missing_columns
from databench.analysis.eta import (
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
    OscillationResult,
    analyze_oscillation_row,
    save_oscillation_bursts,
    save_oscillation_plot,
    OscillationDetectorAnalysis,
)


__all__ = [
    # Base
    "StatFn",
    "AxisFn",
    "Analysis",
    "AnalysisResult",
    "_warn_missing_columns",
    # Deprecated — use Analysis instead
    "AnalysisFn",
    # ETA
    "eta_baselined",
    "EtaByConditionAnalysis",
    "EtaLongitudinalAnalysis",
    "EtaPrePostDiffAnalysis",
    # Longitudinal
    "MeanSEM",
    "LongitudinalAnalysis",
    "longitudinal_summary",
    # Mesomap
    "run_mesomap_hilbert",
    "export_hilbert_envelopes",
    "MesomapHilbertAnalysis",
    # Oscillation
    "OscillationContext",
    "OscillationResult",
    "OscillationDetectorAnalysis",
    "analyze_oscillation_row",
    "save_oscillation_bursts",
    "save_oscillation_plot",
]
