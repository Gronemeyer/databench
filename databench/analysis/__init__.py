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
from databench.analysis.mesomap_hilbert import MesomapHilbertConfig, run_mesomap_hilbert, export_hilbert_envelopes
from databench.analysis.oscillation_detector import (
    OscillationDetectorConfig,
    OscillationContext,
    OscillationResult,
    analyze_oscillation_row,
    analyze_oscillation_dataset,
    save_oscillation_bursts,
    save_oscillation_plot,
    context_from_index,
    OscillationDetectorAnalysis,
)


__all__ = [
    "StatFn",
    "AxisFn",
    "AnalysisFn",
    "Analysis",
    "AnalysisResult",
    "_warn_missing_columns",
    "eta_baselined",
    "EtaByConditionAnalysis",
    "EtaLongitudinalAnalysis",
    "EtaPrePostDiffAnalysis",
    "MeanSEM",
    "LongitudinalAnalysis",
    "MesomapHilbertConfig",
    "run_mesomap_hilbert",
    "export_hilbert_envelopes",
    "OscillationDetectorConfig",
    "OscillationContext",
    "OscillationResult",
    "OscillationDetectorAnalysis",
    "analyze_oscillation_row",
    "analyze_oscillation_dataset",
    "save_oscillation_bursts",
    "save_oscillation_plot",
    "context_from_index",
    "longitudinal_summary",
]
