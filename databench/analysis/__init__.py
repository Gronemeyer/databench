"""Analysis/stats module organized like features (minimal classes + core funcs)."""
from __future__ import annotations

from .base import StatFn, AxisFn, AnalysisFn, Analysis, AnalysisResult
from .longitudinal import MeanSEM, longitudinal_summary, LongitudinalAnalysis
from .mesomap_hilbert import MesomapHilbertConfig, run_mesomap_hilbert, export_hilbert_envelopes
from .oscillation_detector import (
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
