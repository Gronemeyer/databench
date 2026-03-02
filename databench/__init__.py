"""Minimal, reproducible analysis/plotting toolkit for multiindex datasets."""
# ── New public API (v2) ────────────────────────────────────────────────────
from databench.project import Project
from databench.session import Session, SessionGroup, AlignedData, SaveableFigure
from databench.analysis.oscillation import OscillationDetector, OscillationResult
from databench.analysis.eta import EtaAnalysis, EtaResult
from databench._signal.events import make_events, locomotion_events

# ── Legacy API (preserved for existing scripts) ───────────────────────────
from databench.bench import Bench
from databench.config import CONDITION_COLORS, CONDITION_ORDER, resolve_dataset
from databench import analysis, features, plotting, utils, debug
from databench.registry import register_analysis, register_plotter, register_feature

__all__ = [
    # New API
    "Project",
    "Session",
    "SessionGroup",
    "AlignedData",
    "SaveableFigure",
    "OscillationDetector",
    "OscillationResult",
    "EtaAnalysis",
    "EtaResult",
    "make_events",
    "locomotion_events",
    # Legacy
    "Bench",
    "CONDITION_COLORS",
    "CONDITION_ORDER",
    "resolve_dataset",
    "register_analysis",
    "register_plotter",
    "register_feature",
    "analysis",
    "features",
    "plotting",
    "utils",
    "debug",
]
