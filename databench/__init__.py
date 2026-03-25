"""Minimal, reproducible analysis/plotting toolkit for multiindex datasets."""
from databench.project import Project
from databench.session import Session, SessionGroup, AlignedData, SaveableFigure
from databench.analysis.oscillation import OscillationDetector, OscillationResult
from databench.analysis.eta import EtaAnalysis, EtaResult
from databench._signal.events import make_events
from databench.analysis.locomotion import locomotion_events

__all__ = [
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
]
