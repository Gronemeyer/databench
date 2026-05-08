"""Minimal, reproducible analysis/plotting toolkit for multiindex datasets."""

from databench.utils.logger import setup_logging as _setup_logging

_setup_logging()

# Public API re-exports — keep script imports short.
from databench.project import Project
from databench.config import resolve_dataset, dataset_params
from databench.plotting import set_theme
from databench.analysis.oscillation import OscillationDetector
from databench.analysis.eta import EtaAnalysis
from databench.analysis.locomotion import (
    locomotion_bout_events,
    locomotion_bouts,
    locomotion_events,
    quiescent_bouts,
)
from databench.utils.labels import parse_session_day

__all__ = [
    "Project",
    "resolve_dataset",
    "dataset_params",
    "set_theme",
    "OscillationDetector",
    "EtaAnalysis",
    "locomotion_bout_events",
    "locomotion_bouts",
    "locomotion_events",
    "quiescent_bouts",
    "parse_session_day",
]
