"""Minimal, reproducible analysis/plotting toolkit for multiindex datasets."""
from databench.bench import Bench
from databench.provenance import RunContext
from databench.config import CONDITION_COLORS, CONDITION_ORDER, resolve_dataset
from databench import analysis, features, plotting, utils, debug
from databench.registry import (
    register_analysis,
    register_plotter,
    register_feature,
    get_analysis,
    get_plotter,
    get_feature,
)

__all__ = [
    "Bench",
    "RunContext",
    "CONDITION_COLORS",
    "CONDITION_ORDER",
    "resolve_dataset",
    "register_analysis",
    "register_plotter",
    "register_feature",
    "get_analysis",
    "get_plotter",
    "get_feature",
    "analysis",
    "features",
    "plotting",
    "utils",
    "debug",
]
