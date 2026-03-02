"""Minimal, reproducible analysis/plotting toolkit for multiindex datasets."""
from databench.bench import Bench
from databench.config import CONDITION_COLORS, CONDITION_ORDER, resolve_dataset
from databench import analysis, features, plotting, utils, debug
from databench.registry import register_analysis, register_plotter, register_feature

__all__ = [
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
