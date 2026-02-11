"""Minimal, reproducible analysis/plotting toolkit for multiindex datasets."""
from .bench import Bench
from . import analysis, features, plotting, utils, debug

__all__ = [
    "Bench",
    "analysis",
    "features",
    "plotting",
    "utils",
    "debug",
]
