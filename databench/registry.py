"""Explicit decorators for registering features, analyses, and plotters."""
from __future__ import annotations

from typing import List, Type, TypeVar


T = TypeVar("T")

FEATURE_CLASSES: List[Type[T]] = []
ANALYSIS_CLASSES: List[Type[T]] = []
PLOTTER_CLASSES: List[Type[T]] = []


def register_feature(cls: Type[T]) -> Type[T]:
    FEATURE_CLASSES.append(cls)
    return cls


def register_analysis(cls: Type[T]) -> Type[T]:
    ANALYSIS_CLASSES.append(cls)
    return cls


def register_plotter(cls: Type[T]) -> Type[T]:
    PLOTTER_CLASSES.append(cls)
    return cls
