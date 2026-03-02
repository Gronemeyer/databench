"""Decorators for registering features, analyses, and plotters.

Usage::

    @register_analysis
    @dataclass(frozen=True)
    class MyAnalysis(Analysis):
        name: str = "my_analysis"

    @register_analysis(depends_on=["oscillation_detector"])
    @dataclass(frozen=True)
    class MySecondOrder(Analysis):
        name: str = "second_order"

Dependency metadata is stored on the class as ``_depends_on``
and ``_component_kind`` and is used at provenance time.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Type, Union


# Global class lists — consumed by Bench.__init__ to seed instance registries
FEATURE_CLASSES: List[Type] = []
ANALYSIS_CLASSES: List[Type] = []
PLOTTER_CLASSES: List[Type] = []


def _register(
    cls: Type,
    registry: List[Type],
    depends_on: Sequence[str] = (),
    kind: str = "unknown",
) -> Type:
    """Core registration: deduplicate by ``name``, attach dependency metadata."""
    name: Optional[str] = getattr(cls, "name", None)
    if name is not None:
        registry[:] = [c for c in registry if getattr(c, "name", None) != name]
    if cls not in registry:
        registry.append(cls)
    cls._depends_on = tuple(depends_on)
    cls._component_kind = kind
    return cls


def register_feature(
    cls: Optional[Type] = None,
    *,
    depends_on: Sequence[str] = (),
) -> Union[Type, Callable[[Type], Type]]:
    """``@register_feature`` or ``@register_feature(depends_on=[...])``."""
    if cls is not None:
        return _register(cls, FEATURE_CLASSES, kind="feature")

    def wrapper(inner: Type) -> Type:
        return _register(inner, FEATURE_CLASSES, depends_on=depends_on, kind="feature")
    return wrapper


def register_analysis(
    cls: Optional[Type] = None,
    *,
    depends_on: Sequence[str] = (),
) -> Union[Type, Callable[[Type], Type]]:
    """``@register_analysis`` or ``@register_analysis(depends_on=[...])``."""
    if cls is not None:
        return _register(cls, ANALYSIS_CLASSES, kind="analysis")

    def wrapper(inner: Type) -> Type:
        return _register(inner, ANALYSIS_CLASSES, depends_on=depends_on, kind="analysis")
    return wrapper


def register_plotter(
    cls: Optional[Type] = None,
    *,
    depends_on: Sequence[str] = (),
) -> Union[Type, Callable[[Type], Type]]:
    """``@register_plotter`` or ``@register_plotter(depends_on=[...])``."""
    if cls is not None:
        return _register(cls, PLOTTER_CLASSES, kind="plotter")

    def wrapper(inner: Type) -> Type:
        return _register(inner, PLOTTER_CLASSES, depends_on=depends_on, kind="plotter")
    return wrapper
