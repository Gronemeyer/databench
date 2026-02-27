"""Explicit decorators for registering features, analyses, and plotters.

Usage::

    @register_analysis
    @dataclass(frozen=True)
    class MyAnalysis(Analysis):
        name: str = "my_analysis"
        ...

Registered components can be retrieved by name::

    get_analysis("my_analysis")   # returns the *class*
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type


FEATURE_CLASSES: List[Type] = []
ANALYSIS_CLASSES: List[Type] = []
PLOTTER_CLASSES: List[Type] = []

# Name → class lookup tables (populated alongside the lists above)
_FEATURE_BY_NAME: Dict[str, Type] = {}
_ANALYSIS_BY_NAME: Dict[str, Type] = {}
_PLOTTER_BY_NAME: Dict[str, Type] = {}


def _register(cls: Type, registry: List[Type], by_name: Dict[str, Type]) -> Type:
    """Shared logic: deduplicate by ``name`` field, index by name."""
    name: Optional[str] = getattr(cls, "name", None)
    # Deduplicate: if a class with the same name already registered, replace it
    if name is not None and name in by_name:
        old = by_name[name]
        if old in registry:
            registry.remove(old)
    if cls not in registry:
        registry.append(cls)
    if name is not None:
        by_name[name] = cls
    return cls


def register_feature(cls: Type) -> Type:
    return _register(cls, FEATURE_CLASSES, _FEATURE_BY_NAME)


def register_analysis(cls: Type) -> Type:
    return _register(cls, ANALYSIS_CLASSES, _ANALYSIS_BY_NAME)


def register_plotter(cls: Type) -> Type:
    return _register(cls, PLOTTER_CLASSES, _PLOTTER_BY_NAME)


# --- lookup helpers --------------------------------------------------------

def get_feature(name: str) -> Type:
    """Return the registered feature *class* by its ``name`` field."""
    try:
        return _FEATURE_BY_NAME[name]
    except KeyError:
        raise KeyError(f"No feature registered with name={name!r}. Available: {sorted(_FEATURE_BY_NAME)}")


def get_analysis(name: str) -> Type:
    """Return the registered analysis *class* by its ``name`` field."""
    try:
        return _ANALYSIS_BY_NAME[name]
    except KeyError:
        raise KeyError(f"No analysis registered with name={name!r}. Available: {sorted(_ANALYSIS_BY_NAME)}")


def get_plotter(name: str) -> Type:
    """Return the registered plotter *class* by its ``name`` field."""
    try:
        return _PLOTTER_BY_NAME[name]
    except KeyError:
        raise KeyError(f"No plotter registered with name={name!r}. Available: {sorted(_PLOTTER_BY_NAME)}")


# --- unregistration --------------------------------------------------------

def unregister_feature(cls_or_name: Any) -> None:
    _unregister(cls_or_name, FEATURE_CLASSES, _FEATURE_BY_NAME)


def unregister_analysis(cls_or_name: Any) -> None:
    _unregister(cls_or_name, ANALYSIS_CLASSES, _ANALYSIS_BY_NAME)


def unregister_plotter(cls_or_name: Any) -> None:
    _unregister(cls_or_name, PLOTTER_CLASSES, _PLOTTER_BY_NAME)


def _unregister(cls_or_name: Any, registry: List[Type], by_name: Dict[str, Type]) -> None:
    if isinstance(cls_or_name, str):
        cls = by_name.pop(cls_or_name, None)
        if cls is not None and cls in registry:
            registry.remove(cls)
    else:
        if cls_or_name in registry:
            registry.remove(cls_or_name)
        name = getattr(cls_or_name, "name", None)
        if name is not None:
            by_name.pop(name, None)
