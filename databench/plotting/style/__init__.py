"""Style registry for databench plotting.

This subpackage holds self-contained style modules (palettes, rcParams,
axis helpers) that can be versioned and swapped independently of the
core plotting code.

Quick start
-----------
>>> from databench.plotting.style import use
>>> theme = use("cold_field_v5", mode="dark")   # applies rcParams globally

Or import a specific style directly:

>>> from databench.plotting.style.cold_field_v5 import Theme, clean_ax

Trace configuration
-------------------
>>> from databench.plotting.style import get_style, TraceStyle, CONDITION_COLORS
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Re-export trace configuration — canonical location
from databench.plotting.style.trace_config import (  # noqa: F401
    TraceStyle,
    TRACE_STYLES,
    get_style,
    CONDITION_COLORS,
    CONDITION_ORDER,
)

# Re-export the type scale — the sizes scripts pass to ``ax.text``
from databench.plotting.style.gsipe_v1 import (  # noqa: F401
    FONT_SIZES,
    font_size,
)

# Every style module must expose a ``Theme`` class with an ``apply()`` method.
_REGISTRY: dict[str, str] = {
    "cold_field_v5": ".cold_field_v5",
    "gsipe_v1": ".gsipe_v1",
}

_DEFAULT_STYLE = "gsipe_v1"


def available() -> list[str]:
    """Return names of all registered style modules."""
    return list(_REGISTRY)


def get_module(name: str | None = None):
    """Import and return a style module by registered name."""
    name = name or _DEFAULT_STYLE
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown style {name!r}. Available: {available()}"
        )
    return import_module(_REGISTRY[name], package=__package__)


def use(name: str | None = None, *, mode: str = "light") -> Any:
    """Load a style, apply its rcParams, and return the ``Theme`` instance.

    Parameters
    ----------
    name : str, optional
        Registered style name (default: ``"cold_field_v5"``).
    mode : str
        ``"light"`` or ``"dark"``.
    """
    mod = get_module(name)
    return mod.Theme(mode).apply()
