"""Parsers for session/subject/task label strings."""
from __future__ import annotations

import re

__all__ = ["parse_session_day"]

_SES_RE = re.compile(r"ses-(\d+)$")


def parse_session_day(label: str | None, *, default: int | None = ...) -> int | None:
    """Return the integer day from a ``"ses-NN"`` label.

    Examples
    --------
    >>> parse_session_day("ses-04")
    4
    >>> parse_session_day("baseline", default=None)
    None

    Parameters
    ----------
    label : str
        Session label, e.g. ``"ses-04"``.
    default : int | None, optional
        If provided, returned when *label* cannot be parsed; otherwise a
        :class:`ValueError` is raised.
    """
    if isinstance(label, str):
        m = _SES_RE.match(label.strip())
        if m is not None:
            return int(m.group(1))
    if default is ...:
        raise ValueError(f"Unparseable session label: {label!r}")
    return default
