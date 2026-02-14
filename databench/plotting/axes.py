from __future__ import annotations

from dataclasses import dataclass

from databench.analysis.base import AxisFn


@dataclass(frozen=True)
class SessionAxis(AxisFn):
    name: str = "session_n"
    label: str = "Session (days)"


@dataclass(frozen=True)
class SubjectAxis(AxisFn):
    name: str = "Subject"
    label: str = "Subject"


__all__ = ["SessionAxis", "SubjectAxis"]
