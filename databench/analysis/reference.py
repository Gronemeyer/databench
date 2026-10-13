"""Express values relative to a per-animal reference distribution.

A reference is the pooled set of samples an animal produced under a chosen
condition — for example every pupil frame from its habituation sessions.
Other recordings from the same animal are then expressed against it, as a
z-score or a percentile, and thresholds (a median, a 5th percentile) are read
from it.  This removes per-animal offsets such as camera geometry or eye size
without assuming a shared scale across animals.

Two tiers, as in :mod:`databench.analysis.locomotion`:

* :class:`ReferenceDistribution` — pure-array core.  Build one from any
  arrays with :meth:`ReferenceDistribution.from_samples`.
* :func:`subject_references` — thin :class:`SessionGroup` wrapper that pools
  ``extract(session)`` over every session of each subject.

>>> ref = ReferenceDistribution.from_samples(habituation_pupil_mm)
>>> z = ref.zscore(drug_day_pupil_mm)
>>> low = ref.quantile(0.05)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from databench.session import Session, SessionGroup

__all__ = ["ReferenceDistribution", "subject_references"]


@dataclass(frozen=True)
class ReferenceDistribution:
    """Sorted finite reference samples, and values expressed against them."""

    samples: np.ndarray

    @classmethod
    def from_samples(cls, *arrays: np.ndarray) -> "ReferenceDistribution":
        """Pool *arrays*, drop non-finite values, and sort."""
        pooled = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays]) \
            if arrays else np.empty(0)
        return cls(np.sort(pooled[np.isfinite(pooled)]))

    @property
    def n(self) -> int:
        return int(self.samples.size)

    @property
    def mean(self) -> float:
        return float(self.samples.mean()) if self.n else float("nan")

    @property
    def sd(self) -> float:
        return float(self.samples.std(ddof=1)) if self.n > 1 else float("nan")

    def zscore(self, values: np.ndarray) -> np.ndarray:
        """``(values - mean) / sd`` of the reference."""
        return (np.asarray(values, dtype=float) - self.mean) / self.sd

    def percentile(self, values: np.ndarray) -> np.ndarray:
        """Percent of reference samples below each value (mid-rank for ties), 0-100."""
        values = np.asarray(values, dtype=float)
        if not self.n:
            return np.full(values.shape, np.nan)
        below = np.searchsorted(self.samples, values, side="left")
        at_or_below = np.searchsorted(self.samples, values, side="right")
        out = 100.0 * (below + at_or_below) / (2.0 * self.n)
        return np.where(np.isfinite(values), out, np.nan)

    def quantile(self, q: float | np.ndarray) -> float | np.ndarray:
        """Reference quantile(s) at *q* in [0, 1]."""
        return np.quantile(self.samples, q) if self.n else np.full(np.shape(q), np.nan)


def subject_references(
    group: "SessionGroup",
    extract: "Callable[[Session], np.ndarray]",
) -> dict[str, ReferenceDistribution]:
    """One :class:`ReferenceDistribution` per subject, pooling ``extract(session)``.

    *extract* returns the samples a session contributes — e.g.
    ``lambda s: s.signal("pupil", "pupil_diameter_mm")``, or only the frames
    inside quiescent periods.
    """
    pooled: dict[str, list[np.ndarray]] = {}
    for session in group:
        pooled.setdefault(session.subject, []).append(np.asarray(extract(session), dtype=float))
    return {subject: ReferenceDistribution.from_samples(*arrays)
            for subject, arrays in pooled.items()}
