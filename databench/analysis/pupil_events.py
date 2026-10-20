"""Pupil dilation events: prominence peaks on a smoothed diameter trace.

The detection follows ``analysis_scripts/osc-event-onset/05-pupil-peak-oscillation-onset.py``:
a median filter then a Savitzky-Golay filter, and
:func:`databench.signal.peaks.prominence_events` (``scipy.signal.find_peaks``
with a minimum prominence and a minimum peak separation).  Each peak is paired
with the lowest point since the previous peak (its dilation onset) and the
lowest point before the next peak (its constriction trough).

>>> smoothed = smooth_pupil(t, diameter_mm, median_s=1.0, smooth_s=0.5)
>>> events = dilation_events(t, smoothed, prominence=0.10, min_separation_s=5.0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from databench.signal.peaks import prominence_events
from databench.signal.preproc import smooth_median, smooth_savgol

__all__ = ["smooth_pupil", "dilation_events"]


def smooth_pupil(t: np.ndarray, y: np.ndarray, *, median_s: float = 1.0,
                 smooth_s: float = 0.5) -> np.ndarray:
    """Median filter (*median_s*) then Savitzky-Golay (*smooth_s*); NaN samples stay NaN.

    NaN runs are linearly interpolated for filtering and restored afterwards.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return np.full_like(y, np.nan)
    fs = 1.0 / np.nanmedian(np.diff(t))
    filled = np.interp(t, t[finite], y[finite])
    smoothed = smooth_savgol(smooth_median(filled, int(round(median_s * fs)) | 1), smooth_s, fs)
    return np.where(finite, smoothed, np.nan)


def dilation_events(t: np.ndarray, smoothed: np.ndarray, *, prominence: float,
                    min_separation_s: float) -> pd.DataFrame:
    """One row per dilation peak of *smoothed*, with its onset and following trough.

    :func:`~databench.signal.peaks.prominence_events` on the smoothed
    diameter.  Columns: ``onset_s``, ``peak_s``, ``trough_s``,
    ``onset_value``, ``peak_value``, ``trough_value``, ``prominence``,
    ``amplitude`` (peak - onset), ``rise_s`` (peak - onset) and
    ``constriction_s`` (trough - peak).  Values are in the units of
    *smoothed*.  Peaks whose onset-to-trough span contains a NaN sample are
    dropped.
    """
    return prominence_events(t, smoothed, prominence=prominence, min_separation_s=min_separation_s)
