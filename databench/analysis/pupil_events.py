"""Pupil dilation events: prominence peaks on a smoothed diameter trace.

The detection follows ``analysis_scripts/osc-event-onset/05-pupil-peak-oscillation-onset.py``:
a median filter then a Savitzky-Golay filter, and ``scipy.signal.find_peaks``
with a minimum prominence and a minimum peak separation.  Each peak is paired
with the lowest point since the previous peak (its dilation onset) and the
lowest point before the next peak (its constriction trough).

>>> smoothed = smooth_pupil(t, diameter_mm, median_s=1.0, smooth_s=0.5)
>>> events = dilation_events(t, smoothed, prominence=0.10, min_separation_s=5.0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

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

    Columns: ``onset_s``, ``peak_s``, ``trough_s``, ``onset_value``,
    ``peak_value``, ``trough_value``, ``prominence``, ``amplitude``
    (peak - onset), ``rise_s`` (peak - onset) and ``constriction_s``
    (trough - peak).  Values are in the units of *smoothed*.  Peaks whose
    onset-to-trough span contains a NaN sample are dropped.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(smoothed, dtype=float)
    columns = ["onset_s", "peak_s", "trough_s", "onset_value", "peak_value", "trough_value",
               "prominence", "amplitude", "rise_s", "constriction_s"]
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return pd.DataFrame(columns=columns)
    fs = 1.0 / np.nanmedian(np.diff(t))
    filled = np.interp(t, t[finite], y[finite])
    peaks, properties = find_peaks(filled, prominence=prominence,
                                   distance=max(1, int(round(min_separation_s * fs))))
    starts = np.concatenate([[0], peaks[:-1]])
    ends = np.concatenate([peaks[1:], [len(y)]])
    onsets = np.array([s + np.argmin(filled[s:p + 1]) for s, p in zip(starts, peaks)], dtype=int)
    troughs = np.array([p + np.argmin(filled[p:e]) for p, e in zip(peaks, ends)], dtype=int)
    events = pd.DataFrame({
        "onset_s": t[onsets], "peak_s": t[peaks], "trough_s": t[troughs],
        "onset_value": filled[onsets], "peak_value": filled[peaks], "trough_value": filled[troughs],
        "prominence": properties["prominences"],
    })
    events["amplitude"] = events.peak_value - events.onset_value
    events["rise_s"] = events.peak_s - events.onset_s
    events["constriction_s"] = events.trough_s - events.peak_s
    complete = [finite[o:r + 1].all() for o, r in zip(onsets, troughs)]
    return events[complete].reset_index(drop=True)[columns]
