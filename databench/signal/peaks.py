"""Prominence peaks of a 1-D trace, each paired with its onset and trough.

``scipy.signal.find_peaks`` with a minimum prominence and a minimum peak
separation.  Each peak is paired with the lowest point since the previous
peak (its onset) and the lowest point before the next peak (its trough).
The routine is signal-agnostic: smoothing is left to the caller.

>>> events = prominence_events(t, smooth_savgol(dff, 0.2, fs), prominence=0.02, min_separation_s=1.0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

__all__ = ["PEAK_EVENT_COLUMNS", "prominence_events"]

PEAK_EVENT_COLUMNS = ["onset_s", "peak_s", "trough_s", "onset_value", "peak_value", "trough_value",
                      "prominence", "amplitude", "rise_s", "constriction_s"]


def prominence_events(t: np.ndarray, y: np.ndarray, *, prominence: float,
                      min_separation_s: float) -> pd.DataFrame:
    """One row per prominence peak of *y*, with its onset and following trough.

    Columns (:data:`PEAK_EVENT_COLUMNS`): ``onset_s``, ``peak_s``,
    ``trough_s``, ``onset_value``, ``peak_value``, ``trough_value``,
    ``prominence``, ``amplitude`` (peak - onset), ``rise_s`` (peak - onset)
    and ``constriction_s`` (trough - peak).  Values are in the units of *y*;
    *prominence* is too.  The sampling rate is taken from the median step of
    *t*.  NaN samples are linearly interpolated for detection, and peaks
    whose onset-to-trough span contains a NaN sample are dropped.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return pd.DataFrame(columns=PEAK_EVENT_COLUMNS)
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
    return events[complete].reset_index(drop=True)[PEAK_EVENT_COLUMNS]
