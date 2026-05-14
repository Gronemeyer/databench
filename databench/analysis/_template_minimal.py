"""Minimal portable-analysis template.

Use this template when you want a single analysis function that
*takes plain numpy/pandas data and returns plain numpy/pandas data*.
A colleague who does not use databench should be able to copy this
function into their notebook and call it on their own arrays — no
databench import required.

Portability rule
----------------
Functions in ``databench.analysis.*`` MUST NOT import from:

* ``databench.session``
* ``databench.project``
* ``databench.plotting``
* ``databench.config`` (other than the ``TIME_COLUMNS`` registry)

They MAY import from:

* ``databench.signal.*`` (pure-numpy/pandas helpers)
* ``databench.utils.*``
* ``numpy``, ``pandas``, ``scipy``, etc.

If your analysis also wants a "convenience wrapper" that operates on a
:class:`~databench.session.SessionGroup`, put that wrapper in the *same
module* as a separate function — keep the core function reachable
without it.  See ``databench/analysis/locomotion.py`` for the canonical
two-tier shape (``locomotion_bout_events`` core +
``locomotion_events`` wrapper).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_my_metric(
    t: np.ndarray,
    x: np.ndarray,
    *,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Example portable analysis.

    Takes a time array and a signal array; returns one row per detected
    above-threshold crossing.  No SessionGroup, no Plotter, no config
    objects — just numpy in, pandas out.

    Parameters
    ----------
    t : np.ndarray
        Timestamps in seconds.
    x : np.ndarray
        Signal values, same length as *t*.
    threshold : float
        DEVELOPER: replace with whatever criterion makes sense.

    Returns
    -------
    pd.DataFrame
        Columns ``crossing_index``, ``crossing_time``, ``value``.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.shape != x.shape:
        raise ValueError(f"t and x must have the same shape; got {t.shape} vs {x.shape}")

    above = x > threshold
    crossings = np.where(np.diff(above.astype(int)) > 0)[0] + 1
    if crossings.size == 0:
        return pd.DataFrame(columns=["crossing_index", "crossing_time", "value"])
    return pd.DataFrame({
        "crossing_index": crossings,
        "crossing_time": t[crossings],
        "value": x[crossings],
    })


# ── Optional: databench wrapper ────────────────────────────────────────────
#
# If you also want the convenience of "run on every session in a group,"
# put the wrapper in this module — but keep `compute_my_metric` above
# usable on plain arrays.
#
# def compute_my_metric_for_group(group, *, source: str, column: str, **kwargs):
#     rows = []
#     for sess in group:
#         t = sess.time(source)
#         x = sess.signal(source, column)
#         out = compute_my_metric(t, x, **kwargs)
#         out["Subject"] = sess.subject
#         out["Session"] = sess.session
#         out["Task"]    = sess.task
#         rows.append(out)
#     return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
