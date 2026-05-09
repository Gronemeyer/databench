from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import pandas as pd

# Type alias: each entry is a bare string ("ses-11") or a tuple of strings.
DropSpec = Union[str, tuple[str, ...]]
DropList = Sequence[DropSpec]


def session_to_int(session_label: object) -> float:
    m = re.search(r"(\d+)$", str(session_label))
    return float(int(m.group(1))) if m else np.nan


def as_1d(x) -> Optional[np.ndarray]:
    """
    Convert input to a 1-dimensional numpy array.
    
    Parameters
    ----------
    x : array-like
        Input data that can be converted to a numpy array.
    
    Returns
    -------
    np.ndarray or None
        A flattened 1-dimensional numpy array, or None if input is None.
    
    Examples
    --------
    >>> as_1d([1, 2, 3])
    array([1, 2, 3])
    
    >>> as_1d([[1, 2], [3, 4]])
    array([1, 2, 3, 4])
    """
    return np.asarray(x).ravel()


def clean_xy(t, y):
    t = as_1d(t)
    y = as_1d(y)
    m = min(len(t), len(y))
    t = t[:m].astype(float, copy=False)
    y = y[:m].astype(float, copy=False)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    order = np.argsort(t)
    return t[order], y[order]


def get_first(row: pd.Series, keys):
    for key in keys:
        val = row.get(key)
        return val
    return row.get(keys[0])


def drop_rows(df: pd.DataFrame, drop_specs: DropList) -> pd.DataFrame:
    """Drop rows from a DataFrame by index values.

    Parameters
    ----------
    df : DataFrame
        Source data.  Works with both flat and MultiIndex indices.
    drop_specs : sequence of str or tuple[str, ...]
        Values to drop.  Each element is either:

        - A bare **string** — treated as a single value to match against
          *any* index level (wildcard).
        - A **tuple with fewer elements than index levels** — same
          wildcard behavior: any row matching *any* of the values is
          dropped.
        - A **tuple with exactly as many elements as index levels** —
          matched as an exact multi-index key.

    Examples
    --------
    Drop every row whose Session (or any level) is ``"ses-11"``::

        drop_rows(df, ["ses-11"])

    Drop one specific row::

        drop_rows(df, [("STREHAB07", "ses-11", "task-widefield")])

    Mix both styles::

        drop_rows(df, ["ses-11", ("STREHAB02", "ses-01", "task-widefield")])
    """
    if not isinstance(df.index, pd.MultiIndex):
        return df.drop(index=list(drop_specs), errors="ignore")

    n_levels = df.index.nlevels
    full_keys: list[tuple[str, ...]] = []
    mask = pd.Series(False, index=df.index)

    for entry in drop_specs:
        if isinstance(entry, str):
            entry = (entry,)
        if len(entry) == n_levels:
            full_keys.append(entry)
        else:
            values = set(entry)
            for level in range(n_levels):
                mask |= df.index.get_level_values(level).isin(values)

    df = df.loc[~mask]
    if full_keys:
        df = df.drop(index=full_keys, errors="ignore")
    return df


def strip_prefix(value: object, prefix: str) -> str:
    value = str(value)
    return value[len(prefix):] if value.lower().startswith(prefix) else value
