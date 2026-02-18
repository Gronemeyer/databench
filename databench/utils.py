from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


def session_to_int(session_label: object) -> float:
    m = re.search(r"(\d+)$", str(session_label))
    return float(int(m.group(1))) if m else np.nan


def as_1d(x) -> Optional[np.ndarray]:
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


def get_value(row: pd.Series, source: str, feature: str):
    return row.get((source, feature)) if isinstance(row.index, pd.MultiIndex) else row.get(feature)


def get_first(row: pd.Series, keys):
    for key in keys:
        val = row.get(key)
        return val
    return row.get(keys[0])


def drop_rows(df: pd.DataFrame, drop_tuples: tuple) -> pd.DataFrame:
    """Drop specific MultiIndex rows like (Subject, Session, Task)."""
    return df.drop(index=list(drop_tuples), errors="ignore")


def strip_prefix(value: object, prefix: str) -> str:
    value = str(value)
    return value[len(prefix):] if value.lower().startswith(prefix) else value
