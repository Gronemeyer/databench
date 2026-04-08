from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any, Optional

import numpy as np
import pandas as pd


def session_to_int(session_label: object) -> float:
    m = re.search(r"(\d+)$", str(session_label))
    return float(int(m.group(1))) if m else np.nan


def as_1d(x) -> Optional[np.ndarray]:
    """Convert input to a flattened 1D numpy array."""
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


def time_mask(
    time: np.ndarray,
    window: tuple[float, float] | None = None,
) -> tuple[np.ndarray | None, slice | None]:
    """Build a boolean mask and contiguous slice for an optional time window."""
    t = as_1d(time)
    if t is None or len(t) == 0:
        return None, None

    if window is None:
        return np.ones(len(t), dtype=bool), slice(0, len(t))

    lo, hi = window
    if lo > hi:
        lo, hi = hi, lo

    mask = np.isfinite(t) & (t >= lo) & (t <= hi)
    if not bool(mask.any()):
        return None, None

    idx = np.flatnonzero(mask)
    return mask, slice(int(idx[0]), int(idx[-1]) + 1)


def get_first(row: pd.Series, keys):
    for key in keys:
        val = row.get(key)
        if val is not None:
            return val
    return row.get(keys[0])


def _resolve_level_name(df: pd.DataFrame, key: str) -> str | None:
    key_l = str(key).lower()
    for name in df.index.names:
        if name is not None and str(name).lower() == key_l:
            return str(name)
    aliases = {
        "subject": "Subject",
        "session": "Session",
        "task": "Task",
    }
    alias = aliases.get(key_l)
    if alias in df.index.names:
        return alias
    return None


def _as_values(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _mask_for_mapping(df: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    matched_any = False
    for raw_level, raw_val in spec.items():
        level = _resolve_level_name(df, str(raw_level))
        if level is None:
            continue
        matched_any = True
        mask &= df.index.get_level_values(str(level)).isin(_as_values(raw_val))
    if not matched_any:
        return pd.Series(False, index=df.index)
    return mask


def _mask_for_tuple(df: pd.DataFrame, spec: tuple[Any, ...]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    names = list(df.index.names)
    matched_any = False

    for pos, raw_val in enumerate(spec):
        if pos >= len(names):
            break
        level = names[pos]
        if level is None:
            continue
        matched_any = True
        mask &= df.index.get_level_values(str(level)).isin(_as_values(raw_val))

    if not matched_any:
        return pd.Series(False, index=df.index)
    return mask


def _to_drop_mask(df: pd.DataFrame, spec: Any) -> pd.Series:
    if isinstance(spec, dict):
        return _mask_for_mapping(df, spec)

    if isinstance(spec, tuple):
        return _mask_for_tuple(df, spec)

    if isinstance(spec, Iterable) and not isinstance(spec, (str, bytes)):
        mask = pd.Series(False, index=df.index)
        for item in spec:
            mask |= _to_drop_mask(df, item)
        return mask

    return pd.Series(False, index=df.index)


def drop_rows(df: pd.DataFrame, drop_spec: Any) -> pd.DataFrame:
    """Drop rows using intuitive MultiIndex-aware spec(s).

    Supported specs:
    - ``{"session": "ses-00"}``
    - ``("GS29", "ses-00", "task-widefield")``
    - Collections of mixed specs, e.g. ``[{...}, (...)]``
    """
    if not drop_spec:
        return df

    mask = _to_drop_mask(df, drop_spec)
    if not bool(mask.any()):
        return df

    return df.loc[~mask]


def strip_prefix(value: object, prefix: str) -> str:
    value = str(value)
    return value[len(prefix):] if value.lower().startswith(prefix) else value


def label_conditions(
    df: pd.DataFrame,
    mapping: dict[str, str],
    column: str = "Condition",
) -> pd.DataFrame:
    """Map Session values to condition labels on a long table."""
    df[column] = df["Session"].map(mapping)
    return df
