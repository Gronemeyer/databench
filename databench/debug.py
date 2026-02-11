from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd


def log_context(index) -> str:
    if isinstance(index, tuple) and len(index) >= 3:
        return f"Subject={index[0]} | Session={index[1]} | Task={index[2]}"
    return f"Index={index}"


def subset_df(
    df: pd.DataFrame,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        return df
    idx = df.index
    mask = pd.Series(True, index=idx)

    def _norm(value: str, prefix: str) -> str:
        value = str(value)
        return value[len(prefix):] if value.lower().startswith(prefix) else value

    def _match(level_name: str, target: Optional[str], prefix: str) -> Optional[pd.Series]:
        if target is None:
            return None
        values = idx.get_level_values(level_name)
        target_str = str(target)
        exact = values == target_str
        if exact.any():
            return exact
        target_norm = _norm(target_str, prefix)
        return values.map(lambda v: _norm(v, prefix) == target_norm)

    if subject is not None and "Subject" in idx.names:
        m = _match("Subject", subject, "sub-")
        if m is not None:
            mask &= m
    if session is not None and "Session" in idx.names:
        m = _match("Session", session, "ses-")
        if m is not None:
            mask &= m
    if task is not None and "Task" in idx.names:
        m = _match("Task", task, "task-")
        if m is not None:
            mask &= m
    return df.loc[mask]


def get_row(
    df: pd.DataFrame,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
) -> Tuple[tuple, pd.Series]:
    sub = subset_df(df, subject, session, task)
    if sub.empty:
        raise KeyError("No rows matched the provided Subject/Session/Task.")
    idx = sub.index[0]
    return idx, sub.iloc[0]
