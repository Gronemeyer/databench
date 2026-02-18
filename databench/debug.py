from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd


def log_context(index) -> str:
    return f"Subject={index[0]} | Session={index[1]} | Task={index[2]}"


def subset_df(
    df: pd.DataFrame,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
) -> pd.DataFrame:
    idx = df.index
    mask = pd.Series(True, index=idx)

    if subject is not None:
        mask &= idx.get_level_values("Subject") == str(subject)
    if session is not None:
        mask &= idx.get_level_values("Session") == str(session)
    if task is not None:
        mask &= idx.get_level_values("Task") == str(task)
    return df.loc[mask]


def get_row(
    df: pd.DataFrame,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
) -> Tuple[tuple, pd.Series]:
    sub = subset_df(df, subject, session, task)
    idx = sub.index[0]
    return idx, sub.iloc[0]
