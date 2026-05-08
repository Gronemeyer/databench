"""DataTabler: index-aware wrapper around a project DataFrame.

This class centralizes MultiIndex-oriented pandas wrangling patterns so
`Project` can focus on orchestration and provenance.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from databench.utils import drop_rows


class DataTabler:
    """Wrap a pandas DataFrame with index-aware filtering helpers.

    Notes
    -----
    - Methods mutate this instance by default and return ``self`` for chaining.
    - Index level aliases ``subject``, ``session``, ``task`` map to
      ``Subject``, ``Session``, ``Task`` when present.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._drop_spec: Any = ()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @staticmethod
    def _resolve_index_level(df: pd.DataFrame, key: str) -> str | None:
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

    @staticmethod
    def _coerce_values(value: Any) -> list[Any]:
        if isinstance(value, (str, bytes)):
            return [value]
        if isinstance(value, Iterable):
            return list(value)
        return [value]

    def _apply_level_filters(
        self,
        df: pd.DataFrame,
        spec: dict[str, Any],
        *,
        mode: str,
    ) -> pd.DataFrame:
        if not spec:
            return df

        if mode == "exclude":
            mask = pd.Series(False, index=df.index)
        else:
            mask = pd.Series(True, index=df.index)
        for raw_level, raw_val in spec.items():
            level = self._resolve_index_level(df, str(raw_level))
            if level is None:
                continue
            values = self._coerce_values(raw_val)
            level_match = df.index.get_level_values(level).isin(values)
            if mode == "exclude":
                mask |= level_match
            else:
                mask &= level_match

        if mode == "include":
            return df.loc[mask]
        if mode == "exclude":
            return df.loc[~mask]
        raise ValueError("mode must be 'include' or 'exclude'")

    def set_filters(self, drop_spec: Any = ()) -> "DataTabler":
        self._drop_spec = drop_spec
        return self

    def filter_data(self, df: pd.DataFrame, drop_spec: Any = None) -> pd.DataFrame:
        if drop_spec is None:
            drop_spec = self._drop_spec
        return drop_rows(df, drop_spec or ())

    def filter(
        self,
        drop_rows: Any = None,
        *,
        include: dict[str, Any] | None = None,
        exclude: Any = None,
        **kwargs,
    ) -> "DataTabler":
        if drop_rows is not None:
            self.set_filters(drop_rows)

        df = self._df
        if self._drop_spec:
            df = self.filter_data(df)

        if exclude:
            if isinstance(exclude, dict):
                df = self._apply_level_filters(df, exclude, mode="exclude")
            else:
                df = self.filter_data(df, exclude)

        if include:
            df = self._apply_level_filters(df, include, mode="include")

        for level, value in kwargs.items():
            resolved_level = self._resolve_index_level(df, str(level))
            if resolved_level is None:
                continue
            values = self._coerce_values(value)
            mask = df.index.get_level_values(resolved_level).isin(values)
            df = df.loc[mask]

        self._df = df
        return self

    def index_frame(self) -> pd.DataFrame:
        """Return index levels as a regular DataFrame."""
        return self._df.index.to_frame(index=False)

    def add_session_number(
        self,
        df: pd.DataFrame,
        *,
        session_col: str = "Session",
        out_col: str = "session_n",
        pattern: str = r"(\d+)",
    ) -> pd.DataFrame:
        """Add numeric session index extracted from a session label column."""
        out = df.copy()
        out[out_col] = out[session_col].astype(str).str.extract(pattern, expand=False).astype(float)
        return out

    def sorted_sessions(
        self,
        df: pd.DataFrame,
        *,
        session_col: str = "Session",
        session_n_col: str = "session_n",
        pattern: str = r"(\d+)",
    ) -> list[str]:
        """Return session labels sorted by extracted numeric session index."""
        with_n = df if session_n_col in df.columns else self.add_session_number(
            df,
            session_col=session_col,
            out_col=session_n_col,
            pattern=pattern,
        )
        return (
            with_n[[session_col, session_n_col]]
            .drop_duplicates()
            .sort_values(session_n_col)
            [session_col]
            .tolist()
        )

    def pivot_time(
        self,
        df: pd.DataFrame,
        *,
        session_col: str = "Session",
        time_col: str = "rel_time",
        value_col: str = "value",
        aggfunc: str = "mean",
        session_order: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build a session x time matrix and optionally apply a session order."""
        pivot = df.pivot_table(
            index=session_col,
            columns=time_col,
            values=value_col,
            aggfunc=aggfunc,
        )
        if session_order is not None:
            pivot = pivot.reindex([s for s in session_order if s in pivot.index])
        return pivot

    def group_mean_sem(
        self,
        df: pd.DataFrame,
        *,
        group_col: str | list[str],
        value_col: str,
    ) -> pd.DataFrame:
        """Aggregate a value column into mean and sem by one grouping column."""
        return df.groupby(group_col)[value_col].agg(["mean", "sem"]).reset_index()
