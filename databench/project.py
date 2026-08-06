"""Project — open a dataset and select sessions for analysis.

A ``Project`` is **read-only data**: it loads a multiindex dataset, exposes
its subjects / sessions / tasks, and hands out :class:`~databench.session.Session`
objects.  Outputs and provenance live on :class:`~databench.run.Run`, which
you get from :meth:`Project.run`::

    from databench import Project

    proj = Project("hfsa")                       # alias or path
    proj.describe()                              # what's in here?

    sess  = proj.session(subject="GS28", session="ses-01", task="task-spont")
    group = proj.sessions(task="task-spont")

    run = proj.run(name="oscillations", tag="L_VISp")
    run.save_figure(fig, "overview.svg")
    run.finish(notes="...")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from databench.config import (
    Corrections,
    Schema,
    _resolve_dataset_alias_for_output,
    _user_config,
    dataset_params,
    resolve_dataset,
)
from databench.tabler import DataTabler

_DATASET_SUFFIXES = (".pkl", ".pickle", ".h5", ".hdf", ".hdf5", ".parquet", ".csv")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a dataset from disk (.pkl, .h5, .parquet, .csv)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix in {".h5", ".hdf", ".hdf5"}:
        return cast(pd.DataFrame, pd.read_hdf(path, key="HFSA"))
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported file format: {path.suffix!r}. Use .pkl, .h5, .parquet, or .csv."
    )


# ── Exceptions ───────────────────────────────────────────────────────────

class SessionNotFoundError(KeyError):
    """No session matched the given selection criteria."""


class AmbiguousSessionError(ValueError):
    """Multiple sessions matched but a single session was requested."""


class NoSessionsFoundError(KeyError):
    """No sessions matched a group selection."""


# ── Project ──────────────────────────────────────────────────────────────

class Project:
    """Open a dataset and select sessions.

    Parameters
    ----------
    dataset : str or Path
        A ``datasets.toml`` alias (e.g. ``"hfsa"``) or a path to a dataset
        file. Aliases resolve via :func:`~databench.config.resolve_dataset`.
    output_root : str or Path
        Root for run directories (default ``"outputs"``).
    analyst, lab : str
        Recorded in provenance; filled from ``databench.toml`` / env when blank.
    """

    def __init__(
        self,
        dataset: str | Path,
        *,
        output_root: str | Path = "outputs",
        analyst: str = "",
        lab: str = "",
    ) -> None:
        self.dataset_path = self._resolve_path(dataset)
        self.output_root = Path(output_root)
        self.alias = _resolve_dataset_alias_for_output(self.dataset_path)

        user = _user_config()
        self.analyst = analyst or user.get("analyst", "")
        self.lab = lab or user.get("lab", "")

        try:
            self.params: dict = dataset_params(self.alias)
        except (KeyError, FileNotFoundError):
            self.params = {}
        self._schema = Schema.from_dataset_entry(self.params)
        self._corrections = Corrections.from_dataset_entry(self.params)

        df = load_dataset(self.dataset_path)
        self._df = self._apply_task_aliases(df, self._corrections)
        self._tabler = DataTabler(self._df)

    @staticmethod
    def _apply_task_aliases(df: pd.DataFrame, corrections: Corrections) -> pd.DataFrame:
        """Rewrite mistyped task strings on the ``Task`` index level.

        Applied once at load so that selection, grouping, and output naming
        all see the canonical task string.  The mapping itself reaches
        ``provenance.json`` through the dataset params, so the repair stays
        auditable.
        """
        if not corrections.task_aliases or "Task" not in (df.index.names or []):
            return df
        # Rebuild the index from arrays rather than set_levels: an alias can
        # map two distinct task strings onto one, which collapses the level.
        arrays = [
            df.index.get_level_values(name).map(corrections.canonical_task)
            if name == "Task"
            else df.index.get_level_values(name)
            for name in df.index.names
        ]
        df = df.copy()
        df.index = pd.MultiIndex.from_arrays(arrays, names=df.index.names)
        return df

    @staticmethod
    def _resolve_path(dataset: str | Path) -> Path:
        if isinstance(dataset, str) and not (
            "/" in dataset or "\\" in dataset or dataset.endswith(_DATASET_SUFFIXES)
        ):
            return resolve_dataset(dataset)
        return Path(dataset)

    # ── output ───────────────────────────────────────────────────────────

    def run(self, *, name: str = "databench", tag: str = "") -> "Run":
        """Create a :class:`~databench.run.Run` — a versioned output directory."""
        from databench.run import Run
        return Run(self, name=name, tag=tag)

    # ── data ─────────────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        return self._tabler.df

    @property
    def tabler(self) -> DataTabler:
        return self._tabler

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def corrections(self) -> Corrections:
        """Task aliases, task labels, and condition declarations for this dataset."""
        return self._corrections

    @property
    def condition_order(self) -> tuple[str, ...]:
        """Declared condition ordering, or ``()`` when the dataset declares none."""
        return self._corrections.condition_order

    @property
    def condition_colors(self) -> dict[str, str]:
        """Declared condition → colour mapping, or ``{}``."""
        return dict(self._corrections.condition_colors)

    def conditions(self) -> pd.DataFrame:
        """One row per recording: Subject, Session, Task, task_label, condition.

        Resolves each recording's condition through
        :meth:`Corrections.condition_for`, so declared repairs and the value
        the session itself carries are combined in one place rather than in
        every script.
        """
        rows = [
            {
                "Subject": sess.subject,
                "Session": sess.session,
                "Task": sess.task,
                "task_label": sess.task_label,
                "condition": sess.condition,
            }
            for sess in self.sessions()
        ]
        return pd.DataFrame(rows)

    def filter(
        self,
        drop_rows: Any = None,
        *,
        include: dict[str, Any] | None = None,
        exclude: Any = None,
        **kwargs,
    ) -> "Project":
        """Filter the dataset in place and return self (chainable).

        ``include`` / ``exclude`` take ``{level: value(s)}`` mappings;
        ``drop_rows`` and positional ``exclude`` take MultiIndex tuples.
        Keyword args are include-style: ``proj.filter(Task="task-spont")``.
        """
        self._tabler.filter(drop_rows=drop_rows, include=include, exclude=exclude, **kwargs)
        self._df = self._tabler.df
        return self

    # ── selection ────────────────────────────────────────────────────────

    def session(self, *, subject: str, session: str, task: str) -> "Session":
        """Select exactly one session; raises if zero or multiple match."""
        from databench.session import Session

        idx = self.df.index
        mask = (
            (idx.get_level_values("Subject") == subject)
            & (idx.get_level_values("Session") == session)
            & (idx.get_level_values("Task") == task)
        )
        matched = self.df.loc[mask]
        if len(matched) == 0:
            raise SessionNotFoundError(
                f"No session for subject={subject!r}, session={session!r}, task={task!r}.\n"
                f"  Available subjects: {self.subjects}\n"
                f"  Available sessions: {self.all_sessions}\n"
                f"  Available tasks:    {self.tasks}"
            )
        if len(matched) > 1:
            raise AmbiguousSessionError(
                f"{len(matched)} rows match subject={subject!r}, session={session!r}, "
                f"task={task!r}. Use project.sessions() for multi-session selection."
            )
        return Session(row=matched.iloc[0], index=matched.index[0], schema=self._schema, corrections=self._corrections)

    def sessions(
        self,
        *,
        task: str | None = None,
        subject: str | None = None,
        subjects: list[str] | None = None,
        sessions: list[str] | None = None,
    ) -> "SessionGroup":
        """Select a group of sessions. With no filters, returns all sessions."""
        from databench.session import Session, SessionGroup

        idx = self.df.index
        mask = pd.Series(True, index=self.df.index)
        if task is not None:
            mask &= idx.get_level_values("Task") == task
        if subject is not None:
            mask &= idx.get_level_values("Subject") == subject
        if subjects is not None:
            mask &= idx.get_level_values("Subject").isin(subjects)
        if sessions is not None:
            mask &= idx.get_level_values("Session").isin(sessions)

        matched = self.df.loc[mask]
        if len(matched) == 0:
            raise NoSessionsFoundError(
                "No sessions matched.\n"
                f"  Available tasks:    {self.tasks}\n"
                f"  Available sessions: {self.all_sessions}"
            )
        return SessionGroup(
            [Session(row=matched.loc[i], index=i, schema=self._schema, corrections=self._corrections) for i in matched.index]
        )

    def first_session(self) -> "Session":
        """Any session — a one-liner for "what's in this dataset?"."""
        from databench.session import Session
        if len(self.df) == 0:
            raise NoSessionsFoundError("Project has zero rows after filtering.")
        i = self.df.index[0]
        return Session(row=self.df.loc[i], index=i, schema=self._schema, corrections=self._corrections)

    # ── discovery ────────────────────────────────────────────────────────

    @property
    def subjects(self) -> list[str]:
        return sorted(self.df.index.get_level_values("Subject").unique())

    @property
    def all_sessions(self) -> list[str]:
        return sorted(self.df.index.get_level_values("Session").unique())

    @property
    def tasks(self) -> list[str]:
        return sorted(self.df.index.get_level_values("Task").unique())

    def __repr__(self) -> str:
        return f"Project(dataset={self.dataset_path.name!r}, alias={self.alias!r}, rows={len(self.df)})"

    def describe(self) -> str:
        """Print the dataset shape and (declared or introspected) sources."""
        from collections import Counter

        lines = [
            repr(self),
            f"  Subjects ({len(self.subjects)}): {self.subjects}",
            f"  Sessions ({len(self.all_sessions)}): {self.all_sessions}",
            f"  Tasks    ({len(self.tasks)}): {self.tasks}",
            f"  Rows per task: {dict(Counter(self.df.index.get_level_values('Task')))}",
        ]

        schema = self._schema
        if schema.sources or schema.columns:
            lines.append("  Schema (declared):")
            for alias, canonical in sorted(schema.sources.items()):
                lines.append(f"    {alias} -> {canonical}")
            for src in schema.declared_sources():
                cols = schema.columns_for(src)
                if cols:
                    desc = ", ".join(
                        f"{n} ({s.role}, {s.unit})" if s.unit else f"{n} ({s.role})"
                        for n, s in cols
                    )
                    lines.append(f"    {src}: {desc}")
        elif len(self.df):
            sample = self.first_session()
            if sample.sources:
                lines.append("  Sources (introspected):")
                for src in sample.sources:
                    sigs = sample.signals(src)
                    preview = sigs[:4] + (["…"] if len(sigs) > 4 else [])
                    lines.append(f"    {src}: {preview}")

        lines.append(f"  Output root: {self.output_root}")
        text = "\n".join(lines)
        print(text)
        return text
