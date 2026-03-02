"""Project — top-level entry point for the databench public API.

Usage::

    from databench import Project
    from databench.config import resolve_dataset

    project = Project(
        dataset=resolve_dataset(),
        analyst="Jacob Gronemeyer",
        lab="Sipe Lab",
        run_name="oscillations",
        tag="L_VISp-spont",
    )

    session = project.session(subject="GS28", session="ses-01", task="task-spont")
    sessions = project.sessions(task="task-spont")
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd

from databench._io.loader import load_dataset
from databench.config import OutputContext, _detect_script_name


# ── Exceptions ─────────────────────────────────────────────────────────────

class SessionNotFoundError(KeyError):
    """Raised when no session matches the given selection criteria."""


class AmbiguousSessionError(ValueError):
    """Raised when multiple sessions match but a single session was requested."""


class NoSessionsFoundError(KeyError):
    """Raised when no sessions match a group selection."""


# ── Project ────────────────────────────────────────────────────────────────

class Project:
    """Open a dataset and select sessions for analysis.

    Parameters
    ----------
    dataset : Path
        Path to the dataset file (.pkl, .h5, .parquet, .csv).
    output_root : Path
        Root directory for all outputs.
    analyst, lab : str
        Metadata recorded in output provenance.
    run_name : str
        Name for this analysis run (used in output directory structure).
    tag : str
        Short tag appended to run directory name.
    """

    def __init__(
        self,
        dataset: Path,
        *,
        output_root: Path = Path("outputs"),
        analyst: str = "",
        lab: str = "",
        run_name: str = "databench",
        tag: str = "",
    ) -> None:
        self._dataset_path = Path(dataset)
        self._df: pd.DataFrame = load_dataset(self._dataset_path)

        # Build output directory structure
        # outputs/<run_name>_<tag>/<YYMMDD_HHMMSS>/{plots,reports,stats}
        script_name = _detect_script_name()
        tag_suffix = f"_{tag}" if tag else ""
        run_root = Path(output_root) / f"{run_name}{tag_suffix}"
        run_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        run_dir = run_root / run_stamp

        self._context = OutputContext(
            run_dir=run_dir,
            plots_dir=run_dir / "plots",
            stats_dir=run_dir / "stats",
            reports_dir=run_dir / "reports",
            analyst=analyst,
            lab=lab,
            run_name=run_name,
            tag=tag,
            script_name=script_name,
        )
        self._context.ensure_dirs()

    # ── Selection ──────────────────────────────────────────────────────────

    def session(
        self,
        *,
        subject: str,
        session: str,
        task: str,
    ) -> "Session":
        """Select a single session by subject, session, and task.

        Raises
        ------
        SessionNotFoundError
            If no matching row exists.
        AmbiguousSessionError
            If multiple rows match.
        """
        from databench.session import Session

        df = self._df
        idx = df.index

        # Filter by each level
        mask = (
            (idx.get_level_values("Subject") == subject)
            & (idx.get_level_values("Session") == session)
            & (idx.get_level_values("Task") == task)
        )
        matched = df.loc[mask]

        if len(matched) == 0:
            available_subjects = sorted(idx.get_level_values("Subject").unique())
            available_sessions = sorted(idx.get_level_values("Session").unique())
            available_tasks = sorted(idx.get_level_values("Task").unique())
            raise SessionNotFoundError(
                f"No session found for subject={subject!r}, session={session!r}, task={task!r}.\n"
                f"  Available subjects:  {available_subjects}\n"
                f"  Available sessions:  {available_sessions}\n"
                f"  Available tasks:     {available_tasks}"
            )

        if len(matched) > 1:
            raise AmbiguousSessionError(
                f"Multiple rows ({len(matched)}) match subject={subject!r}, "
                f"session={session!r}, task={task!r}. "
                f"Expected exactly one. Use project.sessions() for multi-session selection."
            )

        row = matched.iloc[0]
        index = matched.index[0]
        return Session(row=row, index=index, context=self._context)

    def sessions(
        self,
        *,
        task: str | None = None,
        subject: str | None = None,
        subjects: list[str] | None = None,
        sessions: list[str] | None = None,
    ) -> "SessionGroup":
        """Select a group of sessions by task, subject(s), and/or session label(s).

        At least one filter must be provided.

        Raises
        ------
        NoSessionsFoundError
            If no sessions match the criteria.
        """
        from databench.session import Session, SessionGroup

        df = self._df
        idx = df.index
        mask = pd.Series(True, index=df.index)

        if task is not None:
            mask &= idx.get_level_values("Task") == task
        if subject is not None:
            mask &= idx.get_level_values("Subject") == subject
        if subjects is not None:
            mask &= idx.get_level_values("Subject").isin(subjects)
        if sessions is not None:
            mask &= idx.get_level_values("Session").isin(sessions)

        matched = df.loc[mask]
        if len(matched) == 0:
            filters = []
            if task is not None:
                filters.append(f"task={task!r}")
            if subject is not None:
                filters.append(f"subject={subject!r}")
            if subjects is not None:
                filters.append(f"subjects={subjects!r}")
            if sessions is not None:
                filters.append(f"sessions={sessions!r}")
            available_tasks = sorted(idx.get_level_values("Task").unique())
            available_sessions = sorted(idx.get_level_values("Session").unique())
            raise NoSessionsFoundError(
                f"No sessions found for {', '.join(filters)}.\n"
                f"  Available tasks:    {available_tasks}\n"
                f"  Available sessions: {available_sessions}"
            )

        session_list = []
        for row_idx in matched.index:
            row = matched.loc[row_idx]
            session_list.append(Session(row=row, index=row_idx, context=self._context))

        return SessionGroup(sessions=session_list, context=self._context)

    # ── Discovery ──────────────────────────────────────────────────────────

    @property
    def output_dir(self) -> Path:
        """Root output directory for this project run."""
        return self._context.run_dir

    @property
    def subjects(self) -> list[str]:
        """All unique subject identifiers in the dataset."""
        return sorted(self._df.index.get_level_values("Subject").unique())

    @property
    def all_sessions(self) -> list[str]:
        """All unique session labels in the dataset."""
        return sorted(self._df.index.get_level_values("Session").unique())

    @property
    def tasks(self) -> list[str]:
        """All unique task labels in the dataset."""
        return sorted(self._df.index.get_level_values("Task").unique())

    # ── Reporting ──────────────────────────────────────────────────────────

    def save_report(
        self,
        *results,
        notes: str = "",
        extra_metadata: dict[str, str] | None = None,
    ) -> Path:
        """Write a markdown summary report.

        Collects sections from any result objects that define a
        ``_report_section()`` method, then auto-discovers plots and
        data files from the output directories.

        Parameters
        ----------
        *results
            Zero or more result objects (``OscillationResult``,
            ``EtaResult``, etc.).  Each that has a ``_report_section()``
            method contributes a section to the report.
        notes : str
            Free-form notes appended at the end.
        extra_metadata : dict, optional
            Extra key/value pairs for the header table.

        Returns
        -------
        Path
            Absolute path to the written ``.md`` file.
        """
        from databench._reporting import write_report

        sections = []
        for r in results:
            if hasattr(r, "_report_section"):
                sections.append(r._report_section())

        return write_report(
            self._context,
            sections=sections,
            notes=notes,
            dataset_path=self._dataset_path,
            extra_metadata=extra_metadata,
        )

    def __repr__(self) -> str:
        n = len(self._df)
        return (
            f"Project(dataset={self._dataset_path.name!r}, "
            f"rows={n}, run_name={self._context.run_name!r})"
        )
