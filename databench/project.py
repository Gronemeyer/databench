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

import matplotlib.pyplot as plt
import pandas as pd

from databench._io.loader import load_dataset
from databench.config import FilterConfig, OutputContext, _detect_script_name, _resolve_dataset_alias_for_output
from databench.utils import drop_rows


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
        Name for this analysis run (used in report/provenance metadata).
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
        self.filter_config: FilterConfig | None = None

        # Build output directory structure
        # outputs/<dataset_alias>/<script_name>/<YYMMDD>/<tag>/{plots,reports,stats}
        dataset_alias = _resolve_dataset_alias_for_output(self._dataset_path)
        script_name = _detect_script_name()
        date_stamp = datetime.now().strftime("%y%m%d")
        tag_folder = tag or "untagged"
        run_dir = Path(output_root) / dataset_alias / script_name / date_stamp / tag_folder

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

    @property
    def df(self) -> pd.DataFrame:
        """The project dataset (potentially filtered)."""
        return self._df

    def set_filters(self, drop_rows: tuple = ()) -> "Project":
        """Set dataset filters for later use with ``filter_data()``."""
        self.filter_config = FilterConfig(drop_rows=drop_rows)
        return self

    def filter_data(self, df: pd.DataFrame, drop_rows_list: tuple | None = None) -> pd.DataFrame:
        """Filter rows using stored or provided drop rules."""
        if drop_rows_list is None and self.filter_config is not None:
            drop_rows_list = self.filter_config.drop_rows
        return drop_rows(df, drop_rows_list or ())

    def filter(self, drop_rows: tuple | None = None, **kwargs) -> "Project":
        """Apply filters to the project DataFrame in-place and return self.

        Keyword arguments are matched against index levels::

            project.filter(Task="task-spont")

        Or use ``drop_rows`` for explicit multi-index tuple exclusion.
        """
        if drop_rows is not None:
            self.set_filters(drop_rows)

        df = self.df
        if self.filter_config is not None:
            df = self.filter_data(df)

        for level, value in kwargs.items():
            if level in df.index.names:
                mask = df.index.get_level_values(level) == value
                df = df.loc[mask]

        self._df = df
        return self

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

    # ── Save helpers ───────────────────────────────────────────────────────

    def save_table(
        self,
        df: pd.DataFrame,
        name: str = "table.csv",
    ) -> Path:
        """Save a DataFrame to the stats output directory.

        Returns the path to the saved file.
        """
        out_dir = self._context.stats_dir
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            df.to_parquet(path)
        else:
            path = path.with_suffix(".csv") if path.suffix not in (".csv",) else path
            df.to_csv(path)
        return path

    def save_figure(
        self,
        fig,
        name: str,
        dpi: int = 300,
        bbox_inches: str = "tight",
    ) -> Path:
        """Save a matplotlib figure to the plots output directory."""
        out_dir = self._context.plots_dir
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        return path

    def save_and_close(
        self,
        fig,
        name: str,
        dpi: int = 300,
    ) -> Path:
        """Save a figure and close it. Returns the output path."""
        path = self.save_figure(fig, name, dpi=dpi)
        plt.close(fig)
        return path

    def __repr__(self) -> str:
        n = len(self._df)
        return (
            f"Project(dataset={self._dataset_path.name!r}, "
            f"rows={n}, run_name={self._context.run_name!r})"
        )
