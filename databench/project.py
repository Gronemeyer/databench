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
from typing import Any, cast

import pandas as pd

from databench.config import (
    OutputContext,
    Schema,
    _detect_script_name,
    _resolve_dataset_alias_for_output,
    _user_config,
    resolve_dataset,
)
from databench.tabler import DataTabler


def _parse_run_dir_stamp(name: str) -> str:
    """Parse a run-dir name into an ISO timestamp; returns ``""`` on miss.

    Accepted shapes (oldest-to-newest convention):
      - ``YYMMDD``                 → ``20YY-MM-DDT00:00:00``
      - ``YYMMDD_HHMMSS``          → ``20YY-MM-DDTHH:MM:SS``
      - ``YYMMDD_HHMMSS_<tag>``    → ``20YY-MM-DDTHH:MM:SS``
      - ``YYMMDD_<tag>``           → ``20YY-MM-DDT00:00:00``
    """
    import re
    m = re.match(
        r"^(\d{2})(\d{2})(\d{2})(?:_(\d{2})(\d{2})(\d{2}))?",
        name,
    )
    if not m:
        return ""
    yy, mm, dd, h, mi, ss = m.groups()
    hh = h or "00"
    mn = mi or "00"
    sc = ss or "00"
    return f"20{yy}-{mm}-{dd}T{hh}:{mn}:{sc}"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a dataset from disk.

    Supported formats: ``.pkl``, ``.h5``, ``.parquet``, ``.csv``.
    """
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
        f"Unsupported file format: {path.suffix!r}. "
        "Use .pkl, .h5, .parquet, or .csv."
    )


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
    dataset : Path or str
        Path to a dataset file (.pkl, .h5, .parquet, .csv), **or** an
        alias string registered in ``datasets.toml`` (e.g. ``"hfsa"``).
        When a string is given the alias is resolved internally via
        :func:`~databench.config.resolve_dataset` — callers no longer
        need to import it separately.
    output_root : Path or ``"dataset"``
        Root directory for all outputs.  Defaults to ``Path("outputs")``,
        which writes to ``./outputs/`` relative to the working directory.

        Pass the string ``"dataset"`` to write outputs next to the
        dataset file::

            Project(dataset=path, output_root="dataset")
            # → /path/to/dataset_dir/event-detection/260404/hfsa/...

        This is never used as a fallback — it must be requested
        explicitly.
    analyst, lab : str
        Metadata recorded in output provenance.
    run_name : str
        Name for this analysis run (used in report/provenance metadata).
    tag : str
        Short tag appended to run directory name.
    """

    def __init__(
        self,
        dataset: Path | str,
        *,
        output_root: Path | str = Path("outputs"),
        analyst: str = "",
        lab: str = "",
        run_name: str = "databench",
        tag: str = "",
    ) -> None:
        # Accept either a filesystem path or a datasets.toml alias string.
        # An alias resolves through resolve_dataset; an existing path is
        # used as-is. Strings that look like paths (contain a separator
        # or end in a known extension) skip the alias lookup.
        if isinstance(dataset, str) and not (
            "/" in dataset or "\\" in dataset
            or dataset.endswith((".pkl", ".pickle", ".h5", ".hdf", ".hdf5",
                                 ".parquet", ".csv"))
        ):
            self._dataset_path = resolve_dataset(dataset)
        else:
            self._dataset_path = Path(dataset)
        self._df: pd.DataFrame = load_dataset(self._dataset_path)
        self._tabler = DataTabler(self._df)

        # Fill missing analyst/lab from databench.toml / env vars.
        if not analyst or not lab:
            user = _user_config()
            if not analyst:
                analyst = user.get("analyst", "")
            if not lab:
                lab = user.get("lab", "")

        # Resolve output_root: "dataset" → write next to the dataset file
        if isinstance(output_root, str) and output_root == "dataset":
            resolved_root = self._dataset_path.parent
        else:
            resolved_root = Path(output_root)

        # Build output directory structure
        # outputs/<dataset_alias>/<YYMMDD>/<script_name>[_<tag>]/{plots,reports,stats}
        dataset_alias = _resolve_dataset_alias_for_output(self._dataset_path)

        # Load per-dataset params (session_map, condition_order, ...) from
        # the matching `[datasets.<alias>]` table. Empty dict if missing.
        from databench.config import dataset_params as _dataset_params
        try:
            self.params: dict = _dataset_params(dataset_alias)
        except (KeyError, FileNotFoundError):
            self.params = {}

        # Schema (source aliases + per-column role/unit). Empty schema if
        # the dataset has no [sources] / [schema] sub-tables — every
        # downstream consumer treats an empty Schema as "no metadata."
        self._schema: Schema = Schema.from_dataset_entry(self.params)

        script_name = _detect_script_name()
        run_dir = self._build_run_dir(
            output_root=resolved_root,
            dataset_alias=dataset_alias,
            script_name=script_name,
            tag=tag,
        )

        self._context = OutputContext(
            run_dir=run_dir,
            plots_dir=run_dir / "plots",
            stats_dir=run_dir / "stats",
            reports_dir=run_dir / "reports",
            config_dir=run_dir / "config",
            analyst=analyst,
            lab=lab,
            run_name=run_name,
            tag=tag,
            script_name=script_name,
        )
        self._context.ensure_dirs()
        self._dataset_alias = dataset_alias
        self._io: "ProjectIO | None" = None

    @property
    def io(self) -> "ProjectIO":
        """Unified save surface: ``project.io.figure / table / report``."""
        if self._io is None:
            from databench._io import ProjectIO
            self._io = ProjectIO(self)
        return self._io

    @staticmethod
    def _build_run_dir(
        *,
        output_root: Path,
        dataset_alias: str,
        script_name: str,
        tag: str,
    ) -> Path:
        """Compose the run directory path: <root>/<alias>/<script>/<YYMMDD_HHMMSS>[_<tag>]/.

        The timestamp suffix is full second-precision so repeated runs on
        the same day no longer overwrite each other.  Use
        :meth:`Project.list_runs` / :meth:`Project.last_run` to navigate
        across runs.
        """
        stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        if tag:
            stamp = f"{stamp}_{tag}"
        return output_root / dataset_alias / script_name / stamp

    def _append_run_name(self, name: str) -> Path:
        """Append run_name to a file stem while preserving parent and suffix.

        Retained for ``Session``’s ``align()`` cache key and any internal
        callers; not exposed in the public save surface (``project.io``).
        """
        rel_path = Path(name)
        run_name = str(self._context.run_name).strip()
        if not run_name:
            return rel_path
        return rel_path.with_name(f"{rel_path.stem}_{run_name}{rel_path.suffix}")

    @property
    def df(self) -> pd.DataFrame:
        """The project dataset (potentially filtered)."""
        return self._tabler.df

    @property
    def tabler(self) -> DataTabler:
        """Index-aware dataframe helper owned by this project."""
        return self._tabler

    def filter(
        self,
        drop_rows: Any = None,
        *,
        include: dict[str, Any] | None = None,
        exclude: Any = None,
        **kwargs,
    ) -> "Project":
        """Apply filters to the project DataFrame in-place and return self.

        ``include`` uses index-level mapping filters::

            project.filter(include={"task": "task-spont"})

            project.filter(include={"subject": ["GS28", "GS29"]})

        ``exclude`` uses drop-style specs::

            project.filter(exclude={"session": ["ses-00", "ses-11"]})

            project.filter(exclude=("STREHAB14", "ses-01"))

            project.filter(exclude=["STREHAB14", "ses-01"])

        Positional tuple/list specs follow MultiIndex order from left to
        right (typically ``Subject``, ``Session``, ``Task``).

        ``drop_rows`` accepts the same drop-style specs::

            project.filter(drop_rows=[("STREHAB02", "ses-01", "task-spont")])

        Keyword arguments are also supported as include-style filters::

            project.filter(Task="task-spont")
        """
        self._tabler.filter(
            drop_rows=drop_rows,
            include=include,
            exclude=exclude,
            **kwargs,
        )
        self._df = self._tabler.df
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

        df = self.df
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
        return Session(row=row, index=index, context=self._context, schema=self._schema)

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

        df = self.df
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
            session_list.append(
                Session(row=row, index=row_idx, context=self._context, schema=self._schema)
            )

        return SessionGroup(sessions=session_list, context=self._context)

    # ── Discovery ──────────────────────────────────────────────────────────

    @property
    def output_dir(self) -> Path:
        """Root output directory for this project run."""
        return self._context.run_dir

    @property
    def plots_dir(self) -> Path:
        """Plots output directory."""
        return self._context.plots_dir

    @property
    def stats_dir(self) -> Path:
        """Stats/tables output directory."""
        return self._context.stats_dir

    @property
    def reports_dir(self) -> Path:
        """Reports output directory."""
        return self._context.reports_dir

    @property
    def subjects(self) -> list[str]:
        """All unique subject identifiers in the dataset."""
        return sorted(self.df.index.get_level_values("Subject").unique())

    @property
    def all_sessions(self) -> list[str]:
        """All unique session labels in the dataset."""
        return sorted(self.df.index.get_level_values("Session").unique())

    @property
    def tasks(self) -> list[str]:
        """All unique task labels in the dataset."""
        return sorted(self.df.index.get_level_values("Task").unique())

    @property
    def schema(self) -> Schema:
        """Dataset schema (source aliases + per-column role/unit).

        Loaded from the ``[datasets.<alias>]`` table in ``datasets.toml``
        at construction time.  Empty :class:`Schema` when not declared.
        """
        return self._schema

    def first_session(self) -> "Session":
        """Return any session — a one-liner for "what's in this dataset?".

        Useful for inspecting source/signal availability before deciding
        what to analyse::

            proj.first_session().describe()
        """
        from databench.session import Session
        if len(self.df) == 0:
            raise NoSessionsFoundError("Project has zero rows after filtering.")
        row_idx = self.df.index[0]
        row = self.df.loc[row_idx]
        return Session(row=row, index=row_idx, context=self._context, schema=self._schema)

    # ── Run history ────────────────────────────────────────────────────────

    def list_runs(
        self,
        *,
        script: str | None = None,
        since: str | None = None,
    ) -> pd.DataFrame:
        """List previous runs for this dataset under ``outputs/``.

        Each row corresponds to one run directory and exposes the metadata
        recorded in ``provenance.json`` (written by every call to
        :meth:`ProjectIO.report`).  Pure-pandas — compose with
        ``.query(...)`` for richer filtering.

        Parameters
        ----------
        script : str, optional
            Only include runs whose ``script_name`` matches this stem.
        since : str, optional
            ISO date (``"2026-04-01"``) lower bound on ``created_at``.

        Returns
        -------
        pd.DataFrame
            Columns: ``run_dir``, ``script``, ``tag``, ``datetime``,
            ``has_plots``, ``has_stats``, ``has_report``,
            ``analyst``, ``git_short``, ``git_dirty``.
        """
        import json as _json

        base = self._context.run_dir.parent.parent  # <root>/<alias>/
        if not base.exists():
            return pd.DataFrame(columns=[
                "run_dir", "script", "tag", "datetime",
                "has_plots", "has_stats", "has_report",
                "analyst", "git_short", "git_dirty",
            ])

        rows: list[dict] = []
        script_dirs = (
            [base / script] if script is not None
            else [p for p in base.iterdir() if p.is_dir()]
        )
        for script_dir in script_dirs:
            if not script_dir.is_dir():
                continue
            for run_dir in script_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                prov_path = run_dir / "provenance.json"
                if prov_path.is_file():
                    try:
                        prov = _json.loads(prov_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        prov = {}
                else:
                    prov = {}
                # Prefer the provenance timestamp; otherwise parse the
                # dir name (format: YYMMDD_HHMMSS[_tag] or YYMMDD[_tag]).
                dt = prov.get("created_at", "")
                if not dt:
                    dt = _parse_run_dir_stamp(run_dir.name)
                row = {
                    "run_dir": run_dir,
                    "script": script_dir.name,
                    "tag": prov.get("tag", ""),
                    "datetime": dt,
                    "has_plots": (run_dir / "plots").is_dir()
                                 and any((run_dir / "plots").iterdir()),
                    "has_stats": (run_dir / "stats").is_dir()
                                 and any((run_dir / "stats").iterdir()),
                    "has_report": any(run_dir.glob("reports/*.md"))
                                  or any(run_dir.glob("reports/*.pdf")),
                    "analyst": prov.get("analyst", ""),
                    "git_short": (prov.get("git") or {}).get("short", ""),
                    "git_dirty": (prov.get("git") or {}).get("dirty", False),
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values("datetime", ascending=False).reset_index(drop=True)
        if since is not None:
            df = df[df["datetime"] >= since].reset_index(drop=True)
        return df

    def last_run(self, script: str | None = None) -> Path | None:
        """Return the most recent run directory (or ``None``)."""
        df = self.list_runs(script=script)
        if df.empty:
            return None
        return Path(df.iloc[0]["run_dir"])

    def __repr__(self) -> str:
        n = len(self.df)
        return (
            f"Project(dataset={self._dataset_path.name!r}, "
            f"rows={n}, run_name={self._context.run_name!r})"
        )

    def describe(self) -> str:
        """Print a human-readable summary of the dataset and run dirs.

        The canonical first call when starting a new analysis::

            proj = Project("hfsa")
            proj.describe()

        Output includes index structure (subjects/sessions/tasks), the
        schema-declared sources and columns when available, and where
        outputs from this run will land.
        """
        from collections import Counter
        subjects = self.subjects
        sessions = self.all_sessions
        tasks = self.tasks
        rows_per_task = Counter(self.df.index.get_level_values("Task"))
        lines = [
            repr(self),
            f"  Subjects ({len(subjects)}): {subjects}",
            f"  Sessions ({len(sessions)}): {sessions}",
            f"  Tasks    ({len(tasks)}): {tasks}",
            f"  Rows per task: {dict(rows_per_task)}",
        ]

        # Schema block — prefer the declared schema; otherwise fall back
        # to dynamic introspection on a representative session.
        try:
            sample = self.first_session()
        except Exception:
            sample = None

        schema = self._schema
        if schema.sources or schema.columns:
            lines.append("  Schema (declared):")
            if schema.sources:
                for alias, canonical in sorted(schema.sources.items()):
                    lines.append(f"    {alias} -> {canonical}")
            for src in schema.declared_sources():
                cols = schema.columns_for(src)
                if not cols:
                    continue
                col_strs = [
                    f"{name} ({spec.role}, {spec.unit})" if spec.unit
                    else f"{name} ({spec.role})"
                    for name, spec in cols
                ]
                lines.append(f"    {src}: {', '.join(col_strs)}")
        elif sample is not None:
            srcs = sample.sources
            if srcs:
                lines.append("  Sources (introspected):")
                for src in srcs:
                    sigs = sample.signals(src)
                    preview = sigs[:4] + (["…"] if len(sigs) > 4 else [])
                    lines.append(f"    {src}: {preview}")

        lines.extend([
            f"  Run dir:    {self._context.run_dir}",
            f"  Plots dir:  {self._context.plots_dir}",
            f"  Stats dir:  {self._context.stats_dir}",
            f"  Reports:    {self._context.reports_dir}",
        ])
        # Drop schema-related and noise keys from the params print —
        # they're already shown above.
        user_params = sorted(
            k for k in self.params
            if k not in {"path", "schema", "sources"}
        )
        if user_params:
            lines.append(f"  Params:     {user_params}")
        text = "\n".join(lines)
        print(text)
        return text
