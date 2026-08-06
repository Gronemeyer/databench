from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _detect_script_name() -> str:
    """Return the stem of the top-level script (e.g. 'oscillation-pupil-eta')."""
    main = getattr(sys.modules.get("__main__"), "__file__", None)
    if main:
        return Path(main).stem
    return "interactive"


# -- Time-column fallback registry ------------------------------------------
#
# When ``Session.align()`` cannot find ``(source, time_column)`` in the
# wide row, it consults this list — ordered from most-preferred to
# least-preferred — to recover a time vector at the same sample rate.
#
# Override at the script top with::
#
#     from databench import config
#     config.TIME_COLUMNS = [
#         ("dataqueue", "time_elapsed_s"),
#         ("time",      "master_elapsed_s"),
#     ]
#
# Each entry is ``(source, column)``.  ``master_elapsed_s`` is auto-zeroed
# so its timeline begins at 0.
TIME_COLUMNS: List[tuple[str, str]] = [
    ("dataqueue", "time_elapse_s"),
    ("dataqueue", "time_elapsed_s"),
    ("time",      "master_elapsed_s"),
    ("time",      "queue_elapsed"),
]


# -- Dataset resolver --------------------------------------------------------

def _find_datasets_toml() -> Path:
    """Walk up from cwd to find datasets.toml."""
    cur = Path.cwd()
    for parent in [cur, *cur.parents]:
        candidate = parent / "datasets.toml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "datasets.toml not found. Copy datasets.toml.example to datasets.toml "
        "and fill in your local paths."
    )


# -- User config (analyst / lab) --------------------------------------------

def _user_config() -> Dict[str, str]:
    """Read ``[user]`` table from ``./databench.toml`` then ``~/.databench.toml``.

    Returns an empty dict when no file is found.  Environment variables
    ``DATABENCH_ANALYST`` / ``DATABENCH_LAB`` override file values.
    """
    out: Dict[str, str] = {}
    candidates = [Path.cwd() / "databench.toml", Path.home() / ".databench.toml"]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with path.open("rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        user = data.get("user", {})
        if isinstance(user, dict):
            for key in ("analyst", "lab"):
                val = user.get(key)
                if isinstance(val, str) and val and key not in out:
                    out[key] = val
        if out:
            break
    for key, env in (("analyst", "DATABENCH_ANALYST"), ("lab", "DATABENCH_LAB")):
        env_val = os.environ.get(env)
        if env_val:
            out[key] = env_val
    return out


def _dataset_entry(cfg: dict, alias: str) -> dict:
    """Normalize a ``[datasets]`` entry to a dict with at least ``path``.

    Accepts both shorthand (``alias = "/path"``) and the table form
    (``[datasets.alias] path = ...`` plus arbitrary user keys).
    """
    raw = cfg["datasets"][alias]
    if isinstance(raw, str):
        return {"path": raw}
    if isinstance(raw, dict):
        if "path" not in raw:
            raise KeyError(f"Dataset {alias!r} is missing required 'path' key")
        return dict(raw)
    raise TypeError(
        f"Dataset {alias!r} must be a string or table, got {type(raw).__name__}"
    )


def _resolve_alias(cfg: dict, alias: str | None) -> str:
    """Apply the standard alias-resolution chain (arg → env → default)."""
    if alias is None:
        alias = os.environ.get("DATABENCH_DATASET")
    if alias is None:
        alias = cfg.get("default")
    if alias is None:
        raise KeyError(
            "No alias given, DATABENCH_DATASET not set, and no 'default' in datasets.toml"
        )
    if alias not in cfg.get("datasets", {}):
        available = ", ".join(sorted(cfg.get("datasets", {})))
        raise KeyError(f"Unknown dataset alias {alias!r}. Available: {available}")
    return alias


def resolve_dataset(alias: str | None = None) -> Path:
    """Return the Path for a dataset alias.

    Resolution order for the alias:
      1. ``alias`` argument (if given)
      2. ``DATABENCH_DATASET`` environment variable
      3. ``default`` key in datasets.toml

    Raises KeyError with available aliases when the alias is unknown.
    """
    toml_path = _find_datasets_toml()
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)

    alias = _resolve_alias(cfg, alias)
    entry = _dataset_entry(cfg, alias)

    path = Path(entry["path"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


def dataset_params(alias: str | None = None) -> dict:
    """Return the full ``[datasets.<alias>]`` table as a plain dict.

    Same alias-resolution rules as :func:`resolve_dataset`.  ``path`` is
    always present and resolved to a :class:`~pathlib.Path`; all other
    keys are passed through unchanged from ``datasets.toml``.

    Returns ``{"path": Path(...)}`` for shorthand-only entries.
    """
    toml_path = _find_datasets_toml()
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)

    alias = _resolve_alias(cfg, alias)
    entry = _dataset_entry(cfg, alias)
    entry["path"] = Path(entry["path"])
    return entry


@dataclass(frozen=True)
class ColumnSpec:
    """Semantic description of a single (source, column) pair.

    ``role`` is one of ``"timeseries"``, ``"event"``, ``"scalar"``, or
    ``"time"`` (extensible).  ``unit`` is a free-form string used for
    axis labels (e.g. ``"mm"``, ``"cm/s"``).
    """

    role: str = ""
    unit: str = ""


@dataclass(frozen=True)
class Schema:
    """Dataset schema: source aliases and per-column semantic metadata.

    A schema is read from the ``[datasets.<alias>]`` table in
    ``datasets.toml`` (sub-tables ``sources`` and ``schema``).  It powers
    transparent source aliasing in :class:`~databench.session.Session`,
    schema-aware printing in :meth:`Project.describe`, and the default
    plotting dispatch in :mod:`databench.plotting`.

    The dataclass is intentionally databench-import-free so the
    dataset-producing repo can emit an equivalent JSON sidecar next to a
    ``.pkl`` and have :class:`Project` consume it the same way.
    """

    sources: Mapping[str, str] = field(default_factory=dict)
    columns: Mapping[tuple[str, str], ColumnSpec] = field(default_factory=dict)

    @classmethod
    def from_dataset_entry(cls, entry: Mapping[str, Any]) -> "Schema":
        """Build a Schema from a parsed ``[datasets.<alias>]`` table."""
        raw_sources = entry.get("sources")
        sources: dict[str, str] = {}
        if isinstance(raw_sources, dict):
            for k, v in raw_sources.items():
                if isinstance(k, str) and isinstance(v, str):
                    sources[k] = v

        raw_schema = entry.get("schema")
        columns: dict[tuple[str, str], ColumnSpec] = {}
        if isinstance(raw_schema, dict):
            for key, spec in raw_schema.items():
                if not isinstance(key, str) or "." not in key:
                    continue
                source, column = key.split(".", 1)
                if not isinstance(spec, dict):
                    continue
                columns[(source, column)] = ColumnSpec(
                    role=str(spec.get("role", "")),
                    unit=str(spec.get("unit", "")),
                )
        return cls(sources=sources, columns=columns)

    def resolve_source(self, name: str) -> str:
        """Apply alias rewriting; return *name* unchanged when no alias."""
        return self.sources.get(name, name)

    def role_of(self, source: str, column: str) -> str:
        spec = self.columns.get((self.resolve_source(source), column))
        return spec.role if spec else ""

    def unit_of(self, source: str, column: str) -> str:
        spec = self.columns.get((self.resolve_source(source), column))
        return spec.unit if spec else ""

    def columns_for(self, source: str) -> list[tuple[str, ColumnSpec]]:
        """Columns declared in the schema for *source* (after alias resolution)."""
        canonical = self.resolve_source(source)
        return [
            (col, spec)
            for (src, col), spec in self.columns.items()
            if src == canonical
        ]

    def declared_sources(self) -> list[str]:
        """Canonical source keys mentioned anywhere in the schema."""
        names = set(self.sources.values()) | {src for src, _ in self.columns}
        return sorted(names)


@dataclass(frozen=True)
class Corrections:
    """Acquisition repairs and display labels declared in ``datasets.toml``.

    These are facts about a *dataset*, not about an analysis: a task string
    that was mistyped at acquisition, the short label a task is drawn with,
    or a condition that a session carries by protocol but that never made it
    into ``session_config``.  Declaring them once next to the dataset path
    keeps every script that opens the dataset in agreement, and keeps the
    repair visible in ``provenance.json`` instead of copied into script
    headers.

    Read from the ``[datasets.<alias>]`` table::

        [datasets.etoh2]
        path = "..."
        condition_order = ["Baseline", "Saline", "Low", "High"]

        [datasets.etoh2.task_aliases]
        "task-eothlate" = "task-etohlate"

        [datasets.etoh2.task_labels]
        "task-etohearly" = "early"
        "task-etohlate"  = "late"

        [datasets.etoh2.conditions]
        "ses-11"      = "Baseline"   # every subject at ses-11
        "JG02/ses-11" = "Baseline"   # one subject only; wins over the row's own value

    Attributes
    ----------
    task_aliases
        Mistyped task string → canonical task string.  :class:`Project`
        applies these to the ``Task`` index level at load, so downstream
        code never sees the typo.
    task_labels
        Canonical task string → short label for axes and filenames.
    conditions
        ``(subject_or_None, session)`` → condition.  A subject-scoped entry
        is an explicit repair and overrides the row's own condition value;
        a session-scoped entry is a protocol default and only fills a gap.
    condition_order
        Canonical ordering for grouping and plotting.
    condition_colors
        Condition → colour, for consistent figures across scripts.
    """

    task_aliases: Mapping[str, str] = field(default_factory=dict)
    task_labels: Mapping[str, str] = field(default_factory=dict)
    conditions: Mapping[tuple[Optional[str], str], str] = field(default_factory=dict)
    condition_order: tuple[str, ...] = ()
    condition_colors: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dataset_entry(cls, entry: Mapping[str, Any]) -> "Corrections":
        """Build a Corrections from a parsed ``[datasets.<alias>]`` table."""
        def _str_map(key: str) -> dict[str, str]:
            raw = entry.get(key)
            if not isinstance(raw, dict):
                return {}
            return {str(k): str(v) for k, v in raw.items()}

        conditions: dict[tuple[Optional[str], str], str] = {}
        raw_conditions = entry.get("conditions")
        if isinstance(raw_conditions, dict):
            for key, value in raw_conditions.items():
                key = str(key)
                if "/" in key:
                    subject, session = key.split("/", 1)
                    conditions[(subject, session)] = str(value)
                else:
                    conditions[(None, key)] = str(value)

        raw_order = entry.get("condition_order")
        order = (
            tuple(str(c) for c in raw_order)
            if isinstance(raw_order, (list, tuple))
            else ()
        )

        return cls(
            task_aliases=_str_map("task_aliases"),
            task_labels=_str_map("task_labels"),
            conditions=conditions,
            condition_order=order,
            condition_colors=_str_map("condition_colors"),
        )

    def canonical_task(self, task: str) -> str:
        """Apply task-alias rewriting; return *task* unchanged when no alias."""
        return self.task_aliases.get(task, task)

    def task_label(self, task: str) -> str:
        """Short display label for *task*, falling back to the task string."""
        canonical = self.canonical_task(task)
        return self.task_labels.get(canonical, canonical)

    def condition_for(
        self,
        subject: str,
        session: str,
        *,
        recorded: str | None = None,
    ) -> str:
        """Resolve the condition for one recording.

        *recorded* is the value the session itself carries (typically
        ``session_config.condition``).  A subject-scoped declaration wins
        over it, because such an entry exists precisely to correct the
        recorded value; a session-scoped declaration only applies when the
        recording carries nothing usable.
        """
        scoped = self.conditions.get((subject, session))
        if scoped is not None:
            return scoped
        if recorded is not None and str(recorded).strip().lower() not in (
            "", "nan", "none",
        ):
            return str(recorded)
        return self.conditions.get((None, session), "")

    def __bool__(self) -> bool:
        return bool(
            self.task_aliases
            or self.task_labels
            or self.conditions
            or self.condition_order
            or self.condition_colors
        )


def _resolve_dataset_alias_for_output(input_path: Path | None = None) -> str:
    """Best-effort dataset alias for output folder naming.

    Resolution order:
      1. Alias matching ``input_path`` in ``datasets.toml`` (authoritative)
      2. ``DATABENCH_DATASET`` env var (short alias name only)
      3. ``DATASET`` env var (short alias name only)
      4. ``default`` in ``datasets.toml``
      5. ``"dataset"`` fallback
    """
    try:
        toml_path = _find_datasets_toml()
    except FileNotFoundError:
        return "dataset"

    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)

    datasets: dict = cfg.get("datasets", {})

    # If the actual dataset path is registered in the toml, use that alias.
    if input_path is not None:
        try:
            target = Path(input_path).resolve()
            for name in datasets:
                try:
                    raw_path = _dataset_entry(cfg, name)["path"]
                except (KeyError, TypeError):
                    continue
                if Path(raw_path).resolve() == target:
                    return name
        except Exception:
            pass

    # Env vars are a fallback — only accept short alias names, not file paths.
    for env_var in ("DATABENCH_DATASET", "DATASET"):
        env_val = os.environ.get(env_var)
        if env_val and not Path(env_val).is_absolute():
            return env_val

    default_alias = cfg.get("default")
    if isinstance(default_alias, str) and default_alias:
        return default_alias

    return "dataset"
