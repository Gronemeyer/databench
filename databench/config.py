from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


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


# -- Output context (new API) -----------------------------------------------

@dataclass(frozen=True)
class OutputContext:
    """Lightweight value object carrying output directory information.

    Built internally by :class:`~databench.project.Project`.
    Not intended for direct user construction.
    """

    run_dir: Path
    plots_dir: Path
    stats_dir: Path
    reports_dir: Path
    config_dir: Path
    analyst: str = ""
    lab: str = ""
    run_name: str = "databench"
    tag: str = ""
    script_name: str = field(default_factory=_detect_script_name)

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in (self.run_dir, self.plots_dir, self.stats_dir, self.reports_dir, self.config_dir):
            d.mkdir(parents=True, exist_ok=True)
