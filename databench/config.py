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


@dataclass(frozen=True)
class IOConfig:
    input_path: Path
    output_root: Path = Path("outputs")
    scientist: Optional[str] = None
    analyst: Optional[str] = None
    lab: Optional[str] = None
    run_name: str = "databench"
    tag: str = ""
    script_name: str = field(default_factory=_detect_script_name)


@dataclass(frozen=True)
class FilterConfig:
    drop_rows: Any = ()


@dataclass(frozen=True)
class OutputPaths:
    run_dir: Path
    plots: Path
    reports: Path
    stats: Path
    config: Path


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

    datasets: dict[str, str] = cfg["datasets"]

    if alias is None:
        alias = os.environ.get("DATABENCH_DATASET")
    if alias is None:
        alias = cfg.get("default")
    if alias is None:
        raise KeyError("No alias given, DATABENCH_DATASET not set, and no 'default' in datasets.toml")

    if alias not in datasets:
        available = ", ".join(sorted(datasets))
        raise KeyError(f"Unknown dataset alias {alias!r}. Available: {available}")

    path = Path(datasets[alias])
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


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

    datasets: dict[str, str] = cfg.get("datasets", {})

    # If the actual dataset path is registered in the toml, use that alias.
    if input_path is not None:
        try:
            target = Path(input_path).resolve()
            for name, raw_path in datasets.items():
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
    analyst: str = ""
    lab: str = ""
    run_name: str = "databench"
    tag: str = ""
    script_name: str = field(default_factory=_detect_script_name)

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in (self.run_dir, self.plots_dir, self.stats_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)
