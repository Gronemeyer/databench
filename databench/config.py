from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


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
    drop_rows: tuple = ()  # e.g., (("GS29","ses-04","task-movies"),)


@dataclass(frozen=True)
class OutputPaths:
    run_dir: Path
    plots: Path
    reports: Path
    stats: Path
    config: Path


# -- Default condition color / order maps ------------------------------------
# Import and override in scripts when needed:
#   from databench.config import CONDITION_COLORS, CONDITION_ORDER
#   CONDITION_COLORS["my_cond"] = "#abcdef"

CONDITION_COLORS: Dict[str, str] = {
    "baseline": "#bbabab",
    "saline": "#4289e6",
    "ethanol_low": "#ffa251",
    "ethanol_high": "#ce1818",
}

CONDITION_ORDER: List[str] = [
    "baseline",
    "saline",
    "ethanol_low",
    "ethanol_high",
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
