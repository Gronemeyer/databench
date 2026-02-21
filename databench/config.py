from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class IOConfig:
    input_path: Path
    output_root: Path = Path("outputs")
    scientist: Optional[str] = None
    run_name: str = "databench"
    tag: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))


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
