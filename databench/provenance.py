"""Provenance utilities — helper functions for provenance tracking.

This module provides utility functions for formatting provenance metadata.
The full Bench-coupled provenance system has been removed; these helpers
remain available for future integration with the Project API.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import json
import os
import subprocess


def get_git_hash() -> str | None:
    """Return the current git HEAD hash, or None if unavailable."""
    try:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def format_provenance_value(value: Any) -> str:
    """Convert a value to a string suitable for provenance records."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=True, default=str)
    return str(value)


def iter_plot_images(paths: Iterable[str]) -> List[Path]:
    """Deduplicate plot image paths, preferring .png > .jpg > .svg."""
    allowed = {".png", ".jpg", ".jpeg", ".svg"}
    preference = {".png": 0, ".jpg": 1, ".jpeg": 2, ".svg": 3}
    chosen: Dict[str, Path] = {}
    order: List[str] = []
    for entry in paths:
        path = Path(entry)
        suffix = path.suffix.lower()
        if suffix not in allowed:
            continue
        stem = path.stem
        if stem not in chosen:
            chosen[stem] = path
            order.append(stem)
            continue
        current = chosen[stem]
        if preference.get(suffix, 99) < preference.get(current.suffix.lower(), 99):
            chosen[stem] = path
    return [chosen[stem] for stem in order]
