from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import json
import subprocess

if TYPE_CHECKING:
    from databench.bench import Bench


def _safe_git_hash() -> Optional[str]:
    out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
    return out.decode("utf-8").strip()


def _serialize_config(obj):
    return asdict(obj)


def save_provenance(bench: "Bench", *, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(),
        "git_hash": _safe_git_hash(),
        "io_config": _serialize_config(bench.io_config),
        "filter_config": _serialize_config(bench.filter_config),
        "features": {
            name: {"class": feat.__class__.__name__, "config": _serialize_config(feat)}
            for name, feat in bench._features.items()
        },
        "analyses": {
            name: {"class": analysis.__class__.__name__, "config": _serialize_config(analysis)}
            for name, analysis in bench._analyses.items()
        },
        "plotters": {
            name: {"class": plotter.__class__.__name__, "config": _serialize_config(plotter)}
            for name, plotter in bench._plotters.items()
        },
    }

    path = output_dir / "provenance.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path