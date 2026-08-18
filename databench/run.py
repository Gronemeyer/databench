"""Run — one analysis run's output directory and its provenance.

A :class:`~databench.project.Project` is read-only data.  A ``Run`` is the
place results go::

    proj = Project("hfsa")
    run  = proj.run(name="locomotion-bouts", tag="canonical")

    run.save_figure(fig, "overview.svg")
    run.save_table(bouts, "bouts.csv")
    run.finish(notes="...")          # writes provenance.json + report.md

Outputs land in
``outputs/<alias>/<script>/<YYMMDD_HHMMSS>[_<tag>]/`` with ``plots/``,
``tables/``, a top-level ``provenance.json`` (the single source of truth),
and a human-readable ``report.md``.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import matplotlib.pyplot as plt
import pandas as pd

from databench import provenance
from databench.config import _detect_script_name

if TYPE_CHECKING:
    from databench.project import Project


def _build_run_dir(output_root: Path, alias: str, script: str, tag: str) -> Path:
    stamp = datetime.now().strftime("%y%m%d")
    if tag:
        stamp = f"{stamp}_{tag}"
    return output_root / alias / script / stamp


class Run:
    """An output directory plus the provenance for what produced it."""

    def __init__(self, project: "Project", *, name: str = "databench", tag: str = "") -> None:
        self.project = project
        self.name = name
        self.tag = tag
        self.script = _detect_script_name()
        self.dir = _build_run_dir(project.output_root, project.alias, self.script, tag)
        self.plots_dir = self.dir / "plots"
        self.tables_dir = self.dir / "tables"
        self.dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Run(name={self.name!r}, dir={self.dir})"

    @staticmethod
    def _announce(path: Path) -> None:
        print(f"[databench] wrote {path}")

    # ── figures ──────────────────────────────────────────────────────────

    def save_figure(
        self,
        fig,
        name: str,
        *,
        dpi: int = 300,
        bbox_inches: str = "tight",
        formats: Iterable[str] | None = None,
        close: bool = True,
    ) -> Path:
        """Save *fig* to ``plots/``; ``formats`` writes extra suffixes too."""
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        path = self.plots_dir / name
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        self._announce(path)
        for fmt in formats or ():
            alt = path.with_suffix(f".{fmt.lstrip('.')}")
            if alt != path:
                fig.savefig(alt, dpi=dpi, bbox_inches=bbox_inches)
                self._announce(alt)
        if close:
            plt.close(fig)
        return path

    # ── tables ───────────────────────────────────────────────────────────

    def save_table(self, df: pd.DataFrame, name: str, *, index: bool = False) -> Path:
        """Save *df* to ``tables/``; format follows the suffix (.csv/.parquet/.json)."""
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        path = self.tables_dir / name
        if path.suffix == ".parquet":
            df.to_parquet(path, index=index)
        elif path.suffix == ".json":
            df.to_json(path, orient="records", indent=2)
        else:
            path = path.with_suffix(".csv")
            df.to_csv(path, index=index)
        self._announce(path)
        return path

    def save_json(self, payload: Mapping[str, Any], name: str) -> Path:
        """Save an arbitrary JSON payload to ``tables/``."""
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        path = self.tables_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        self._announce(path)
        return path

    @contextmanager
    def pdf(self, name: str):
        """Multi-page PDF context manager rooted at ``plots/``::

        >>> with run.pdf("report.pdf") as pdf:
        ...     pdf.savefig(fig)
        """
        from matplotlib.backends.backend_pdf import PdfPages

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        path = self.plots_dir / name
        with PdfPages(path) as pdf:
            yield pdf
        self._announce(path)

    # ── provenance ───────────────────────────────────────────────────────

    def finish(self, *, notes: str = "") -> Path:
        """Write ``provenance.json`` + ``report.md``; return the run directory."""
        prov = provenance.capture(self.project, self)
        prov_path = provenance.write(prov, self.dir / "provenance.json")
        self._announce(prov_path)
        report = provenance.render_report(prov, self, notes=notes)
        report_path = self.dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        self._announce(report_path)
        return self.dir
