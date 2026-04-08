"""Markdown report generation for the new public API.

Produces a human-readable ``.md`` summary that mirrors the old
summary style: metadata header, analysis parameters,
embedded plot links, and free-form notes.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from databench.config import OutputContext


# ── Git helper ─────────────────────────────────────────────────────────────

def _git_hash() -> str | None:
    try:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


# ── Formatting helpers ─────────────────────────────────────────────────────

def _fmt(value: Any) -> str:
    """Format a value for display in the report."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _relative(base: Path, target: Path | str) -> str:
    """Compute a relative path string from *base* directory to *target*."""
    try:
        return os.path.relpath(str(Path(target)), start=str(base))
    except Exception:
        return str(target)


def _collect_files(directory: Path, suffixes: Sequence[str] | None = None) -> list[Path]:
    """List files in *directory*, optionally filtered by suffix."""
    if not directory.is_dir():
        return []
    files = sorted(directory.iterdir())
    if suffixes:
        files = [f for f in files if f.suffix.lower() in suffixes]
    return files


# ── Report section protocol ───────────────────────────────────────────────

class ReportSection:
    """A section that an analysis result contributes to the markdown report.

    Attributes
    ----------
    heading : str
        Markdown heading (without ``##``).
    params : dict
        Key/value pairs to list under the heading.
    notes : str
        Free-form description.
    figures : list of Path
        Paths to figure files to embed.
    tables : list of Path
        Paths to CSV/data files to reference.
    """

    def __init__(
        self,
        heading: str,
        *,
        params: dict[str, Any] | None = None,
        notes: str = "",
        figures: list[Path] | None = None,
        tables: list[Path] | None = None,
    ) -> None:
        self.heading = heading
        self.params = params or {}
        self.notes = notes
        self.figures = figures or []
        self.tables = tables or []


# ── Main report writer ────────────────────────────────────────────────────

def write_report(
    context: OutputContext,
    *,
    sections: Sequence[ReportSection] | None = None,
    notes: str = "",
    dataset_path: Path | str = "",
    extra_metadata: dict[str, str] | None = None,
) -> Path:
    """Write a markdown summary report to the reports directory.

    Parameters
    ----------
    context : OutputContext
        Output context for this run.
    sections : list of ReportSection, optional
        Analysis-specific sections to include.
    notes : str
        Free-form notes appended at the end.
    dataset_path : Path or str
        Path to the input dataset (for provenance).
    extra_metadata : dict, optional
        Additional key/value pairs for the header.

    Returns
    -------
    Path
        Absolute path to the written ``.md`` file.
    """
    lines: list[str] = []
    now = datetime.now()
    date_prefix = now.strftime("%y%m%d")
    script = context.script_name or "unknown"

    # ── Title ──────────────────────────────────────────────────────────────
    lines.append(f"# {context.run_name}")
    if context.tag:
        lines.append(f"**Tag:** {context.tag}")
    lines.append("")

    # ── Metadata table ─────────────────────────────────────────────────────
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Created | {now.strftime('%Y-%m-%d %H:%M:%S')} |")
    lines.append(f"| Script | `{script}` |")
    if context.analyst:
        lines.append(f"| Analyst | {context.analyst} |")
    if context.lab:
        lines.append(f"| Lab | {context.lab} |")
    if dataset_path:
        lines.append(f"| Dataset | `{dataset_path}` |")
    git = _git_hash()
    if git:
        lines.append(f"| Git | `{git}` |")
    if extra_metadata:
        for k, v in extra_metadata.items():
            lines.append(f"| {k} | {v} |")
    lines.append("")

    # ── Analysis sections ──────────────────────────────────────────────────
    if sections:
        for sec in sections:
            lines.append(f"## {sec.heading}")
            lines.append("")

            if sec.notes:
                lines.append(sec.notes)
                lines.append("")

            if sec.params:
                lines.append("**Parameters:**")
                lines.append("")
                for k, v in sec.params.items():
                    lines.append(f"- `{k}` = {_fmt(v)}")
                lines.append("")

            if sec.figures:
                lines.append("**Figures:**")
                lines.append("")
                for fig_path in sec.figures:
                    rel = _relative(context.reports_dir, fig_path)
                    alt = fig_path.stem.replace("_", " ")
                    lines.append(f"![{alt}]({rel})")
                    lines.append("")

            if sec.tables:
                lines.append("**Tables:**")
                lines.append("")
                for tbl_path in sec.tables:
                    rel = _relative(context.reports_dir, tbl_path)
                    lines.append(f"- [{tbl_path.name}]({rel})")
                lines.append("")

    # ── Auto-discovered outputs ────────────────────────────────────────────
    # If no sections provided explicit figures, list everything in plots/
    section_figs = {f for s in (sections or []) for f in s.figures}
    extra_plots = [
        p for p in _collect_files(context.plots_dir, [".svg", ".png", ".pdf", ".jpg"])
        if p not in section_figs
    ]
    if extra_plots:
        lines.append("## Plots")
        lines.append("")
        for p in extra_plots:
            rel = _relative(context.reports_dir, p)
            alt = p.stem.replace("_", " ")
            lines.append(f"![{alt}]({rel})")
            lines.append("")

    extra_tables = [
        p for p in _collect_files(context.stats_dir, [".csv", ".json"])
    ]
    extra_tables.extend(_collect_files(context.run_dir, [".json"]))
    section_tbls = {t for s in (sections or []) for t in s.tables}
    extra_tables = sorted({t for t in extra_tables if t not in section_tbls})
    if extra_tables:
        lines.append("## Data files")
        lines.append("")
        for p in extra_tables:
            rel = _relative(context.reports_dir, p)
            lines.append(f"- [{p.name}]({rel})")
        lines.append("")

    # ── Notes ──────────────────────────────────────────────────────────────
    if notes:
        lines.append("---")
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append(notes)
        lines.append("")

    # ── Write ──────────────────────────────────────────────────────────────
    context.reports_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{date_prefix}_{script}_summary" if script != "unknown" else f"{date_prefix}_summary"
    run_suffix = str(context.run_name).strip()
    filename = f"{base_name}_{run_suffix}.md" if run_suffix else f"{base_name}.md"
    path = context.reports_dir / filename
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path
