"""Unified save surface for analysis runs.

``ProjectIO`` is the single namespace for persisting figures, tables, and
reports.  Access it via ``project.io``::

    project.io.figure(fig, "overview.svg", sidecar=plotter.recipe(result))
    project.io.table(result.events, "bursts.csv")
    project.io.report(result, notes="...")

All paths are written under the project's run directory.  ``report()``
also writes ``provenance.json`` and ``config/params.json``.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from databench.project import Project


_KIND_TO_ATTR = {
    "stats": "stats_dir",
    "plots": "plots_dir",
    "reports": "reports_dir",
    "config": "config_dir",
    "run": "run_dir",
}


def clickLink(url: str, text: str) -> str:
    """Return an OSC-8 hyperlink string for terminal output."""
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


class ProjectIO:
    """Save figures, tables, and reports under a Project's run directory."""

    def __init__(self, project: "Project") -> None:
        self._project = project

    @property
    def context(self):
        return self._project._context

    @property
    def run_dir(self) -> Path:
        """Top-level directory for this run (delegate to context)."""
        return self.context.run_dir

    @staticmethod
    def _announce_created_path(path: Path) -> None:
        """Print created file paths so runs show concrete output locations."""
        resolved = path.resolve()
        print(f"[databench] wrote {clickLink(resolved.as_uri(), resolved.name)}")

    def path(self, name: str | None = None, *, kind: str = "stats") -> Path:
        """Return ``{kind}_dir / name`` (or just ``{kind}_dir`` if name is None).

        ``kind`` ∈ {"stats", "plots", "reports", "config", "run"}.

        Escape hatch for files that don't fit ``figure``/``table``/``report``
        (e.g. handing a path to a third-party PDF/HTML builder).
        """
        try:
            attr = _KIND_TO_ATTR[kind]
        except KeyError as e:
            raise ValueError(
                f"Unknown kind {kind!r}; expected one of {list(_KIND_TO_ATTR)}"
            ) from e
        base: Path = getattr(self.context, attr)
        base.mkdir(parents=True, exist_ok=True)
        return base if name is None else base / name

    # ── Figures ────────────────────────────────────────────────────────────

    def figure(
        self,
        fig,
        name: str,
        *,
        suptitle: str | None = None,
        tight: bool = False,
        formats: Iterable[str] | None = None,
        dpi: int = 300,
        bbox_inches: str = "tight",
        sidecar: dict[str, Any] | None = None,
        close: bool = True,
    ) -> Path:
        """Save *fig* to the plots directory; optionally write a recipe sidecar.

        Parameters
        ----------
        fig
            Matplotlib Figure.
        name : str
            File name (suffix decides the primary format).
        suptitle : str, optional
            If given, applied via ``fig.suptitle(..., y=1.02)`` before save.
        tight : bool
            If True, call ``fig.tight_layout()`` before save.
        formats : iterable of str, optional
            Additional suffixes (e.g. ``("png", "svg")``) to write next to
            *name*'s base. The primary save still goes to *name*; each
            requested suffix is written to ``<stem>.<fmt>``.
        dpi, bbox_inches
            Forwarded to ``Figure.savefig``.
        sidecar : dict, optional
            JSON-serialisable description of how the figure was built.
            Written next to the primary figure as ``<name>.recipe.json``.
        close : bool
            Close the figure after saving (default).

        Returns
        -------
        Path
            Path to the primary file.
        """
        if suptitle:
            fig.suptitle(suptitle, y=1.02)
        if tight:
            fig.tight_layout()

        out_dir = self.context.plots_dir
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        self._announce_created_path(path)

        if formats:
            for fmt in formats:
                fmt_clean = fmt.lstrip(".")
                alt = path.with_suffix(f".{fmt_clean}")
                if alt != path:
                    fig.savefig(alt, dpi=dpi, bbox_inches=bbox_inches)
                    self._announce_created_path(alt)

        if sidecar is not None:
            recipe_path = path.with_suffix(path.suffix + ".recipe.json")
            recipe_path.write_text(
                json.dumps(sidecar, indent=2, default=str), encoding="utf-8"
            )
            self._announce_created_path(recipe_path)
        if close:
            plt.close(fig)
        return path

    # ── Tables ─────────────────────────────────────────────────────────────

    def table(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        index: bool = False,
    ) -> Path:
        """Save *df* under ``stats/``.  Format inferred from suffix."""
        out_dir = self.context.stats_dir
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            df.to_parquet(path, index=index)
        elif path.suffix == ".json":
            df.to_json(path, orient="records", indent=2)
        else:
            if path.suffix != ".csv":
                path = path.with_suffix(".csv")
            df.to_csv(path, index=index)
        self._announce_created_path(path)
        return path

    def tables(
        self,
        mapping: Mapping[str, pd.DataFrame],
        *,
        prefix: str = "",
        suffix: str = ".csv",
        index: bool = False,
    ) -> dict[str, Path]:
        """Save many DataFrames at once.

        Each ``key`` in *mapping* becomes ``{prefix}_{key}{suffix}`` (or
        ``{key}{suffix}`` when prefix is empty).
        """
        out: dict[str, Path] = {}
        for key, df in mapping.items():
            stem = f"{prefix}_{key}" if prefix else str(key)
            out[key] = self.table(df, f"{stem}{suffix}", index=index)
        return out

    def dict_table(
        self,
        mapping: Mapping[str, Mapping[str, Any]],
        name: str,
        *,
        key_column: str = "feature",
    ) -> Path | None:
        """Save a ``{key: {col: val, ...}}`` mapping as a CSV.

        Empty mappings are skipped (returns ``None``).
        """
        if not mapping:
            return None
        df = (
            pd.DataFrame(mapping)
            .T.reset_index()
            .rename(columns={"index": key_column})
        )
        return self.table(df, name)

    def json(self, payload: dict, name: str) -> Path:
        """Save an arbitrary JSON payload under ``stats/``."""
        out_dir = self.context.stats_dir
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        self._announce_created_path(path)
        return path

    def text(self, name: str, content: str, *, kind: str = "stats") -> Path:
        """Save plain text under ``{kind}/`` (default: ``stats/``)."""
        path = self.path(name, kind=kind)
        path.write_text(content, encoding="utf-8")
        self._announce_created_path(path)
        return path

    def dump(
        self,
        scope: Mapping[str, Any],
        *,
        subdir: str = "_debug",
        max_rows: int | None = None,
    ) -> list[Path]:
        """Dump every DataFrame in *scope* to ``run_dir/<subdir>/<name>.csv``.

        Intended for early-iteration debugging::

            proj.io.dump(locals())

        Skips names starting with ``_`` and the project / IO objects themselves.
        Returns the list of paths written.
        """
        out_dir = self.run_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for name, value in scope.items():
            if name.startswith("_") or not isinstance(value, pd.DataFrame):
                continue
            path = out_dir / f"{name}.csv"
            df = value if max_rows is None else value.head(max_rows)
            df.to_csv(path, index=False)
            self._announce_created_path(path)
            written.append(path)
        return written

    @contextmanager
    def pdf(self, name: str):
        """Context-manager wrapper around ``PdfPages`` rooted at ``reports/``.

        Example
        -------
        >>> with proj.io.pdf("traces.pdf") as pdf:
        ...     pdf.savefig(fig)
        """
        from matplotlib.backends.backend_pdf import PdfPages

        path = self.path(name, kind="reports")
        with PdfPages(path) as pdf:
            yield pdf
        self._announce_created_path(path)

    # ── Report + provenance ────────────────────────────────────────────────

    def params(self) -> Path | None:
        """Snapshot UPPER_CASE module-level constants from the running script."""
        from databench import _provenance
        snap = _provenance.snapshot_main_params()
        if not snap:
            return None
        path = _provenance.write_params(self.context, snap)
        self._announce_created_path(path)
        return path

    def report(
        self,
        *results,
        notes: str = "",
        extra_metadata: dict[str, str] | None = None,
    ) -> Path:
        """Write the markdown report + ``provenance.json`` + params snapshot."""
        from databench._reporting import write_report
        from databench import _provenance

        sections = []
        for r in results:
            if hasattr(r, "_report_section"):
                sections.append(r._report_section())

        prov = _provenance.capture(
            self.context,
            self._project._dataset_path,
            dataset_alias=getattr(self._project, "_dataset_alias", None),
        )
        provenance_path = _provenance.write(self.context, prov)
        self._announce_created_path(provenance_path)
        snap = _provenance.snapshot_main_params()
        if snap:
            params_path = _provenance.write_params(self.context, snap)
            self._announce_created_path(params_path)

        report_path = write_report(
            self.context,
            sections=sections,
            notes=notes,
            dataset_path=self._project._dataset_path,
            extra_metadata=extra_metadata,
            provenance=prov,
        )
        self._announce_created_path(report_path)
        return report_path
