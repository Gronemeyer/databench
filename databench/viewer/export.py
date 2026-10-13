"""Write viewer pages that open without Python.

Two flavours, both built from the same ``index.html``:

``export(root)``
    ``outputs/index.html`` — metadata is embedded, figures are linked by
    relative path.  Small, covers every run, needs the folder beside it.

``export_standalone(root, ...)`` / ``export_run(root, run_dir)``
    One file with the figures embedded as data URIs too, so it can be emailed
    on its own.  Scope it with filters, or take a single run.  Browsers treat
    each local file as its own origin and prompt before reading siblings, so
    embedding is what makes a page truly self-contained.

For each figure only one image is embedded: the SVG when there is one, since
it stays sharp at any zoom, unless it is much larger than the PNG beside it
(true of heatmaps and brain maps, where the SVG just wraps a bitmap).
"""
from __future__ import annotations

import base64
import mimetypes
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from databench.viewer.server import _INDEX_HTML, _is_run_dir, discover, run_detail, table_preview

INDEX_NAME = "index.html"
RUN_PAGE_NAME = "report.html"
PREVIEW_ROWS = 100
TEXT_LIMIT = 100_000
EMBED_LIMIT = 12_000_000        # skip any single image bigger than this
SVG_PENALTY = 2.0               # prefer the PNG when the SVG is this much bigger
_TABLE_SUFFIXES = {".csv", ".tsv", ".parquet"}
_TEXT_SUFFIXES = {".md", ".txt", ".json"}
_RASTER = {".png", ".jpg", ".jpeg", ".gif"}


# ── embedding ────────────────────────────────────────────────────────────


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _pick_image(files: list[dict]) -> dict | None:
    """The one image to embed for a figure: SVG unless a PNG is much smaller."""
    svg = next((f for f in files if f["path"].lower().endswith(".svg")), None)
    raster = next((f for f in files if Path(f["path"]).suffix.lower() in _RASTER), None)
    if svg and raster and svg["size"] > raster["size"] * SVG_PENALTY:
        svg = None
    for choice in (svg, raster):
        if choice and choice["size"] <= EMBED_LIMIT:
            return choice
    return None


def _embed_figures(root: Path, detail: dict) -> dict[str, str]:
    """Data URIs for one image per figure; marks the chosen files ``embedded``."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in detail["files"]:
        if f["kind"] == "figure":
            groups[f["path"].rsplit(".", 1)[0]].append(f)
    assets = {}
    for files in groups.values():
        choice = _pick_image(files)
        if choice is None:
            continue
        key = f"{detail['id']}/{choice['path']}"
        assets[key] = _data_uri(root / key)
        choice["embedded"] = True
    return assets


# ── snapshot ─────────────────────────────────────────────────────────────


def build_snapshot(root: Path, *, runs: list[dict] | None = None, embed: bool = False, base: str = "") -> dict:
    """Everything the viewer would otherwise fetch from the server."""
    root = root.resolve()
    runs = discover(root) if runs is None else runs
    details, previews, texts, assets = {}, {}, {}, {}
    for summary in runs:
        detail = run_detail(root, summary["id"])
        detail.pop("abs_path", None)  # meaningless on the recipient's machine
        details[summary["id"]] = detail
        for f in detail["files"]:
            if f["reserved"]:
                continue
            key = f"{summary['id']}/{f['path']}"
            path = root / key
            suffix = path.suffix.lower()
            if suffix in _TABLE_SUFFIXES:
                previews[key] = table_preview(path, PREVIEW_ROWS)
            elif suffix in _TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="replace")
                texts[key] = {"text": text[:TEXT_LIMIT], "truncated": len(text) > TEXT_LIMIT}
        if embed:
            assets.update(_embed_figures(root, detail))
    return {
        "root": str(root),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "embed": embed,
        "base": base,  # run pages sit inside the run dir, so relative links drop this prefix
        "runs": runs,
        "details": details,
        "previews": previews,
        "texts": texts,
        "assets": assets,
    }


def _write_page(snapshot: dict, out: Path) -> Path:
    import json

    from databench.viewer.static import render

    # Embedded pages also get a script-free rendering (phones preview .html
    # without running JavaScript).  Images live only in that markup; the
    # script reads them back out, so they aren't stored twice.
    static_html = render(snapshot) if snapshot["embed"] else ""
    payload = json.dumps({**snapshot, "assets": {}}, default=str, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")  # can't close the <script> tag early
    data_tag = f'<script id="databench-data" type="application/json">{payload}</script>\n'
    html = _INDEX_HTML.read_text(encoding="utf-8")
    html = html.replace("<!--databench-static-->", static_html, 1).replace("<!--databench-data-->", data_tag, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# ── filters ──────────────────────────────────────────────────────────────


def _as_iso(stamp: str) -> str:
    """``260801`` or ``2026-08-01`` → ``2026-08-01``."""
    stamp = stamp.strip()
    return f"20{stamp[:2]}-{stamp[2:4]}-{stamp[4:6]}" if re.fullmatch(r"\d{6}", stamp) else stamp


def select_runs(runs: list[dict], *, alias: str = "", script: str = "",
                tag: str = "", since: str = "", run: str = "") -> list[dict]:
    """Filter discovered runs; ``script``/``tag`` match as substrings."""
    since_iso = _as_iso(since) if since else ""
    return [
        r for r in runs
        if (not alias or r["alias"] == alias)
        and (not script or script in (r["script"] or ""))
        and (not tag or tag in (r["tag"] or ""))
        and (not run or r["id"].startswith(run))
        and (not since_iso or (r["created"] or "") >= since_iso)
    ]


def _default_name(**filters: str) -> str:
    parts = [v.strip("/").replace("/", "-") for v in filters.values() if v]
    return "standalone-" + ("-".join(parts) if parts else "all") + ".html"


# ── entry points ─────────────────────────────────────────────────────────


def export(root: Path) -> Path:
    """Write ``<root>/index.html``: all runs, figures linked relatively."""
    root = root.resolve()
    return _write_page(build_snapshot(root), root / INDEX_NAME)


def refresh(root: Path) -> Path | None:
    """:func:`export`, unless *root* sits inside another run (pipeline sub-outputs)."""
    root = root.resolve()
    if not root.is_dir() or any(_is_run_dir(p) for p in (root, *root.parents)):
        return None
    return export(root)


def export_standalone(root: Path, *, out: Path | None = None, **filters: str) -> tuple[Path, int]:
    """Write one self-contained page for the runs matching *filters*."""
    root = root.resolve()
    runs = select_runs(discover(root), **filters)
    if not runs:
        raise ValueError("no runs match those filters")
    out = Path(out) if out else root / _default_name(**filters)
    return _write_page(build_snapshot(root, runs=runs, embed=True), out), len(runs)


def export_run(root: Path, run_dir: Path) -> Path | None:
    """Write ``report.html`` inside *run_dir* — that one run, fully self-contained."""
    root, run_dir = root.resolve(), run_dir.resolve()
    try:
        run_id = run_dir.relative_to(root).as_posix()
    except ValueError:
        return None
    detail = run_detail(root, run_id)
    if detail is None:
        return None
    summary = {k: v for k, v in detail.items() if k not in ("provenance", "files", "abs_path")}
    summary["files"] = [f["path"] for f in detail["files"]]
    snapshot = build_snapshot(root, runs=[summary], embed=True, base=run_id)
    return _write_page(snapshot, run_dir / RUN_PAGE_NAME)
