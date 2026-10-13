"""Local HTTP server that discovers runs under ``outputs/`` and serves the viewer.

Stdlib only.  Every request to ``/api/runs`` rescans the output root, so new
runs appear without restarting.

A *run* is any directory whose name starts with a ``YYMMDD`` stamp (the
format :class:`~databench.run.Run` writes).  Files belong to their nearest
stamped ancestor, so runs that use ``plots/``/``tables/`` subfolders still
work, and a pipeline run that nests other runs only lists its own files.
"""
from __future__ import annotations

import csv
import json
import mimetypes
import os
import re
import subprocess
import sys
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

_STAMP = re.compile(r"^(\d{2})(\d{2})(\d{2})(?:_(\d{2})(\d{2})(\d{2}))?(?:_(.+))?$")
_IMAGE = {".png", ".svg", ".jpg", ".jpeg", ".gif"}
_FIGURE = _IMAGE | {".pdf", ".tif", ".tiff", ".eps", ".mp4"}
_TABLE = {".csv", ".tsv", ".parquet", ".json"}
_DOC = {".md", ".txt"}
_RESERVED = {"provenance.json", "report.md", "report.html"}
_SKIP = {".DS_Store", "Thumbs.db"}
_GENERATED = {".html"}  # viewer pages written into the outputs tree, not run outputs
_INDEX_HTML = Path(__file__).with_name("index.html")

mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("text/markdown", ".md")


# ── discovery ────────────────────────────────────────────────────────────


def _parse_stamp(name: str) -> tuple[str | None, str]:
    """``260525_191022_canonical`` → (``2026-05-25T19:10:22``, ``canonical``)."""
    m = _STAMP.match(name)
    if not m:
        return None, ""
    yy, mo, dd, hh, mi, ss, tag = m.groups()
    try:
        dt = datetime(2000 + int(yy), int(mo), int(dd), int(hh or 0), int(mi or 0), int(ss or 0))
    except ValueError:
        return None, ""
    return dt.isoformat(), tag or ""


def _is_run_dir(path: Path) -> bool:
    return _parse_stamp(path.name)[0] is not None


def _run_files(run_dir: Path) -> list[Path]:
    """Files owned by *run_dir* — descends into subfolders but not nested runs."""
    out: list[Path] = []
    for root, dirs, files in os.walk(run_dir):
        root_p = Path(root)
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not _is_run_dir(root_p / d))
        out += [
            root_p / f for f in sorted(files)
            if f not in _SKIP and not f.startswith(".") and Path(f).suffix.lower() not in _GENERATED
        ]
    return out


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _notes_from_report(path: Path) -> str:
    """Return the ``## Notes`` section that :meth:`Run.finish` appends."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(r"^## Notes\s*\n(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    return m.group(1).strip() if m else ""


def _kind(path: Path) -> str:
    s = path.suffix.lower()
    if s in _FIGURE:
        return "figure"
    if s in _TABLE:
        return "table"
    if s in _DOC:
        return "doc"
    return "other"


def _normalize(prov: dict) -> dict:
    """Flatten current and legacy provenance schemas into one set of fields.

    Legacy files (pre-``Run``) use top-level ``run_name``/``tag``/``script``,
    ``git`` instead of ``code``, and ``dataset.absolute_path``.
    """
    run = prov.get("run") or {}
    code = prov.get("code") or prov.get("git") or {}
    ds = prov.get("dataset") or {}
    env = prov.get("env") or {}
    legacy_script = prov.get("script") if isinstance(prov.get("script"), dict) else {}
    script_path = code.get("script") or legacy_script.get("relative_path") or ""
    return {
        "name": run.get("name") or prov.get("run_name") or "",
        "tag": run.get("tag") or prov.get("tag") or "",
        "script": run.get("script") or (Path(script_path).stem if script_path else ""),
        "script_path": script_path,
        "alias": ds.get("alias") or "",
        "dataset_path": ds.get("path") or ds.get("absolute_path") or "",
        "rows": ds.get("rows"),
        "sha256": ds.get("sha256") or "",
        "git": {k: code.get(k) for k in ("hash", "short", "branch", "dirty", "permalink")},
        "env": env,
        "databench": env.get("databench") or prov.get("databench_version") or "",
        "analyst": prov.get("analyst") or "",
        "lab": prov.get("lab") or "",
    }


def _summarize(root: Path, run_dir: Path, files: list[Path]) -> dict:
    rel = run_dir.relative_to(root)
    stamp_date, stamp_tag = _parse_stamp(run_dir.name)
    prov = _read_json(run_dir / "provenance.json")
    meta = _normalize(prov)
    parts = rel.parts
    counts = {"figure": 0, "table": 0, "doc": 0, "other": 0}
    stems: set[tuple[str, str]] = set()
    for f in files:
        if f.name in _RESERVED and f.parent == run_dir:
            continue
        k = _kind(f)
        if k == "figure":
            stems.add((str(f.parent), f.stem))  # svg+png+pdf of one figure count once
        else:
            counts[k] += 1
    counts["figure"] = len(stems)
    return {
        "id": rel.as_posix(),
        "script": meta["script"] or (parts[-2] if len(parts) >= 2 else ""),
        "name": meta["name"],
        "alias": meta["alias"] or (parts[-3] if len(parts) >= 3 else ""),
        "tag": meta["tag"] or stamp_tag,
        "created": prov.get("created_at") or stamp_date,
        "git": meta["git"]["short"] or "",
        "dirty": bool(meta["git"]["dirty"]),
        "meta": meta,
        "has_provenance": bool(prov),
        "notes": _notes_from_report(run_dir / "report.md"),
        "counts": counts,
        "files": [f.relative_to(run_dir).as_posix() for f in files],
    }


def discover(root: Path) -> list[dict]:
    """Every run under *root*, newest first."""
    root = root.resolve()
    runs = []
    for dirpath, dirs, _ in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        p = Path(dirpath)
        if p == root or not _is_run_dir(p):
            continue
        files = _run_files(p)
        if files:
            runs.append(_summarize(root, p, files))
    runs.sort(key=lambda r: r["created"] or "", reverse=True)
    return runs


def run_detail(root: Path, run_id: str) -> dict | None:
    root = root.resolve()
    run_dir = _safe_path(root, run_id)
    if run_dir is None or not run_dir.is_dir():
        return None
    files = _run_files(run_dir)
    detail = _summarize(root, run_dir, files)
    detail["provenance"] = _read_json(run_dir / "provenance.json")
    detail["abs_path"] = str(run_dir.resolve())
    detail["files"] = [
        {
            "path": f.relative_to(run_dir).as_posix(),
            "kind": _kind(f),
            "size": f.stat().st_size,
            "reserved": f.name in _RESERVED and f.parent == run_dir,
        }
        for f in files
    ]
    return detail


def table_preview(path: Path, max_rows: int) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd

            df = pd.read_parquet(path)
        except Exception as exc:  # pyarrow missing, corrupt file, ...
            return {"error": f"cannot preview parquet: {exc}"}
        head = df.head(max_rows).astype(str)
        return {"header": list(map(str, df.columns)), "rows": head.values.tolist(), "total": len(df)}
    delim = "\t" if suffix == ".tsv" else ","
    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter=delim)
        header = next(reader, [])
        for row in reader:
            rows.append(row)
            if len(rows) >= max_rows:
                break
    return {"header": header, "rows": rows, "total": max(_count_lines(path) - 1, len(rows))}


def _count_lines(path: Path, chunk: int = 1 << 20) -> int:
    """Newline count in binary chunks — far faster than parsing, exact for unquoted CSVs."""
    n, last = 0, b"\n"
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            n += block.count(b"\n")
            last = block[-1:]
    return n + (last != b"\n")


# ── HTTP ─────────────────────────────────────────────────────────────────


def _safe_path(root: Path, rel: str) -> Path | None:
    """Resolve *rel* under *root*, refusing anything that escapes it."""
    target = (root / unquote(rel)).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


def _reveal(path: Path) -> None:
    if sys.platform == "darwin":
        cmd = ["open", "-R", str(path)] if path.is_file() else ["open", str(path)]
    elif sys.platform.startswith("win"):
        cmd = ["explorer", f"/select,{path}"]
    else:
        cmd = ["xdg-open", str(path if path.is_dir() else path.parent)]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, payload, status: int = 200) -> None:
            self._send(status, json.dumps(payload, default=str).encode(), "application/json")

        def _error(self, status: int, msg: str) -> None:
            self._json({"error": msg}, status)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, _INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/runs":
                    self._json({"root": str(root.resolve()), "runs": discover(root)})
                elif url.path == "/api/run":
                    detail = run_detail(root, q.get("id", ""))
                    self._json(detail) if detail else self._error(404, "run not found")
                elif url.path == "/api/table":
                    path = _safe_path(root, q.get("path", ""))
                    if path is None or not path.is_file():
                        return self._error(404, "file not found")
                    self._json(table_preview(path, int(q.get("rows", 200))))
                elif url.path.startswith("/files/"):
                    self._serve_file(url.path[len("/files/"):])
                else:
                    self._error(404, "not found")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self._error(500, f"{type(exc).__name__}: {exc}")

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/api/reveal":
                return self._error(404, "not found")
            length = int(self.headers.get("Content-Length") or 0)
            body = _read_body(self.rfile.read(length))
            path = _safe_path(root, body.get("path", ""))
            if path is None or not path.exists():
                return self._error(404, "path not found")
            _reveal(path)
            self._json({"ok": True})

        def _serve_file(self, rel: str) -> None:
            path = _safe_path(root, rel)
            if path is None or not path.is_file():
                return self._error(404, "file not found")
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype == "application/json":
                ctype += "; charset=utf-8"
            self._send(200, path.read_bytes(), ctype, {"Content-Disposition": "inline"})

    return Handler


def _read_body(raw: bytes) -> dict:
    try:
        data = json.loads(raw or b"{}")
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def serve(root: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Bind the first free port at or above *port*."""
    root = root.resolve()
    for candidate in range(port, port + 50):
        try:
            return ThreadingHTTPServer((host, candidate), make_handler(root))
        except OSError:
            continue
    raise OSError(f"no free port in {port}–{port + 49}")
