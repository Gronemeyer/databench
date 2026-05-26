"""Provenance — the single record needed to reconstruct and share a run.

One run writes one ``provenance.json``.  It answers five questions:

* **what data**   — dataset alias, path, content hash, row count
* **what code**   — git commit, dirty flag, script path, GitHub permalink
* **what params** — the script's UPPER_CASE module constants
* **what env**    — python + key package versions
* **when / who**  — timestamp, analyst, lab

``capture()`` builds the dict; :class:`~databench.run.Run` writes it.
``render_report()`` turns the same dict into a human-readable ``report.md``.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from importlib import metadata as _md
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from databench.project import Project
    from databench.run import Run

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ── version ──────────────────────────────────────────────────────────────

def _databench_version() -> str:
    """Resolve the running databench version (build tag → git → metadata)."""
    try:
        from databench._version import __version__  # type: ignore[import-not-found]
        return __version__
    except Exception:
        pass
    described = _git(["describe", "--tags", "--always", "--dirty"])
    if described:
        return described
    try:
        return _md.version("databench")
    except _md.PackageNotFoundError:
        return "unknown"


# ── git ──────────────────────────────────────────────────────────────────

def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_info(script: Path | None) -> dict[str, Any]:
    full = _git(["rev-parse", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    remote = _git(["config", "--get", "remote.origin.url"])
    dirty = bool(_git(["status", "--porcelain"]))

    web = None
    if remote:
        r = remote.removesuffix(".git")
        if r.startswith("git@github.com:"):
            r = "https://github.com/" + r[len("git@github.com:"):]
        if "github.com/" in r:
            web = r

    permalink = None
    if web and full and script is not None:
        try:
            rel = script.resolve().relative_to(_REPO_ROOT).as_posix()
            permalink = f"{web}/blob/{full}/{rel}"
        except ValueError:
            pass

    return {
        "hash": full,
        "short": full[:7] if full else None,
        "branch": branch,
        "dirty": dirty,
        "permalink": permalink,
    }


# ── environment ──────────────────────────────────────────────────────────

def _env_info() -> dict[str, str]:
    env = {"python": sys.version.split()[0], "databench": _databench_version()}
    for pkg in ("numpy", "pandas", "scipy", "matplotlib"):
        try:
            env[pkg] = _md.version(pkg)
        except _md.PackageNotFoundError:
            pass
    return env


# ── dataset ──────────────────────────────────────────────────────────────

def file_sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    """Streamed SHA-256 of a file — verifies a shared run used the same data."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ── params ───────────────────────────────────────────────────────────────

def snapshot_main_params() -> dict[str, Any]:
    """JSON-serialisable UPPER_CASE module constants from the running script."""
    main = sys.modules.get("__main__")
    if main is None:
        return {}
    out: dict[str, Any] = {}
    for k, v in vars(main).items():
        if not k.isupper() or k.startswith("_"):
            continue
        try:
            json.dumps(v, default=str)
        except Exception:
            continue
        out[k] = v
    return out


def _script_path() -> Path | None:
    p = getattr(sys.modules.get("__main__"), "__file__", None)
    return Path(p).resolve() if p else None


# ── capture / write ──────────────────────────────────────────────────────

def capture(project: "Project", run: "Run") -> dict[str, Any]:
    """Build the provenance record for a run."""
    script = _script_path()
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analyst": project.analyst,
        "lab": project.lab,
        "run": {"name": run.name, "tag": run.tag, "script": run.script},
        "dataset": {
            "alias": project.alias,
            "path": str(project.dataset_path),
            "sha256": file_sha256(project.dataset_path),
            "rows": int(len(project.df)),
        },
        "code": {
            **_git_info(script),
            "script": script.relative_to(_REPO_ROOT).as_posix()
            if script and script.is_relative_to(_REPO_ROOT) else None,
        },
        "params": snapshot_main_params(),
        "env": _env_info(),
    }


def write(prov: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    return path


# ── report ───────────────────────────────────────────────────────────────

def render_report(prov: dict[str, Any], run: "Run", *, notes: str = "") -> str:
    """Render provenance + an index of saved files as Markdown."""
    code, ds = prov["code"], prov["dataset"]
    lines = [f"# {prov['run']['name']}", ""]
    if prov["run"]["tag"]:
        lines.append(f"**Tag:** {prov['run']['tag']}\n")

    lines += ["| Field | Value |", "|---|---|"]
    lines.append(f"| Created | {prov['created_at']} |")
    lines.append(f"| Script | `{prov['run']['script']}` |")
    if prov["analyst"]:
        lines.append(f"| Analyst | {prov['analyst']} |")
    if prov["lab"]:
        lines.append(f"| Lab | {prov['lab']} |")
    lines.append(f"| Dataset | `{ds['alias']}` ({ds['rows']} rows) |")
    lines.append(f"| Data SHA-256 | `{ds['sha256'][:16]}…` |")
    short = code.get("short")
    if short:
        label = short + ("-dirty" if code.get("dirty") else "")
        link = f"[`{label}`]({code['permalink']})" if code.get("permalink") else f"`{label}`"
        lines.append(f"| Git | {link} (`{code.get('branch')}`) |")
    lines.append(f"| Env | {', '.join(f'{k} {v}' for k, v in prov['env'].items())} |")
    lines.append("")

    if code.get("hash") and prov["run"]["script"]:
        lines += [
            "**Reproduce:**", "",
            f"```bash\ngit checkout {code['hash']} && python {prov['run']['script']}\n```",
            "",
        ]

    if prov["params"]:
        lines += ["## Parameters", ""]
        lines += [f"- `{k}` = {v}" for k, v in prov["params"].items()]
        lines.append("")

    plots = _list_dir(run.plots_dir)
    if plots:
        lines += ["## Plots", ""]
        for p in plots:
            if p.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg"}:
                lines.append(f"![{p.stem}](plots/{p.name})\n")
            else:
                lines.append(f"- [plots/{p.name}](plots/{p.name})")
        lines.append("")

    tables = _list_dir(run.tables_dir)
    if tables:
        lines += ["## Tables", ""]
        lines += [f"- [tables/{p.name}](tables/{p.name})" for p in tables]
        lines.append("")

    if notes:
        lines += ["---", "", "## Notes", "", notes, ""]

    return "\n".join(lines).rstrip() + "\n"


def _list_dir(d: Path) -> list[Path]:
    return sorted(d.iterdir()) if d.is_dir() else []
