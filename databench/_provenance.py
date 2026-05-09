"""Provenance capture for analysis runs.

Captures the minimum needed to reproduce a run: databench version, git
state, dataset path, script path, environment versions.  ``capture()``
returns a JSON-serialisable dict; ``write()`` persists it as
``provenance.json`` in the run directory.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from importlib import metadata as _md
from pathlib import Path
from typing import Any

from databench.config import OutputContext


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_info(repo_root: Path, script_path: Path | None) -> dict[str, Any]:
    full = _git(["rev-parse", "HEAD"], repo_root)
    short = _git(["rev-parse", "--short", "HEAD"], repo_root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    remote = _git(["config", "--get", "remote.origin.url"], repo_root)
    status = _git(["status", "--porcelain"], repo_root)
    dirty = bool(status)

    # Normalise GitHub remote (https or ssh) into web URL
    web_url: str | None = None
    if remote:
        r = remote.strip()
        if r.startswith("git@github.com:"):
            r = "https://github.com/" + r[len("git@github.com:") :]
        if r.endswith(".git"):
            r = r[:-4]
        if "github.com/" in r:
            web_url = r

    permalink: str | None = None
    if web_url and full and script_path is not None:
        try:
            rel = script_path.resolve().relative_to(repo_root).as_posix()
            permalink = f"{web_url}/blob/{full}/{rel}"
        except ValueError:
            permalink = None

    return {
        "hash": full,
        "short": short,
        "branch": branch,
        "dirty": dirty,
        "remote_url": remote,
        "web_url": web_url,
        "permalink": permalink,
    }


def _env_info() -> dict[str, str]:
    info: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("numpy", "pandas", "scipy", "matplotlib"):
        try:
            info[pkg] = _md.version(pkg)
        except _md.PackageNotFoundError:
            pass
    return info


def _databench_version() -> str:
    """Resolve the running databench version.

    Resolution order:
    1. ``databench._version.__version__`` written by setuptools-scm at
       build/install time.
    2. ``git describe --tags --always --dirty`` against the repo root,
       so editable checkouts without a build still report a meaningful
       version.
    3. ``importlib.metadata.version("databench")``.
    4. ``"unknown"``.
    """
    try:
        from databench._version import __version__  # type: ignore[import-not-found]
        return __version__
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    described = _git(["describe", "--tags", "--always", "--dirty"], repo_root)
    if described:
        return described

    try:
        return _md.version("databench")
    except _md.PackageNotFoundError:
        return "unknown"


def _detect_script_path() -> Path | None:
    main = sys.modules.get("__main__")
    p = getattr(main, "__file__", None)
    if p:
        return Path(p).resolve()
    return None


def capture(
    context: OutputContext,
    dataset_path: Path | str,
    *,
    dataset_alias: str | None = None,
) -> dict[str, Any]:
    """Build a provenance dict for the current run."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = _detect_script_path()

    script_info: dict[str, Any] = {}
    if script_path is not None:
        script_info["absolute_path"] = str(script_path)
        try:
            script_info["relative_path"] = script_path.relative_to(repo_root).as_posix()
        except ValueError:
            script_info["relative_path"] = None

    return {
        "databench_version": _databench_version(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": context.run_name,
        "tag": context.tag,
        "analyst": context.analyst,
        "lab": context.lab,
        "git": _git_info(repo_root, script_path),
        "script": script_info,
        "dataset": {
            "absolute_path": str(Path(dataset_path)) if dataset_path else None,
            "alias": dataset_alias,
        },
        "env": _env_info(),
    }


def write(context: OutputContext, provenance: dict[str, Any]) -> Path:
    """Write the provenance dict as ``provenance.json`` in run_dir."""
    context.run_dir.mkdir(parents=True, exist_ok=True)
    path = context.run_dir / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    return path


def snapshot_main_params() -> dict[str, Any]:
    """Scrape UPPER_CASE module-level constants from the running script."""
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


def write_params(context: OutputContext, params: dict[str, Any]) -> Path:
    """Write a params snapshot as ``config/params.json`` in run_dir."""
    config_dir = context.run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "params.json"
    path.write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")
    return path
