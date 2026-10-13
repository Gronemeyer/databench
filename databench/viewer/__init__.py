"""Browse previously run analyses in a local web page.

    python -m databench.viewer            # serves ./outputs, opens a browser
    python -m databench.viewer --root /path/to/outputs --port 9000 --no-browser
    python -m databench.viewer --export                      # outputs/index.html (figures linked)
    python -m databench.viewer --standalone --alias etoh2    # one file, figures embedded

Runs are discovered on every page load, so anything a script writes through
:class:`~databench.run.Run` shows up without restarting the viewer.
"""
from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from databench.viewer.server import discover, serve

__all__ = ["discover", "serve", "main"]

_REPO_OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"


def _default_root() -> Path:
    cwd = Path.cwd() / "outputs"
    return cwd if cwd.is_dir() else _REPO_OUTPUTS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="databench-view", description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="outputs directory (default: ./outputs)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    parser.add_argument(
        "--export", action="store_true",
        help="write <root>/index.html (metadata embedded, figures linked), then exit",
    )
    parser.add_argument(
        "--standalone", action="store_true",
        help="write one file with the figures embedded too — scope it with the filters below",
    )
    parser.add_argument("--alias", default="", help="only runs from this dataset")
    parser.add_argument("--script", default="", help="only runs whose script name contains this")
    parser.add_argument("--tag", default="", help="only runs whose tag contains this")
    parser.add_argument("--since", default="", help="only runs on/after this date (260801 or 2026-08-01)")
    parser.add_argument("--run", default="", help="only this run id (an outputs-relative path)")
    parser.add_argument("--out", type=Path, default=None, help="output file for --standalone")
    args = parser.parse_args(argv)

    root = (args.root or _default_root()).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"outputs directory not found: {root}")

    if args.standalone:
        from databench.viewer.export import export_standalone

        filters = {k: getattr(args, k) for k in ("alias", "script", "tag", "since", "run")}
        try:
            out, n = export_standalone(root, out=args.out, **filters)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"[databench] wrote {out} — {n} run(s), {out.stat().st_size / 1e6:.1f} MB")
        return

    if args.export:
        from databench.viewer.export import export

        print(f"[databench] wrote {export(root)}")
        return

    httpd = serve(root, args.host, args.port)
    url = f"http://{args.host}:{httpd.server_address[1]}/"
    print(f"[databench] viewing {root}\n[databench] {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
