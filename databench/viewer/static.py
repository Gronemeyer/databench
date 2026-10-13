"""Plain-HTML rendering of a snapshot, for viewers that don't run JavaScript.

Phones rarely open a local ``.html`` file in a real browser.  iOS always hands
it to Quick Look (Files, Mail, Messages), and mail/chat apps on Android use
similar previewers.  These render HTML and CSS but never run scripts, and the
viewer is drawn by JavaScript, so without this markup the page opens blank.

When JavaScript does run, the page reads the embedded images out of this
markup (so they are stored once) and swaps in the interactive viewer.
"""
from __future__ import annotations

import re
from collections import defaultdict
from html import escape

STATIC_ROWS = 50


def _e(value) -> str:
    return escape("" if value is None else str(value), quote=True)


def _size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    return f"{n / 1024:.0f} KB" if n < 1 << 20 else f"{n / (1 << 20):.1f} MB"


def _when(iso: str | None) -> str:
    if not iso:
        return "—"
    date, _, time = iso.partition("T")
    return date if not time or time.startswith("00:00:00") else f"{date} {time[:5]}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# ── provenance tables ────────────────────────────────────────────────────


def _value(v) -> str:
    if v is None or v == "":
        return '<span class="nil">—</span>'
    if isinstance(v, bool):
        return f'<span class="bool-{str(v).lower()}">{str(v).lower()}</span>'
    if isinstance(v, (int, float)):
        return _e(v)
    if isinstance(v, str):
        return f'<a href="{_e(v)}">{_e(v)}</a>' if re.match(r"https?://", v) else _e(v)
    if isinstance(v, list):
        if not v:
            return '<span class="nil">empty list</span>'
        if all(not isinstance(x, (dict, list)) for x in v):
            return "".join(f'<span class="tok">{_e("—" if x is None else x)}</span>' for x in v)
        return '<div class="stack">' + "".join(f"<div>{_value(x)}</div>" for x in v) + "</div>"
    if isinstance(v, dict):
        return _kv(v, nested=True) if v else '<span class="nil">empty</span>'
    return _e(v)


def _kv(obj: dict, nested: bool = False) -> str:
    rows = "".join(f"<tr><th>{_e(k)}</th><td>{_value(v)}</td></tr>" for k, v in obj.items())
    return f'<table class="kvt{" nested" if nested else ""}"><tbody>{rows}</tbody></table>'


def _block(title: str, body: str, count: int | str = "", open_: bool = False) -> str:
    return (f'<details class="file"{" open" if open_ else ""}><summary><span class="fname">{_e(title)}</span>'
            f'<span class="size">{_e(count)}</span></summary><div class="body">{body}</div></details>')


# ── sections ─────────────────────────────────────────────────────────────


def _card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="cs">{sub}</div>' if sub else ""
    return f'<div class="card"><div class="cl">{label}</div><div class="cv">{value}</div>{sub_html}</div>'


def _cards(detail: dict, n_figs: int, n_tables: int, n_docs: int) -> str:
    m = detail.get("meta") or {}
    git, env = m.get("git") or {}, m.get("env") or {}
    ds_file = re.split(r"[\\/]", m["dataset_path"])[-1] if m.get("dataset_path") else ""
    ds_sub = " · ".join(x for x in (f"{m['rows']} rows" if m.get("rows") is not None else "", _e(ds_file)) if x)
    cards = [_card("Dataset", _e(detail.get("alias") or "—"), ds_sub), _card("Created", _e(_when(detail.get("created"))))]
    if git.get("short"):
        label = _e(git["short"]) + (' <span class="pill warn">dirty</span>' if git.get("dirty") else "")
        sub = " · ".join(_e(x) for x in (git.get("branch"), m.get("script_path")) if x)
        cards.append(_card("Code", f'<span class="mono">{label}</span>', sub))
    size = sum(f["size"] for f in detail["files"])
    cards.append(_card("Outputs", f"{_plural(n_figs, 'figure')} · {_plural(n_tables, 'table')}",
                       " · ".join(x for x in (_plural(n_docs, "doc") if n_docs else "", _size(size)) if x)))
    if env.get("python"):
        cards.append(_card("Environment", f"python {_e(env['python'])}",
                           _e(f"databench {m['databench'].split('+')[0]}") if m.get("databench") else ""))
    return f'<div class="cards">{"".join(cards)}</div>'


def _figures(detail: dict, assets: dict[str, str]) -> tuple[str, int]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in detail["files"]:
        if f["kind"] == "figure":
            groups[f["path"].rsplit(".", 1)[0]].append(f)
    items = []
    for stem, files in groups.items():
        chosen = next((f for f in files if f.get("embedded")), None)
        name = _e(stem.rsplit("/", 1)[-1])
        if chosen:
            key = f"{detail['id']}/{chosen['path']}"
            items.append(f'<figure><img alt="{name}" data-asset="{_e(key)}" src="{assets[key]}">'
                         f"<figcaption>{name}</figcaption></figure>")
        else:
            exts = ", ".join(f["path"].rsplit(".", 1)[-1] for f in files)
            items.append(f'<figure class="missing"><figcaption>{name} <span class="nil">({_e(exts)}, not embedded)</span></figcaption></figure>')
    if not items:
        return "", 0
    return f'<section><h3>Figures <span class="n">{len(items)}</span></h3><div class="sfigs">{"".join(items)}</div></section>', len(items)


def _notes(text: str) -> str:
    if not text:
        return ""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    body = "".join("<p>" + re.sub(r"`([^`]+)`", r"<code>\1</code>", _e(p)) + "</p>" for p in paras)
    return f'<section><h3>Notes</h3><div class="notes md">{body}</div></section>'


def _table(preview: dict) -> str:
    if "error" in preview:
        return f'<div class="hint">{_e(preview["error"])}</div>'
    rows = preview["rows"][:STATIC_ROWS]
    head = "".join(f"<th>{_e(h)}</th>" for h in preview["header"])
    body = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in r) + "</tr>" for r in rows)
    note = f"First {len(rows)} of {preview['total']:,} rows" if len(rows) < preview["total"] else f"{preview['total']:,} rows"
    return (f'<div class="tablewrap"><table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
            f'<div class="hint">{note} · {len(preview["header"])} columns</div>')


def _files(detail: dict, snapshot: dict) -> str:
    out = []
    for kind, title in (("table", "Tables"), ("doc", "Documents")):
        blocks = []
        for f in detail["files"]:
            if f["kind"] != kind or f["reserved"]:
                continue
            key = f"{detail['id']}/{f['path']}"
            if key in snapshot["previews"]:
                body = _table(snapshot["previews"][key])
            elif key in snapshot["texts"]:
                body = f'<pre>{_e(snapshot["texts"][key]["text"])}</pre>'
            else:
                body = '<div class="hint">No preview.</div>'
            blocks.append(_block(f["path"], body, _size(f["size"])))
        if blocks:
            out.append(f'<section><h3>{title} <span class="n">{len(blocks)}</span></h3><div class="files">{"".join(blocks)}</div></section>')
    return "".join(out)


def _provenance(detail: dict) -> str:
    prov = dict(detail.get("provenance") or {})
    if not prov:
        return ""
    params = prov.pop("params", None) or {}
    blocks = [_block("Parameters", _kv(params), len(params), open_=True)] if params else []
    blocks.append(_block("Run record", _kv(prov), len(prov), open_=not params))
    return f'<section><h3>Provenance</h3><div class="files">{"".join(blocks)}</div></section>'


def _run(detail: dict, snapshot: dict, anchor: str) -> str:
    figs_html, n_figs = _figures(detail, snapshot["assets"])
    n_tables = sum(f["kind"] == "table" and not f["reserved"] for f in detail["files"])
    n_docs = sum(f["kind"] == "doc" and not f["reserved"] for f in detail["files"])
    tag = f' <span class="pill tag">{_e(detail["tag"])}</span>' if detail.get("tag") else ""
    return (f'<article class="srun" id="{anchor}"><div class="title"><h2>{_e(detail.get("script") or detail["id"])}</h2>{tag}</div>'
            f'<div class="path mono">{_e(detail["id"])}</div>'
            f"{_cards(detail, n_figs, n_tables, n_docs)}{figs_html}{_notes(detail.get('notes', ''))}"
            f"{_files(detail, snapshot)}{_provenance(detail)}</article>")


def render(snapshot: dict) -> str:
    """The whole snapshot as static HTML, newest run first."""
    details = snapshot["details"]
    runs = [r for r in snapshot["runs"] if r["id"] in details]
    header = (f'<div class="stop">databench outputs · {_plural(len(runs), "run")} · generated {_e(_when(snapshot["generated"]))}'
              f'<br>Open in a browser for search and the full-size figure viewer.</div>')
    toc = ""
    if len(runs) > 1:
        items = "".join(f'<li><a href="#r{i}">{_e(r["script"])}</a>'
                        f'{f" <span class=\"pill tag\">{_e(r["tag"])}</span>" if r.get("tag") else ""}'
                        f' <span class="nil">{_e(_when(r.get("created")))}</span></li>' for i, r in enumerate(runs))
        toc = f'<ol class="stoc">{items}</ol>'
    body = "".join(_run(details[r["id"]], snapshot, f"r{i}") for i, r in enumerate(runs))
    return f'<div id="static" class="static">{header}{toc}{body}</div>'
