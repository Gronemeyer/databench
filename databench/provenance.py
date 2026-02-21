from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

import json
import os

if TYPE_CHECKING:
    from databench.bench import Bench


def _format_provenance_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=True, default=str)
    return str(value)


def _shared_params(param_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not param_list:
        return {}
    keys = set.intersection(*(set(params.keys()) for params in param_list))
    shared: Dict[str, Any] = {}
    for key in sorted(keys):
        values = [params.get(key) for params in param_list]
        first = values[0]
        if all(v == first for v in values) and first is not None:
            shared[key] = first
    return shared


def _relative_path(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(str(to_path), start=str(from_path))


def _iter_plot_images(paths: Iterable[str]) -> List[Path]:
    allowed = {".png", ".jpg", ".jpeg", ".svg"}
    preference = {".png": 0, ".jpg": 1, ".jpeg": 2, ".svg": 3}
    chosen: Dict[str, Path] = {}
    order: List[str] = []
    for entry in paths:
        path = Path(entry)
        suffix = path.suffix.lower()
        if suffix not in allowed:
            continue
        stem = path.stem
        if stem not in chosen:
            chosen[stem] = path
            order.append(stem)
            continue
        current = chosen[stem]
        if preference.get(suffix, 99) < preference.get(current.suffix.lower(), 99):
            chosen[stem] = path
    return [chosen[stem] for stem in order]


def _append_bullets(lines: List[str], items: Iterable[str]) -> None:
    for item in items:
        lines.append(f"- {item}")
        lines.append("")


def _class_path(module: Optional[str], name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    if module:
        return f"{module}.{name}"
    return name


def _extract_plot_colors(bench: "Bench", entry: Optional[Dict[str, Any]]) -> List[str]:
    if not entry:
        return []
    kwargs = entry.get("kwargs", {})
    if isinstance(kwargs, dict):
        color = kwargs.get("color")
        if color:
            return [str(color)]
        colors = kwargs.get("colors")
        if isinstance(colors, (list, tuple)):
            return [str(c) for c in colors if c is not None]
        feature = kwargs.get("feature")
        if feature is not None:
            if hasattr(feature, "color") and getattr(feature, "color"):
                return [str(getattr(feature, "color"))]
            if isinstance(feature, dict):
                feat_color = feature.get("color")
                if feat_color:
                    return [str(feat_color)]
            if isinstance(feature, str):
                feat = bench._features.get(feature)
                if feat is not None and getattr(feat, "color", None):
                    return [str(getattr(feat, "color"))]
    return []


def _pair_plots_with_usage(
    plot_paths: List[Path],
    usage_plots: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pairs = []
    usage_idx = 0
    for plot_path in plot_paths:
        entry = usage_plots[usage_idx] if usage_idx < len(usage_plots) else None
        if usage_idx < len(usage_plots):
            usage_idx += 1
        pairs.append({"path": plot_path, "usage": entry})
    return pairs


def _feature_plot_map(entries: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapped: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        name = entry.get("feature_name")
        path = entry.get("path")
        if not name or not path:
            continue
        if name in mapped:
            continue
        mapped[name] = {
            "path": Path(path),
            "plotter_class": entry.get("plotter_class"),
            "plotter_module": entry.get("plotter_module"),
            "plotter_doc": entry.get("plotter_doc"),
            "plotter_params": entry.get("plotter_params"),
        }
    return mapped


def _collect_features(bench: "Bench") -> List[Dict[str, Any]]:
    used_feature_names = {
        name
        for entry in bench._usage.get("features", [])
        for name in entry.get("names", [])
    }
    features = []
    for feat in bench.data:
        if feat.name not in used_feature_names:
            continue
        try:
            params = asdict(feat)
        except Exception:
            params = {"name": feat.name, "label": feat.label}
        module = feat.__class__.__module__
        class_name = feat.__class__.__name__
        features.append(
            {
                "name": feat.name,
                "class": class_name,
                "class_module": module,
                "class_path": _class_path(module, class_name),
                "doc": (feat.__class__.__doc__ or "").strip() or None,
                "params": params,
            }
        )
    return features


def _collect_analyses(bench: "Bench") -> List[Dict[str, Any]]:
    return [
        {
            "name": entry.get("name"),
            "class": entry.get("class"),
            "class_module": entry.get("class_module"),
            "class_path": _class_path(entry.get("class_module"), entry.get("class")),
            "doc": entry.get("doc"),
            "params": {**entry.get("config", {}), **entry.get("kwargs", {})},
        }
        for entry in bench._usage.get("analyses", [])
    ]


def _collect_plotters(bench: "Bench") -> List[Dict[str, Any]]:
    return [
        {
            "name": entry.get("name"),
            "class": entry.get("class"),
            "class_module": entry.get("class_module"),
            "class_path": _class_path(entry.get("class_module"), entry.get("class")),
            "doc": entry.get("doc"),
            "params": {**entry.get("config", {}), **entry.get("kwargs", {})},
        }
        for entry in bench._usage.get("plots", [])
    ]


def build_provenance_payload(bench: "Bench") -> Dict[str, Any]:
    created_at = datetime.now().isoformat()
    features = _collect_features(bench)
    analyses = _collect_analyses(bench)
    plotters = _collect_plotters(bench)

    payload = {
        "created_at": created_at,
        "git_hash": bench._get_git_hash(),
        "io_config": None if bench.io_config is None else asdict(bench.io_config),
        "filter_config": None if bench.filter_config is None else asdict(bench.filter_config),
        "notes": bench._provenance_notes,
        "features": features,
        "analyses": analyses,
        "plotters": plotters,
        "usage": bench._usage,
    }
    return {
        "created_at": created_at,
        "features": features,
        "analyses": analyses,
        "plotters": plotters,
        "payload": payload,
    }


def write_provenance_summary(
    path: Path,
    bench: "Bench",
    created_at: str,
    features: List[Dict[str, Any]],
    analyses: List[Dict[str, Any]],
    plotters: List[Dict[str, Any]],
) -> None:
    lines: List[str] = ["# Provenance Summary", ""]
    lines.append(f"Created: {created_at}")
    lines.append("")
    lines.append(f"Git: {bench._get_git_hash()}")

    io_cfg = bench.io_config
    if io_cfg is not None:
        lines.append("")
        lines.append(f"Input: {_format_provenance_value(io_cfg.input_path)}")
        lines.append("")
        if io_cfg.scientist:
            lines.append(f"Scientist: {io_cfg.scientist}")
            lines.append("")
        lines.append(f"Run name: {io_cfg.run_name}")
        lines.append("")
        lines.append(f"Tag: {io_cfg.tag}")
        lines.append("")
        
    if bench.filter_config is not None:
        lines.append("")
        lines.append(f"Filters: {_format_provenance_value(asdict(bench.filter_config))}")

    if bench._provenance_notes:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("#### Notes:")
        lines.append("")
        lines.append(bench._provenance_notes)

    plot_paths = _iter_plot_images(bench._saved_outputs.get("figures", []))
    usage_plots = bench._usage.get("plots", [])
    feature_plot_map = _feature_plot_map(bench._saved_outputs.get("feature_plots", []))
    feature_plot_paths = {entry["path"] for entry in feature_plot_map.values()}

    if features:
        lines.append("")
        lines.append("<div style=\"page-break-after: always;\"></div>")
        lines.append("")
        lines.append("# Features")
        lines.append("")
        feature_names = [feat.get("name") for feat in features if feat.get("name")]
        for name in feature_names:
            feat = next((f for f in features if f.get("name") == name), None)
            if feat is None:
                continue
            class_path = feat.get("class_path") or feat.get("class")
            if feat != features[0]:
                lines.append("<div style=\"page-break-after: always;\"></div>")
                lines.append("")
            lines.append("")
            lines.append(f"## {name}")
            lines.append("")
            plot_entry = feature_plot_map.get(name)
            plot_path = plot_entry.get("path") if plot_entry else None
            if plot_path is not None:
                rel_path = _relative_path(path.parent, plot_path)
                alt_text = plot_path.stem.replace("_", " ")
                lines.append(f"![{alt_text}]({rel_path})")
                lines.append("")
            if class_path:
                lines.append(f"\t\tClass: ({class_path})")
                lines.append("")
            if feat.get("doc"):
                lines.append(f"\t\tDoc: {feat.get('doc')}")
                lines.append("")
            lines.append(f"\t\tgit: {bench._get_git_hash()}")
            lines.append("")
            params = dict(feat.get("params", {}))
            params.pop("name", None)
            ordered_keys = [
                key for key in ("label", "plotter", "color", "source", "unit") if key in params
            ]
            for key in ordered_keys + sorted(k for k in params.keys() if k not in ordered_keys):
                value = params.get(key)
                if value is None:
                    continue
                lines.append(f"\t\t{key}={_format_provenance_value(value)};")
            lines.append("")

    analysis_plot_names = [name for name in feature_plot_map.keys() if name not in feature_names]
    if analyses or analysis_plot_names:
        lines.append("")
        lines.append("## Analyses")
        lines.append("")
        if analyses:
            for analysis in analyses:
                class_path = analysis.get("class_path") or analysis.get("class")
                name = analysis.get("name")
                lines.append(f"### {name}")
                lines.append("")
                if class_path:
                    lines.append(f"\t\tClass: ({class_path})")
                    lines.append("")
                if analysis.get("doc"):
                    lines.append(f"\t\tDoc: {analysis.get('doc')}")
                    lines.append("")
                lines.append(f"\t\tgit: {bench._get_git_hash()}")
                lines.append("")
                params = dict(analysis.get("params", {}))
                params.pop("name", None)
                for key in sorted(params.keys()):
                    value = params.get(key)
                    if value is None:
                        continue
                    lines.append(f"\t\t{key}={_format_provenance_value(value)};")
                lines.append("")
        if analysis_plot_names:
            lines.append("### Analysis plots")
            lines.append("")
            for name in analysis_plot_names:
                plot_meta = feature_plot_map.get(name, {})
                class_path = _class_path(plot_meta.get("plotter_module"), plot_meta.get("plotter_class"))
                plot_doc = plot_meta.get("plotter_doc")
                plot_params = plot_meta.get("plotter_params")
                lines.append(f"#### {name}")
                lines.append("")
                plot_entry = feature_plot_map.get(name)
                plot_path = plot_entry.get("path") if plot_entry else None
                if plot_path is not None:
                    rel_path = _relative_path(path.parent, plot_path)
                    alt_text = plot_path.stem.replace("_", " ")
                    lines.append(f"![{alt_text}]({rel_path})")
                    lines.append("")
                if class_path:
                    lines.append(f"\t\tClass: ({class_path})")
                    lines.append("")
                if plot_doc:
                    lines.append(f"\t\tDoc: {plot_doc}")
                    lines.append("")
                lines.append(f"\t\tgit: {bench._get_git_hash()}")
                lines.append("")
                if plot_params:
                    params = dict(plot_params)
                    params.pop("name", None)
                    for key in sorted(params.keys()):
                        value = params.get(key)
                        if value is None:
                            continue
                        lines.append(f"\t\t{key}={_format_provenance_value(value)};")
                    lines.append("")

    remaining_plot_paths = [path for path in plot_paths if path not in feature_plot_paths]
    if remaining_plot_paths:
        lines.append("## Plots")
        lines.append("")
        summary_dir = path.parent
        for item in _pair_plots_with_usage(remaining_plot_paths, usage_plots):
            entry = item.get("usage")
            plot_path = item.get("path")
            class_path = None
            if entry:
                class_path = _class_path(entry.get("class_module"), entry.get("class"))
            if class_path:
                lines.append(f"### {class_path}")
                lines.append("")
            rel_path = _relative_path(summary_dir, plot_path)
            alt_text = plot_path.stem.replace("_", " ")
            lines.append(f"![{alt_text}]({rel_path})")
            lines.append("")
            colors = _extract_plot_colors(bench, entry)
            if colors:
                lines.append(f"Colors: {', '.join(colors)}")
                lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def save_provenance(bench: "Bench", *, output_dir: Path, name: str = "provenance.json") -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    built = build_provenance_payload(bench)
    payload = built["payload"]

    path = output_dir / name
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    tag = ""
    if bench.io_config is not None:
        tag = bench.io_config.tag or ""
    date_prefix = datetime.now().strftime("%y%m%d")
    summary_name = f"{date_prefix}_{tag}_summary.md" if tag else f"{date_prefix}_summary.md"
    summary_path = output_dir / summary_name
    write_provenance_summary(
        summary_path,
        bench,
        built["created_at"],
        built["features"],
        built["analyses"],
        built["plotters"],
    )
    return path, summary_path