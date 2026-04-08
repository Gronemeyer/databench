"""Overview plots for DataKit pickled datasets.

Public API
----------
plot_subject_overviews   Generate per-subject multi-variable overview figures.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench._utils import as_1d
from databench.analysis._signal.preproc import remove_outliers_iqr, smooth_dense
from databench.analysis._signal.remap import remap_to_timebase


# ── Layout defaults ────────────────────────────────────────────────────────
DEFAULT_GAP_SECONDS: float = 2.0
DEFAULT_FIGSIZE: tuple[int, int] = (16, 10)
DEFAULT_LINE_WIDTH: float = 0.5
DEFAULT_LINE_ALPHA: float = 0.7
DEFAULT_SPAN_ALPHA: float = 0.2
DEFAULT_LABEL_Y: float = 1.15
DEFAULT_TRIM_START_S: float = 0.5
DEFAULT_TRIM_END_S: float = 15.0
DEFAULT_USE_MASTER_TIME: bool = True


# ── Data discovery ─────────────────────────────────────────────────────────
TIME_FEATURE_PRIORITY = (
    "time_elapsed_s",
    "master_elapsed_s",
    "queue_elapsed",
    "time_s",
    "timestamp",
    "time",
)

MASTER_TIME_FEATURES = ("master_elapsed_s", "queue_elapsed", "time_elapsed_s", "time")

PREFERRED_VARIABLES = (
    {
        "label": "Meso dF/F",
        "source": "meso_mean",
        "features": ("dF_F", "Mean", "mean"),
    },
    {
        "label": "Pupil diameter",
        "source": "pupil",
        "features": ("pupil_diameter_mm", "diameter_mm", "pupil_diameter", "diameter"),
    },
    {
        "label": "Treadmill speed",
        "source": "treadmill",
        "features": ("speed_mm", "speed_mm_s", "speed"),
    },
    {
        "label": "Treadmill distance",
        "source": "treadmill",
        "features": ("distance_mm", "distance"),
    },
)


# ── Internal helpers ───────────────────────────────────────────────────────

def _normalize_time(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return arr - float(arr[0])


def _feature_map(frame: pd.DataFrame) -> dict[str, set[str]]:
    sources: dict[str, set[str]] = {}
    for source, feature in frame.columns:
        sources.setdefault(str(source), set()).add(str(feature))
    return sources


def _pick_feature(features: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    feature_set = {str(item) for item in features}
    for candidate in candidates:
        if candidate in feature_set:
            return candidate
    lower_map = {item.lower(): item for item in feature_set}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match:
            return match
    return None


def _pick_time_feature(features: Iterable[str]) -> Optional[str]:
    return _pick_feature(features, TIME_FEATURE_PRIORITY)


def _discover_variables(frame: pd.DataFrame) -> list[dict[str, str]]:
    features_by_source = _feature_map(frame)
    variables: list[dict[str, str]] = []
    for spec in PREFERRED_VARIABLES:
        source = spec["source"]
        feature = _pick_feature(features_by_source.get(source, []), spec["features"])
        if feature:
            variables.append({"source": source, "feature": feature, "label": spec["label"]})
    return variables


def _build_time_feature_map(frame: pd.DataFrame) -> dict[str, str]:
    features_by_source = _feature_map(frame)
    time_features: dict[str, str] = {}
    for source, features in features_by_source.items():
        time_feature = _pick_time_feature(features)
        if time_feature:
            time_features[source] = time_feature
    return time_features


def _estimate_duration(row: pd.Series, fallback: Any) -> float:
    return float(np.nanmax(fallback))


def _get_master_time(row: pd.Series, time_feature_map: dict[str, str]) -> Any:
    time_feature = time_feature_map["time"]
    time_value = row.get(("time", time_feature))
    master = as_1d(time_value)
    return _normalize_time(master)


def _process_signal(data: np.ndarray, var_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Outlier-remove and smooth a trace based on its variable name."""
    if data.ndim > 1:
        data = data[:, 0]
    processed = data.copy()
    valid_mask = ~np.isnan(processed)
    name = var_name.lower()
    if "speed" in name:
        processed, out_mask = remove_outliers_iqr(processed, k=2.0)
        valid_mask &= out_mask
        processed[valid_mask] = smooth_dense(
            processed[valid_mask], median_size=1, window=7, polyorder=2,
        )
    if "pupil" in name or "diameter" in name:
        processed, out_mask = remove_outliers_iqr(processed, k=1.5)
        valid_mask &= out_mask
        processed[valid_mask] = smooth_dense(
            processed[valid_mask], median_size=1, window=15, polyorder=3,
        )
    return processed, valid_mask


def _index_value(idx: object, names: Iterable[str], key: str) -> Any:
    idx = cast(tuple, idx)
    names_list = list(names)
    value = idx[names_list.index(key)]
    return None if value is None else str(value)


def _session_day_label(idx: object, names: Iterable[str]) -> str:
    session = _index_value(idx, names, "Session")
    match = re.search(r"(\d+)", str(session))
    return f"Day {int(match.group(1))}"  # type: ignore[union-attr]


def _is_speed_trace(var: dict[str, str]) -> bool:
    label = var.get("label", "").lower()
    feature = var.get("feature", "").lower()
    return "speed" in label or "speed" in feature


def _subject_level(frame: pd.DataFrame) -> int:
    return list(frame.index.names).index("Subject")


# ── Core plotting ──────────────────────────────────────────────────────────

def _plot_overview_for_frame(
    frame: pd.DataFrame,
    *,
    variables: list[dict[str, str]],
    time_feature_map: dict[str, str],
    subject_label: str,
    subject_value: str,
    gap_seconds: float,
    figsize: tuple[int, int],
    line_width: float,
    line_alpha: float,
    span_alpha: float,
    label_y: float,
    trim_start_s: float,
    trim_end_s: float,
    use_master_time: bool,
) -> Any:
    from databench.plotting import get_theme
    index_names = frame.index.names

    sessions = []
    for idx, row in frame.iterrows():
        label = _session_day_label(idx if isinstance(idx, tuple) else (idx,), cast(Iterable[str], index_names))
        master_time = _get_master_time(row, time_feature_map)
        duration = _estimate_duration(row, master_time)
        effective_duration = duration - max(trim_start_s, 0.0) - max(trim_end_s, 0.0)
        sessions.append(
            {
                "index": idx,
                "label": label,
                "row": row,
                "master_time": master_time,
                "duration": effective_duration,
            }
        )

    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
    if n_vars == 1:
        axes = [axes]

    # Use theme colors cycled over sessions
    theme_colors = get_theme().colors
    n_colors = len(theme_colors)

    session_boundaries = []
    total_duration = 0.0
    cumulative = 0.0
    for idx, session in enumerate(sessions):
        start_time = cumulative
        end_time = cumulative + session["duration"]
        session_boundaries.append(
            {
                "label": session["label"],
                "start": start_time,
                "end": end_time,
                "color": theme_colors[idx % n_colors],
            }
        )
        cumulative += session["duration"] + gap_seconds
        total_duration = cumulative - gap_seconds if cumulative > gap_seconds else cumulative

    for var_index, var in enumerate(variables):
        ax = axes[var_index]
        cumulative = 0.0
        y_min = None
        y_max = None
        for sess_index, session in enumerate(sessions):
            row = session["row"]
            data = as_1d(row.get((var["source"], var["feature"])))

            time_feature = time_feature_map.get(var["source"])
            time_value = row.get((var["source"], time_feature)) if time_feature else None
            time_data = _normalize_time(as_1d(time_value))

            # Align source values onto master timebase
            values = remap_to_timebase(
                session["master_time"], time_data, data, method="linear",
            )
            times = session["master_time"]

            if trim_start_s > 0:
                keep = times >= trim_start_s
                times = times[keep]
                values = values[keep]

            if trim_end_s > 0:
                end_limit = np.nanmax(times) - trim_end_s
                keep = times <= end_limit
                times = times[keep]
                values = values[keep]

            if var["source"] == "treadmill" and _is_speed_trace(var):
                valid_mask = np.isfinite(values)
            else:
                values, valid_mask = _process_signal(
                    values, f"{var['source']}:{var['feature']}:{var['label']}"
                )
            times = times + cumulative
            valid_mask = valid_mask & np.isfinite(values)

            color = theme_colors[sess_index % n_colors]
            ax.plot(
                times[valid_mask],
                values[valid_mask],
                linewidth=line_width,
                alpha=line_alpha,
                color=color,
            )
            finite_vals = values[valid_mask]
            if len(finite_vals) > 0:
                vmin = float(np.nanmin(finite_vals))
                vmax = float(np.nanmax(finite_vals))
                y_min = vmin if y_min is None else min(y_min, vmin)
                y_max = vmax if y_max is None else max(y_max, vmax)
            ax.axvspan(
                cumulative,
                cumulative + session["duration"],
                alpha=span_alpha,
                color=color,
            )

            cumulative += session["duration"] + gap_seconds

        ax.set_ylabel(var["label"])
        ax.set_title(var["label"])
        ax.grid(True, alpha=0.3)
        if y_min is not None and y_max is not None:
            padding = 0.05 * (y_max - y_min)
            ax.set_ylim(y_min - padding, y_max + padding)

    ax_top = axes[0]
    y_position = label_y
    for boundary in session_boundaries:
        mid_time = (boundary["start"] + boundary["end"]) / 2
        ax_top.text(
            mid_time,
            y_position,
            boundary["label"],
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=boundary["color"], alpha=0.3),
            transform=ax_top.get_xaxis_transform(),
        )

    for ax in axes:
        ax.set_xlim(0, total_duration)
        ax.set_xticks([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    fig.suptitle(f"{subject_label} overview: {len(sessions)} sessions, {len(variables)} variables")
    fig.tight_layout()
    return fig


# ── Public API ─────────────────────────────────────────────────────────────

def plot_subject_overviews(
    dataset: pd.DataFrame,
    *,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    line_width: float = DEFAULT_LINE_WIDTH,
    line_alpha: float = DEFAULT_LINE_ALPHA,
    span_alpha: float = DEFAULT_SPAN_ALPHA,
    label_y: float = DEFAULT_LABEL_Y,
    trim_start_s: float = DEFAULT_TRIM_START_S,
    trim_end_s: float = DEFAULT_TRIM_END_S,
    use_master_time: bool = DEFAULT_USE_MASTER_TIME,
) -> dict[str, Any]:
    """Generate per-subject multi-variable overview figures.

    Parameters
    ----------
    dataset : pd.DataFrame
        Multi-index DataFrame with (Subject, Session, ...) index and
        (source, feature) column tuples.
    gap_seconds : float
        Visual gap between concatenated sessions.
    figsize : (int, int)
        Figure size.
    line_width, line_alpha : float
        Trace rendering parameters.
    span_alpha : float
        Background session span alpha.
    label_y : float
        Y-position for session labels.
    trim_start_s, trim_end_s : float
        Trim first/last N seconds of each session.
    use_master_time : bool
        Align traces to master time.

    Returns
    -------
    dict[str, Figure]
        Mapping from subject name to matplotlib Figure.
    """
    frame = dataset.sort_index()

    variables = _discover_variables(frame)
    time_feature_map = _build_time_feature_map(frame)
    subject_level = _subject_level(frame)
    subject_name = frame.index.names[subject_level] or "Subject"
    subject_values = frame.index.get_level_values(subject_level).unique()

    figures: dict[str, Any] = {}
    for subject in subject_values:
        subject_frame = cast(pd.DataFrame, frame.xs(subject, level=subject_level, drop_level=True))
        fig = _plot_overview_for_frame(
            subject_frame,
            variables=variables,
            time_feature_map=time_feature_map,
            subject_label=f"{subject_name} {subject}",
            subject_value=str(subject),
            gap_seconds=gap_seconds,
            figsize=figsize,
            line_width=line_width,
            line_alpha=line_alpha,
            span_alpha=span_alpha,
            label_y=label_y,
            trim_start_s=trim_start_s,
            trim_end_s=trim_end_s,
            use_master_time=use_master_time,
        )
        if fig is not None:
            figures[str(subject)] = fig

    return figures
