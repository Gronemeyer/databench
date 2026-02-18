"""Overview plots for DataKit pickled datasets.

Usage:
  python scripts/plot/overview.py --dataset path/to/dataset.pkl --output plots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional, Any, cast

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Global configuration (defaults used by CLI and plot styling)
# -----------------------------------------------------------------------------
DATASET = r"/Volumes/ake.bin/Projects/HFSA/260129_HFSA-full.pkl"
DEFAULT_GAP_SECONDS = 2.0
DEFAULT_FIGSIZE = (16, 10)
DEFAULT_DPI = 150
DEFAULT_COLORMAP = "tab10"
DEFAULT_LINE_WIDTH = 0.5
DEFAULT_LINE_ALPHA = 0.7
DEFAULT_SPAN_ALPHA = 0.2
DEFAULT_LABEL_Y = 1.15
DEFAULT_TRIM_START_S = 0.5
DEFAULT_TRIM_END_S = 15.0
DEFAULT_USE_MASTER_TIME = True
DEFAULT_OUTPUT_FORMAT = "png"

# Outlier removal + smoothing (compact replica of explorer.py behavior)
OUTLIER_PUPIL_IQR_K = 1.5
OUTLIER_SPEED_IQR_K = 2.0
SMOOTH_PUPIL_METHOD = "savgol"
SMOOTH_PUPIL_WINDOW = 15
SMOOTH_PUPIL_POLYORDER = 3
SMOOTH_SPEED_METHOD = "savgol"
SMOOTH_SPEED_WINDOW = 7
SMOOTH_SPEED_POLYORDER = 2

# Treadmill-style speed smoothing (mirrors TreadmillSourceV2)
TMILL_GAP_THRESHOLD_S = 0.5
TMILL_INTERPOLATE_HZ = 20
TMILL_SMOOTH_WINDOW = 5
TMILL_SMOOTH_POLYORDER = 2
TMILL_MEDIAN_SIZE = 3
TMILL_MIN_POINTS = 10
TMILL_ZERO_PAD_OFFSET_S = 0.05
# Optional exclusion list for known-corrupt traces to avoid scaling skew.
# Example:
EXCLUDE_TRACES = (
    {"subject": "STREHAB14", "session": "ses-01", "source": "meso_mean", "feature": "dF_F"},
    {"subject": "STREHAB07", "session": "ses-11", "source": "meso_mean", "feature": "dF_F"},
    {"subject": "STREHAB07", "session": "ses-11", "source": "pupil", "feature": "diameter_mm"},
    {"subject": "STREHAB07", "session": "ses-11", "source": "treadmill", "feature": "speed_mm"},
    {"subject": "STREHAB07", "session": "ses-11", "source": "treadmill", "feature": "distance_mm"},

)
#EXCLUDE_TRACES: tuple[dict[str, str], ...] = ()


TIME_FEATURE_PRIORITY = (
    "time_elapsed_s",
    "master_elapsed_s",
    "queue_elapsed",
    "time_s",
    "timestamp",
    "time",
)

MASTER_TIME_SOURCES = ("time", "meso_mean", "treadmill", "pupil")
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


def _ensure_multiindex_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame


def _as_numeric_array(value: object) -> Any:
    if isinstance(value, pd.Series):
        value = value.to_numpy()
    arr = np.asarray(value)
    return arr.ravel()


def _normalize_time(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return arr - float(arr[0])


def _remove_outliers_iqr(data: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    if data.ndim > 1:
        data = data[:, 0]
    valid_mask = ~np.isnan(data)
    valid = data[valid_mask]
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    inlier_mask = (data >= lower) & (data <= upper)
    return data, valid_mask & inlier_mask


def _smooth_data(
    data: np.ndarray, *, method: str, window_length: int, polyorder: int
) -> Any:
    win = min(window_length, len(data))
    if win % 2 == 0:
        win -= 1
    if method == "median":
        from scipy.ndimage import median_filter

        return median_filter(data, size=win)
    if method == "rolling_mean":
        return pd.Series(data).rolling(window=win, center=True, min_periods=1).mean().to_numpy()
    if method == "savgol":
        if win >= polyorder + 2:
            from scipy import signal

            return signal.savgol_filter(data, win, polyorder)
    return data


def _smooth_speed_trace(ts_s: np.ndarray, speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import savgol_filter
    from scipy.ndimage import median_filter

    ts_s = np.asarray(ts_s, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)
    order = np.argsort(ts_s)
    ts_s = ts_s[order]
    speed = speed[order]

    speed_filtered = speed.copy()
    if len(speed_filtered) > TMILL_SMOOTH_WINDOW:
        win = TMILL_SMOOTH_WINDOW
        if win % 2 == 0:
            win = max(3, win - 1)
        win = min(win, len(speed_filtered) if len(speed_filtered) % 2 == 1 else len(speed_filtered) - 1)
        if win >= 3:
            speed_filtered = median_filter(speed_filtered, size=TMILL_MEDIAN_SIZE)
            if win >= TMILL_SMOOTH_POLYORDER + 1:
                speed_filtered = savgol_filter(speed_filtered, win, TMILL_SMOOTH_POLYORDER)

    session_duration = ts_s[-1] - ts_s[0]
    n_points = int(session_duration * TMILL_INTERPOLATE_HZ)
    if n_points < TMILL_MIN_POINTS:
        n_points = len(ts_s)
    time_grid = np.linspace(ts_s[0], ts_s[-1], n_points)
    speed_interp = np.interp(time_grid, ts_s, speed_filtered)  # type: ignore[arg-type]

    for i, t in enumerate(time_grid):
        distances = np.abs(ts_s - t)
        if distances.size and np.min(distances) > TMILL_GAP_THRESHOLD_S:
            speed_interp[i] = 0.0

    final_times = []
    final_speeds = []
    for i in range(len(time_grid)):
        final_times.append(time_grid[i])
        final_speeds.append(speed_interp[i])
        if i < len(time_grid) - 1:
            gap = time_grid[i + 1] - time_grid[i]
            if gap > 2 / TMILL_INTERPOLATE_HZ:
                final_times.append(time_grid[i] + TMILL_ZERO_PAD_OFFSET_S)
                final_speeds.append(0.0)
                final_times.append(time_grid[i + 1] - TMILL_ZERO_PAD_OFFSET_S)
                final_speeds.append(0.0)

    return np.asarray(final_times), np.asarray(final_speeds)


def _process_signal(data: np.ndarray, var_name: str) -> tuple[np.ndarray, np.ndarray]:
    if data.ndim > 1:
        data = data[:, 0]
    processed = data.copy()
    valid_mask = ~np.isnan(processed)
    name = var_name.lower()
    if "speed" in name:
        processed, out_mask = _remove_outliers_iqr(processed, OUTLIER_SPEED_IQR_K)
        valid_mask &= out_mask
        smoothed = _smooth_data(
            processed[valid_mask],
            method=SMOOTH_SPEED_METHOD,
            window_length=SMOOTH_SPEED_WINDOW,
            polyorder=SMOOTH_SPEED_POLYORDER,
        )
        processed[valid_mask] = smoothed
    if "pupil" in name or "diameter" in name:
        processed, out_mask = _remove_outliers_iqr(processed, OUTLIER_PUPIL_IQR_K)
        valid_mask &= out_mask
        smoothed = _smooth_data(
            processed[valid_mask],
            method=SMOOTH_PUPIL_METHOD,
            window_length=SMOOTH_PUPIL_WINDOW,
            polyorder=SMOOTH_PUPIL_POLYORDER,
        )
        processed[valid_mask] = smoothed
    return processed, valid_mask


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
    selected = _pick_feature(features, TIME_FEATURE_PRIORITY)
    if selected:
        return selected
    return None


def _first_numeric_array(series: pd.Series, max_scan: int = 10) -> Any:
    for value in series.iloc[:max_scan]:
        arr = _as_numeric_array(value)
        return arr
    return _as_numeric_array(series.iloc[0])


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
    max_val = np.nanmax(fallback)
    return float(max_val)


def _get_master_time(row: pd.Series, time_feature_map: dict[str, str]) -> Any:
    time_feature = time_feature_map["time"]
    time_value = row.get(("time", time_feature))
    master = _as_numeric_array(time_value)
    return _normalize_time(master)


def _sync_values_to_time(
    *,
    values: Any,
    source_time: Any,
    master_time: Any,
    use_master_time: bool,
    pad_zero_edges: bool,
) -> tuple[Any, Any]:
    time_arr = np.asarray(source_time, dtype=np.float64)
    value_arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(time_arr)
    time_arr = time_arr[order]
    value_arr = value_arr[order]

    synced = np.interp(master_time, time_arr, value_arr)
    left = master_time < time_arr[0]
    right = master_time > time_arr[-1]
    synced[left | right] = 0.0
    return master_time, synced


def _index_value(idx: object, names: Iterable[str], key: str) -> Any:
    idx = cast(tuple, idx)
    names_list = list(names)
    value = idx[names_list.index(key)]
    return None if value is None else str(value)


def _session_day_label(idx: object, names: Iterable[str]) -> str:
    session = _index_value(idx, names, "Session")
    match = re.search(r"(\d+)", str(session))
    return f"Day {int(match.group(1))}"  # type: ignore[union-attr]


def _is_excluded_trace(subject: str, idx: object, names: Iterable[str], var: dict[str, str]) -> bool:
    return False


def _is_speed_trace(var: dict[str, str]) -> bool:
    label = var.get("label", "").lower()
    feature = var.get("feature", "").lower()
    return "speed" in label or "speed" in feature


def _subject_level(frame: pd.DataFrame) -> int:
    names = list(frame.index.names)
    return names.index("Subject")


def _plot_overview_for_frame(
    frame: pd.DataFrame,
    *,
    variables: list[dict[str, str]],
    time_feature_map: dict[str, str],
    subject_label: str,
    subject_value: str,
    gap_seconds: float,
    figsize: tuple[int, int],
    colormap: str,
    line_width: float,
    line_alpha: float,
    span_alpha: float,
    label_y: float,
    trim_start_s: float,
    trim_end_s: float,
    use_master_time: bool,
) -> Any:
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

    colors = cm.get_cmap(colormap)(np.linspace(0, 1, 10))

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
                "color": colors[idx % 10],
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
            data = _as_numeric_array(row.get((var["source"], var["feature"])))

            time_feature = time_feature_map.get(var["source"])
            time_value = row.get((var["source"], time_feature)) if time_feature else None
            time_data = _as_numeric_array(time_value)
            time_data = _normalize_time(time_data)

            if var["source"] == "treadmill" and _is_speed_trace(var):
                smooth_times, smooth_values = _smooth_speed_trace(time_data, data)
                time_data = smooth_times
                data = smooth_values

            times, values = _sync_values_to_time(
                values=data,
                source_time=time_data,
                master_time=session["master_time"],
                use_master_time=use_master_time,
                pad_zero_edges=_is_speed_trace(var),
            )

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

            color = colors[sess_index % 10]
            ax.plot(
                times[valid_mask],
                values[valid_mask],
                linewidth=line_width,
                alpha=line_alpha,
                color=color,
            )
            finite_vals = values[valid_mask]
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
        y_min_val = cast(float, y_min)
        y_max_val = cast(float, y_max)
        padding = 0.05 * (y_max_val - y_min_val)
        ax.set_ylim(y_min_val - padding, y_max_val + padding)

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


def plot_subject_overviews(
    dataset: pd.DataFrame,
    *,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    colormap: str = DEFAULT_COLORMAP,
    line_width: float = DEFAULT_LINE_WIDTH,
    line_alpha: float = DEFAULT_LINE_ALPHA,
    span_alpha: float = DEFAULT_SPAN_ALPHA,
    label_y: float = DEFAULT_LABEL_Y,
    trim_start_s: float = DEFAULT_TRIM_START_S,
    trim_end_s: float = DEFAULT_TRIM_END_S,
    use_master_time: bool = DEFAULT_USE_MASTER_TIME,
) -> dict[str, Any]:
    frame = _ensure_multiindex_columns(dataset)
    frame = frame.sort_index()

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
            colormap=colormap,
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


def _parse_figsize(value: str) -> tuple[int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("figsize must be in the form 'width,height'")
    return int(parts[0]), int(parts[1])


def _load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return cast(pd.DataFrame, pd.read_hdf(path))
    return pd.read_pickle(path)


def _resolve_output_path(base: Path, subject: str, fmt: str) -> Path:
    if base.suffix:
        if "{subject}" in base.name:
            filename = base.name.format(subject=subject)
            return base.with_name(filename)
        stem = base.stem
        return base.with_name(f"{stem}_{subject}{base.suffix}")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"overview_{subject}.{fmt}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-subject overviews from a pickled dataset.")
    parser.add_argument("--dataset", type=Path, default=DATASET, help="Path to the pickled dataset (.pkl)")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path (directory or file template, e.g. overview_{subject}.png)",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open the plot window")
    parser.add_argument(
        "--gap",
        type=float,
        default=DEFAULT_GAP_SECONDS,
        help=f"Gap between sessions in seconds (default: {DEFAULT_GAP_SECONDS})",
    )
    parser.add_argument(
        "--figsize",
        type=_parse_figsize,
        default=f"{DEFAULT_FIGSIZE[0]},{DEFAULT_FIGSIZE[1]}",
        help="Figure size 'width,height'",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"Output DPI (default: {DEFAULT_DPI})")
    parser.add_argument(
        "--colormap",
        type=str,
        default=DEFAULT_COLORMAP,
        help=f"Matplotlib colormap name (default: {DEFAULT_COLORMAP})",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=DEFAULT_LINE_WIDTH,
        help=f"Line width (default: {DEFAULT_LINE_WIDTH})",
    )
    parser.add_argument(
        "--line-alpha",
        type=float,
        default=DEFAULT_LINE_ALPHA,
        help=f"Line alpha (default: {DEFAULT_LINE_ALPHA})",
    )
    parser.add_argument(
        "--span-alpha",
        type=float,
        default=DEFAULT_SPAN_ALPHA,
        help=f"Session span alpha (default: {DEFAULT_SPAN_ALPHA})",
    )
    parser.add_argument(
        "--label-y",
        type=float,
        default=DEFAULT_LABEL_Y,
        help=f"Session label y-position (default: {DEFAULT_LABEL_Y})",
    )
    parser.add_argument(
        "--trim-start",
        type=float,
        default=DEFAULT_TRIM_START_S,
        help=f"Trim first N seconds of each session (default: {DEFAULT_TRIM_START_S})",
    )
    parser.add_argument(
        "--trim-end",
        type=float,
        default=DEFAULT_TRIM_END_S,
        help=f"Trim last N seconds of each session (default: {DEFAULT_TRIM_END_S})",
    )
    parser.add_argument(
        "--format",
        choices=("png", "svg", "pdf"),
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output format when --output is a directory (default: {DEFAULT_OUTPUT_FORMAT})",
    )
    parser.add_argument(
        "--use-master-time",
        default=DEFAULT_USE_MASTER_TIME,
        action=argparse.BooleanOptionalAction,
        help="Align traces to dataqueue master time when available",
    )
    args = parser.parse_args()

    dataset = _load_dataset(args.dataset)
    figures = plot_subject_overviews(
        dataset,
        gap_seconds=args.gap,
        figsize=args.figsize,
        colormap=args.colormap,
        line_width=args.line_width,
        line_alpha=args.line_alpha,
        span_alpha=args.span_alpha,
        label_y=args.label_y,
        trim_start_s=args.trim_start,
        trim_end_s=args.trim_end,
        use_master_time=args.use_master_time,
    )

    if args.output:
        for subject, fig in figures.items():
            path = _resolve_output_path(args.output, subject, args.format)
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
