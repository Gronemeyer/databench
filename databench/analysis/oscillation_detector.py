from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from databench.analysis.base import AnalysisFn, Analysis, AnalysisResult
from databench.registry import register_analysis
from databench.utils import as_1d, get_first, strip_prefix

@dataclass(frozen=True)
class OscillationDetectorConfig:
    fs: float = 50.0
    band: Tuple[float, float] = (3.1, 4.3)
    order: int = 4
    k: float = 4.0
    min_duration_s: float = 0.5
    merge_gap_s: float = 0.25
    signal_candidates: Tuple[str, ...] = ("roi", "signal", "value", "trace", "y")
    time_candidates: Tuple[str, ...] = (
        "sample",
        "samples",
        "frame",
        "frames",
        "t",
        "time",
        "timestamp",
    )


def _resolve_signal(
    row: pd.Series,
    source: Optional[str],
    signal_key: Optional[str],
    candidates: Sequence[str],
):
    if signal_key:
        key = (source, signal_key) if source and isinstance(row.index, pd.MultiIndex) else signal_key
        return row.get(key)

    keys = [(source, c) for c in candidates] if source and isinstance(row.index, pd.MultiIndex) else list(candidates)
    val = get_first(row, keys)
    if val is not None:
        return val

    if not isinstance(row.index, pd.MultiIndex):
        numeric_cols = [c for c in row.index if pd.api.types.is_numeric_dtype(row[c])]
        if numeric_cols:
            return row[numeric_cols[0]]
    return None


def _resolve_time(
    row: pd.Series,
    source: Optional[str],
    time_key: Optional[str],
    candidates: Sequence[str],
):
    if time_key:
        key = (source, time_key) if source and isinstance(row.index, pd.MultiIndex) else time_key
        return row.get(key)

    keys = [(source, c) for c in candidates] if source and isinstance(row.index, pd.MultiIndex) else list(candidates)
    return get_first(row, keys)


def _as_time_vector(tt: Optional[np.ndarray], n: int, fs: float) -> np.ndarray:
    if tt is None or n == 0:
        return np.arange(n) / fs

    tt = np.asarray(tt, dtype=float).ravel()
    if tt.size < n:
        n = tt.size
    tt = tt[:n]

    # If max close to n, likely sample indices
    if np.nanmax(tt) >= (0.9 * n):
        return tt / fs

    # Heuristic: very large max implies timestamps (already in seconds)
    if np.nanmax(tt) > 10 * n / fs:
        return tt

    return tt


def _bandpass_env(x: np.ndarray, fs: float, band: Tuple[float, float], order: int) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    sos = signal.butter(order, [band[0] / nyq, band[1] / nyq], btype="band", output="sos")
    xf = signal.sosfiltfilt(sos, x)
    env = np.abs(np.asarray(signal.hilbert(xf)))
    return xf, env


def _robust_threshold(env: np.ndarray, k: float) -> Tuple[float, float, float]:
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med)))
    robust_std = 1.4826 * mad
    return med + k * robust_std, med, robust_std


def _segments_from_mask(mask: np.ndarray) -> list[Tuple[int, int]]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    segments = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segments.append((start, prev))
            start = i
            prev = i
    segments.append((start, prev))
    return segments


def _merge_gaps(segments: Iterable[Tuple[int, int]], min_gap: int) -> list[Tuple[int, int]]:
    merged: list[Tuple[int, int]] = []
    for s, e in segments:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s - pe - 1 <= min_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def _apply_min_duration(segments: Iterable[Tuple[int, int]], min_len: int) -> list[Tuple[int, int]]:
    return [(s, e) for s, e in segments if (e - s + 1) >= min_len]


def _burst_table(
    t: np.ndarray,
    env: np.ndarray,
    bursts: Sequence[Tuple[int, int]],
    fs: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_s": [float(t[s]) for s, _ in bursts],
            "end_s": [float(t[e]) for _, e in bursts],
            "duration_s": [(e - s + 1) / fs for s, e in bursts],
            "peak_env": [float(env[s : e + 1].max()) for s, e in bursts],
        }
    )


@dataclass(frozen=True)
class OscillationContext:
    subject: Optional[str] = None
    session: Optional[str] = None
    task: Optional[str] = None
    source: Optional[str] = None
    signal_key: Optional[str] = None
    time_key: Optional[str] = None

    def label(self) -> str:
        parts = []
        if self.subject is not None:
            parts.append(f"Subject={self.subject}")
        if self.session is not None:
            parts.append(f"Session={self.session}")
        if self.task is not None:
            parts.append(f"Task={self.task}")
        return " | ".join(parts) if parts else "Index=unknown"

    def slug(self) -> str:
        parts = []
        if self.subject is not None:
            parts.append(f"sub-{strip_prefix(self.subject, 'sub-')}")
        if self.session is not None:
            parts.append(f"ses-{strip_prefix(self.session, 'ses-')}")
        if self.task is not None:
            parts.append(f"task-{strip_prefix(self.task, 'task-')}")
        return "_".join(parts) if parts else "row"


@dataclass(frozen=True)
class OscillationResult:
    context: OscillationContext
    band: Tuple[float, float]
    t: np.ndarray
    x: np.ndarray
    xf: np.ndarray
    env: np.ndarray
    thr: float
    med: float
    robust_std: float
    bursts: list[Tuple[int, int]]
    table: pd.DataFrame


@dataclass(frozen=True)
class OscillationDetector(AnalysisFn):
    name: str = "oscillation_detector"

    def _run_impl(
        self,
        row: pd.Series,
        cfg: OscillationDetectorConfig,
        source: Optional[str] = None,
        signal_key: Optional[str] = None,
        time_key: Optional[str] = None,
        debug: bool = False,
        context: Optional[str] = None,
    ):
        raw = _resolve_signal(row, source, signal_key, cfg.signal_candidates)
        x = as_1d(raw)
        if x is None or x.size == 0:
            if debug and context:
                print(f"[oscillation_detector] No signal | {context}")
            return {}

        time_raw = _resolve_time(row, source, time_key, cfg.time_candidates)
        t = _as_time_vector(as_1d(time_raw), len(x), cfg.fs)
        n = min(len(t), len(x))
        x = x[:n]
        t = t[:n]

        xf, env = _bandpass_env(x, cfg.fs, cfg.band, cfg.order)
        thr, med, robust_std = _robust_threshold(env, cfg.k)

        segments = _segments_from_mask(env > thr)
        min_gap = int(round(cfg.merge_gap_s * cfg.fs))
        merged = _merge_gaps(segments, min_gap)
        min_len = int(round(cfg.min_duration_s * cfg.fs))
        bursts = _apply_min_duration(merged, min_len)

        table = _burst_table(t, env, bursts, cfg.fs)

        if debug and context:
            print(f"[oscillation_detector] bursts={len(bursts)} | {context}")

        return {
            "t": t,
            "x": x,
            "xf": xf,
            "env": env,
            "thr": thr,
            "med": med,
            "robust_std": robust_std,
            "bursts": bursts,
            "table": table,
        }


def context_from_index(
    index,
    source: Optional[str] = None,
    signal_key: Optional[str] = None,
    time_key: Optional[str] = None,
) -> OscillationContext:
    if isinstance(index, tuple) and len(index) >= 3:
        subject, session, task = index[:3]
    else:
        subject = session = task = None
    return OscillationContext(
        subject=subject,
        session=session,
        task=task,
        source=source,
        signal_key=signal_key,
        time_key=time_key,
    )


def analyze_oscillation_row(
    row: pd.Series,
    cfg: OscillationDetectorConfig,
    *,
    source: Optional[str] = None,
    signal_key: Optional[str] = None,
    time_key: Optional[str] = None,
    debug: bool = False,
    context: Optional[OscillationContext] = None,
) -> Optional[OscillationResult]:
    ctx = context or OscillationContext(source=source, signal_key=signal_key, time_key=time_key)
    out = OscillationDetector().run(
        row,
        cfg=cfg,
        source=source,
        signal_key=signal_key,
        time_key=time_key,
        debug=debug,
        context=ctx.label(),
    )
    if not out:
        return None
    return OscillationResult(context=ctx, band=cfg.band, **out)


def analyze_oscillation_dataset(
    df: pd.DataFrame,
    cfg: OscillationDetectorConfig,
    *,
    source: Optional[str] = None,
    signal_key: Optional[str] = None,
    time_key: Optional[str] = None,
    debug: bool = False,
) -> list[OscillationResult]:
    results: list[OscillationResult] = []
    for idx, row in df.iterrows():
        ctx = context_from_index(idx, source=source, signal_key=signal_key, time_key=time_key)
        out = analyze_oscillation_row(
            row,
            cfg,
            source=source,
            signal_key=signal_key,
            time_key=time_key,
            debug=debug,
            context=ctx,
        )
        if out is not None:
            results.append(out)
    return results


def save_oscillation_bursts(
    result: OscillationResult,
    out_dir: Path,
    filename: Optional[str] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_key = result.context.signal_key or "signal"
    name = filename or f"{result.context.slug()}_{signal_key}_bursts_table.csv"
    path = out_dir / name
    result.table.to_csv(path, index=False)
    return path


def save_oscillation_plot(
    result: OscillationResult,
    out_dir: Path,
    *,
    overlay: Optional[Tuple[np.ndarray, str]] = None,
    overlay_subplot: bool = False,
    time_window: Optional[Tuple[float, float]] = None,
    filename: Optional[str] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_key = result.context.signal_key or "signal"
    name = filename or f"{result.context.slug()}_{signal_key}_bursts_overlay.png"
    title = f"{result.context.label()} | Signal={signal_key}" if result.context else None
    fig, _ = plot_oscillation_overlay(
        result.t,
        result.x,
        result.xf,
        result.env,
        result.thr,
        result.bursts,
        signal_key,
        band=result.band,
        overlay=overlay,
        overlay_subplot=overlay_subplot,
        window=time_window,
        title=title,
    )
    path = out_dir / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


@register_analysis
@dataclass(frozen=True)
class OscillationDetectorAnalysis(Analysis):
    name: str = "oscillation_detector"

    def run(
        self,
        row: pd.Series,
        cfg: OscillationDetectorConfig,
        *,
        source: Optional[str] = None,
        signal_key: Optional[str] = None,
        time_key: Optional[str] = None,
        debug: bool = False,
        context: Optional[OscillationContext] = None,
    ) -> AnalysisResult:
        result = analyze_oscillation_row(
            row,
            cfg,
            source=source,
            signal_key=signal_key,
            time_key=time_key,
            debug=debug,
            context=context,
        )
        return AnalysisResult(
            name=self.name,
            data=result,
            table=None if result is None else result.table,
            context=None if result is None else result.context,
            meta={"band": cfg.band},
        )

    def plot(
        self,
        result: AnalysisResult,
        *,
        overlay: Optional[Tuple[np.ndarray, str]] = None,
        overlay_subplot: bool = False,
        time_window: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
    ):
        if result.data is None:
            return None
        res: OscillationResult = result.data
        return plot_oscillation_overlay(
            res.t,
            res.x,
            res.xf,
            res.env,
            res.thr,
            res.bursts,
            res.context.signal_key or "signal",
            band=res.band,
            overlay=overlay,
            overlay_subplot=overlay_subplot,
            window=time_window,
            title=title or f"{res.context.label()} | Signal={res.context.signal_key}",
        )

    def save(
        self,
        result: AnalysisResult,
        *,
        stats_dir: Path,
        plots_dir: Optional[Path] = None,
        overlay: Optional[Tuple[np.ndarray, str]] = None,
        overlay_subplot: bool = False,
        time_window: Optional[Tuple[float, float]] = None,
        save_plot: bool = True,
    ) -> list[Path]:
        if result.data is None:
            return []
        res: OscillationResult = result.data
        paths: list[Path] = [save_oscillation_bursts(res, stats_dir)]
        if save_plot and plots_dir is not None:
            paths.append(
                save_oscillation_plot(
                    res,
                    plots_dir,
                    overlay=overlay,
                    overlay_subplot=overlay_subplot,
                    time_window=time_window,
                )
            )
        return paths