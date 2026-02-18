from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, cast
from pathlib import Path

import numpy as np
import pandas as pd
from databench.analysis.base import AnalysisFn, Analysis, AnalysisResult
from databench.debug import log_context
from databench.utils import strip_prefix
from databench.plotting import plot_stacked_envelopes, plot_spectrogram_panel
from scipy import signal


@dataclass(frozen=True)
class MesomapHilbertConfig:
    fs: float = 50.0 # sampling frequency (Hz) to define Nyquist f/2
    band_lo: float = 3.2 # bandpass low end
    band_hi: float = 4.0 # bandpass high end
    win_s: float = 4.0 # spectrogram window size (s)
    overlap_frac: float = 0.950 # spectrogram overlap fraction
    fmax: float = 12.0 # max frequency to keep in spectrogram (Hz) y axis
    targets: Dict[str, str] = field(default_factory=lambda: {
        "VISp": "L_VISp",
        "MOs": "L_MOs",
        "SSp-ll": "L_SSp-ll",
    })


def detrend_zscore_1d(x: np.ndarray) -> np.ndarray:
    xd = signal.detrend(x, type="linear")
    sd = xd.std(ddof=1)
    if sd == 0:
        sd = 1.0
    return (xd - xd.mean()) / sd


def bandpass_1d(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    sos = signal.butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def spectrogram_db(x: np.ndarray, fs: float, win_s: float, overlap_frac: float):
    nperseg = int(win_s * fs)
    noverlap = int(nperseg * overlap_frac)
    f, tt, Sxx = signal.spectrogram(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,  # type: ignore[arg-type]
        scaling="density",
        mode="psd",
    )
    return f, tt, 10 * np.log10(Sxx + 1e-12)


@dataclass(frozen=True)
class MesomapHilbert(AnalysisFn):
    name: str = "mesomap_hilbert"

    def _run_impl(
        self,
        row: pd.Series,
        cfg: MesomapHilbertConfig,
        source: str = "mesomap",
        debug: bool = False,
        context: Optional[str] = None,
    ):
        regions = [c[1] for c in row.index if c[0] == source]

        resolved: Dict[str, str] = {}
        for key, col in cfg.targets.items():
            if col in regions:
                resolved[key] = col
                continue
            candidates = [region for region in regions if region.startswith("L_") and (key in region)]
            if candidates:
                resolved[key] = candidates[0]
            elif regions:
                resolved[key] = regions[0]
            else:
                resolved[key] = col

        signals = {}
        for k, col in resolved.items():
            x = row.get((source, col))
            signals[k] = detrend_zscore_1d(np.asarray(x, dtype=float))

        stacked = []
        for region in regions:
            x = row.get((source, region))
            arr = np.asarray(x, dtype=float)
            stacked.append(arr.ravel())
        n = min(len(arr) for arr in stacked)
        X_all = np.vstack([arr[:n] for arr in stacked])
        X_all = signal.detrend(X_all, axis=1, type="linear")
        g = X_all.mean(axis=0)
        g = detrend_zscore_1d(g)
        signals["GLOBAL"] = g

        n = len(g)
        t = np.arange(n) / cfg.fs

        envs = {}
        specs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for k, x0 in signals.items():
            xb = bandpass_1d(x0, cfg.fs, cfg.band_lo, cfg.band_hi)
            hb = cast(np.ndarray, signal.hilbert(xb))
            envs[k] = np.abs(hb)

            f, tt, Sdb = spectrogram_db(xb, cfg.fs, cfg.win_s, cfg.overlap_frac)
            m = (f >= 0) & (f <= cfg.fmax)
            specs[k] = (f[m], tt, Sdb[m, :])

        overlap = {}
        g_env = envs["GLOBAL"]
        for k in resolved.keys():
            overlap[k] = float(np.corrcoef(envs[k], g_env)[0, 1])

        if debug:
            print(f"[mesomap] Resolved: {resolved} | {context}")

        return {
            "t": t,
            "signals": signals,
            "envs": envs,
            "specs": specs,
            "resolved": resolved,
            "overlap": overlap,
        }


def run_mesomap_hilbert(
    row: pd.Series,
    cfg: MesomapHilbertConfig,
    source: str = "mesomap",
    debug: bool = False,
    context: Optional[str] = None,
):
    return MesomapHilbert().run(row, cfg=cfg, source=source, debug=debug, context=context)


def export_hilbert_envelopes(
    df: pd.DataFrame,
    cfg: MesomapHilbertConfig,
    out_dir: Path,
    source: str = "mesomap",
    analysis_name: str = "hilbert_env",
    debug: bool = False,
):
    """Export Hilbert envelope CSV for each (Subject, Session, Task)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        ctx = log_context(idx)
        out = run_mesomap_hilbert(row, cfg, source=source, debug=debug, context=ctx)
        subject, session, task = idx[:3]
        subject = strip_prefix(subject, "sub-")
        session = strip_prefix(session, "ses-")
        task = strip_prefix(task, "task-")

        keys = list(out["resolved"].keys())
        keys.append("GLOBAL")

        data = {"t_s": out["t"]}
        for k in keys:
            data[f"env_{k}"] = out["envs"][k]

        df_env = pd.DataFrame(data)
        filename = f"sub-{subject}_ses-{session}_task-{task}_{analysis_name}.csv"
        df_env.to_csv(out_dir / filename, index=False)


@dataclass(frozen=True)
class MesomapHilbertAnalysis(Analysis):
    name: str = "mesomap_hilbert"

    def run(
        self,
        row: pd.Series,
        cfg: MesomapHilbertConfig,
        *,
        source: str = "mesomap",
        debug: bool = False,
        context: Optional[str] = None,
    ) -> AnalysisResult:
        out = run_mesomap_hilbert(row, cfg, source=source, debug=debug, context=context)
        return AnalysisResult(name=self.name, data=out, meta={"source": source, "cfg": cfg})

    def plot(
        self,
        result: AnalysisResult,
        *,
        kind: str = "envelopes",
        keys: Optional[list[str]] = None,
        **kwargs,
    ):
        out = result.data
        if kind == "envelopes":
            use_keys = keys or list(out["resolved"].keys())
            use_keys.append("GLOBAL")
            cfg = result.meta.get("cfg")
            return plot_stacked_envelopes(
                out["t"],
                out["envs"],
                use_keys,
                kwargs.pop("band_lo", cfg.band_lo if cfg else 3.2),
                kwargs.pop("band_hi", cfg.band_hi if cfg else 4.0),
                **kwargs,
            )
        if kind == "spectrograms":
            use_keys = keys or list(out["resolved"].keys())
            use_keys.append("GLOBAL")
            cfg = result.meta.get("cfg")
            fmax = kwargs.pop("fmax", cfg.fmax if cfg else 12.0)
            win_s = kwargs.pop("win_s", cfg.win_s if cfg else 4.0)
            overlap_frac = kwargs.pop("overlap_frac", cfg.overlap_frac if cfg else 0.95)
            return plot_spectrogram_panel(out["specs"], use_keys, fmax, win_s, overlap_frac)
        raise ValueError(f"Unknown plot kind: {kind!r}")
