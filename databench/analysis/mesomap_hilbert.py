from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, cast
from pathlib import Path

import numpy as np
import pandas as pd
from databench.analysis.base import AnalysisFn, Analysis, AnalysisResult
from databench.debug import log_context
from databench.utils import strip_prefix
from databench.plotting import plot_stacked_envelopes, plot_spectrogram_panel
from scipy import signal


# Default mesomap Hilbert parameters
_DEFAULT_FS = 50.0
_DEFAULT_BAND_LO = 3.2
_DEFAULT_BAND_HI = 4.0
_DEFAULT_WIN_S = 4.0
_DEFAULT_OVERLAP_FRAC = 0.950
_DEFAULT_FMAX = 12.0
_DEFAULT_TARGETS: Dict[str, str] = {
    "VISp": "L_VISp",
    "MOs": "L_MOs",
    "SSp-ll": "L_SSp-ll",
}


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
        *,
        fs: float = _DEFAULT_FS,
        band_lo: float = _DEFAULT_BAND_LO,
        band_hi: float = _DEFAULT_BAND_HI,
        win_s: float = _DEFAULT_WIN_S,
        overlap_frac: float = _DEFAULT_OVERLAP_FRAC,
        fmax: float = _DEFAULT_FMAX,
        targets: Optional[Dict[str, str]] = None,
        source: str = "mesomap",
        debug: bool = False,
        context: Optional[str] = None,
    ):
        if targets is None:
            targets = _DEFAULT_TARGETS
        regions = [c[1] for c in row.index if c[0] == source]

        resolved: Dict[str, str] = {}
        for key, col in targets.items():
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
        t = np.arange(n) / fs

        envs = {}
        specs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for k, x0 in signals.items():
            xb = bandpass_1d(x0, fs, band_lo, band_hi)
            hb = cast(np.ndarray, signal.hilbert(xb))
            envs[k] = np.abs(hb)

            f, tt, Sdb = spectrogram_db(xb, fs, win_s, overlap_frac)
            m = (f >= 0) & (f <= fmax)
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
    *,
    fs: float = _DEFAULT_FS,
    band_lo: float = _DEFAULT_BAND_LO,
    band_hi: float = _DEFAULT_BAND_HI,
    win_s: float = _DEFAULT_WIN_S,
    overlap_frac: float = _DEFAULT_OVERLAP_FRAC,
    fmax: float = _DEFAULT_FMAX,
    targets: Optional[Dict[str, str]] = None,
    source: str = "mesomap",
    debug: bool = False,
    context: Optional[str] = None,
):
    return MesomapHilbert().run(
        row,
        fs=fs,
        band_lo=band_lo,
        band_hi=band_hi,
        win_s=win_s,
        overlap_frac=overlap_frac,
        fmax=fmax,
        targets=targets,
        source=source,
        debug=debug,
        context=context,
    )


def export_hilbert_envelopes(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    fs: float = _DEFAULT_FS,
    band_lo: float = _DEFAULT_BAND_LO,
    band_hi: float = _DEFAULT_BAND_HI,
    win_s: float = _DEFAULT_WIN_S,
    overlap_frac: float = _DEFAULT_OVERLAP_FRAC,
    fmax: float = _DEFAULT_FMAX,
    targets: Optional[Dict[str, str]] = None,
    source: str = "mesomap",
    analysis_name: str = "hilbert_env",
    debug: bool = False,
):
    """Export Hilbert envelope CSV for each (Subject, Session, Task)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        ctx = log_context(idx)
        out = run_mesomap_hilbert(
            row,
            fs=fs,
            band_lo=band_lo,
            band_hi=band_hi,
            win_s=win_s,
            overlap_frac=overlap_frac,
            fmax=fmax,
            targets=targets,
            source=source,
            debug=debug,
            context=ctx,
        )
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
    fs: float = _DEFAULT_FS
    band_lo: float = _DEFAULT_BAND_LO
    band_hi: float = _DEFAULT_BAND_HI
    win_s: float = _DEFAULT_WIN_S
    overlap_frac: float = _DEFAULT_OVERLAP_FRAC
    fmax: float = _DEFAULT_FMAX
    targets: Optional[Dict[str, str]] = None
    source: str = "mesomap"

    def run(
        self,
        row: pd.Series,
    ) -> AnalysisResult:
        out = run_mesomap_hilbert(
            row,
            fs=self.fs,
            band_lo=self.band_lo,
            band_hi=self.band_hi,
            win_s=self.win_s,
            overlap_frac=self.overlap_frac,
            fmax=self.fmax,
            targets=self.targets,
            source=self.source,
        )
        return AnalysisResult(
            name=self.name,
            data=out,
            meta={
                "source": self.source,
                "band_lo": self.band_lo,
                "band_hi": self.band_hi,
                "fmax": self.fmax,
                "win_s": self.win_s,
                "overlap_frac": self.overlap_frac,
            },
        )

    def plot(
        self,
        result: AnalysisResult,
    ):
        """Plot envelopes from a MesomapHilbert result."""
        out = result.data
        meta = result.meta or {}
        use_keys = list(out["resolved"].keys())
        use_keys.append("GLOBAL")
        return plot_stacked_envelopes(
            out["t"],
            out["envs"],
            use_keys,
            meta.get("band_lo", _DEFAULT_BAND_LO),
            meta.get("band_hi", _DEFAULT_BAND_HI),
        )
