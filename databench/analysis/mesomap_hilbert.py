from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, cast
from pathlib import Path

import numpy as np
import pandas as pd
from databench.analysis.base import Analysis, AnalysisResult
from databench.debug import log_context
from databench._utils import strip_prefix
from databench.plotting.mesomap import plot_stacked_envelopes, plot_spectrogram_panel
from databench.analysis._signal.preproc import detrend_zscore_1d
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


def bandpass_1d(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    from databench.analysis._signal.bandpass import bandpass_envelope
    filtered, _ = bandpass_envelope(x, fs, (lo, hi), order)
    return filtered


def spectrogram_db(x: np.ndarray, fs: float, win_s: float, overlap_frac: float):
    nperseg = int(win_s * fs)
    noverlap = int(nperseg * overlap_frac)
    freqs, spec_time, power = signal.spectrogram(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,  # type: ignore[arg-type]
        scaling="density",
        mode="psd",
    )
    return freqs, spec_time, 10 * np.log10(power + 1e-12)


def _run_mesomap_hilbert_impl(
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
    all_regions = np.vstack([arr[:n] for arr in stacked])
    all_regions = signal.detrend(all_regions, axis=1, type="linear")
    global_mean = all_regions.mean(axis=0)
    global_mean = detrend_zscore_1d(global_mean)
    signals["GLOBAL"] = global_mean

    n = len(global_mean)
    t = np.arange(n) / fs

    envelopes = {}
    spectrograms: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for region_key, region_signal in signals.items():
        bandpassed = bandpass_1d(region_signal, fs, band_lo, band_hi)
        analytic = cast(np.ndarray, signal.hilbert(bandpassed))
        envelopes[region_key] = np.abs(analytic)

        freqs, spec_time, power_db = spectrogram_db(bandpassed, fs, win_s, overlap_frac)
        freq_mask = (freqs >= 0) & (freqs <= fmax)
        spectrograms[region_key] = (freqs[freq_mask], spec_time, power_db[freq_mask, :])

    overlap = {}
    global_envelope = envelopes["GLOBAL"]
    for region_key in resolved.keys():
        overlap[region_key] = float(np.corrcoef(envelopes[region_key], global_envelope)[0, 1])

    if debug:
        print(f"[mesomap] Resolved: {resolved} | {context}")

    return {
        "t": t,
        "signals": signals,
        "envs": envelopes,
        "specs": spectrograms,
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
    return _run_mesomap_hilbert_impl(
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
