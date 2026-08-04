"""
Oscillation detection on a raw TIFF stack.

Computes a whole-field ΔF/F trace from a (frames, Y, X) TIFF stack, then runs
it through the existing :class:`OscillationDetector` (reusing all detection +
plotting logic) via a lightweight session shim — no Project/dataset required.

Pipeline:
    1. Per-frame spatial mean F(t)   (read in chunks; the stack may be many GB)
    2. ΔF/F = (F - F0) / F0,  F0 = median(F)  (global baseline)
    3. Bandpass + Hilbert envelope → threshold → burst epochs
    4. Overview figure + burst-events CSV written next to the TIFF

Usage:
    python Scripts/oscillations/tif-oscillation-detector.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from databench.analysis.oscillation import OscillationDetector
from databench.plotting import set_theme

set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────

TIF_PATH = Path(r"C:\Users\Admin\Desktop\260307_sub-SB20_ses-02_task-spont_saline.tif")

FS = 30.0                 # acquisition frame rate (Hz)
BAND = (3.0, 5.0)         # bandpass range (Hz); must stay below Nyquist = FS/2
ORDER = 4
THRESHOLD = 0.02          # fixed envelope threshold (ΔF/F units); None → adaptive
THRESHOLD_K = 4.0         # multiplier when THRESHOLD is None
MIN_DURATION = 1.0        # s
MERGE_GAP = 0.5           # s

CHUNK_FRAMES = 500        # frames per read when computing the mean trace

# Metadata for labels / output naming (parsed loosely from the filename).
SUBJECT = "SB20"
SESSION = "ses-02"
TASK = "task-spont"


# ─── 1. Per-frame mean trace (chunked read) ───────────────────────────────

def mean_trace(path: Path, chunk: int = CHUNK_FRAMES) -> np.ndarray:
    """Spatial mean of every frame, read in chunks to bound memory use."""
    with tifffile.TiffFile(path) as tif:
        n_frames = tif.series[0].shape[0]
        out = np.empty(n_frames, dtype=np.float64)
        for i in range(0, n_frames, chunk):
            j = min(i + chunk, n_frames)
            block = tif.asarray(key=range(i, j)).astype(np.float64)
            out[i:j] = block.mean(axis=(1, 2))
            print(f"  mean trace: {j}/{n_frames} frames", end="\r")
    print()
    return out


# ─── Session shim ─────────────────────────────────────────────────────────

class _TraceSession:
    """Minimal stand-in for a Session exposing just what OscillationDetector needs."""

    def __init__(self, trace: np.ndarray, fs: float, subject, session, task):
        self._trace = np.asarray(trace, dtype=float)
        self._t = np.arange(self._trace.size) / fs
        self.subject, self.session, self.task = subject, session, task

    def signal(self, source, signal):
        return self._trace

    def _time_for_align(self, source, time_column):
        return self._t


# ─── Run ──────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = TIF_PATH.parent / f"{TIF_PATH.stem}_oscillation"
    out_dir.mkdir(exist_ok=True)
    slug = TIF_PATH.stem
    trace_path = out_dir / f"{slug}_dff_trace.csv"

    # Reuse a previously computed mean trace if present — avoids re-reading the
    # (potentially many-GB) TIFF stack on parameter sweeps.
    if trace_path.exists():
        print(f"Reusing cached trace: {trace_path}")
        f = np.loadtxt(trace_path, delimiter=",", skiprows=1, usecols=1)
    else:
        if not TIF_PATH.exists():
            raise FileNotFoundError(TIF_PATH)
        print(f"Reading {TIF_PATH.name} …")
        f = mean_trace(TIF_PATH)
    print(f"  trace: {f.size} frames, F range [{f.min():.1f}, {f.max():.1f}]")

    f0 = float(np.median(f))
    dff = (f - f0) / f0
    print(f"  dF/F baseline F0 = {f0:.2f} (median); dF/F range "
          f"[{dff.min():.4f}, {dff.max():.4f}]")

    session = _TraceSession(dff, FS, SUBJECT, SESSION, TASK)

    detector = OscillationDetector(
        source="tif",
        signal="dff",
        fs=FS,
        band_hz=BAND,
        filter_order=ORDER,
        threshold=THRESHOLD,
        threshold_k=THRESHOLD_K,
        min_duration_s=MIN_DURATION,
        merge_gap_s=MERGE_GAP,
    )
    result = detector.run(session)

    fig = result.plot_overview()
    fig_path = out_dir / f"{slug}_dff_overview.svg"
    fig.savefig(fig_path, bbox_inches="tight")

    events_path = out_dir / f"{slug}_dff_bursts.csv"
    result.events.to_csv(events_path, index=False)

    # Save the raw + ΔF/F trace for reuse (skip if loaded from cache).
    if not trace_path.exists():
        np.savetxt(
            trace_path,
            np.column_stack([session._t, f, dff]),
            delimiter=",",
            header="time_s,F_mean,dff",
            comments="",
        )

    total = float(result.events["duration_s"].sum()) if not result.events.empty else 0.0
    print(f"\nDone — {len(result.bursts)} bursts (total {total:.1f} s), "
          f"threshold={result.threshold_value:.5f}")
    print(f"  Overview: {fig_path}")
    print(f"  Events:   {events_path}")
    print(f"  Trace:    {trace_path}")
    # Block so the interactive window stays open when run as a script.
    plt.show(block=True)



if __name__ == "__main__":
    main()
