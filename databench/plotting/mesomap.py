from __future__ import annotations

from typing import Dict, Iterable
import matplotlib.pyplot as plt
import numpy as np

from databench.plotting import style_axes


def plot_stacked_envelopes(t: np.ndarray, envs: Dict[str, np.ndarray], keys: Iterable[str],
                           band_lo: float, band_hi: float, ds: int = 10):
    keys = list(keys)
    fig = plt.figure(figsize=(12, 7))
    for i, k in enumerate(keys, start=1):
        ax = plt.subplot(len(keys), 1, i)
        ax.plot(t[::ds], envs[k][::ds], linewidth=1.0)
        ax.set_ylabel(k)
        if i < len(keys):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)")
        style_axes(ax)
    plt.suptitle(
        f"2–5 Hz envelope (Hilbert amplitude), bandpass {band_lo:.0f}–{band_hi:.0f} Hz",
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_spectrogram_panel(specs: Dict[str, tuple], keys: Iterable[str], fmax: float,
                           win_s: float, overlap_frac: float):
    keys = list(keys)
    fig = plt.figure(figsize=(12, 8))
    for i, k in enumerate(keys, start=1):
        f, tt, Sdb = specs[k]
        ax = plt.subplot(2, 2, i)
        ax.pcolormesh(tt, f, Sdb, shading="auto")
        ax.set_ylim(0, fmax)
        ax.set_title(k)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hz")
        style_axes(ax)
    plt.suptitle(
        f"Spectrograms after 2–5 Hz bandpass (win={win_s:.1f}s, overlap={overlap_frac:.3f})",
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return fig