"""Power spectra inside epochs, peaks above the aperiodic (1/f) background, and wavelet power."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import fftconvolve


def epoch_spectra(
    t: np.ndarray,
    x: np.ndarray,
    epochs: pd.DataFrame | np.ndarray,
    *,
    fs: float,
    segment_s: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch power spectrum of *x* inside each epoch.

    *epochs* is a table with ``start_s`` / ``end_s`` or an ``(n, 2)`` array
    of times on the clock *t*.  Each epoch's samples are cut from *x*,
    non-finite samples dropped, and the Welch estimate taken with
    ``segment_s`` Hann segments (50 % overlap, mean removed).  Epochs with
    fewer samples than one segment are skipped.

    Returns ``(freqs, spectra)``: *spectra* has one row per usable epoch, and
    zero rows when none is long enough.
    """
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    pairs = epochs[["start_s", "end_s"]].to_numpy(float) if isinstance(epochs, pd.DataFrame) else np.asarray(epochs, float)
    nperseg = int(round(segment_s * fs))
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    rows = []
    for start, end in pairs.reshape(-1, 2):
        segment = x[np.searchsorted(t, start):np.searchsorted(t, end)]
        segment = segment[np.isfinite(segment)]
        if segment.size >= nperseg:
            rows.append(signal.welch(segment, fs=fs, nperseg=nperseg)[1])
    return freqs, np.vstack(rows) if rows else np.empty((0, freqs.size))


def mean_spectrum(spectra: np.ndarray) -> np.ndarray:
    """Geometric mean of spectra (rows), so each spectrum weighs equally on a log scale."""
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim != 2 or spectra.shape[0] == 0:
        return np.full(spectra.shape[-1], np.nan)
    with np.errstate(divide="ignore"):
        return np.exp(np.nanmean(np.log(spectra), axis=0))


def aperiodic_peak(
    freqs: np.ndarray,
    power: np.ndarray,
    *,
    peak_hz: Tuple[float, float] = (2.5, 6.5),
    fit_hz: Tuple[float, float] = (1.0, 12.0),
) -> dict:
    """Largest excess of *power* over a power-law background, inside *peak_hz*.

    The background is a straight-line fit of log power on log frequency over
    *fit_hz*, leaving out *peak_hz*.  ``peak_gain`` is power ÷ background at
    the peak frequency (1 = no excess); ``residual`` is that ratio at every
    frequency.  All values are NaN when *power* is not finite over the fit.
    """
    freqs, power = np.asarray(freqs, dtype=float), np.asarray(power, dtype=float)
    in_fit = (freqs >= fit_hz[0]) & (freqs <= fit_hz[1])
    in_peak = (freqs >= peak_hz[0]) & (freqs <= peak_hz[1])
    background = in_fit & ~in_peak & (power > 0)
    if background.sum() < 2 or not np.isfinite(power[in_peak]).all():
        return {"peak_hz": np.nan, "peak_gain": np.nan, "residual": np.full(freqs.shape, np.nan)}
    slope, intercept = np.polyfit(np.log(freqs[background]), np.log(power[background]), 1)
    with np.errstate(divide="ignore"):
        residual = power / np.exp(intercept + slope * np.log(freqs))
    best = np.flatnonzero(in_peak)[np.argmax(residual[in_peak])]
    return {"peak_hz": float(freqs[best]), "peak_gain": float(residual[best]), "residual": residual}


def morlet_power(x: np.ndarray, fs: float, freqs: np.ndarray, *, width: float = 6.0) -> np.ndarray:
    """Morlet wavelet power of *x* at each of *freqs*: one row per frequency, one column per sample.

    Each wavelet has *width* cycles (Gaussian σ = width / 2πf, cut at ±3.5σ)
    and unit energy.  The mean is removed and non-finite samples are set to
    zero before convolution.
    """
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x - np.nanmean(x), 0.0)
    power = np.empty((len(freqs), x.size))
    for row, freq in enumerate(np.asarray(freqs, dtype=float)):
        sigma = width / (2 * np.pi * freq)
        lag = np.arange(-3.5 * sigma, 3.5 * sigma + 0.5 / fs, 1.0 / fs)
        wavelet = np.exp(2j * np.pi * freq * lag - lag ** 2 / (2 * sigma ** 2))
        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))
        power[row] = np.abs(fftconvolve(x, wavelet, mode="same")) ** 2
    return power
