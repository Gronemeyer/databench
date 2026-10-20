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


def epoch_cross_spectra(
    t: np.ndarray,
    x: np.ndarray,
    reference: np.ndarray,
    epochs: pd.DataFrame | np.ndarray,
    *,
    fs: float,
    segment_s: float = 4.0,
    overlap: float = 0.5,
    detrend: str = "constant",
) -> dict:
    """Welch spectra of each column of *x* and its cross-spectrum with *reference*, pooled over epochs.

    *x* is ``(T,)`` or ``(T, N)`` on the clock *t*; *reference* is ``(T,)``.
    *epochs* is a table with ``start_s`` / ``end_s`` or an ``(n, 2)`` array.
    Inside each epoch, samples where *reference* or any column is non-finite
    are dropped, and epochs with fewer samples than one ``segment_s`` Hann
    segment are skipped.  Each epoch's Welch estimates are weighted by its
    sample count.  ``cross`` follows :func:`scipy.signal.csd` with the
    reference first, ``E[conj(R) X]``, so a column that lags the reference
    has a negative phase.

    Returns a dict: ``freqs`` (F,), ``power`` (N, F), ``reference_power`` (F,),
    ``cross`` (N, F) complex, ``analysed_s`` and ``n_epochs``.  The spectra
    are NaN when no epoch is long enough.
    """
    t, reference = np.asarray(t, dtype=float), np.asarray(reference, dtype=float)
    x = np.asarray(x, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    pairs = epochs[["start_s", "end_s"]].to_numpy(float) if isinstance(epochs, pd.DataFrame) else np.asarray(epochs, float)
    nperseg = int(round(segment_s * fs))
    options = dict(fs=fs, nperseg=nperseg, noverlap=int(nperseg * overlap), detrend=detrend, axis=0)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    power = np.zeros((x.shape[1], freqs.size))
    cross = np.zeros((x.shape[1], freqs.size), dtype=complex)
    reference_power = np.zeros(freqs.size)
    n_samples, n_epochs = 0, 0
    for start, end in pairs.reshape(-1, 2):
        first, last = np.searchsorted(t, [start, end])
        segment, ref = x[first:last], reference[first:last]
        keep = np.isfinite(ref) & np.isfinite(segment).all(axis=1)
        segment, ref = segment[keep], ref[keep]
        if ref.size < nperseg:
            continue
        power += ref.size * signal.welch(segment, **options)[1].T
        cross += ref.size * signal.csd(ref[:, None], segment, **options)[1].T
        reference_power += ref.size * signal.welch(ref, **{**options, "axis": -1})[1]
        n_samples += ref.size
        n_epochs += 1
    scale = 1.0 / n_samples if n_samples else np.nan
    return {"freqs": freqs, "power": power * scale, "reference_power": reference_power * scale,
            "cross": cross * scale, "analysed_s": n_samples / fs, "n_epochs": n_epochs}


def epoch_cross_spectral_matrix(
    t: np.ndarray,
    x: np.ndarray,
    epochs: pd.DataFrame | np.ndarray,
    *,
    fs: float,
    segment_s: float = 4.0,
    overlap: float = 0.5,
    detrend: str | bool = "constant",
) -> dict:
    """Welch cross-spectral matrix of every pair of columns of *x*, pooled over epochs.

    Uses the segments, Hann window, per-segment detrending, one-sided density
    scaling and epoch weighting of :func:`epoch_cross_spectra`, in one pass:
    entry ``[k, l]`` equals that function's ``cross`` for column ``l`` with
    column ``k`` as the reference, ``E[conj(x_k) x_l]``, and the diagonal is
    each column's power.  *x* is ``(T, N)`` on the clock *t*; *epochs* is a
    table with ``start_s`` / ``end_s`` or an ``(n, 2)`` array.  Inside each
    epoch, samples where any column is non-finite are dropped, and epochs
    with fewer samples than one segment are skipped.  Memory grows with
    ``N²``, so *x* is meant to hold components (tens to a few hundred), not
    pixels.

    Returns a dict: ``freqs`` (F,), ``csd`` (N, N, F) complex, ``analysed_s``
    and ``n_epochs``.  ``csd`` is NaN when no epoch is long enough.
    """
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    pairs = epochs[["start_s", "end_s"]].to_numpy(float) if isinstance(epochs, pd.DataFrame) else np.asarray(epochs, float)
    nperseg = int(round(segment_s * fs))
    step = nperseg - int(nperseg * overlap)
    window = signal.get_window("hann", nperseg)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    one_sided = np.full(freqs.size, 2.0)
    one_sided[0] = 1.0
    if nperseg % 2 == 0:
        one_sided[-1] = 1.0
    density = one_sided / (fs * np.sum(window ** 2))
    csd = np.zeros((x.shape[1], x.shape[1], freqs.size), dtype=complex)
    n_samples, n_epochs = 0, 0
    for start, end in pairs.reshape(-1, 2):
        first, last = np.searchsorted(t, [start, end])
        segment = x[first:last]
        segment = segment[np.isfinite(segment).all(axis=1)]
        if segment.shape[0] < nperseg:
            continue
        starts = np.arange(0, segment.shape[0] - nperseg + 1, step)
        frames = np.stack([segment[s:s + nperseg] for s in starts])          # (segments, nperseg, N)
        if detrend:
            frames = signal.detrend(frames, axis=1, type=detrend)
        spectra = np.fft.rfft(frames * window[None, :, None], axis=1)         # (segments, F, N)
        csd += segment.shape[0] * np.einsum("sfk,sfl->klf", spectra.conj(), spectra) * density / starts.size
        n_samples += segment.shape[0]
        n_epochs += 1
    scale = 1.0 / n_samples if n_samples else np.nan
    return {"freqs": freqs, "csd": csd * scale, "analysed_s": n_samples / fs, "n_epochs": n_epochs}


def band_coupling(
    freqs: np.ndarray,
    power: np.ndarray,
    reference_power: np.ndarray,
    cross: np.ndarray,
    band: Tuple[float, float],
) -> dict:
    """Band power of each row of *power* and its coherence and lag with the reference, over ``[band[0], band[1])``.

    ``power`` is the PSD summed over the band times the bin width.
    ``coherence`` is ``|Σ cross|² / (Σ power · Σ reference_power)`` over the
    band bins.  ``lag_s`` is the phase of ``Σ cross`` divided by 2π times the
    band's centre frequency, which is weighted by ``|cross|``.  It is
    positive when the row lags the reference, using the sign convention of
    :func:`epoch_cross_spectra`.  Each value is a 1-D array with one entry
    per row.
    """
    freqs = np.asarray(freqs, dtype=float)
    power, cross = np.atleast_2d(power), np.atleast_2d(cross)
    in_band = (freqs >= band[0]) & (freqs < band[1])
    step = float(np.median(np.diff(freqs)))
    band_cross = cross[:, in_band].sum(axis=1)
    band_power = power[:, in_band].sum(axis=1)
    weight = np.abs(cross[:, in_band])
    with np.errstate(invalid="ignore", divide="ignore"):
        centre_hz = (weight * freqs[in_band]).sum(axis=1) / weight.sum(axis=1)
        coherence = np.abs(band_cross) ** 2 / (band_power * np.asarray(reference_power)[in_band].sum())
        lag_s = -np.angle(band_cross) / (2 * np.pi * centre_hz)
    return {"power": band_power * step, "coherence": coherence, "lag_s": lag_s}


def aperiodic_fit(
    freqs: np.ndarray,
    power: np.ndarray,
    *,
    fit_hz: Tuple[float, float] = (1.0, 8.0),
    method: str = "specparam",
    aperiodic_mode: str = "fixed",
    max_n_peaks: int = 3,
    peak_width_limits: Tuple[float, float] = (0.5, 3.0),
    robust_iterations: int = 3,
) -> dict:
    """Aperiodic (1/f) component of a power spectrum over *fit_hz*.

    ``method="specparam"`` fits the spectrum with specparam's ``SpectralModel``
    (Donoghue et al. 2020): an aperiodic component (``aperiodic_mode``
    ``"fixed"``, or ``"knee"`` when *fit_hz* reaches below ~1 Hz) plus up to
    *max_n_peaks* Gaussian peaks of width *peak_width_limits* (Hz).
    ``method="robust"`` fits a line of log10 power on log10 frequency, then
    refits *robust_iterations* times leaving out bins above the fit + 1 SD of
    the residuals.

    Returns a dict: ``method``, ``offset`` (log10 power), ``exponent``,
    ``knee`` (NaN unless fitted), ``r_squared`` over *fit_hz* (of the full
    specparam model; of the line over the kept bins for ``"robust"``),
    ``peaks`` (``(n, 3)`` centre Hz, log10 height, width Hz; empty for
    ``"robust"``), and ``background``, a callable from frequency to power in
    the units of *power*.  Values are NaN, and ``background`` returns NaN,
    when the fit fails.
    """
    freqs, power = np.asarray(freqs, dtype=float), np.asarray(power, dtype=float)
    keep = (freqs >= fit_hz[0]) & (freqs <= fit_hz[1]) & (power > 0) & np.isfinite(power)
    failed = {"method": method, "offset": np.nan, "exponent": np.nan, "knee": np.nan, "r_squared": np.nan,
              "peaks": np.empty((0, 3)), "background": lambda f: np.full(np.shape(f), np.nan)}
    if keep.sum() < 4:
        return failed
    if method == "robust":
        x, y = np.log10(freqs[keep]), np.log10(power[keep])
        used = np.ones(x.size, dtype=bool)
        for _ in range(robust_iterations + 1):
            slope, offset = np.polyfit(x[used], y[used], 1)
            residual = y - (offset + slope * x)
            used = residual <= residual[used].std(ddof=1) if used.sum() > 2 else used
        r_squared = 1 - np.var(residual[used]) / np.var(y[used])
        return {**failed, "offset": float(offset), "exponent": float(-slope), "r_squared": float(r_squared),
                "background": lambda f: 10.0 ** (offset + slope * np.log10(np.asarray(f, dtype=float)))}
    if method != "specparam":
        raise ValueError(f"method must be 'specparam' or 'robust', not {method!r}")
    from specparam import SpectralModel  # optional dependency
    model = SpectralModel(aperiodic_mode=aperiodic_mode, max_n_peaks=max_n_peaks,
                          peak_width_limits=peak_width_limits, verbose=False)
    try:
        model.fit(freqs[keep], power[keep], list(fit_hz))
    except Exception:  # noqa: BLE001 — a failed fit is reported as NaN
        return failed
    params = np.asarray(model.results.get_params("aperiodic"), dtype=float)
    if not np.isfinite(params).all():
        return failed
    offset, knee, exponent = (params[0], params[1], params[2]) if aperiodic_mode == "knee" \
        else (params[0], 0.0, params[1])
    peaks = np.atleast_2d(np.asarray(model.results.get_params("peak"), dtype=float)).reshape(-1, 3)
    r_squared = float(model.results.get_metrics("gof_rsquared"))
    return {"method": method, "offset": float(offset), "exponent": float(exponent),
            "knee": float(knee) if aperiodic_mode == "knee" else np.nan, "r_squared": r_squared,
            "peaks": peaks[np.isfinite(peaks).all(axis=1)],
            "background": lambda f: 10.0 ** (offset - np.log10(knee + np.asarray(f, dtype=float) ** exponent))}


def colored_noise(n: int, fs: float, psd, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Gaussian noise of *n* samples whose one-sided PSD is ``psd(f)`` (units²/Hz).

    Each positive Fourier coefficient gets a complex Gaussian amplitude with
    ``E|X|² = psd(f) · fs · n / 2`` and the DC term is zero, so a Welch or
    periodogram estimate of the result averages ``psd(f)``.  Non-finite
    values of ``psd`` are set to 0.
    """
    rng = np.random.default_rng() if rng is None else rng
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    target = np.zeros(freqs.size)
    target[1:] = np.nan_to_num(np.asarray(psd(freqs[1:]), dtype=float), nan=0.0, posinf=0.0)
    amplitude = np.sqrt(target * fs * n / 2.0)
    coefficients = amplitude * (rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)) / np.sqrt(2)
    if n % 2 == 0:
        coefficients[-1] = amplitude[-1] * rng.standard_normal()
    return np.fft.irfft(coefficients, n)
