"""Spatio-spectral decomposition (SSD): spatial filters that maximise narrowband power against flanking bands.

Nikulin, Nolte & Curio (2011) NeuroImage 55:1528.  Both covariances come from a
cross-spectral matrix (:func:`databench.signal.spectral.epoch_cross_spectral_matrix`):
the real part summed over the bins of the signal band and of the flanking bands.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy import linalg


def _band_covariance(freqs: np.ndarray, csd: np.ndarray, inside: np.ndarray) -> np.ndarray:
    return csd[:, :, inside].real.sum(axis=2)


def ssd(
    freqs: np.ndarray,
    csd: np.ndarray,
    band: Tuple[float, float],
    flanks: Sequence[Tuple[float, float]],
    *,
    shrinkage: float = 0.01,
) -> dict:
    """SSD of the channels of *csd* ``(N, N, F)``: signal band *band* against the union of *flanks*.

    Bands are closed intervals ``[low, high]`` in Hz; the DC bin is never a
    flank.  The flank covariance is regularised by ``shrinkage`` x its mean
    eigenvalue x identity.  Components come from the generalized eigenproblem
    ``S w = λ R w``, sorted by λ, largest first.

    Returns a dict: ``ratio`` (N,) — λ x flank bins ÷ signal bins, the
    component's mean PSD per bin in the signal band over its mean PSD per bin
    in the flanks; ``filters`` (N, N), one filter per column; ``patterns``
    (N, N), one forward model per column (Haufe et al. 2014,
    ``S w / (wᵀ S w)``).
    """
    freqs = np.asarray(freqs, dtype=float)
    in_signal = (freqs >= band[0]) & (freqs <= band[1])
    in_flank = np.zeros(freqs.size, dtype=bool)
    for low, high in flanks:
        in_flank |= (freqs >= low) & (freqs <= high)
    in_flank &= (freqs > 0) & ~in_signal
    signal_cov = _band_covariance(freqs, csd, in_signal)
    flank_cov = _band_covariance(freqs, csd, in_flank)
    flank_cov = flank_cov + shrinkage * np.trace(flank_cov) / len(flank_cov) * np.eye(len(flank_cov))
    values, filters = linalg.eigh(signal_cov, flank_cov)
    values, filters = values[::-1], filters[:, ::-1]
    patterns = signal_cov @ filters / np.einsum("nk,nm,mk->k", filters, signal_cov, filters)
    return {"ratio": values * in_flank.sum() / in_signal.sum(), "filters": filters, "patterns": patterns}


def ssd_scan(
    freqs: np.ndarray,
    csd: np.ndarray,
    centres: Iterable[float],
    *,
    half_width_hz: float = 0.5,
    flank_hz: Tuple[float, float] = (1.0, 2.0),
    shrinkage: float = 0.01,
) -> dict:
    """Top SSD component at each centre frequency.

    Signal band ``centre ± half_width_hz``; flanks ``centre - flank_hz[1] …
    centre - flank_hz[0]`` and ``centre + flank_hz[0] … centre + flank_hz[1]``.

    Returns a dict: ``centre_hz`` (C,), ``ratio`` (C,), ``filters`` (C, N) and
    ``patterns`` (C, N), each pattern scaled so its largest-magnitude entry is
    +1.  Filters and patterns are in the channel space of *csd*; with *csd*
    from components of a pixel basis ``V`` (pixels × N), ``V @ filter`` and
    ``V @ pattern`` are the pixel filter and pattern.
    """
    centres = np.asarray(list(centres), dtype=float)
    ratios, filters, patterns = [], [], []
    for centre in centres:
        result = ssd(freqs, csd, (centre - half_width_hz, centre + half_width_hz),
                     [(centre - flank_hz[1], centre - flank_hz[0]), (centre + flank_hz[0], centre + flank_hz[1])],
                     shrinkage=shrinkage)
        pattern = result["patterns"][:, 0]
        ratios.append(result["ratio"][0])
        filters.append(result["filters"][:, 0])
        patterns.append(pattern / pattern[np.argmax(np.abs(pattern))])
    return {"centre_hz": centres, "ratio": np.array(ratios), "filters": np.array(filters), "patterns": np.array(patterns)}
