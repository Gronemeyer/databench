"""Oscillation burst detection — public analysis API.

Two tiers, as in :mod:`databench.analysis.locomotion`:

* :func:`oscillation_bursts` / :func:`burst_events` — pure-array core.  A
  colleague with a time vector and a trace calls these directly; no
  ``Project`` or ``Session`` is involved.  :func:`bosc_episodes` is the
  threshold-free alternative: power above the trace's own 1/f background::

      from databench.analysis.oscillation import oscillation_bursts
      bursts = oscillation_bursts(t, dff, fs=50.0, band_hz=(2.0, 5.0))

* :class:`OscillationDetector` (one ``Session``) and
  :func:`oscillation_events` (a ``SessionGroup``, as an ETA events table) —
  thin wrappers that pull the arrays out of a session and call the core.

Session usage::

    from databench.analysis.oscillation import OscillationDetector

    detector = OscillationDetector(
        source="mesomap",
        signal="L_VISp",
        fs=50.0,
        band_hz=(2.0, 4.0),
    )
    result = detector.run(session)
    run = proj.run(name="oscillation-detector")
    run.save_figure(result.plot_overview(aligned=aligned, pupil="pupil_diameter_mm"), "overview.svg")
    run.save_table(result.events, "bursts.csv")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import chi2

from databench.utils.logger import get_logger, log_run
from databench.signal.bandpass import bandpass_envelope, robust_threshold
from databench.signal.spectral import morlet_power
from databench.signal.epoching import (
    END_IDX, END_S, REQUIRED_COLUMNS, START_IDX, START_S, detect_epochs, make_events,
)

if TYPE_CHECKING:
    from databench.session import Session, SessionGroup
    from databench.types import EventsTable

_log = get_logger(__name__)


# ── OscillationDetector ────────────────────────────────────────────────────

@dataclass(frozen=True)
class OscillationDetector:
    """Detect oscillatory bursts in a neural signal via Hilbert envelope thresholding.

    Parameters
    ----------
    source : str
        Data source name (e.g. ``"mesomap"``).
    signal : str
        Signal name within the source (e.g. ``"L_VISp"``).
    fs : float
        Sampling rate in Hz.
    band_hz : (float, float)
        Bandpass filter frequency range in Hz.
    filter_order : int
        Butterworth filter order.
    threshold : float or None
        Fixed absolute threshold.  If ``None``, uses adaptive
        ``median + threshold_k × robust_std``.
    threshold_k : float
        Multiplier for adaptive threshold (only used when *threshold* is None).
    min_duration_s : float
        Minimum burst duration in seconds.
    merge_gap_s : float
        Merge bursts separated by less than this gap (seconds).
    time_column : str
        Name of the time column within the source.
    """

    source: str
    signal: str
    fs: float = 50.0
    band_hz: Tuple[float, float] = (3.1, 4.3)
    filter_order: int = 4
    threshold: float | None = None
    threshold_k: float = 4.0
    min_duration_s: float = 0.5
    merge_gap_s: float = 0.25
    time_column: str = "time_elapsed_s"

    @log_run
    def run(self, session: "Session") -> "OscillationResult":
        """Run oscillation detection on a single session.

        Parameters
        ----------
        session : Session
            The session to analyze.

        Returns
        -------
        OscillationResult
            Contains detected events, signal arrays, and plotting/saving methods.

        Raises
        ------
        ValueError
            If parameters are invalid (e.g. band exceeds Nyquist).
        SignalNotFoundError
            If the source or signal is not found in the session.
        """
        self._validate()

        # Extract signal and time arrays.
        # Use align-time fallback so scripts do not need local time synthesis.
        x = session.signal(self.source, self.signal)
        t_raw = session._time_for_align(self.source, self.time_column)

        # Align lengths
        n = min(len(t_raw), len(x))
        x = x[:n]
        t = t_raw[:n].astype(float)

        # Bandpass + Hilbert envelope — require signal longer than sosfiltfilt padlen
        min_len = 3 * (2 * self.filter_order + 1) + 1
        if n < min_len:
            _log.warning(
                "Signal too short for bandpass filter "
                f"({n} < {min_len} samples): "
                f"{session.subject}/{session.session}/{session.task} — skipping"
            )
            nothing = np.full_like(x, np.nan, dtype=float)
            return OscillationResult(
                events=burst_events(t, nothing, nothing, np.nan, fs=self.fs),
                time=t, raw_signal=x,
                filtered_signal=np.full_like(x, np.nan),
                envelope=np.full_like(x, np.nan),
                threshold_value=np.nan,
                bursts=[],
                subject=session.subject,
                session=session.session,
                task=session.task,
                source=self.source,
                signal_name=self.signal,
                band_hz=self.band_hz,
                fs=self.fs,
                _session=session,
            )
        filtered, env = bandpass_envelope(x, self.fs, self.band_hz, self.filter_order)

        # Threshold
        if self.threshold is not None:
            thr = self.threshold
        else:
            thr, _, _ = robust_threshold(env, self.threshold_k)

        events = burst_events(
            t, filtered, env, thr, fs=self.fs,
            min_duration_s=self.min_duration_s, merge_gap_s=self.merge_gap_s,
        )

        # Derive sample-index pairs for plotting compatibility
        bursts = list(zip(
            events[START_IDX].astype(int), events[END_IDX].astype(int)
        )) if not events.empty else []

        return OscillationResult(
            events=events,
            time=t,
            raw_signal=x,
            filtered_signal=filtered,
            envelope=env,
            threshold_value=thr,
            bursts=bursts,
            subject=session.subject,
            session=session.session,
            task=session.task,
            source=self.source,
            signal_name=self.signal,
            band_hz=self.band_hz,
            fs=self.fs,
            _session=session,
        )

    def _validate(self) -> None:
        """Check parameters and raise clear errors."""
        if self.fs <= 0:
            raise ValueError(f"fs must be positive, got {self.fs}")
        nyq = self.fs / 2
        lo, hi = self.band_hz
        if lo <= 0:
            raise ValueError(f"band_hz lower bound must be positive, got {lo}")
        if hi <= lo:
            raise ValueError(
                f"band_hz upper ({hi}) must be greater than lower ({lo})"
            )
        if hi >= nyq:
            raise ValueError(
                f"band_hz upper ({hi} Hz) must be less than Nyquist frequency "
                f"({nyq} Hz for fs={self.fs})"
            )
        if self.filter_order < 1:
            raise ValueError(f"filter_order must be ≥ 1, got {self.filter_order}")
        if self.min_duration_s < 0:
            raise ValueError(f"min_duration_s must be ≥ 0, got {self.min_duration_s}")
        if self.merge_gap_s < 0:
            raise ValueError(f"merge_gap_s must be ≥ 0, got {self.merge_gap_s}")


# ── Pure-array core ────────────────────────────────────────────────────────

def _instantaneous_frequency(filtered: np.ndarray, fs: float) -> np.ndarray:
    """Hz between consecutive samples, from the unwrapped analytic phase."""
    phase = np.unwrap(np.angle(hilbert(filtered)))
    return np.diff(phase) * fs / (2 * np.pi)


def burst_events(
    t: np.ndarray,
    filtered: np.ndarray,
    envelope: np.ndarray,
    threshold: float,
    *,
    fs: float,
    min_duration_s: float = 0.5,
    merge_gap_s: float = 0.25,
) -> pd.DataFrame:
    """Bursts where *envelope* exceeds *threshold*, as an EpochTable.

    Takes an already band-passed signal and its envelope — use
    :func:`oscillation_bursts` to start from the raw trace.  Adds to the
    standard epoch columns:

    ``peak_env``
        Envelope maximum within the burst.
    ``frequency_hz``
        Median instantaneous frequency of *filtered* within the burst.
    ``confirmed_s``
        First above-threshold sample at which the burst has lasted
        *min_duration_s* — the earliest moment it could be known to be a
        burst.  ``start_s`` is set with hindsight; use ``confirmed_s`` when a
        state defined by the burst must not look ahead in time (for example,
        the rate of an event that ends the bursts).
    """
    t = np.asarray(t, dtype=float)
    events = detect_epochs(
        np.asarray(envelope) > threshold, t, event_type="oscillation_burst",
        min_duration_s=min_duration_s, merge_gap_s=merge_gap_s,
    )
    frequency = _instantaneous_frequency(np.asarray(filtered, dtype=float), fs)
    pairs = list(zip(events[START_IDX].astype(int), events[END_IDX].astype(int)))
    events["peak_env"] = [float(np.max(envelope[s:e + 1])) for s, e in pairs]
    events["frequency_hz"] = [float(np.nanmedian(frequency[s:e])) if e > s else np.nan for s, e in pairs]
    above = np.asarray(envelope) > threshold
    events["confirmed_s"] = [_confirmation_time(t, above, s, e, min_duration_s) for s, e in pairs]
    return events


def _confirmation_time(t, above, start, end, min_duration_s) -> float:
    """Time of the first above-threshold sample in ``[start, end]`` at least *min_duration_s* after *start*."""
    dt = float(np.nanmedian(np.diff(t))) if t.size > 1 else 0.0
    idx = start + np.flatnonzero(above[start:end + 1] & (t[start:end + 1] - t[start] + dt >= min_duration_s))
    return float(t[idx[0]]) if idx.size else float(t[end])


def oscillation_bursts(
    t: np.ndarray,
    x: np.ndarray,
    *,
    fs: float,
    band_hz: Tuple[float, float] = (3.1, 4.3),
    filter_order: int = 4,
    threshold: float | None = None,
    threshold_k: float = 4.0,
    min_duration_s: float = 0.5,
    merge_gap_s: float = 0.25,
) -> pd.DataFrame:
    """Detect band-limited bursts in a raw 1-D trace.

    Butterworth band-pass (:func:`~databench.signal.bandpass.bandpass_envelope`),
    Hilbert envelope, then a fixed *threshold* or ``median + threshold_k ×
    robust_std`` (:func:`~databench.signal.bandpass.robust_threshold`), then
    :func:`burst_events`.  The same pipeline as :class:`OscillationDetector`,
    on plain arrays.
    """
    x = np.asarray(x, dtype=float)
    filtered, envelope = bandpass_envelope(x, fs, band_hz, filter_order)
    thr = threshold if threshold is not None else robust_threshold(envelope, threshold_k)[0]
    return burst_events(t, filtered, envelope, thr, fs=fs,
                        min_duration_s=min_duration_s, merge_gap_s=merge_gap_s)


def bosc_episodes(
    t: np.ndarray,
    x: np.ndarray,
    *,
    fs: float,
    band_hz: Tuple[float, float],
    fit_hz: Tuple[float, float] = (1.0, 12.0),
    reference: np.ndarray | None = None,
    freqs_per_octave: int = 8,
    percentile: float = 0.95,
    min_cycles: float = 3.0,
    width: float = 6.0,
) -> dict:
    """Oscillatory episodes in *band_hz* by the BOSC rule (Caplan et al. 2001; Whitten et al. 2011).

    Morlet power (:func:`~databench.signal.spectral.morlet_power`) at
    log-spaced frequencies across *fit_hz*.  The background is a straight-line
    fit of log mean power on log frequency, over the *reference* samples
    (default: all) and the *fit_hz* frequencies outside *band_hz*.  Under a
    background-only signal, wavelet power at one frequency is distributed as
    background × χ²(2) / 2, so the power threshold is background ×
    ``chi2.ppf(percentile, 2) / 2``.  A frequency is detected where power
    stays above its threshold for at least *min_cycles* cycles; a sample is
    oscillating when any *band_hz* frequency is detected.

    Returns a dict:

    ``freqs``, ``mean_power``, ``background``, ``threshold``
        Per frequency across *fit_hz*.
    ``band_freqs``, ``band_ratio``, ``detected``
        The *band_hz* frequencies, power ÷ background (rows × samples), and
        the detection mask.
    ``oscillating``
        Boolean per sample.
    ``episodes``
        EpochTable of contiguous oscillating runs, with ``frequency_hz`` (the
        band frequency detected for the most samples, weighted by power ÷
        background) and ``confirmed_s`` (``start_s`` + *min_cycles* cycles at
        that frequency, capped at ``end_s``: the earliest moment the episode
        could be known).
    """
    t = np.asarray(t, dtype=float)
    n_octaves = np.log2(fit_hz[1] / fit_hz[0])
    freqs = fit_hz[0] * 2.0 ** (np.arange(int(round(n_octaves * freqs_per_octave)) + 1) / freqs_per_octave)
    power = morlet_power(x, fs, freqs, width=width)
    reference = np.ones(t.size, dtype=bool) if reference is None else np.asarray(reference, dtype=bool)
    mean_power = power[:, reference].mean(axis=1) if reference.any() else power.mean(axis=1)
    in_band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    slope, intercept = np.polyfit(np.log(freqs[~in_band]), np.log(mean_power[~in_band]), 1)
    background = np.exp(intercept + slope * np.log(freqs))
    threshold = background * chi2.ppf(percentile, 2) / 2
    band_freqs = freqs[in_band]
    ratio = power[in_band] / background[in_band, None]
    detected = np.zeros(ratio.shape, dtype=bool)
    for row, freq in enumerate(band_freqs):
        runs = detect_epochs(power[in_band][row] > threshold[in_band][row], t, event_type="cycle_run",
                             min_duration_s=min_cycles / freq)
        for start, end in zip(runs[START_IDX].astype(int), runs[END_IDX].astype(int)):
            detected[row, start:end + 1] = True
    oscillating = detected.any(axis=0)
    episodes = detect_epochs(oscillating, t, event_type="oscillation_episode")
    pairs = list(zip(episodes[START_IDX].astype(int), episodes[END_IDX].astype(int)))
    episodes["frequency_hz"] = [float(band_freqs[np.argmax((detected[:, a:b + 1] * ratio[:, a:b + 1]).sum(axis=1))])
                                for a, b in pairs]
    episodes["confirmed_s"] = np.minimum(episodes[START_S] + min_cycles / episodes["frequency_hz"], episodes[END_S])
    return {"freqs": freqs, "mean_power": mean_power, "background": background, "threshold": threshold,
            "band_freqs": band_freqs, "band_ratio": ratio, "detected": detected,
            "oscillating": oscillating, "episodes": episodes}


# ── Session wrappers ───────────────────────────────────────────────────────


def oscillation_events(
    sessions: "SessionGroup | list[Session]",
    detector: OscillationDetector,
    *,
    event_types: tuple[str, ...] = ("onset", "offset"),
) -> "EventsTable":
    """Burst onsets and offsets across sessions, as a point-event table.

    The oscillation counterpart of
    :func:`~databench.analysis.locomotion.locomotion_events`: a SessionGroup
    wrapper that runs *detector* on each session and returns ``Subject``, ``Session``, ``Task``,
    ``EventType`` and ``event_time`` rows ready for
    :meth:`~databench.analysis.eta.EtaAnalysis.run`.  ``"onset"`` is a burst
    ``start_s`` and ``"offset"`` its ``end_s``.
    """
    columns = {"onset": START_S, "offset": END_S}
    unknown = set(event_types) - set(columns)
    if unknown:
        raise ValueError(f"Unknown oscillation event type(s) {sorted(unknown)}. Expected {sorted(columns)}.")
    tables = []
    for sess in sessions:
        events = detector.run(sess).events
        if events.empty:
            continue
        tables.append(make_events(
            {etype: events[columns[etype]].to_numpy() for etype in event_types},
            subject=sess.subject, session=sess.session, task=sess.task,
        ))
    if not tables:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    return pd.concat(tables, ignore_index=True)


# ── OscillationResult ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class OscillationResult:
    """Result of oscillation detection on a single session.

    Attributes
    ----------
    events : pd.DataFrame
        Burst events table with columns: ``start_s``, ``end_s``,
        ``duration_s``, ``peak_env``, ``frequency_hz`` (median instantaneous
        frequency of the filtered signal within the burst).
    time : np.ndarray
        Time array.
    raw_signal : np.ndarray
        Original signal.
    filtered_signal : np.ndarray
        Bandpass-filtered signal.
    envelope : np.ndarray
        Hilbert envelope.
    threshold_value : float
        Threshold used for detection.
    bursts : list of (int, int)
        Sample-index pairs for each burst.
    subject, session, task : str
        Session identifiers.
    source, signal_name : str
        Source and signal that were analyzed.
    band_hz : (float, float)
        Bandpass frequency range.
    fs : float
        Sampling rate.
    """

    # Core detection outputs
    events: pd.DataFrame
    time: np.ndarray
    raw_signal: np.ndarray
    filtered_signal: np.ndarray
    envelope: np.ndarray
    threshold_value: float
    bursts: list[Tuple[int, int]]

    # Context
    subject: str
    session: str
    task: str
    source: str
    signal_name: str
    band_hz: Tuple[float, float]
    fs: float

    # Private
    _session: "Session" = field(repr=False)

    # ── Plotting ───────────────────────────────────────────────────────────

    def plot_overview(
        self,
        *,
        aligned: "AlignedData | None" = None,
        pupil: str | None = None,
        speed: str | None = None,
        smooth_pupil_s: float = 0.5,
        smooth_speed_s: float = 0.2,
        window: Tuple[float, float] | None = None,
    ):
        """Plot a multi-panel oscillation overview.

        Returns a matplotlib ``Figure``.  Persist via ``run.save_figure(...)``.
        """
        from databench.plotting.oscillation import plot_oscillation_overview

        pupil_arr = None
        speed_arr = None
        aligned_t = None
        if aligned is not None:
            aligned_t = aligned.df[aligned.time_column].to_numpy()
            if pupil and pupil in aligned.df.columns:
                pupil_arr = aligned.df[pupil].to_numpy()
            if speed and speed in aligned.df.columns:
                speed_arr = aligned.df[speed].to_numpy()

        fig = plot_oscillation_overview(
            time=self.time,
            raw_signal=self.raw_signal,
            filtered_signal=self.filtered_signal,
            envelope=self.envelope,
            threshold_value=self.threshold_value,
            bursts=self.bursts,
            signal_name=self.signal_name,
            band_hz=self.band_hz,
            subject=self.subject,
            session=self.session,
            task=self.task,
            aligned_time=aligned_t,
            pupil=pupil_arr,
            speed=speed_arr,
            smooth_pupil_s=smooth_pupil_s,
            smooth_speed_s=smooth_speed_s,
            window=window,
        )
        return fig

    def plot_bursts(
        self,
        *,
        aligned: "AlignedData | None" = None,
        max_examples: int = 12,
        pad_s: float = 5.0,
        fs: float | None = None,
        pupil: str | None = None,
        speed: str | None = None,
        smooth_pupil_s: float = 0.5,
        smooth_speed_s: float = 0.2,
    ) -> list:
        """Plot zoomed windows around the longest bursts.

        Returns a list of matplotlib ``Figure`` objects (longest burst first).
        Persist via ``run.save_figure(...)``.
        """
        from databench.plotting.oscillation import plot_oscillation_burst

        if not self.bursts:
            return []

        actual_fs = fs or self.fs

        pupil_arr = None
        speed_arr = None
        aligned_t = None
        if aligned is not None:
            aligned_t = aligned.df[aligned.time_column].to_numpy()
            if pupil and pupil in aligned.df.columns:
                pupil_arr = aligned.df[pupil].to_numpy()
            if speed and speed in aligned.df.columns:
                speed_arr = aligned.df[speed].to_numpy()

        # Sort by duration (longest first)
        burst_order = sorted(
            range(len(self.bursts)),
            key=lambda i: self.bursts[i][1] - self.bursts[i][0],
            reverse=True,
        )
        n_examples = min(max_examples, len(burst_order))

        figures = []
        for bi in burst_order[:n_examples]:
            fig = plot_oscillation_burst(
                time=self.time,
                raw_signal=self.raw_signal,
                filtered_signal=self.filtered_signal,
                envelope=self.envelope,
                threshold_value=self.threshold_value,
                bursts=self.bursts,
                signal_name=self.signal_name,
                band_hz=self.band_hz,
                subject=self.subject,
                session=self.session,
                task=self.task,
                burst_idx=bi,
                fs=actual_fs,
                pad_s=pad_s,
                aligned_time=aligned_t,
                pupil=pupil_arr,
                speed=speed_arr,
                smooth_pupil_s=smooth_pupil_s,
                smooth_speed_s=smooth_speed_s,
            )
            figures.append(fig)
        return figures

    # ── Plotter factories ──────────────────────────────────────────────────

    def overview_plotter(self, **kwargs):
        """Return an :class:`OscillationOverviewPlotter` configured for this result."""
        from databench.plotting.oscillation import OscillationOverviewPlotter
        return OscillationOverviewPlotter(**kwargs)

    def burst_plotter(self, **kwargs):
        """Return an :class:`OscillationBurstPlotter` configured for this result."""
        from databench.plotting.oscillation import OscillationBurstPlotter
        return OscillationBurstPlotter(**kwargs)
