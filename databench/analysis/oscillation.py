"""Oscillation detector and result — public analysis API.

Usage::

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
from typing import Tuple

import numpy as np
import pandas as pd

from databench.utils.logger import get_logger, log_run
from databench.signal.bandpass import bandpass_envelope, robust_threshold
from databench.signal.epoching import detect_epochs, START_IDX, END_IDX

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
            return OscillationResult(
                events=pd.DataFrame(),
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

        # Detect bursts via unified epoch pipeline
        mask = env > thr
        events = detect_epochs(
            mask, t,
            event_type="oscillation_burst",
            min_duration_s=self.min_duration_s,
            merge_gap_s=self.merge_gap_s,
        )
        # Oscillation-specific: peak envelope within each burst
        events["peak_env"] = [
            float(env[int(s):int(e) + 1].max())
            for s, e in zip(events[START_IDX], events[END_IDX])
        ] if not events.empty else []

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


# ── OscillationResult ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class OscillationResult:
    """Result of oscillation detection on a single session.

    Attributes
    ----------
    events : pd.DataFrame
        Burst events table with columns: ``start_s``, ``end_s``,
        ``duration_s``, ``peak_env``.
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
