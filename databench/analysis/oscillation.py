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
    result.plot_overview(aligned=aligned, pupil="pupil_diameter_mm").save("overview.svg")
    result.events.to_csv("bursts.csv")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from databench._utils._logger import get_logger, log_run
from databench.analysis._signal.bandpass import bandpass_envelope, robust_threshold
from databench.analysis._signal.epoching import detect_epochs, START_IDX, END_IDX

_log = get_logger(__name__)

from databench.config import OutputContext


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
                _context=session._context,
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
            _context=session._context,
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
    _context: OutputContext = field(repr=False)
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
    ) -> "SaveableFigure":
        """Plot a multi-panel oscillation overview.

        Parameters
        ----------
        aligned : AlignedData, optional
            Time-aligned auxiliary data (for pupil / speed traces).
        pupil : str, optional
            Column name for pupil signal in *aligned*.
        speed : str, optional
            Column name for speed signal in *aligned*.
        smooth_pupil_s, smooth_speed_s : float
            Smoothing window in seconds.
        window : (float, float), optional
            Time window to zoom into. ``None`` shows full session.

        Returns
        -------
        SaveableFigure
            Wraps the matplotlib Figure with a ``.save()`` method.
        """
        from databench.session import SaveableFigure
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
        return SaveableFigure(fig, self._context)

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
    ) -> list["SaveableFigure"]:
        """Plot zoomed windows around the longest bursts.

        Parameters
        ----------
        aligned : AlignedData, optional
            Time-aligned auxiliary data.
        max_examples : int
            Maximum number of burst examples to plot.
        pad_s : float
            Seconds of context around each burst.
        fs : float, optional
            Sampling rate override (defaults to ``self.fs``).
        pupil, speed : str, optional
            Column names in *aligned*.

        Returns
        -------
        list of SaveableFigure
        """
        from databench.session import SaveableFigure
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
        for rank, bi in enumerate(burst_order[:n_examples], start=1):
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
            sf = SaveableFigure(fig, self._context)
            sf.save(f"burst_{rank:02d}.svg")
            figures.append(sf)
        return figures

    # ── Saving ─────────────────────────────────────────────────────────────

    def save_events(self, name: str = "bursts.csv") -> Path:
        """Save the burst events table as CSV.

        Parameters
        ----------
        name : str
            Filename for the CSV.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        out_dir = self._context.stats_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        self.events.to_csv(path, index=False)
        return path

    def save_summary(self, name: str = "summary.json") -> Path:
        """Save a JSON summary of the detection run.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        out_dir = self._context.run_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name

        summary = {
            "analysis": "oscillation_detection",
            "created_at": datetime.now().isoformat(),
            "analyst": self._context.analyst,
            "lab": self._context.lab,
            "run_name": self._context.run_name,
            "tag": self._context.tag,
            "subject": self.subject,
            "session": self.session,
            "task": self.task,
            "source": self.source,
            "signal": self.signal_name,
            "band_hz": list(self.band_hz),
            "fs": self.fs,
            "threshold": self.threshold_value,
            "n_bursts": len(self.bursts),
            "total_burst_duration_s": float(self.events["duration_s"].sum()) if not self.events.empty else 0.0,
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path

    # ── Reporting ──────────────────────────────────────────────────────────

    def _report_section(self) -> "ReportSection":
        """Build a :class:`~databench._reporting.ReportSection` for this result."""
        from databench._reporting import ReportSection

        total = float(self.events["duration_s"].sum()) if not self.events.empty else 0.0
        notes = (
            f"Detected **{len(self.bursts)} bursts** "
            f"(total {total:.1f} s) in `{self.source}/{self.signal_name}` "
            f"for {self.subject} / {self.session} / {self.task}.\n\n"
            f"Bandpass {self.band_hz[0]}–{self.band_hz[1]} Hz, "
            f"threshold {self.threshold_value:.5f}, fs {self.fs} Hz."
        )
        params = {
            "source": self.source,
            "signal": self.signal_name,
            "band_hz": self.band_hz,
            "fs": self.fs,
            "threshold": self.threshold_value,
            "n_bursts": len(self.bursts),
            "total_burst_duration_s": round(total, 2),
        }
        # Collect any figures/tables already saved
        figures = sorted(self._context.plots_dir.glob("*.svg")) + sorted(self._context.plots_dir.glob("*.png"))
        tables = sorted(self._context.stats_dir.glob("*.csv"))
        return ReportSection(
            heading="Oscillation Detection",
            params=params,
            notes=notes,
            figures=figures,
            tables=tables,
        )
