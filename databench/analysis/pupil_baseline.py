"""Pupil baseline normalization using session-1 quiescent epochs as anchor.

For each subject, derive a single baseline pupil diameter (median across all
samples falling inside quiescent epochs of ``ses-01``), then express pupil
values from every other session as a dimensionless ratio relative to that
baseline.

The intent is to make pupil traces comparable within-subject across
sessions when the camera setup is not calibrated to a physical reference,
and to remove the systematic between-animal offsets that come from
arbitrary mm scaling.

Public API
----------
* :func:`compute_session1_baselines` — one row per subject.
* :func:`apply_baseline_normalization` — long-format pupil_raw / pupil_norm.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from databench.analysis.locomotion import locomotion_bouts, quiescent_bouts
from databench.utils.logger import get_logger

if TYPE_CHECKING:
    from databench.session import Session, SessionGroup

_log = get_logger(__name__)

__all__ = [
    "compute_session1_baselines",
    "apply_baseline_normalization",
]


# ── Helpers ────────────────────────────────────────────────────────────────


def _quiescent_pupil_samples(
    sess: "Session",
    *,
    pupil_source: str,
    pupil_column: str,
    speed_source: str,
    speed_column: str,
    speed_scale_to_cms: float,
    quiescence_speed_cms: float,
    min_quiescent_duration_s: float,
    tolerance_s: float = 0.25,
) -> tuple[np.ndarray, int]:
    """Return (pupil samples inside quiescent epochs, number of epochs used).

    Pupil is merge_asof-aligned onto the treadmill timebase, so the bout
    indices computed from speed directly index the aligned pupil column.
    """
    ad = sess.align(
        {speed_source: [speed_column], pupil_source: [pupil_column]},
        reference=speed_source,
        tolerance_s=tolerance_s,
    )
    df = ad.df
    if df.empty or pupil_column not in df.columns or speed_column not in df.columns:
        return np.array([]), 0

    t = df["time_elapsed_s"].to_numpy(dtype=float)
    speed_cms = df[speed_column].to_numpy(dtype=float) / speed_scale_to_cms
    pupil = df[pupil_column].to_numpy(dtype=float)

    # locomotion_bouts needs a clean, sorted, monotonic time vector with a
    # matching speed array. Pupil is allowed to have NaNs (blink artifacts
    # etc.) — those are dropped after epoch detection.
    valid = np.isfinite(t) & np.isfinite(speed_cms)
    if valid.sum() < 3:
        return np.array([]), 0
    t = t[valid]
    speed_cms = speed_cms[valid]
    pupil = pupil[valid]

    order = np.argsort(t)
    t = t[order]
    speed_cms = speed_cms[order]
    pupil = pupil[order]

    loco = locomotion_bouts(
        t,
        speed_cms,
        min_speed_cms=quiescence_speed_cms,
        min_duration_s=1.0,
        merge_gap_s=0.5,
    )
    quiet = quiescent_bouts(t, loco, min_duration_s=min_quiescent_duration_s)
    if not quiet:
        return np.array([]), 0

    parts: list[np.ndarray] = []
    for s, e in quiet:
        seg = pupil[s : e + 1]
        parts.append(seg[np.isfinite(seg)])

    if not parts:
        return np.array([]), 0
    return np.concatenate(parts), len(quiet)


# ── Public API ─────────────────────────────────────────────────────────────


def compute_session1_baselines(
    sessions: "SessionGroup | list[Session]",
    *,
    pupil_source: str = "pupil",
    pupil_column: str = "pupil_diameter_mm",
    speed_source: str = "treadmill",
    speed_column: str = "speed_mm",
    speed_scale_to_cms: float = 10.0,
    quiescence_speed_cms: float = 0.5,
    min_quiescent_duration_s: float = 2.0,
    baseline_session: str = "ses-01",
) -> pd.DataFrame:
    """Return one row per subject: Subject, baseline_pupil, n_samples, n_epochs.

    The baseline is the *median* raw pupil diameter across all samples that
    fall inside a quiescent epoch (treadmill speed < ``quiescence_speed_cms``
    for at least ``min_quiescent_duration_s``) within ``baseline_session``.
    Median (not mean) keeps the baseline robust to blink artifacts.

    Raises
    ------
    ValueError
        If any subject in *sessions* is missing ``baseline_session`` entirely.
        This is a real data-availability problem, not an edge case worth
        silently skipping.
    """
    session_list = list(sessions)
    subjects_all = sorted({s.subject for s in session_list})
    subjects_with_baseline = {
        s.subject for s in session_list if s.session == baseline_session
    }
    missing = sorted(set(subjects_all) - subjects_with_baseline)
    if missing:
        raise ValueError(
            f"Subjects missing baseline session {baseline_session!r}: {missing}. "
            "Cannot compute pupil baselines without a session-1 anchor."
        )

    samples_by_subject: dict[str, list[np.ndarray]] = {}
    epochs_by_subject: dict[str, int] = {}

    for sess in session_list:
        if sess.session != baseline_session:
            continue
        try:
            pupil_samples, n_epochs = _quiescent_pupil_samples(
                sess,
                pupil_source=pupil_source,
                pupil_column=pupil_column,
                speed_source=speed_source,
                speed_column=speed_column,
                speed_scale_to_cms=speed_scale_to_cms,
                quiescence_speed_cms=quiescence_speed_cms,
                min_quiescent_duration_s=min_quiescent_duration_s,
            )
        except Exception as exc:
            _log.warning(f"Baseline extraction failed for {sess.label}: {exc!r}")
            continue

        samples_by_subject.setdefault(sess.subject, []).append(pupil_samples)
        epochs_by_subject[sess.subject] = (
            epochs_by_subject.get(sess.subject, 0) + n_epochs
        )

    rows: list[dict] = []
    for subj in sorted(subjects_with_baseline):
        sample_arrays = samples_by_subject.get(subj, [])
        pooled = (
            np.concatenate(sample_arrays) if sample_arrays else np.array([])
        )
        n_samples = int(pooled.size)
        n_epochs = int(epochs_by_subject.get(subj, 0))
        baseline = float(np.median(pooled)) if n_samples > 0 else float("nan")

        if n_samples < 100:
            warnings.warn(
                f"Subject {subj}: only {n_samples} quiescent pupil samples in "
                f"{baseline_session}; baseline may be unreliable.",
                stacklevel=2,
            )
        if n_epochs < 3:
            warnings.warn(
                f"Subject {subj}: only {n_epochs} quiescent epochs used in "
                f"{baseline_session}; baseline may be unreliable.",
                stacklevel=2,
            )

        rows.append({
            "Subject": subj,
            "baseline_pupil": baseline,
            "n_samples": n_samples,
            "n_quiescent_epochs_used": n_epochs,
        })

    return pd.DataFrame(rows)


def apply_baseline_normalization(
    sessions: "SessionGroup | list[Session]",
    baselines: pd.DataFrame,
    *,
    pupil_source: str = "pupil",
    pupil_column: str = "pupil_diameter_mm",
    time_column: str = "time_elapsed_s",
    **_unused,  # accept the same source/column kwargs as compute_session1_baselines
) -> pd.DataFrame:
    """Return a long-format table of normalized pupil for every session.

    Columns: ``Subject``, ``Session``, ``Task``, ``time_elapsed_s``,
    ``pupil_raw``, ``pupil_norm``.  ``pupil_norm = pupil_raw / baseline``
    where *baseline* is looked up per subject from *baselines*.  Sessions
    whose subject has no valid baseline are skipped with a warning.
    """
    if "Subject" not in baselines.columns or "baseline_pupil" not in baselines.columns:
        raise ValueError(
            "baselines DataFrame must have 'Subject' and 'baseline_pupil' "
            "columns (produced by compute_session1_baselines)."
        )

    baseline_map: dict[str, float] = {
        str(row["Subject"]): float(row["baseline_pupil"])
        for _, row in baselines.iterrows()
    }

    frames: list[pd.DataFrame] = []
    for sess in sessions:
        base = baseline_map.get(sess.subject)
        if base is None or not np.isfinite(base) or base <= 0:
            _log.warning(
                f"No usable baseline for subject {sess.subject!r}; "
                f"skipping {sess.label}"
            )
            continue
        try:
            t = sess.time(pupil_source, time_column)
            p = sess.signal(pupil_source, pupil_column)
        except Exception as exc:
            _log.warning(f"Could not read pupil for {sess.label}: {exc!r}")
            continue

        m = min(t.size, p.size)
        if m == 0:
            continue
        t = np.asarray(t[:m], dtype=float)
        p = np.asarray(p[:m], dtype=float)
        finite = np.isfinite(t) & np.isfinite(p)
        if not finite.any():
            continue

        frames.append(pd.DataFrame({
            "Subject": sess.subject,
            "Session": sess.session,
            "Task": sess.task,
            "time_elapsed_s": t[finite],
            "pupil_raw": p[finite],
            "pupil_norm": p[finite] / base,
        }))

    if not frames:
        return pd.DataFrame(columns=[
            "Subject", "Session", "Task",
            "time_elapsed_s", "pupil_raw", "pupil_norm",
        ])
    return pd.concat(frames, ignore_index=True)
