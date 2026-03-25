from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any, Iterable, Mapping, Tuple
import numpy as np
import pandas as pd

from databench.features.base import FeatureFn
from databench.registry import register_feature
from databench.utils import clean_xy, as_1d, get_first

SOURCE_COLOR = "#2ca02c"


def _segments_from_mask(mask: np.ndarray) -> list[Tuple[int, int]]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    segments = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segments.append((start, prev))
            start = i
            prev = i
    segments.append((start, prev))
    return segments


def _merge_gaps_by_time(
    segments: Iterable[Tuple[int, int]],
    t: np.ndarray,
    min_gap_s: float,
) -> list[Tuple[int, int]]:
    merged: list[Tuple[int, int]] = []
    for s, e in segments:
        if not merged:
            merged.append((s, e))
            continue
        prev_start, prev_end = merged[-1]
        gap = float(t[s] - t[prev_end])
        if gap <= min_gap_s:
            merged[-1] = (prev_start, e)
        else:
            merged.append((s, e))
    return merged


def _apply_min_duration_by_time(
    segments: Iterable[Tuple[int, int]],
    t: np.ndarray,
    min_duration_s: float,
) -> list[Tuple[int, int]]:
    sample_interval = float(np.nanmedian(np.diff(t))) if t.size > 1 else 0.0
    keep = []
    for s, e in segments:
        duration = float(t[e] - t[s] + sample_interval)
        if duration >= min_duration_s:
            keep.append((s, e))
    return keep


def _locomotion_bouts(
    t: np.ndarray,
    speed_cms: np.ndarray,
    *,
    min_speed_cms: float,
    min_duration_s: float,
    merge_gap_s: float,
) -> list[Tuple[int, int]]:
    mask = speed_cms >= min_speed_cms
    segments = _segments_from_mask(mask)
    if not segments:
        return []
    if merge_gap_s > 0:
        segments = _merge_gaps_by_time(segments, t, merge_gap_s)
    if min_duration_s > 0:
        segments = _apply_min_duration_by_time(segments, t, min_duration_s)
    return segments


def locomotion_bout_events(
    t: np.ndarray | pd.DataFrame,
    speed_cms: np.ndarray | None = None,
    *,
    min_speed_cms: float = 0.5,
    min_duration_s: float = 1.0,
    merge_gap_s: float = 0.5,
    as_table: bool = False,
    context: Mapping[str, Any] | None = None,
    group_cols: tuple[str, ...] = ("Subject", "Session", "Task"),
    time_col: str = "time_elapsed_s",
    speed_col: str = "speed_mm",
    speed_scale_to_cms: float = 10.0,
) -> tuple[list[Tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray] | pd.DataFrame:
    """
    Return bouts with onset/offset indices and times.

    Parameters
    ----------
    as_table : bool
        If True, return a DataFrame with one row per bout and columns:
        ['bout_id', 'onset_idx', 'offset_idx', 'onset_t', 'offset_t'].
        If False (default), return the tuple
        (bouts, onset_idx, offset_idx, onset_t, offset_t).
    context : mapping[str, Any] | None
        Optional constant columns to append to each returned bout row when
        `as_table=True` (e.g., Subject/Session/Task identifiers).
    group_cols, time_col, speed_col, speed_scale_to_cms
        Used only when `t` is a pandas DataFrame. The function groups by
        `group_cols`, reads per-group time/speed arrays from `time_col` and
        `speed_col`, converts speed to cm/s by dividing by `speed_scale_to_cms`,
        and returns one concatenated events table.
    """
    if isinstance(t, pd.DataFrame):
        tables: list[pd.DataFrame] = []
        for key, group in t.groupby(list(group_cols), sort=False):
            group = group.sort_values(time_col)
            t_raw = group[time_col].to_numpy()
            v_raw = group[speed_col].to_numpy()
            valid = np.isfinite(t_raw) & np.isfinite(v_raw)
            time_valid = t_raw[valid]
            speed_valid = v_raw[valid]
            if time_valid.size < 3:
                continue

            local_context = {col: val for col, val in zip(group_cols, key if isinstance(key, tuple) else (key,))}
            if context:
                local_context.update(dict(context))

            bout_table = locomotion_bout_events(
                time_valid,
                speed_valid / speed_scale_to_cms,
                min_speed_cms=min_speed_cms,
                min_duration_s=min_duration_s,
                merge_gap_s=merge_gap_s,
                as_table=True,
                context=local_context,
            )
            if bout_table.empty:
                continue
            tables.append(bout_table)

        base_cols = ["bout_id", "onset_idx", "offset_idx", "onset_t", "offset_t"]
        extra_cols = list(group_cols)
        if context:
            extra_cols.extend([c for c in context.keys() if c not in extra_cols])
        out_cols = base_cols + extra_cols
        out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=out_cols)
        for col in out_cols:
            if col not in out.columns:
                out[col] = np.nan
        return out[out_cols]

    bouts = _locomotion_bouts(
        t,
        speed_cms,
        min_speed_cms=min_speed_cms,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
    )
    onset_idx = np.array([s for s, _ in bouts], dtype=int)
    offset_idx = np.array([e for _, e in bouts], dtype=int)
    onset_t = t[onset_idx].astype(float)
    offset_t = t[offset_idx].astype(float)
    if as_table:
        table = pd.DataFrame(
            {
                "bout_id": np.arange(len(bouts), dtype=int),
                "onset_idx": onset_idx,
                "offset_idx": offset_idx,
                "onset_t": onset_t,
                "offset_t": offset_t,
            }
        )
        if context:
            for key, value in context.items():
                table[key] = value
        return table
    return bouts, onset_idx, offset_idx, onset_t, offset_t


@dataclass(frozen=True)
class LocomotionBoutEventsExtractor:
    """Reusable, explicit bout-event extractor for long-table pipelines."""

    min_speed_cms: float = 0.5
    min_duration_s: float = 1.0
    merge_gap_s: float = 0.5
    group_cols: tuple[str, ...] = ("Subject", "Session", "Task")
    time_col: str = "time_elapsed_s"
    speed_col: str = "speed_mm"
    speed_scale_to_cms: float = 10.0

    @classmethod
    def from_feature(
        cls,
        feat: "LocomotionBoutFeature",
        *,
        group_cols: tuple[str, ...] = ("Subject", "Session", "Task"),
        time_col: str = "time_elapsed_s",
        speed_col: str = "speed_mm",
        speed_scale_to_cms: float = 10.0,
    ) -> "LocomotionBoutEventsExtractor":
        """Build an extractor from a LocomotionBoutFeature's thresholds.

        Replaces the 7-parameter constructor boilerplate::

            bouts_feat = LocomotionBoutsCount()
            extractor = LocomotionBoutEventsExtractor.from_feature(bouts_feat)
        """
        return cls(
            min_speed_cms=feat.min_speed_cms,
            min_duration_s=feat.min_duration_s,
            merge_gap_s=feat.merge_gap_s,
            group_cols=group_cols,
            time_col=time_col,
            speed_col=speed_col,
            speed_scale_to_cms=speed_scale_to_cms,
        )

    def run(self, long: pd.DataFrame) -> pd.DataFrame:
        return locomotion_bout_events(
            long,
            min_speed_cms=self.min_speed_cms,
            min_duration_s=self.min_duration_s,
            merge_gap_s=self.merge_gap_s,
            as_table=True,
            group_cols=self.group_cols,
            time_col=self.time_col,
            speed_col=self.speed_col,
            speed_scale_to_cms=self.speed_scale_to_cms,
        )

    def __call__(self, long: pd.DataFrame) -> pd.DataFrame:
        return self.run(long)


def epoch_indices(
    t: np.ndarray,
    t0: float,
    *,
    window: tuple[float, float] = (-2.0, 1.0),
) -> Tuple[int, int] | None:
    """Return inclusive index bounds for an epoch; None if window exceeds data range."""
    t_start = t0 + window[0]
    t_end = t0 + window[1]
    i0 = int(np.searchsorted(t, t_start, side="left"))
    i1 = int(np.searchsorted(t, t_end, side="right") - 1)
    return i0, i1


def extract_epoch_interpolated(
    t: np.ndarray,
    y: np.ndarray,
    t0: float,
    *,
    window: tuple[float, float] = (-2.0, 1.0),
    dt: float = 0.02,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Interpolate an epoch onto a uniform grid; returns (rel_t, y_interp)."""
    valid = np.isfinite(t) & np.isfinite(y)
    t_valid = t[valid]
    y_valid = y[valid]
    bounds = epoch_indices(t_valid, t0, window=window)
    rel_t = np.arange(window[0], window[1] + 1e-12, dt)
    target_times = t0 + rel_t
    interpolated_values = np.interp(target_times, t_valid, y_valid)
    return rel_t, interpolated_values


def _bout_stats(
    t: np.ndarray,
    speed_cms: np.ndarray,
    bouts: Iterable[Tuple[int, int]],
) -> tuple[list[float], list[float], list[float]]:
    mean_speeds: list[float] = []
    distances_m: list[float] = []
    durations_s: list[float] = []
    sample_interval = float(np.nanmedian(np.diff(t))) if t.size > 1 else 0.0
    for s, e in bouts:
        s = int(s)
        e = int(e)
        duration = float(t[e] - t[s] + sample_interval)
        durations_s.append(duration)
        mean_speeds.append(float(np.nanmean(np.abs(speed_cms[s : e + 1]))))
        time_steps = np.diff(t[s : e + 1])
        dist_cm = float(np.nansum(speed_cms[s + 1 : e + 1] * time_steps))
        distances_m.append(dist_cm / 100.0)
    return mean_speeds, distances_m, durations_s


def _nanmean_no_warn(values) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(values))


@register_feature
@dataclass(frozen=True)
class MeanSpeedCMS(FeatureFn):
    """Mean treadmill speed per session in cm/s."""

    name: str = "speed_mean_cms"
    label: str = "Speed (cm/s)"
    color: str = SOURCE_COLOR
    source: str = "treadmill"
    plotter: str = "boxplot"

    def _run_impl(self, row) -> float:
        t, spd_mm = clean_xy(
            get_first(row, [("treadmill", "time_elapsed_s"), ("encoder", "time_elapsed_s")]),
            get_first(row, [("treadmill", "speed_mm"), ("encoder", "speed")]),
        )
        return float(np.nanmean(spd_mm) / 10.0)


@register_feature
@dataclass(frozen=True)
class StdSpeedCMS(FeatureFn):
    """Standard deviation of treadmill speed in cm/s."""

    name: str = "speed_std_cms"
    label: str = "Speed SD (cm/s)"
    color: str = SOURCE_COLOR
    source: str = "treadmill"

    def _run_impl(self, row) -> float:
        t, spd_mm = clean_xy(
            get_first(row, [("treadmill", "time_elapsed_s"), ("encoder", "time_elapsed_s")]),
            get_first(row, [("treadmill", "speed_mm"), ("encoder", "speed")]),
        )
        return float(np.nanstd(spd_mm) / 10.0)


@register_feature
@dataclass(frozen=True)
class TotalDistanceM(FeatureFn):
    """Total distance traveled in meters during the session."""

    name: str = "distance_m"
    label: str = "Distance (m)"
    color: str = SOURCE_COLOR
    source: str = "treadmill"
    plotter: str = "boxplot"

    def _run_impl(self, row) -> float:
        dist_mm = as_1d(get_first(row, [("treadmill", "distance_mm"), ("encoder", "distance")]))
        dist_mm = dist_mm[np.isfinite(dist_mm)]
        return float((dist_mm[-1] - dist_mm[0]) / 1000.0)


@dataclass(frozen=True)
class LocomotionBoutFeature(FeatureFn):
    """Base class for locomotion-bout features with shared thresholds."""

    color: str = SOURCE_COLOR
    source: str = "treadmill"
    plotter: str = "boxplot"
    min_speed_cms: float = 5
    min_duration_s: float = 1.5
    merge_gap_s: float = 1.5

    def _bout_params(self) -> dict[str, float]:
        return {
            "min_speed_cms": self.min_speed_cms,
            "min_duration_s": self.min_duration_s,
            "merge_gap_s": self.merge_gap_s,
        }

    def _get_bouts(self, row):
        t, spd_mm = clean_xy(
            get_first(row, [("treadmill", "time_elapsed_s"), ("encoder", "time_elapsed_s")]),
            get_first(row, [("treadmill", "speed_mm"), ("encoder", "speed")]),
        )
        speed_cms = spd_mm / 10.0
        bouts, _, _, _, _ = locomotion_bout_events(t, speed_cms, **self._bout_params())
        return t, speed_cms, bouts

    def _get_bout_events(self, row):
        out = self._get_bouts(row)
        t, speed_cms, _ = out
        return locomotion_bout_events(
            t,
            speed_cms,
            **self._bout_params(),
        )


@register_feature
@dataclass(frozen=True)
class LocomotionBoutsCount(LocomotionBoutFeature):
    """Count of locomotion bouts per session."""

    name: str = "locomotion_bouts_n"
    label: str = "Locomotion bouts (n)"

    def _run_impl(self, row) -> float:
        out = self._get_bouts(row)
        _, _, bouts = out
        return float(len(bouts))


@register_feature
@dataclass(frozen=True)
class LocomotionBoutSpeedMeanCMS(LocomotionBoutFeature):
    """Mean speed across locomotion bouts in cm/s."""

    name: str = "locomotion_bout_speed_mean_cms"
    label: str = "Bout speed (cm/s)"

    def _run_impl(self, row) -> float:
        out = self._get_bouts(row)
        t, speed_cms, bouts = out
        mean_speeds, _, _ = _bout_stats(t, speed_cms, bouts)
        return _nanmean_no_warn(mean_speeds)


@register_feature
@dataclass(frozen=True)
class LocomotionBoutDistanceM(LocomotionBoutFeature):
    """Mean distance across locomotion bouts in meters."""

    name: str = "locomotion_bout_distance_m"
    label: str = "Bout distance (m)"

    def _run_impl(self, row) -> float:
        out = self._get_bouts(row)
        t, speed_cms, bouts = out
        _, distances_m, _ = _bout_stats(t, speed_cms, bouts)
        return _nanmean_no_warn(distances_m)


@register_feature
@dataclass(frozen=True)
class LocomotionBoutDurationS(LocomotionBoutFeature):
    """Mean duration across locomotion bouts in seconds."""

    name: str = "locomotion_bout_duration_s"
    label: str = "Bout duration (s)"

    def _run_impl(self, row) -> float:
        out = self._get_bouts(row)
        t, speed_cms, bouts = out
        _, _, durations_s = _bout_stats(t, speed_cms, bouts)
        return _nanmean_no_warn(durations_s)
