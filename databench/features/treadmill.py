from __future__ import annotations

from dataclasses import dataclass
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
        ps, pe = merged[-1]
        gap = float(t[s] - t[pe])
        if gap <= min_gap_s:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def _apply_min_duration_by_time(
    segments: Iterable[Tuple[int, int]],
    t: np.ndarray,
    min_duration_s: float,
) -> list[Tuple[int, int]]:
    dt = float(np.nanmedian(np.diff(t))) if t.size > 1 else 0.0
    keep = []
    for s, e in segments:
        duration = float(t[e] - t[s] + dt)
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


def detect_locomotion_bouts(
    t: np.ndarray,
    speed_cms: np.ndarray,
    *,
    min_speed_cms: float = 0.5,
    min_duration_s: float = 1.0,
    merge_gap_s: float = 0.5,
) -> list[Tuple[int, int]]:
    """Detect locomotion bouts from a speed trace in cm/s."""
    return _locomotion_bouts(
        t,
        speed_cms,
        min_speed_cms=min_speed_cms,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
    )


def _bout_stats(
    t: np.ndarray,
    speed_cms: np.ndarray,
    bouts: Iterable[Tuple[int, int]],
) -> tuple[list[float], list[float], list[float]]:
    mean_speeds: list[float] = []
    distances_m: list[float] = []
    durations_s: list[float] = []
    dt_med = float(np.nanmedian(np.diff(t))) if t.size > 1 else 0.0
    for s, e in bouts:
        s = int(s)
        e = int(e)
        if e < s:
            continue
        duration = float(t[e] - t[s] + dt_med)
        durations_s.append(duration)
        mean_speeds.append(float(np.nanmean(np.abs(speed_cms[s : e + 1]))))
        if e == s:
            distances_m.append(0.0)
            continue
        dt = np.diff(t[s : e + 1])
        dist_cm = float(np.nansum(speed_cms[s + 1 : e + 1] * dt))
        distances_m.append(dist_cm / 100.0)
    return mean_speeds, distances_m, durations_s


@register_feature
@dataclass(frozen=True)
class MeanSpeedCMS(FeatureFn):
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
        if t is None:
            return np.nan
        return float(np.nanmean(spd_mm) / 10.0)


@register_feature
@dataclass(frozen=True)
class StdSpeedCMS(FeatureFn):
    name: str = "speed_std_cms"
    label: str = "Speed SD (cm/s)"
    color: str = SOURCE_COLOR
    source: str = "treadmill"

    def _run_impl(self, row) -> float:
        t, spd_mm = clean_xy(
            get_first(row, [("treadmill", "time_elapsed_s"), ("encoder", "time_elapsed_s")]),
            get_first(row, [("treadmill", "speed_mm"), ("encoder", "speed")]),
        )
        if t is None:
            return np.nan
        return float(np.nanstd(spd_mm) / 10.0)


@register_feature
@dataclass(frozen=True)
class TotalDistanceM(FeatureFn):
    name: str = "distance_m"
    label: str = "Distance (m)"
    color: str = SOURCE_COLOR
    source: str = "treadmill"
    plotter: str = "boxplot"

    def _run_impl(self, row) -> float:
        dist_mm = as_1d(get_first(row, [("treadmill", "distance_mm"), ("encoder", "distance")]))
        if dist_mm is not None and dist_mm.size >= 2:
            dist_mm = dist_mm[np.isfinite(dist_mm)]
            if dist_mm.size >= 2:
                return float((dist_mm[-1] - dist_mm[0]) / 1000.0)

        t, spd_mm = clean_xy(
            get_first(row, [("treadmill", "time_elapsed_s"), ("encoder", "time_elapsed_s")]),
            get_first(row, [("treadmill", "speed_mm"), ("encoder", "speed")]),
        )
        if t is None:
            return np.nan
        dt = np.diff(t)
        if dt.size == 0:
            return np.nan
        dt = np.clip(dt, 0, np.nanpercentile(dt, 99))
        return float(np.nansum(spd_mm[1:] * dt) / 1000.0)


@dataclass(frozen=True)
class LocomotionBoutFeature(FeatureFn):
    color: str = SOURCE_COLOR
    source: str = "treadmill"
    plotter: str = "boxplot"
    min_speed_cms: float = 5
    min_duration_s: float = 1.5
    merge_gap_s: float = 1.5

    def _get_bouts(self, row):
        t, spd_mm = clean_xy(
            get_first(row, [("treadmill", "time_elapsed_s"), ("encoder", "time_elapsed_s")]),
            get_first(row, [("treadmill", "speed_mm"), ("encoder", "speed")]),
        )
        if t is None:
            return None
        speed_cms = spd_mm / 10.0
        bouts = _locomotion_bouts(
            t,
            speed_cms,
            min_speed_cms=self.min_speed_cms,
            min_duration_s=self.min_duration_s,
            merge_gap_s=self.merge_gap_s,
        )
        return t, speed_cms, bouts


@register_feature
@dataclass(frozen=True)
class LocomotionBoutsCount(LocomotionBoutFeature):
    name: str = "locomotion_bouts_n"
    label: str = "Locomotion bouts (n)"

    def _run_impl(self, row) -> float:
        out = self._get_bouts(row)
        if out is None:
            return np.nan
        _, _, bouts = out
        return float(len(bouts))


@register_feature
@dataclass(frozen=True)
class LocomotionBoutSpeedMeanCMS(LocomotionBoutFeature):
    name: str = "locomotion_bout_speed_mean_cms"
    label: str = "Bout speed (cm/s)"

    def _run_impl(self, row) -> float:
        out = self._get_bouts(row)
        if out is None:
            return np.nan
        t, speed_cms, bouts = out
        if not bouts:
            return np.nan
        mean_speeds, _, _ = _bout_stats(t, speed_cms, bouts)
        return float(np.nanmean(mean_speeds)) if mean_speeds else np.nan


@register_feature
@dataclass(frozen=True)
class LocomotionBoutDistanceM(LocomotionBoutFeature):
    name: str = "locomotion_bout_distance_m"
    label: str = "Bout distance (m)"

    def _run_impl(self, row) -> float:
        out = self._get_bouts(row)
        if out is None:
            return np.nan
        t, speed_cms, bouts = out
        if not bouts:
            return np.nan
        _, distances_m, _ = _bout_stats(t, speed_cms, bouts)
        return float(np.nanmean(distances_m)) if distances_m else np.nan


@register_feature
@dataclass(frozen=True)
class LocomotionBoutDurationS(LocomotionBoutFeature):
    name: str = "locomotion_bout_duration_s"
    label: str = "Bout duration (s)"

    def _run_impl(self, row) -> float:
        out = self._get_bouts(row)
        if out is None:
            return np.nan
        t, speed_cms, bouts = out
        if not bouts:
            return np.nan
        _, _, durations_s = _bout_stats(t, speed_cms, bouts)
        return float(np.nanmean(durations_s)) if durations_s else np.nan
