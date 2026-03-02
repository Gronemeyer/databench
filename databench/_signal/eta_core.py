"""Core event-triggered average computation."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from databench._signal.epochs import extract_epoch_interpolated


def eta_baselined(
    df: pd.DataFrame,
    event_times: np.ndarray,
    roi_columns: Sequence[str],
    *,
    time_column: str = "time_elapsed_s",
    window: tuple[float, float] = (-2.0, 2.0),
    dt: float = 0.05,
    baseline: tuple[float, float] = (-2.0, -1.0),
    bout_intervals: Optional[np.ndarray] = None,
    exclude_events_in_bouts: bool = True,
    baseline_exclude_bouts: bool = False,
    min_clean_baseline_points: int = 3,
    fallback_to_full_baseline: bool = True,
) -> pd.DataFrame:
    """Compute baseline-subtracted ETA traces for each event × ROI.

    Parameters
    ----------
    df : DataFrame
        Long-format table with *time_column* and all *roi_columns*.
    event_times : array
        1-D array of event timestamps.
    roi_columns : sequence of str
        Columns to extract per-event epochs from.
    time_column : str
        Name of the time column.
    window : (float, float)
        Peri-event window in seconds.
    dt : float
        Interpolation step size in seconds.
    baseline : (float, float)
        Baseline window for subtraction.

    Returns
    -------
    DataFrame
        Columns: ``event_id``, ``rel_time``, ``ROI``, ``value``.
    """
    df = df.sort_values(time_column)
    time_values = df[time_column].to_numpy()

    # Optionally filter events inside bout intervals
    if exclude_events_in_bouts and bout_intervals is not None and len(bout_intervals) > 0:
        keep = np.ones(len(event_times), dtype=bool)
        for i, et in enumerate(event_times):
            for onset, offset in bout_intervals:
                if onset <= et <= offset:
                    keep[i] = False
                    break
        event_times = event_times[keep]

    output_frames: list[pd.DataFrame] = []
    for event_id, event_time in enumerate(event_times):
        for roi in roi_columns:
            if roi not in df.columns:
                continue  # ROI not available for this session — skip
            roi_values = df[roi].to_numpy()
            rel_t, roi_epoch = extract_epoch_interpolated(
                time_values, roi_values, event_time, window=window, dt=dt,
            )
            if rel_t is None:
                continue

            baseline_mask_full = (rel_t >= baseline[0]) & (rel_t <= baseline[1])
            baseline_mask = baseline_mask_full.copy()

            if baseline_exclude_bouts and bout_intervals is not None and len(bout_intervals) > 0:
                abs_t = event_time + rel_t
                in_bout = np.zeros(abs_t.shape, dtype=bool)
                for onset_t, offset_t in bout_intervals:
                    in_bout |= (abs_t >= float(onset_t)) & (abs_t <= float(offset_t))
                baseline_mask = baseline_mask & ~in_bout

                clean_count = int(np.sum(baseline_mask & np.isfinite(roi_epoch)))
                if clean_count < min_clean_baseline_points and fallback_to_full_baseline:
                    baseline_mask = baseline_mask_full

            baseline_value = (
                np.nanmean(roi_epoch[baseline_mask])
                if baseline_mask.any()
                else np.nan
            )
            roi_epoch = roi_epoch - baseline_value

            output_frames.append(
                pd.DataFrame(
                    {
                        "event_id": event_id,
                        "rel_time": rel_t,
                        "ROI": roi,
                        "value": roi_epoch,
                    }
                )
            )

    if not output_frames:
        return pd.DataFrame(columns=["event_id", "rel_time", "ROI", "value"])
    return pd.concat(output_frames, ignore_index=True)
