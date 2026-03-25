"""Standalone utility functions extracted from the former Bench class.

These are helpers for building long-format tables, session feature tables,
and labeling conditions on DataFrames.  They work on plain DataFrames and
do not require any coordinator object.

Usage::

    from databench.bench import build_long, build_session_table, label_conditions

    long = build_long(df, sources=[("mesomap", ["L_VISp"]), ("treadmill", ["speed_mm"])])
    table = build_session_table(df, features=[speed_mean, pupil_mean])
    long  = label_conditions(long, {"ses-01": "baseline", "ses-02": "saline"})
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from databench.features.base import FeatureFn
from databench.utils import session_to_int
from databench._utils._logger import get_logger

_logger = get_logger("databench")


# -- Low-level helpers -------------------------------------------------------

def _extract_trace(
    row: pd.Series,
    source: str,
    feature: str,
    index: Optional[int],
    label: str,
) -> Optional[np.ndarray]:
    """Pull a single trace from a multi-index row."""
    x = row.get((source, feature))
    arr = np.asarray(x)
    if arr.ndim > 1 and index is not None:
        arr = arr[index]
    return arr


def _source_timeseries(
    row: pd.Series,
    source: str,
    features: Iterable[str],
    time_column: str,
    index: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Build a DataFrame of aligned time + feature columns for one source."""
    t = _extract_trace(row, source, time_column, index, "Source")
    t_arr = np.atleast_1d(t).astype(float, copy=False)
    data: Dict[str, Any] = {time_column: t_arr}
    for feature_name in features:
        x = _extract_trace(row, source, feature_name, index, "Feature")
        data[feature_name] = np.atleast_1d(x)
    return pd.DataFrame(data)


# -- Public utility functions ------------------------------------------------

def build_session_table(
    df: pd.DataFrame,
    features: Iterable[FeatureFn],
) -> pd.DataFrame:
    """Compute a wide feature table from a raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format dataset with a (Subject, Session, Task) MultiIndex.
    features : iterable of FeatureFn
        Feature extractors to apply to each row.

    Returns
    -------
    pd.DataFrame
        One row per session with a column per feature, sorted by
        Subject and session number.
    """
    use_features = list(features)
    _logger.info(
        f"Build session table: features={len(use_features)} df_shape={df.shape}"
    )
    rows: List[dict] = []
    for _, row in df.iterrows():
        out: Dict[str, Any] = {}
        for feat in use_features:
            out[feat.name] = feat.run(row)
        rows.append(out)
    table = pd.DataFrame(rows, index=df.index)
    table["session_n"] = df.index.get_level_values("Session").map(session_to_int)
    return table.sort_values(["Subject", "session_n"])


def build_long(
    df: pd.DataFrame,
    source_features: Optional[Iterable[tuple]] = None,
    sources: Optional[Iterable[tuple]] = None,
    tol: float = 0.25,
    time_column: str = "time_elapsed_s",
    reference_source: Optional[str] = None,
) -> pd.DataFrame:
    """Build a long table by aligning multiple source timeseries.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format dataset with a (Subject, Session, Task) MultiIndex.
    source_features / sources : iterable
        Ordered list of `(source, features)` or `(source, features, indices)`
        tuples.  `sources` is an alias for `source_features`.
    tol : float
        Tolerance in seconds for `pd.merge_asof`.
    time_column : str
        Name of the time column within each source.
    reference_source : str | None
        Source whose time base becomes the output index.

    Returns
    -------
    pd.DataFrame
        Long-format table with Subject, Session, Task columns prepended.
    """
    sf = source_features or sources
    if sf is None:
        raise ValueError("Provide source_features (or sources=) argument.")

    source_features_list = []
    _logger.info("Build long table")
    for entry in sf:
        source = entry[0]
        features = entry[1]
        indices = entry[2] if len(entry) > 2 else None
        source_features_list.append((source, list(features), indices))

    ref_idx = 0
    if reference_source is not None:
        for i, (source, _, _) in enumerate(source_features_list):
            if source == reference_source:
                ref_idx = i
                break

    ref_source, ref_features, ref_indices = source_features_list[ref_idx]
    merge_sources = [
        entry for i, entry in enumerate(source_features_list) if i != ref_idx
    ]

    frames: List[pd.DataFrame] = []
    if ref_indices is None:
        ref_index_list: list = []
    elif isinstance(ref_indices, (list, tuple, np.ndarray)):
        ref_index_list = list(ref_indices)
    else:
        ref_index_list = [ref_indices]

    for idx, row in df.iterrows():
        if ref_indices is None:
            out = _source_timeseries(
                row, ref_source, ref_features, time_column, index=None,
            )
            if out is None:
                continue
        else:
            roi_frames: List[pd.DataFrame] = []
            base_time = None
            for ref_index in ref_index_list:
                roi_df = _source_timeseries(
                    row, ref_source, ref_features, time_column, index=ref_index,
                )
                if roi_df is None:
                    continue

                if base_time is None:
                    base_time = roi_df[time_column].to_numpy()
                elif not np.array_equal(roi_df[time_column].to_numpy(), base_time):
                    raise ValueError(
                        f"Source {ref_source!r} ROI timebases differ; cannot align per-ROI columns."
                    )

                rename = {
                    feature_name: f"{feature_name}_roi{ref_index}"
                    for feature_name in ref_features
                }
                roi_frames.append(roi_df.rename(columns=rename))

            if base_time is None:
                continue

            out = pd.concat(
                [roi_frames[0][[time_column]]]
                + [frame.drop(columns=[time_column]) for frame in roi_frames],
                axis=1,
            )

        out = out.sort_values(time_column)

        for source, features, _ in merge_sources:
            ts = _source_timeseries(
                row, source, features, time_column, index=None,
            )
            if ts is not None:
                ts = ts.dropna(subset=[time_column])
                out = pd.merge_asof(
                    out,
                    ts.sort_values(time_column),
                    on=time_column,
                    direction="nearest",
                    tolerance=tol,
                )
            else:
                for feature_name in features:
                    out[feature_name] = np.nan

        subj, ses, task = idx  # type: ignore[misc]
        out.insert(0, "Task", task)
        out.insert(0, "Session", ses)
        out.insert(0, "Subject", subj)

        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def label_conditions(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    column: str = "Condition",
) -> pd.DataFrame:
    """Map Session values to condition labels on a long table.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format table (must have a `"Session"` column).
    mapping : dict
        `{"ses-01": "baseline", "ses-02": "saline", ...}`
    column : str
        Name of the new column.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with the new column added (modified in place).
    """
    df[column] = df["Session"].map(mapping)
    return df
