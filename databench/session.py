"""Session, SessionGroup, AlignedData, and SaveableFigure.

These are the data-selection and data-alignment objects in the public API.
Users get them from :class:`~databench.project.Project`, not by importing
directly (though direct import works fine).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench.analysis.base import FeatureFn
from databench.config import OutputContext
from databench._utils import session_to_int


# ── Exceptions ─────────────────────────────────────────────────────────────

class SignalNotFoundError(KeyError):
    """Raised when a requested source/signal is not in the session data."""


def _extract_trace(
    row: pd.Series,
    source: str,
    feature: str,
    index: Optional[int],
) -> Optional[np.ndarray]:
    """Pull a single trace from a multi-index row."""
    x = row.get((source, feature))
    if x is None:
        return None
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
    t = _extract_trace(row, source, time_column, index)
    if t is None:
        return None
    t_arr = np.atleast_1d(t).astype(float, copy=False)
    data: dict[str, Any] = {time_column: t_arr}
    for feature_name in features:
        x = _extract_trace(row, source, feature_name, index)
        if x is None:
            data[feature_name] = np.full(t_arr.shape, np.nan)
        else:
            data[feature_name] = np.atleast_1d(x)
    return pd.DataFrame(data)


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
        Ordered list of ``(source, features)`` or
        ``(source, features, indices)`` tuples. ``sources`` is an alias for
        ``source_features``.
    tol : float
        Tolerance in seconds for ``pd.merge_asof``.
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

    frames: list[pd.DataFrame] = []
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
            roi_frames: list[pd.DataFrame] = []
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


# ── AlignedData ────────────────────────────────────────────────────────────

class AlignedData:
    """Lightweight wrapper around a time-aligned DataFrame.

    Attributes
    ----------
    df : pd.DataFrame
        Flat table with one row per timepoint.  Columns include the
        time column, all requested signal columns, and (for multi-session)
        Subject / Session / Task identifiers.
    reference : str
        Name of the reference source used for alignment.
    sources : dict[str, list[str]]
        Mapping from source name to signal names that were aligned.
    time_column : str
        Name of the time column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        reference: str,
        sources: dict[str, list[str]],
        time_column: str,
    ) -> None:
        self._df = df
        self._reference = reference
        self._sources = sources
        self._time_column = time_column

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def reference(self) -> str:
        return self._reference

    @property
    def sources(self) -> dict[str, list[str]]:
        return self._sources

    @property
    def time_column(self) -> str:
        return self._time_column

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    def __repr__(self) -> str:
        rows, cols = self._df.shape
        return f"AlignedData(rows={rows}, columns={cols}, reference={self._reference!r})"


# ── SaveableFigure ─────────────────────────────────────────────────────────

class SaveableFigure:
    """Thin wrapper around a matplotlib Figure with a convenient ``.save()`` method.

    Usage::

        result.plot_overview(...).save("overview.svg")
        result.plot_overview(...).fig   # raw matplotlib Figure
    """

    def __init__(self, fig: plt.Figure, context: OutputContext) -> None:
        self._fig = fig
        self._context = context

    @property
    def fig(self) -> plt.Figure:
        """The underlying matplotlib Figure."""
        return self._fig

    def save(
        self,
        name: str,
        *,
        dpi: int = 300,
        folder: str = "plots",
        bbox_inches: str = "tight",
    ) -> Path:
        """Save the figure to the project's output directory.

        Parameters
        ----------
        name : str
            Filename (e.g. ``"overview.svg"``).
        dpi : int
            Resolution for raster formats.
        folder : str
            Subdirectory under the run directory (``"plots"`` by default).

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        out_dir = self._context.run_dir / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        self._fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(self._fig)
        return path

    def show(self) -> None:
        """Display the figure (interactive backends only)."""
        self._fig.show()

    def close(self) -> None:
        """Close the figure to free memory."""
        plt.close(self._fig)

    def __repr__(self) -> str:
        return f"SaveableFigure({self._fig.number})"


# ── Session ────────────────────────────────────────────────────────────────

class Session:
    """A single experimental session (one row of the wide dataset).

    Provides access to raw signals and time-alignment across sources.
    Obtained via :meth:`Project.session`, not by direct construction.
    """

    def __init__(
        self,
        row: pd.Series,
        index: tuple,
        context: OutputContext,
    ) -> None:
        self._row = row
        self._index = index
        self._context = context

    # ── Identity ───────────────────────────────────────────────────────────

    @property
    def subject(self) -> str:
        return self._index[0]

    @property
    def session(self) -> str:
        return self._index[1]

    @property
    def task(self) -> str:
        return self._index[2]

    @property
    def label(self) -> str:
        return f"Subject={self.subject} | Session={self.session} | Task={self.task}"

    # ── Signal access ──────────────────────────────────────────────────────

    def signal(self, source: str, name: str) -> np.ndarray:
        """Extract a 1-D signal array from this session.

        Parameters
        ----------
        source : str
            Data source name (e.g. ``"mesomap"``, ``"pupil"``, ``"treadmill"``).
        name : str
            Signal name within the source (e.g. ``"L_VISp"``, ``"pupil_diameter_mm"``).

        Raises
        ------
        SignalNotFoundError
            If the source/signal combination is not found.
        """
        value = self._row.get((source, name))
        if value is None:
            available = self._available_signals(source)
            sources = self._available_sources()
            msg = (
                f"Signal {name!r} not found in source {source!r}.\n"
                f"  Available signals for {source!r}: {available}"
            )
            if not available:
                msg += f"\n  Available sources: {sources}"
            raise SignalNotFoundError(msg)
        arr = np.asarray(value)
        if arr.ndim == 0:
            # Scalar stored in the dataset — promote to 1-element array
            arr = arr.reshape(1)
        elif arr.ndim > 1:
            arr = arr.ravel()
        return arr

    def time(self, source: str, column: str = "time_elapsed_s") -> np.ndarray:
        """Extract the time array for a given source.

        Parameters
        ----------
        source : str
            Data source name.
        column : str
            Name of the time column (default: ``"time_elapsed_s"``).
        """
        value = self._row.get((source, column))
        if value is None:
            raise SignalNotFoundError(
                f"Time column {column!r} not found in source {source!r}."
            )
        arr = np.asarray(value, dtype=float).ravel()
        return arr

    def _time_for_align(self, source: str, column: str = "time_elapsed_s") -> np.ndarray:
        """Return a time vector for alignment, with dataset-level fallbacks.

        Fallback order when ``(source, column)`` is missing:
        1. ``(source, "time_elapse_s")``   — typo variant in some datasets
        2. ``("dataqueue", "time_elapse_s")``
        3. ``("dataqueue", "time_elapsed_s")``
        4. ``("time", "master_elapsed_s")`` — frame-locked acquisition clock
        5. ``("time", "queue_elapsed")``    — general event queue

        A candidate is skipped when its length is more than double the
        source's longest signal (likely a different-rate time vector).
        """
        # Determine the expected signal length for this source so we can
        # reject wildly mismatched time vectors.
        max_sig_len = 0
        if isinstance(self._row.index, pd.MultiIndex):
            for src, name in self._row.index:
                if src == source and name != column:
                    val = self._row.get((src, name))
                    if val is not None:
                        arr = np.asarray(val)
                        if arr.ndim >= 1:
                            max_sig_len = max(max_sig_len, arr.size)

        candidates = [
            (source, column),
            (source, "time_elapse_s"),
            ("dataqueue", "time_elapse_s"),
            ("dataqueue", "time_elapsed_s"),
            ("time", "master_elapsed_s"),
            ("time", "queue_elapsed"),
        ]
        for src, col in candidates:
            value = self._row.get((src, col))
            if value is not None:
                arr = np.asarray(value, dtype=float).ravel()
                # Skip candidates whose length is wildly incompatible with
                # the source's signals (> 2× longer suggests a different-rate
                # time vector).
                if max_sig_len > 0 and src != source and arr.size > 2 * max_sig_len:
                    continue
                # master_elapsed_s carries an acquisition-start offset;
                # zero it so the timeline begins at 0 like source-local
                # time columns.
                if col == "master_elapsed_s" and arr.size > 0:
                    arr = arr - arr[0]
                return arr

        raise SignalNotFoundError(
            f"Time column {column!r} not found in source {source!r}, and no "
            "alignment fallback was available (dataqueue/time sources missing)."
        )

    def _available_signals(self, source: str, time_column: str = "time_elapsed_s") -> list[str]:
        """List signal names available for a given source.

        Filters to array-valued entries whose length is comparable to the
        source's time array (i.e. real timeseries, not metadata that happens
        to be stored as a short array).
        """
        if not isinstance(self._row.index, pd.MultiIndex):
            return list(self._row.index)
        # Get time array length as reference
        t_val = self._row.get((source, time_column))
        t_len = len(np.asarray(t_val)) if t_val is not None else 0
        min_len = max(1, int(t_len * 0.5))  # must be at least 50 % of time length
        signals = []
        for src, name in self._row.index:
            if src == source and name != time_column:
                val = self._row.get((src, name))
                arr = np.asarray(val)
                if arr.ndim >= 1 and arr.size >= min_len:
                    signals.append(name)
        return sorted(signals)

    def _available_sources(self) -> list[str]:
        """List source names present in this session's data."""
        if not isinstance(self._row.index, pd.MultiIndex):
            return []
        return sorted({src for src, _name in self._row.index})

    # ── Alignment ──────────────────────────────────────────────────────────

    def align(
        self,
        sources: dict[str, list[str]] | list[str],
        *,
        reference: str,
        tolerance_s: float = 0.25,
        time_column: str = "time_elapsed_s",
    ) -> AlignedData:
        """Align signals from multiple sources onto a common time base.

        Uses ``pd.merge_asof`` to align each non-reference source onto the
        reference source's time base.

        Parameters
        ----------
        sources : dict[str, list[str]] or list[str]
            Either a mapping from source name to list of signal column names
            (e.g. ``{"mesomap": ["L_VISp"], "pupil": ["pupil_diameter_mm"]}``),
            or a plain list of source names (e.g. ``["mesomap", "pupil"]``).
            When a list is given, all available signals for each source are
            included automatically.
        reference : str
            Name of the reference source (its time base becomes the output index).
        tolerance_s : float
            Maximum allowed time difference for nearest-match alignment.
        time_column : str
            Name of the time column within each source.

        Returns
        -------
        AlignedData
            Wrapped DataFrame with one row per timepoint.

        Raises
        ------
        SignalNotFoundError
            If a requested source or signal is not found.
        ValueError
            If the reference source is not in *sources*.
        """
        # Normalise list[str] → dict[str, list[str]]
        if isinstance(sources, list):
            sources = {src: self._available_signals(src, time_column) for src in sources}

        if reference not in sources:
            raise ValueError(
                f"Reference source {reference!r} must be included in sources. "
                f"Got sources: {list(sources.keys())}"
            )

        # Build reference DataFrame
        ref_signals = sources[reference]
        ref_t = self._time_for_align(reference, time_column)
        ref_arrays: dict[str, np.ndarray] = {}
        for sig_name in ref_signals:
            sig = self.signal(reference, sig_name)
            if np.asarray(sig).ndim == 0:
                continue
            ref_arrays[sig_name] = np.asarray(sig)
        # Truncate all arrays to the shortest common length
        n_ref = min(len(ref_t), *(len(v) for v in ref_arrays.values())) if ref_arrays else len(ref_t)
        ref_data: dict[str, np.ndarray] = {time_column: ref_t[:n_ref]}
        for k, v in ref_arrays.items():
            ref_data[k] = v[:n_ref]
        out = pd.DataFrame(ref_data).sort_values(time_column)

        # Merge each non-reference source
        for src_name, sig_names in sources.items():
            if src_name == reference:
                continue
            src_t = self._time_for_align(src_name, time_column)
            src_arrays: dict[str, np.ndarray] = {}
            for sig_name in sig_names:
                sig = self.signal(src_name, sig_name)
                if np.asarray(sig).ndim == 0:
                    continue
                src_arrays[sig_name] = np.asarray(sig)
            if not src_arrays:
                continue
            n_src = min(len(src_t), *(len(v) for v in src_arrays.values()))
            src_data: dict[str, np.ndarray] = {time_column: src_t[:n_src]}
            for k, v in src_arrays.items():
                src_data[k] = v[:n_src]
            src_df = pd.DataFrame(src_data).dropna(subset=[time_column]).sort_values(time_column)
            out = pd.merge_asof(
                out,
                src_df,
                on=time_column,
                direction="nearest",
                tolerance=tolerance_s,
            )

        # Prepend session identifiers
        out.insert(0, "Task", self.task)
        out.insert(0, "Session", self.session)
        out.insert(0, "Subject", self.subject)

        return AlignedData(
            df=out,
            reference=reference,
            sources=sources,
            time_column=time_column,
        )

    def __repr__(self) -> str:
        return f"Session(subject={self.subject!r}, session={self.session!r}, task={self.task!r})"


# ── SessionGroup ───────────────────────────────────────────────────────────

class SessionGroup:
    """A group of sessions for multi-session analyses.

    Obtained via :meth:`Project.sessions`, not by direct construction.
    Supports iteration, indexing, and ``len()``.
    """

    def __init__(
        self,
        sessions: list[Session],
        context: OutputContext,
    ) -> None:
        self._sessions = sessions
        self._context = context

    def __len__(self) -> int:
        return len(self._sessions)

    def __iter__(self) -> Iterator[Session]:
        return iter(self._sessions)

    def __getitem__(self, index: int) -> Session:
        return self._sessions[index]

    @property
    def subjects(self) -> list[str]:
        """Unique subjects in this group."""
        return sorted({s.subject for s in self._sessions})

    @property
    def session_labels(self) -> list[str]:
        """Unique session labels in this group."""
        return sorted({s.session for s in self._sessions})

    def align(
        self,
        sources: dict[str, list[str]] | list[str],
        *,
        reference: str,
        tolerance_s: float = 0.25,
        time_column: str = "time_elapsed_s",
    ) -> AlignedData:
        """Align signals across all sessions in the group.

        Calls :meth:`Session.align` per session and concatenates results.

        Parameters
        ----------
        sources, reference, tolerance_s, time_column
            Same as :meth:`Session.align`.

        Returns
        -------
        AlignedData
            Combined aligned data with Subject/Session/Task columns.
        """
        frames: list[pd.DataFrame] = []
        for session in self._sessions:
            aligned = session.align(
                sources,
                reference=reference,
                tolerance_s=tolerance_s,
                time_column=time_column,
            )
            frames.append(aligned.df)

        combined = pd.concat(frames, ignore_index=True)
        return AlignedData(
            df=combined,
            reference=reference,
            sources=sources,
            time_column=time_column,
        )

    def __repr__(self) -> str:
        n = len(self._sessions)
        subjects = self.subjects
        return f"SessionGroup(n={n}, subjects={subjects})"
