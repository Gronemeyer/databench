"""Session, SessionGroup, AlignedData, and SaveableFigure.

These are the data-selection and data-alignment objects in the public API.
Users get them from :class:`~databench.project.Project`, not by importing
directly (though direct import works fine).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench.config import OutputContext


# ── Exceptions ─────────────────────────────────────────────────────────────

class SignalNotFoundError(KeyError):
    """Raised when a requested source/signal is not in the session data."""


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
            raise SignalNotFoundError(
                f"Signal {name!r} not found in source {source!r}.\n"
                f"  Available signals for {source!r}: {available}"
            )
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
        ref_t = self.time(reference, time_column)
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
            src_t = self.time(src_name, time_column)
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
