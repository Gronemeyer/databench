"""Session, SessionGroup, and AlignedData.

These are the data-selection and data-alignment objects in the public API.
Users get them from :class:`~databench.project.Project`, not by importing
directly (though direct import works fine).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench.config import Schema
from databench.utils.labels import parse_session_day


def _fuzzy_suggest(needle: str, haystack: list[str], n: int = 3) -> str:
    """Return a 'Did you mean …?' suffix for not-found errors.

    Returns an empty string when nothing reasonable matches.
    """
    import difflib
    if not needle or not haystack:
        return ""
    matches = difflib.get_close_matches(str(needle), haystack, n=n, cutoff=0.4)
    if not matches:
        return ""
    rendered = ", ".join(repr(m) for m in matches)
    return f"\n  Did you mean {rendered}?"


# ── Exceptions ─────────────────────────────────────────────────────────────

class SignalNotFoundError(KeyError):
    """Raised when a requested source/signal is not in the session data."""


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
        schema: Schema | None = None,
    ) -> None:
        self._row = row
        self._index = index
        self._schema = schema if schema is not None else Schema()

    @property
    def schema(self) -> Schema:
        """The dataset schema (source aliases + per-column role/unit)."""
        return self._schema

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
    def day(self) -> int | None:
        """Integer day parsed from a ``ses-NN`` session label, or ``None``."""
        return parse_session_day(self.session, default=None)

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
            If the dataset's schema declares an alias for this name (e.g.
            ``pupil -> pupil_dlc``) the alias is applied transparently.
        name : str
            Signal name within the source (e.g. ``"L_VISp"``, ``"pupil_diameter_mm"``).

        Raises
        ------
        SignalNotFoundError
            If the source/signal combination is not found.  The message
            includes a fuzzy suggestion when a near-match exists.
        """
        canonical = self._schema.resolve_source(source)
        value = self._row.get((canonical, name))
        if value is None:
            available = self.signals(canonical)
            sources = self.sources
            msg = (
                f"Signal {name!r} not found in source {canonical!r}"
                + (f" (resolved from alias {source!r})" if canonical != source else "")
                + ".\n"
                f"  Available signals for {canonical!r}: {available}"
            )
            if not available:
                msg += f"\n  Available sources: {sources}"
                msg += _fuzzy_suggest(source, sources)
            else:
                msg += _fuzzy_suggest(name, available)
            raise SignalNotFoundError(msg)
        arr = np.asarray(value)
        if arr.ndim == 0:
            # Scalar stored in the dataset â€” promote to 1-element array
            arr = arr.reshape(1)
        elif arr.ndim > 1:
            arr = arr.ravel()
        return arr

    def time(self, source: str, column: str = "time_elapsed_s") -> np.ndarray:
        """Extract the time array for a given source.

        Parameters
        ----------
        source : str
            Data source name; aliases declared in the schema are applied.
        column : str
            Name of the time column (default: ``"time_elapsed_s"``).
        """
        canonical = self._schema.resolve_source(source)
        value = self._row.get((canonical, column))
        if value is None:
            available = self.signals(canonical)
            sources = self.sources
            msg = (
                f"Time column {column!r} not found in source {canonical!r}"
                + (f" (resolved from alias {source!r})" if canonical != source else "")
                + "."
            )
            if not available:
                msg += f"\n  Available sources: {sources}"
                msg += _fuzzy_suggest(source, sources)
            else:
                msg += f"\n  Available columns for {canonical!r}: {available}"
                msg += _fuzzy_suggest(column, available)
            raise SignalNotFoundError(msg)
        arr = np.asarray(value, dtype=float).ravel()
        return arr

    def _time_for_align(self, source: str, column: str = "time_elapsed_s") -> np.ndarray:
        """Return a time vector for alignment with registry fallbacks.

        Resolution order:
        1. ``(source, column)`` — the requested column.
        2. ``(source, "time_elapse_s")`` — common typo variant.
        3. Each ``(src, col)`` in :data:`databench.config.TIME_COLUMNS`.

        A candidate is skipped when its length is more than double the
        source's longest signal.  ``master_elapsed_s`` is auto-zeroed so
        the timeline starts at 0.  Raises :class:`SignalNotFoundError`
        when no candidate satisfies the constraints.
        """
        from databench.config import TIME_COLUMNS

        canonical = self._schema.resolve_source(source)

        # Determine the expected signal length for this source so we can
        # reject wildly mismatched time vectors.
        max_sig_len = 0
        if isinstance(self._row.index, pd.MultiIndex):
            for src, name in self._row.index:
                if src == canonical and name != column:
                    val = self._row.get((src, name))
                    if val is not None:
                        arr = np.asarray(val)
                        if arr.ndim >= 1:
                            max_sig_len = max(max_sig_len, arr.size)

        candidates: list[tuple[str, str]] = [
            (canonical, column),
            (canonical, "time_elapse_s"),
            *TIME_COLUMNS,
        ]
        for src, col in candidates:
            value = self._row.get((src, col))
            if value is None:
                continue
            arr = np.asarray(value, dtype=float).ravel()
            if max_sig_len > 0 and src != canonical and arr.size > 2 * max_sig_len:
                continue
            if col == "master_elapsed_s" and arr.size > 0:
                arr = arr - arr[0]
            return arr

        raise SignalNotFoundError(
            f"Time column {column!r} not found in source {canonical!r}; no "
            "fallback in databench.config.TIME_COLUMNS matched either."
        )

    @property
    def sources(self) -> list[str]:
        """Source names present in this session's data (canonical keys)."""
        if not isinstance(self._row.index, pd.MultiIndex):
            return []
        return sorted({src for src, _name in self._row.index})

    def signals(
        self,
        source: str,
        *,
        time_column: str = "time_elapsed_s",
    ) -> list[str]:
        """Signal names available for *source* (canonical or aliased).

        Filters to array-valued entries whose length is comparable to the
        source's time array (i.e. real timeseries, not metadata that
        happens to be stored as a short array).
        """
        canonical = self._schema.resolve_source(source)
        if not isinstance(self._row.index, pd.MultiIndex):
            return list(self._row.index)
        t_val = self._row.get((canonical, time_column))
        t_len = len(np.asarray(t_val)) if t_val is not None else 0
        if t_val is None:
            t_len = 0
        else:
            t_arr = np.asarray(t_val)
            t_len = t_arr.size if t_arr.ndim >= 1 else 0
        min_len = max(1, int(t_len * 0.5))
        names: list[str] = []
        for src, name in self._row.index:
            if src == canonical and name != time_column:
                val = self._row.get((src, name))
                arr = np.asarray(val)
                if arr.ndim >= 1 and arr.size >= min_len:
                    names.append(name)
        return sorted(names)

    # ── Discovery ──────────────────────────────────────────────────────────

    def describe(self, *, time_column: str = "time_elapsed_s") -> str:
        """Print a human-readable summary of sources, signals, and durations.

        Useful from the REPL or a notebook cell when authoring a new
        analysis::

            sess = proj.session(subject="GS28", session="ses-01", task="task-spont")
            sess.describe()

        Returns the same text it prints, so it composes with logging.
        """
        lines = [self.label, "Sources:"]
        for src in self.sources:
            sigs = self.signals(src, time_column=time_column)
            try:
                t = self.time(src, time_column)
                span = f"{t[0]:.1f}–{t[-1]:.1f}s, n={t.size}" if t.size else "empty"
            except SignalNotFoundError:
                span = "no time column"
            preview = sigs[:6] + (["…"] if len(sigs) > 6 else [])
            lines.append(f"  {src:<10} {preview}  ({span})")
        text = "\n".join(lines)
        print(text)
        return text

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
        # Normalise list[str] â†’ dict[str, list[str]]
        if isinstance(sources, list):
            sources = {src: self.signals(src, time_column=time_column) for src in sources}

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

    def __init__(self, sessions: list[Session]) -> None:
        self._sessions = sessions

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

    def to_frame(
        self,
        extractor: "Callable[[Session], Iterable[dict] | None]",
    ) -> pd.DataFrame:
        """Build a long-form DataFrame by calling *extractor* on each session.

        ``extractor(session)`` returns an iterable of row dicts (or ``None``
        / empty iterable to skip the session).  ``Subject``, ``Session``,
        and ``Task`` columns are auto-prepended to every row.

        Example
        -------
        >>> def per_session(sess):
        ...     speed = sess.signal("treadmill", "speed_mm")
        ...     yield {"day": sess.day, "mean_speed_mm": float(np.nanmean(speed))}
        >>> table = group.to_frame(per_session)
        """
        rows: list[dict] = []
        for session in self._sessions:
            produced = extractor(session)
            if produced is None:
                continue
            for row in produced:
                rows.append({
                    "Subject": session.subject,
                    "Session": session.session,
                    "Task": session.task,
                    **row,
                })
        return pd.DataFrame(rows)

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

    def describe(self) -> str:
        """Print a human-readable summary of subjects × sessions × tasks."""
        from collections import Counter
        n = len(self._sessions)
        subjects = self.subjects
        sessions = self.session_labels
        tasks = sorted({s.task for s in self._sessions})
        per_subj = Counter(s.subject for s in self._sessions)
        lines = [
            f"SessionGroup: {n} sessions",
            f"  Subjects ({len(subjects)}): {subjects}",
            f"  Sessions ({len(sessions)}): {sessions}",
            f"  Tasks    ({len(tasks)}): {tasks}",
            f"  Per-subject session counts: {dict(per_subj)}",
        ]
        text = "\n".join(lines)
        print(text)
        return text
