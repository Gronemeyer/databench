"""Dataset export to DuckDB — the wide frame as SQL tables and views.

:func:`write_duckdb` turns a dataset frame (rows = recordings, columns =
``(source, feature)``, a pickle or a lazy archive) into one ``.duckdb``
file:

``recordings``
    one row per recording: ``subject, session, task``, any label columns
    passed in (e.g. ``task_label``, ``condition``), then every scalar,
    string or dict cell as ``<source>__<feature>`` (dicts as JSON).
``<source>``
    one row per sample for each source with 1-D arrays: ``subject,
    session, task, sample``, one column per feature.  A feature missing
    from a recording (e.g. a parcel) is NULL there.
``<source>__<feature>``
    a 1-D feature whose length differs from its source's
    ``time_elapsed_s`` in any recording gets a table of its own.

:func:`add_views` then adds the views whose tables exist: ``sessions``,
``coverage``, ``parcels``, ``mesomap_long``, ``pupil`` and ``locomotion``.
Each view carries a comment (``SELECT * FROM duckdb_views()``).

Requires the optional ``duckdb`` package.  From the shell::

    databench-duckdb etoh2 ~/Desktop/etoh2.duckdb
    databench-duckdb --views ~/Desktop/etoh2.duckdb   # (re)create views only
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from databench.store import LazyCell, cell_shape, resolve
from databench.utils.logger import get_logger

logger = get_logger(__name__)

KEYS = ("subject", "session", "task")
TIME = "time_elapsed_s"


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError("DuckDB export needs the duckdb package: pip install duckdb") from exc
    return duckdb


def _is_vector(value: Any) -> bool:
    if isinstance(value, LazyCell):
        return value.dtype is not None and value.ndim == 1
    return isinstance(value, np.ndarray) and value.ndim == 1


def _sql_name(*parts: str) -> str:
    return "__".join(parts).replace("-", "_").replace(" ", "_")


def _column(arr: np.ndarray) -> np.ndarray | pd.Series:
    """Object arrays become typed pandas columns, or strings as a fallback."""
    if arr.dtype != object:
        return arr
    s = pd.Series(arr).infer_objects()
    if s.dtype == object:
        s = s.map(lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))
    return s


def _scalar(value: Any) -> Any:
    value = resolve(value)
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, default=str)
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), default=str)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _append(con, table: str, chunk: pd.DataFrame, existing: set[str]) -> None:
    """Insert *chunk* into *table*, creating it or adding columns as needed."""
    con.register("chunk", chunk)
    if table not in existing:
        con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM chunk')
        existing.add(table)
    else:
        have = {r[0] for r in con.execute(f'DESCRIBE "{table}"').fetchall()}
        for name, typ, *_ in con.execute("DESCRIBE chunk").fetchall():
            if name not in have:
                con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {typ}')
        con.execute(f'INSERT INTO "{table}" BY NAME SELECT * FROM chunk')
    con.unregister("chunk")


def write_duckdb(
    df: pd.DataFrame,
    path: str | Path,
    *,
    labels: pd.DataFrame | None = None,
    views: bool = True,
) -> list[str]:
    """Write the dataset frame *df* to a new DuckDB file at *path*.

    *labels* is an optional frame indexed like *df* (Subject, Session,
    Task) whose columns are added to ``recordings`` after the keys.  Cells
    are read one recording at a time, so a lazy archive never needs to fit
    in memory.  Returns the names of the tables written.
    """
    duckdb = _duckdb()
    path = Path(path).expanduser()
    if path.exists():
        raise FileExistsError(path)

    vector_cols = {c for c in df.columns if df[c].map(_is_vector).any()}
    split: set[tuple] = set()
    for _, row in df.iterrows():
        for col in vector_cols:
            t = row.get((col[0], TIME))
            if _is_vector(row[col]) and t is not None and _is_vector(t) \
                    and cell_shape(row[col])[0] != cell_shape(t)[0]:
                split.add(col)
    if split:
        logger.info(f"features in their own tables: {sorted(split)}")

    tmp = path.with_name(path.name + ".partial")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    existing: set[str] = set()
    rec_rows = []
    t0 = time.time()
    try:
        for i, (idx, row) in enumerate(df.iterrows()):
            keys = dict(zip(KEYS, idx))
            rec = dict(keys)
            if labels is not None:
                rec.update(labels.loc[idx].to_dict())
            by_source: dict[str, dict] = defaultdict(dict)
            for (src, feat), value in row.items():
                if (src, feat) not in vector_cols:
                    rec[_sql_name(src, feat)] = _scalar(value)
                elif _is_vector(value):  # otherwise missing in this recording
                    arr = _column(np.asarray(resolve(value)))
                    if (src, feat) in split:
                        chunk = pd.DataFrame({**keys, "sample": np.arange(len(arr)), feat: arr})
                        _append(con, _sql_name(src, feat), chunk, existing)
                    else:
                        by_source[src][feat] = arr
            for src, cols in by_source.items():
                n = len(next(iter(cols.values())))
                chunk = pd.DataFrame({**keys, "sample": np.arange(n), **cols})
                _append(con, _sql_name(src), chunk, existing)
            rec_rows.append(rec)
            logger.debug(f"[{i + 1}/{len(df)}] {'/'.join(map(str, idx))} {time.time() - t0:.0f}s")

        # Object dtype keeps None as NULL; a pandas string column would reach DuckDB as 'nan'.
        recordings = pd.DataFrame(rec_rows, dtype=object)
        # A column that mixes types across recordings (number vs dict) goes in as text.
        for c in recordings.columns:
            types = {type(x) for x in recordings[c] if x is not None}
            if len(types) > 1 and not types <= {int, float, bool}:
                recordings[c] = recordings[c].map(lambda x: None if x is None else str(x))
        _append(con, "recordings", recordings, existing)
        if views:
            add_views(con)
        con.execute("CHECKPOINT")
    finally:
        con.close()
    tmp.replace(path)
    logger.info(f"wrote {path} ({path.stat().st_size / 1e9:.2f} GB, {len(existing)} tables) "
                f"in {time.time() - t0:.0f}s")
    return sorted(existing)


# ── views ────────────────────────────────────────────────────────────────

LABELS = "subject, session, task, task_label, condition"

_SESSION_FIELDS = {
    "sex": "session_config__sex",
    "genotype": "session_config__genotype",
    "cage_id": "session_config__cage_id",
    "dob": "session_config__DOB",
    "weight_g": "session_config__weight_g",
    "inj_vol_ml": "session_config__inj_vol_mL",
    "duration_s": "session_config__duration",
    "protocol": "session_config__protocol",
}
_SESSION_CASTS = {"duration_s": "INTEGER"}


def _columns(con, table: str) -> list[str]:
    return [r[0] for r in con.execute(f'DESCRIBE "{table}"').fetchall()]


def _view_sql(con) -> dict[str, tuple[str, str]]:
    """``{view: (comment, sql)}`` for the views this file's tables support."""
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "recordings" not in tables:
        return {}
    rec = set(_columns(con, "recordings"))
    labels = ", ".join(c for c in KEYS + ("task_label", "condition") if c in rec)
    views: dict[str, tuple[str, str]] = {}

    fields = ", ".join(
        (f"TRY_CAST({col} AS {_SESSION_CASTS[name]})" if name in _SESSION_CASTS else col) + f" AS {name}"
        for name, col in _SESSION_FIELDS.items() if col in rec
    )
    views["sessions"] = (
        "One row per recording with its labels and animal/session variables.",
        f"""SELECT {labels},
                   TRY_CAST(regexp_extract(session, '\\d+') AS INTEGER) AS ses_num
                   {', ' + fields if fields else ''}
            FROM recordings""",
    )

    timed = sorted(t for t in tables if t != "recordings" and TIME in _columns(con, t))
    if timed:
        views["coverage"] = (
            "Samples, time span and median sampling rate of each source in each recording.",
            " UNION ALL ".join(
                f"""SELECT subject, session, task, '{t}' AS source, count(*) AS n_samples,
                           min({TIME}) AS t_start_s, max({TIME}) AS t_stop_s,
                           1 / median(dt) AS rate_hz
                    FROM (SELECT *, {TIME} - lag({TIME})
                                    OVER (PARTITION BY subject, session, task ORDER BY sample) AS dt
                          FROM "{t}")
                    GROUP BY ALL"""
                for t in timed
            ),
        )

    other = f"subject, session, task, sample, {TIME}"
    if "mesomap" in tables:
        views["parcels"] = (
            "Mesomap parcels: hemisphere, region and how many recordings have samples for each one.",
            f"""SELECT parcel, left(parcel, 1) AS hemisphere, substr(parcel, 3) AS region,
                       count(*) FILTER (WHERE n > 0) AS n_recordings
                FROM (UNPIVOT (SELECT subject, session, task,
                                      count(COLUMNS(* EXCLUDE ({other}))) AS "\\0"
                               FROM mesomap GROUP BY ALL)
                      ON COLUMNS(* EXCLUDE (subject, session, task))
                      INTO NAME parcel VALUE n)
                GROUP BY ALL ORDER BY region, hemisphere""",
        )
        views["mesomap_long"] = (
            "Mesomap traces in long form: one row per frame per parcel (ΔF/F). NULL samples are dropped.",
            f"""SELECT {', '.join('r.' + c for c in labels.split(', '))}, m.sample,
                       m.{TIME} AS time_s, m.parcel,
                       left(m.parcel, 1) AS hemisphere, substr(m.parcel, 3) AS region, m.dff
                FROM (UNPIVOT mesomap ON COLUMNS(* EXCLUDE ({other}))
                      INTO NAME parcel VALUE dff) m
                JOIN recordings r USING (subject, session, task)""",
        )
    if "pupil_dlc" in tables:
        views["pupil"] = (
            "Pupil diameter (DeepLabCut, mm) with recording labels.",
            f"""SELECT {labels}, sample, {TIME} AS time_s, pupil_diameter_mm
                FROM pupil_dlc JOIN recordings USING (subject, session, task)""",
        )
    if "treadmill" in tables:
        views["locomotion"] = (
            "Treadmill speed (mm/s) and distance with recording labels. The encoder logs only while "
            "the wheel moves, so a gap between samples means the animal was still, not missing data.",
            f"""SELECT {labels}, sample, {TIME} AS time_s, speed_mm AS speed_mm_s, distance_mm,
                       {TIME} - lag({TIME})
                           OVER (PARTITION BY subject, session, task ORDER BY sample) AS gap_s
                FROM treadmill JOIN recordings USING (subject, session, task)""",
        )
    return views


def add_views(con) -> list[str]:
    """Create (or replace) the convenience views on an open DuckDB connection."""
    views = _view_sql(con)
    for name, (comment, sql) in views.items():
        con.execute(f'CREATE OR REPLACE VIEW "{name}" AS {sql}')
        con.execute(f"""COMMENT ON VIEW "{name}" IS '{comment.replace("'", "''")}'""")
    return list(views)


# ── command line ─────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="databench-duckdb",
                                     description="Export a databench dataset to a DuckDB file.")
    parser.add_argument("dataset", nargs="?", help="datasets.toml alias or path to a .pkl/.npz")
    parser.add_argument("out", help="DuckDB file to write")
    parser.add_argument("--views", action="store_true",
                        help="only (re)create the views in an existing file")
    args = parser.parse_args(argv)

    if args.views:
        with _duckdb().connect(str(Path(args.out).expanduser())) as con:
            logger.info(f"views: {add_views(con)}")
        return
    if args.dataset is None:
        parser.error("dataset is required unless --views is given")

    from databench.project import Project

    proj = Project(args.dataset)
    labels = proj.conditions().set_index(["Subject", "Session", "Task"])
    write_duckdb(proj.df, args.out, labels=labels)


if __name__ == "__main__":
    main()
