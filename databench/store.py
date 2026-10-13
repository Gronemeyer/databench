"""Dataset archive — one file that opens without reading its data.

A dataset archive (``.npz``) holds the same wide frame as a dataset pickle
(rows = recordings, columns = ``(source, feature)``) but keeps every large
cell out of the frame:

* ``__frame__.pkl`` — the frame with each cell larger than
  :data:`INLINE_MAX_BYTES` set to ``None``.  The index, columns,
  ``session_config`` strings and other scalars stay here, so opening a
  :class:`~databench.project.Project` reads only this member.
* one member per large cell — ``.npy`` for arrays, ``.pkl`` for anything
  else — named ``<Subject>/<Session>/<Task>/<source>/<feature>``.
* ``__manifest__.json`` — which frame cell each member belongs to, with its
  shape and dtype.

:func:`read_archive` returns the frame with a :class:`LazyCell` in place of
each large cell.  Members are stored uncompressed and 64-byte aligned, so
:class:`ArchiveStore` maps the file once and returns numeric arrays as
zero-copy views: a cell costs memory only for the pages a script touches.
The map is copy-on-write — writing into a returned array never changes
the file.

The archive is a plain zip of ``.npy`` files, so it opens without
databench too::

    npz = np.load("dataset.npz", allow_pickle=True)
    npz["GS26/ses-01/task-spont/mesomap/L_VISp"]
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import mmap
import pickle
import struct
import threading
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FORMAT = "databench-archive"
FORMAT_VERSION = 1
FRAME_MEMBER = "__frame__.pkl"
MANIFEST_MEMBER = "__manifest__.json"
INLINE_MAX_BYTES = 64 * 1024
"""Cells whose serialised size is at most this stay inside the frame."""

_ALIGN = 64
_ALIGN_EXTRA_ID = 0xD935          # zip extra-field id used for alignment padding
_LOCAL_HEADER_SIZE = 30
_NPY_HEADER_READERS = {
    (1, 0): np.lib.format.read_array_header_1_0,
    (2, 0): np.lib.format.read_array_header_2_0,
}


def is_archive(path: str | Path) -> bool:
    """Whether *path* names a dataset archive (by suffix)."""
    return Path(path).suffix == ".npz"


# ── reading ──────────────────────────────────────────────────────────────

class ArchiveStore:
    """Read the members of one archive through a single memory map.

    Numeric ``.npy`` members come back as views into the map; object arrays,
    compressed or unaligned members and ``.pkl`` members are read into
    memory on each call.  The map and its views live as long as anything
    references them.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._zip = zipfile.ZipFile(self.path)
        with open(self.path, "rb") as f:
            self._map = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
        self._layout: dict[str, tuple[int, tuple, np.dtype, bool] | None] = {}
        self._lock = threading.Lock()

    def read_bytes(self, member: str) -> bytes:
        return self._zip.read(member)

    def read(self, member: str) -> Any:
        """The value stored in *member*."""
        if not member.endswith(".npy"):
            return pickle.loads(self._zip.read(member))
        layout = self._array_layout(member)
        if layout is None:
            with self._zip.open(member) as f:
                return np.lib.format.read_array(f, allow_pickle=True)
        offset, shape, dtype, fortran = layout
        return np.ndarray(shape, dtype=dtype, buffer=self._map, offset=offset,
                          order="F" if fortran else "C")

    def _array_layout(self, member: str) -> tuple[int, tuple, np.dtype, bool] | None:
        """Absolute data offset, shape, dtype and order of a mappable member.

        ``None`` when the member cannot be mapped (compressed, object dtype,
        unaligned, or an ``.npy`` header version without a public reader).
        """
        with self._lock:
            if member in self._layout:
                return self._layout[member]
            info = self._zip.getinfo(member)
            layout = None
            if info.compress_type == zipfile.ZIP_STORED:
                name_len, extra_len = struct.unpack_from("<HH", self._map, info.header_offset + 26)
                start = info.header_offset + _LOCAL_HEADER_SIZE + name_len + extra_len
                header = io.BytesIO(self._map[start:start + min(info.file_size, 1 << 16)])
                try:
                    reader = _NPY_HEADER_READERS.get(np.lib.format.read_magic(header))
                    if reader is not None:
                        shape, fortran, dtype = reader(header)
                        offset = start + header.tell()
                        if not dtype.hasobject and offset % max(dtype.alignment, 1) == 0:
                            layout = (offset, shape, dtype, fortran)
                except ValueError:
                    layout = None
            self._layout[member] = layout
            return layout


class LazyCell:
    """A dataset cell that stays on disk until its value is asked for.

    Archive frames hold these in place of large cells.  ``np.asarray(cell)``
    and :meth:`load` read the value; :attr:`shape`, :attr:`ndim`,
    :attr:`size` and :attr:`dtype` come from the manifest and read nothing.
    :class:`~databench.session.Session` resolves cells itself, so analysis
    code sees the same arrays and dicts a pickled dataset gives.
    """

    __slots__ = ("store", "member", "shape", "dtype")

    def __init__(self, store: ArchiveStore, member: str, shape: tuple,
                 dtype: np.dtype | None) -> None:
        self.store = store
        self.member = member
        self.shape = tuple(shape)
        self.dtype = dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    def load(self) -> Any:
        """Read the cell's value from the archive."""
        return self.store.read(self.member)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        arr = np.asarray(self.load(), dtype=dtype)
        return arr.copy() if copy else arr

    def __reduce__(self):
        descr = None if self.dtype is None else np.lib.format.dtype_to_descr(self.dtype)
        return _reopen_cell, (str(self.store.path), self.member, self.shape, descr)

    def __repr__(self) -> str:
        kind = f"dtype={self.dtype}" if self.dtype is not None else "pickled"
        return f"LazyCell({self.member!r}, shape={self.shape}, {kind})"


def resolve(value: Any) -> Any:
    """Load *value* if it is a :class:`LazyCell`; otherwise return it unchanged."""
    return value.load() if isinstance(value, LazyCell) else value


def cell_shape(value: Any) -> tuple:
    """Shape of a cell value without loading a :class:`LazyCell`."""
    if isinstance(value, LazyCell):
        return value.shape
    return np.asarray(value).shape


_STORES: dict[tuple[str, int, int], ArchiveStore] = {}


def open_store(path: str | Path) -> ArchiveStore:
    """One :class:`ArchiveStore` per archive file per process."""
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _STORES:
        _STORES[key] = ArchiveStore(path)
    return _STORES[key]


def _reopen_cell(path: str, member: str, shape: tuple, descr: Any) -> LazyCell:
    dtype = None if descr is None else np.lib.format.descr_to_dtype(descr)
    return LazyCell(open_store(path), member, shape, dtype)


def read_manifest(path: str | Path) -> dict:
    """The archive's manifest (format, source pickle, cell table)."""
    manifest = json.loads(open_store(path).read_bytes(MANIFEST_MEMBER))
    if manifest.get("format") != FORMAT:
        raise ValueError(
            f"{path} is not a databench dataset archive (no {MANIFEST_MEMBER}). "
            "Convert a dataset pickle with analysis_scripts/Tools/convert-dataset.py."
        )
    if manifest.get("version", 0) > FORMAT_VERSION:
        raise ValueError(
            f"{path} uses archive format version {manifest['version']}; "
            f"this databench reads up to {FORMAT_VERSION}."
        )
    return manifest


def read_archive(path: str | Path) -> pd.DataFrame:
    """The dataset frame, with a :class:`LazyCell` in place of each large cell."""
    store = open_store(path)
    manifest = read_manifest(path)
    frame: pd.DataFrame = pickle.loads(store.read_bytes(FRAME_MEMBER))

    by_column: dict[int, list[dict]] = {}
    for cell in manifest["cells"]:
        by_column.setdefault(cell["col"], []).append(cell)
    for col, cells in by_column.items():
        values = frame.iloc[:, col].to_numpy(dtype=object, copy=True)
        for cell in cells:
            dtype = None if cell["dtype"] is None else np.lib.format.descr_to_dtype(cell["dtype"])
            values[cell["row"]] = LazyCell(store, cell["member"], cell["shape"], dtype)
        frame.isetitem(col, values)
    return frame


def archive_sha256(path: str | Path) -> str:
    """SHA-256 of the archive's member table: each member's name, size and CRC-32.

    Changes whenever any member's content changes, without reading the
    data — the fingerprint :mod:`databench.provenance` records for archives.
    """
    h = hashlib.sha256()
    for info in open_store(path)._zip.infolist():
        h.update(f"{info.filename}\0{info.file_size}\0{info.CRC:08x}\n".encode())
    return h.hexdigest()


# ── writing ──────────────────────────────────────────────────────────────

def write_archive(
    df: pd.DataFrame,
    path: str | Path,
    *,
    inline_max_bytes: int = INLINE_MAX_BYTES,
    source: dict | None = None,
) -> dict:
    """Write *df* as a dataset archive at *path* and return its manifest.

    Every object-dtype cell whose serialised size exceeds *inline_max_bytes*
    becomes its own member; everything else stays in the frame.  *source*
    (e.g. the original pickle's name and SHA-256) is recorded in the
    manifest.  The file is written beside *path* and renamed into place,
    so a reader never sees a partial archive.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    frame = df.copy()
    unique_rows = df.index.is_unique
    cells: list[dict] = []

    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for col in range(df.shape[1]):
            if not pd.api.types.is_object_dtype(df.dtypes.iloc[col]):
                continue
            values = df.iloc[:, col].to_numpy(dtype=object, copy=True)
            moved = False
            for row, value in enumerate(values):
                encoded = _encode(value, inline_max_bytes)
                if encoded is None:
                    continue
                payload, suffix, shape, descr = encoded
                stem = _member_stem(df.index[row], df.columns[col])
                member = (stem if unique_rows else f"{row:05d}/{stem}") + suffix
                _write_member(zf, member, payload)
                cells.append({"row": row, "col": col, "member": member,
                              "shape": list(shape), "dtype": descr})
                values[row] = None
                moved = True
            if moved:
                frame.isetitem(col, values)

        manifest = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "source": source or {},
            "inline_max_bytes": inline_max_bytes,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "cells": cells,
        }
        _write_member(zf, FRAME_MEMBER, pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL))
        _write_member(zf, MANIFEST_MEMBER, json.dumps(manifest).encode("utf-8"))
    tmp.replace(path)
    return manifest


def _encode(value: Any, inline_max_bytes: int) -> tuple[bytes, str, tuple, Any] | None:
    """Serialised member for a large cell, or ``None`` to keep it inline."""
    if value is None or isinstance(value, (str, bytes, int, float, bool, np.generic)):
        return None
    if isinstance(value, np.ndarray):
        buf = io.BytesIO()
        np.lib.format.write_array(buf, value, allow_pickle=True)
        payload = buf.getvalue()
        if len(payload) <= inline_max_bytes:
            return None
        return payload, ".npy", value.shape, np.lib.format.dtype_to_descr(value.dtype)
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) <= inline_max_bytes:
        return None
    try:
        shape = np.shape(value)
    except ValueError:
        shape = ()
    return payload, ".pkl", shape, None


def _member_stem(index: Any, column: Any) -> str:
    """``Subject/Session/Task/source/feature`` with ``/`` escaped in each part."""
    parts = (index if isinstance(index, tuple) else (index,)) + (
        column if isinstance(column, tuple) else (column,)
    )
    return "/".join(str(p).replace("%", "%25").replace("/", "%2F") for p in parts)


def _write_member(zf: zipfile.ZipFile, name: str, payload: bytes) -> None:
    """Store *payload* uncompressed with its data 64-byte aligned in the file.

    Padding goes in a local-header extra field (the scheme Android's
    ``zipalign`` uses), so ``.npy`` data — whose header numpy pads to 64
    bytes — lands on an aligned offset and maps without a copy.
    """
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    data_start = zf.start_dir + _LOCAL_HEADER_SIZE + len(name.encode("utf-8")) + 4
    pad = -data_start % _ALIGN
    info.extra = struct.pack("<HH", _ALIGN_EXTRA_ID, pad) + bytes(pad)
    zf.writestr(info, payload)
