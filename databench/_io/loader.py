"""Dataset loading from various file formats."""
from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a dataset from disk.

    Supported formats: ``.pkl``, ``.h5``, ``.parquet``, ``.csv``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix in {".h5", ".hdf", ".hdf5"}:
        return cast(pd.DataFrame, pd.read_hdf(path, key="HFSA"))
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported file format: {path.suffix!r}. "
        f"Use .pkl, .h5, .parquet, or .csv."
    )
