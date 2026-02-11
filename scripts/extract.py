from __future__ import annotations

from pathlib import Path

import pandas as pd

from databench import Bench
from databench.debug import subset_df


DATASET_PATH = Path(r"/Volumes/ake.bin/Projects/RO1_ETOH/260116_dataset_mvp.pkl")

# Optional filters
SUBJECT = None
SESSION = None
TASK = None

# One CSV per output name. Each output can try multiple (source, feature) candidates.
# The first match found in df.columns is used.
EXTRACT_FEATURES = {
    "locomotion_speed_mm": [
        ("treadmill", "speed_mm"),
        ("encoder", "speed"),
    ],
    "locomotion_distance_mm": [
        ("treadmill", "distance_mm"),
        ("encoder", "distance"),
    ],
    "pupil_diameter_mm": [
        ("pupil", "pupil_diameter_mm"),
        ("pupil", "diameter_mm"),
    ],
    # "roi_R_VISp": [("mesomap", "R_VISp")],
}


def main() -> None:
    bench = Bench()
    bench.setup(DATASET_PATH, run_name="feature_extraction", tag="ETOH_R01")

    df = bench.load()
    if SUBJECT is not None or SESSION is not None or TASK is not None:
        df = subset_df(df, subject=SUBJECT, session=SESSION, task=TASK)

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns (source, feature).")

    available = set(df.columns)
    for out_name, candidates in EXTRACT_FEATURES.items():
        if not candidates:
            print(f"Skipping {out_name}: no candidates provided.")
            continue

        chosen = None
        for source, feature in candidates:
            if (source, feature) in available:
                chosen = (source, feature)
                break

        if chosen is None:
            cand_str = ", ".join([f"({s}, {f})" for s, f in candidates])
            print(f"No match found for {out_name}. Tried: {cand_str}")
            continue

        source, feature = chosen
        file_name = f"{out_name}.csv".replace("/", "-")
        out_path = bench.export_feature_report(
            df,
            source=source,
            feature=feature,
            name=file_name,
            folder="stats",
        )
        print(f"Saved {out_name} CSV to: {out_path}")


if __name__ == "__main__":
    main()
