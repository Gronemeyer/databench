from __future__ import annotations

from pathlib import Path

import pandas as pd

from databench import Bench
from databench.debug import subset_df


from databench.config import resolve_dataset
DATASET = resolve_dataset()

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


bench = Bench()
bench.setup(
    DATASET, 
    analyst="Jacob Gronemeyer", 
    lab="Sipe Lab", 
    run_name="feature_extraction", 
    tag="ETOH_R01")

df = bench.df
df = subset_df(df, subject=SUBJECT, session=SESSION, task=TASK)

available = set(df.columns)

with bench.run("feature-extraction") as run:
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
        out_path = run.export_feature_report(
            df,
            source=source,
            feature=feature,
            name=file_name,
            folder="stats",
        )
        print(f"Saved {out_name} CSV to: {out_path}")

    run.save_run_summary(
        notes="Batch feature extraction to CSV for locomotion, pupil, and ROI signals.",
    )
