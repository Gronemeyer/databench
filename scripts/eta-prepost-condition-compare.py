#%%
"""
Compare ETA pre/post dF/F differences for locomotion onset vs offset.

This script demonstrates:
1) explicit procedural flow (load -> build_long -> label -> analyze -> plot -> save)
2) custom analysis + plotter classes reusing databench interfaces
3) DataFrame-based locomotion bout event API
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench import Bench
from databench.analysis.eta import EtaPrePostDiffAnalysis
from databench.config import resolve_dataset
from databench.features.treadmill import LocomotionBoutEventsExtractor
from databench.plotting.eta import EtaPrePostDiffBoxplot


#%%
# Procedural workflow

pickle_path = resolve_dataset()
project_root = Path(__file__).resolve().parents[1]
output_root = project_root / "outputs"

bench = Bench()
bench.setup(pickle_path, output_root=output_root, analyst="Jacob Gronemeyer", lab="Sipe Lab", run_name="260215", tag="eta-prepost-condition_vis-primary-secondary")

bouts_feature = bench.get_feature("locomotion_bouts_n")

#roi_cols = ["L_MOp", "R_MOp", "L_MOs", "R_MOs"]
roi_cols = ["L_VISp", "L_VISa", "L_SSp-ll", "L_SSp-bfd"]  # alternate set of ROIs to compare

ses_to_cond = {
    "ses-01": "baseline",
    "ses-02": "saline",
    "ses-03": "ethanol_low",
    "ses-04": "ethanol_high",
}

analysis = EtaPrePostDiffAnalysis(
    roi_cols=tuple(roi_cols),
    task="task-spont",
    pre_window=(-1.0, 0.0),
    post_window=(0.0, 1.0),
    dt=0.02,
    reverse_for_offset=False,
    bout_events_extractor=LocomotionBoutEventsExtractor.from_feature(bouts_feature),
)

plotter = EtaPrePostDiffBoxplot(
    rois=tuple(roi_cols),
    task=analysis.task,
    condition_col=analysis.condition_col,
)

with bench.run("eta-prepost-condition") as run:
    (run
        .build_long(sources=[
            ("mesomap", roi_cols),
            ("treadmill", ["speed_mm"]),
        ], tol=0.25, time_column="time_elapsed_s", reference_source="mesomap")
        .label_conditions(ses_to_cond))

    result = run.analyze(analysis, df=run.long)
    run.plot(plotter, result, save="eta_prepost_diff_boxplot.png")
    run.save_tables(result, prefix="eta_prepost_diff")
    run.save_run_summary(
        notes="ETA pre/post difference comparison across conditions for onset/offset locomotion events.",
    )

# %%