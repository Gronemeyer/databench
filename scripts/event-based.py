#%%
"""
Event-based analysis in blessed databench style.

This script demonstrates:
1) explicit procedural flow (load -> build_long -> label -> analyze -> plot -> save)
2) decorator-registered custom analysis and plotter
3) DataFrame-based locomotion bout event API
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench import Bench
from databench.analysis.eta import EtaByConditionAnalysis
from databench.config import resolve_dataset
from databench.features.treadmill import LocomotionBoutEventsExtractor
from databench.plotting.eta import EtaConditionPlotter


#%%
# Procedural workflow

pickle_path = resolve_dataset()
project_root = Path(__file__).resolve().parents[1]
output_root = project_root / "outputs"

bench = Bench()
bench.setup(pickle_path, output_root=output_root, analyst="Jacob Gronemeyer", lab="Sipe Lab", run_name="260218", tag="spont-mop-eta").load()

bouts_feature = bench.get_feature("locomotion_bouts_n")

roi_cols = ["L_MOp", "R_MOp", "L_MOs", "R_MOs"]
plot_rois = roi_cols

analysis = EtaByConditionAnalysis(
    roi_cols=tuple(roi_cols),
    task="task-spont",
    event_types=("onset", "offset"),
    window=(-1.0, 3.0),
    dt=0.02,
    baseline=(-5.0, 0.0),
    bout_events_extractor=LocomotionBoutEventsExtractor.from_feature(bouts_feature),
)

plot_onset = EtaConditionPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="onset",
    baseline=analysis.baseline,
)
plot_offset = EtaConditionPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="offset",
    baseline=analysis.baseline,
)

ses_to_cond = {
    "ses-01": "baseline",
    "ses-02": "saline",
    "ses-03": "ethanol_low",
    "ses-04": "ethanol_high",
}

(bench
    .build_long(sources=[
        ("mesomap", roi_cols),
        ("pupil", ["pupil_diameter_mm"]),
        ("treadmill", ["speed_mm"]),
    ], tol=0.25, time_column="time_elapsed_s", reference_source="mesomap")
    .label_conditions(ses_to_cond)
    .analyze(analysis)
    .plot(plot_onset, save="eta_onset_rois.png")
    .plot(plot_offset, save="eta_offset_rois.png")
    .save_tables(prefix="eta")
)

bench.save_run_summary(
    notes="Event-based ETA workflow using registered analysis/plotter.",
)
bench.save_provenance()

# %%