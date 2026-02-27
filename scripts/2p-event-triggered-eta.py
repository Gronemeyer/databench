#%%
"""
Event-triggered averages for ROI traces aligned to locomotion onset/offset.

This script mirrors the event-based workflow:
- explicit procedural flow (load -> build_long -> label -> analyze -> plot -> save)
- databench Analysis + Plotter classes
- locomotion bout event extraction
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
from databench.plotting.eta import EtaSubjectPlotter

#%%
# Procedural workflow

pickle_path = resolve_dataset()
project_root = Path(__file__).resolve().parents[1]
output_root = project_root / "outputs"

bench = Bench()
bench.setup(pickle_path, output_root=output_root, analyst="Jacob Gronemeyer", lab="Sipe Lab", run_name="260217", tag="2p-locomotion-eta").load()

raw = bench.df

roi_feature = "deltaf_f"
mean_feature = f"{roi_feature}_mean"
roi_cols = [mean_feature]
plot_rois = roi_cols

mean_col = ("suite2p", mean_feature)
if mean_col not in raw.columns:
    def mean_suite2p_trace(row: pd.Series) -> np.ndarray | float:
        trace_matrix = row.get(("suite2p", roi_feature))
        if trace_matrix is None:
            return np.nan
        arr = np.asarray(trace_matrix)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            return np.nanmean(arr, axis=0)
        return np.nan

    raw[mean_col] = raw.apply(mean_suite2p_trace, axis=1)

task = "task-gratings"
min_speed_cms = 0.05
min_duration_s = 1.0
merge_gap_s = 0.5

analysis = EtaByConditionAnalysis(
    name="eta_by_locomotion",
    roi_cols=tuple(roi_cols),
    task=task,
    event_types=("onset", "offset"),
    window=(-1.0, 3.0),
    dt=0.02,
    baseline=(-5.0, 0.0),
    bout_events_extractor=LocomotionBoutEventsExtractor(
        group_cols=("Subject", "Session", "Task"),
        time_col="time_elapsed_s",
        speed_col="speed_mm",
        speed_scale_to_cms=10.0,
        min_speed_cms=min_speed_cms,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
    ),
)

eta_cond_colors = {
    "baseline": "#bbabab",
    "saline": "#4289e6",
    "ethanol_low": "#ffa251",
    "ethanol_high": "#ce1818",
    "low": "#ffa251",
    "high": "#ce1818",
}
eta_cond_order = ("baseline", "saline", "low", "high")

plot_onset = EtaSubjectPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="onset",
    baseline=analysis.baseline,
    condition_colors=eta_cond_colors,
    condition_order=eta_cond_order,
)
plot_offset = EtaSubjectPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="offset",
    baseline=analysis.baseline,
    condition_colors=eta_cond_colors,
    condition_order=eta_cond_order,
)

bench.build_long(
    raw,
    sources=[
        ("suite2p", [mean_feature]),
        ("encoder", ["speed_mm"]),
    ],
    tol=0.25,
    time_column="time_elapsed_s",
    reference_source="suite2p",
)

# Map session_config injection → Condition
config_cols = ["injection"]
cfg = raw["session_config"][config_cols].copy()
cfg = cfg.rename(columns={"injection": "Condition"})
cfg["Condition"] = cfg["Condition"].astype(str).str.strip().str.lower()
condition_map = {
    "baseline": "baseline",
    "saline": "saline",
    "ethanol_low": "low",
    "ethanol_high": "high",
    "low": "low",
    "high": "high",
}
cfg["Condition"] = cfg["Condition"].map(condition_map)
cfg["Condition"] = pd.Categorical(
    cfg["Condition"],
    categories=["baseline", "saline", "low", "high"],
    ordered=True,
)

# Join condition labels into long table
bench._long = bench.long.join(cfg, on=["Subject", "Session", "Task"])

if task not in set(bench.long["Task"].unique()):
    raise ValueError(f"Requested task {task!r} not found in long table.")

(bench
    .analyze(analysis)
    .plot(plot_onset, save="eta_locomotion_onset.png")
    .plot(plot_offset, save="eta_locomotion_offset.png")
    .save_tables(prefix="eta_locomotion"))

bench.save_run_summary(
    notes="Event-triggered ETA for ROI traces aligned to locomotion onset/offset.",
)
bench.save_provenance()

# %%