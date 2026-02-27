#%%
"""
Event-based ETA analysis for longitudinal widefield recordings.

 event-triggered averages pooled across all 10 days
 longitudinal day-by-day visualizations across sessions
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench import Bench
from databench.analysis.eta import EtaLongitudinalAnalysis
from databench.features.treadmill import LocomotionBoutEventsExtractor
from databench.plotting.eta import (
    EtaAllDaysAveragePlotter,
    EtaLongitudinalHeatmapPlotter,
    EtaLongitudinalMetricPlotter,
)


def _infer_default_rois(df: pd.DataFrame, preferred: list[str], fallback_n: int = 6) -> list[str]:
    if not isinstance(df.columns, pd.MultiIndex):
        return preferred

    mesomap_cols = [
        feat
        for source, feat in df.columns
        if source == "mesomap" and feat != "time_elapsed_s"
    ]
    mesomap_cols = list(dict.fromkeys(mesomap_cols))
    selected = [r for r in preferred if r in mesomap_cols]
    if selected:
        return selected
    return mesomap_cols[:fallback_n]


#%%
# Procedural workflow
pickle_path = Path(r"/Users/jakegronemeyer/Desktop/4jake/260212_ETOH-HFSA_dataset.pkl")
project_root = Path(__file__).resolve().parents[1]
output_root = project_root / "outputs"

bench = Bench()
bench.setup(pickle_path, 
            output_root=output_root, 
            analyst="Jacob Gronemeyer",
            lab="Sipe Lab",
            run_name="260222", 
            tag="widefield-10day-eta_2s").load()

bouts_feature = bench.get_feature("locomotion_bouts_n")

preferred_rois = ["L_MOp", "R_MOp", "L_MOs", "R_MOs", "L_VISp", "R_VISp"]
roi_cols = _infer_default_rois(bench.df, preferred_rois, fallback_n=6)
if not roi_cols:
    raise ValueError("No mesomap ROI columns were found in the dataset.")
plot_rois = roi_cols

analysis = EtaLongitudinalAnalysis(
    roi_cols=tuple(roi_cols),
    task="task-widefield",
    event_types=("onset", "offset"),
    window=(-1.0, 1.0),
    dt=0.02,
    baseline=(-2, 0.0),
    baseline_exclude_bouts=True,
    min_clean_baseline_points=3,
    fallback_to_full_baseline=True,
    metric_window=(0.0, 2.0),
    bout_events_extractor=LocomotionBoutEventsExtractor.from_feature(bouts_feature),
)

plot_avg_onset = EtaAllDaysAveragePlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="onset",
    baseline=analysis.baseline,
)
plot_avg_offset = EtaAllDaysAveragePlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="offset",
    baseline=analysis.baseline,
)

plot_long_heatmap_onset = EtaLongitudinalHeatmapPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="onset",
)
plot_long_heatmap_offset = EtaLongitudinalHeatmapPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="offset",
)

plot_long_metric_onset = EtaLongitudinalMetricPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="onset",
    metric_window=analysis.metric_window,
)
plot_long_metric_offset = EtaLongitudinalMetricPlotter(
    rois=tuple(plot_rois),
    task=analysis.task,
    event_type="offset",
    metric_window=analysis.metric_window,
)

(bench
    .build_long(sources=[
        ("mesomap", roi_cols),
        ("treadmill", ["speed_mm"]),
    ], tol=0.25, time_column="time_elapsed_s", reference_source="mesomap")
    .analyze(analysis)
    .plot(plot_avg_onset, save="eta_onset_rois_10day_avg.png")
    .plot(plot_avg_offset, save="eta_offset_rois_10day_avg.png")
    .plot(plot_long_heatmap_onset, save="eta_onset_longitudinal_heatmap.png")
    .plot(plot_long_heatmap_offset, save="eta_offset_longitudinal_heatmap.png")
    .plot(plot_long_metric_onset, save="eta_onset_longitudinal_metric.png")
    .plot(plot_long_metric_offset, save="eta_offset_longitudinal_metric.png")
    .save_tables(prefix="eta_widefield_10day"))

bench.save_run_summary(
    notes=(
        "Widefield event-based ETA workflow for a longitudinal 10-day dataset. "
        "Includes pooled 10-day averages and day-wise longitudinal visualizations."
    ),
)

bench.save_provenance()
# %%