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


#%%

from databench.config import resolve_dataset
DATASET = resolve_dataset()

bench = Bench()
bench.setup(
    DATASET, 
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name="widefield-10day", 
    tag="ROI-speed_eta-2s")

bouts_feature = bench.get_feature("locomotion_bouts_n")

REGIONS = ["L_MOp", "R_MOp", "L_MOs", "R_MOs", "L_VISp", "R_VISp"]

analysis = EtaLongitudinalAnalysis(
    roi_cols=tuple(REGIONS),
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
    rois=tuple(REGIONS),
    task=analysis.task,
    event_type="onset",
    baseline=analysis.baseline,
)
plot_avg_offset = EtaAllDaysAveragePlotter(
    rois=tuple(REGIONS),
    task=analysis.task,
    event_type="offset",
    baseline=analysis.baseline,
)
plot_long_heatmap_onset = EtaLongitudinalHeatmapPlotter(
    rois=tuple(REGIONS),
    task=analysis.task,
    event_type="onset",
)
plot_long_heatmap_offset = EtaLongitudinalHeatmapPlotter(
    rois=tuple(REGIONS),
    task=analysis.task,
    event_type="offset",
)
plot_long_metric_onset = EtaLongitudinalMetricPlotter(
    rois=tuple(REGIONS),
    task=analysis.task,
    event_type="onset",
    metric_window=analysis.metric_window,
)
plot_long_metric_offset = EtaLongitudinalMetricPlotter(
    rois=tuple(REGIONS),
    task=analysis.task,
    event_type="offset",
    metric_window=analysis.metric_window,
)

with bench.run("widefield-10day-eta") as run:
    (run
        .build_long(sources=[
            ("mesomap", REGIONS),
            ("treadmill", ["speed_mm"]),
        ], tol=0.25, time_column="time_elapsed_s", reference_source="mesomap"))

    result = run.analyze(analysis, df=run.long)
    run.plot(plot_avg_onset, result, save="eta_onset_rois_10day_avg.png")
    run.plot(plot_avg_offset, result, save="eta_offset_rois_10day_avg.png")
    run.plot(plot_long_heatmap_onset, result, save="eta_onset_longitudinal_heatmap.png")
    run.plot(plot_long_heatmap_offset, result, save="eta_offset_longitudinal_heatmap.png")
    run.plot(plot_long_metric_onset, result, save="eta_onset_longitudinal_metric.png")
    run.plot(plot_long_metric_offset, result, save="eta_offset_longitudinal_metric.png")
    run.save_tables(result, prefix="eta_widefield_10day")
    run.save_run_summary(
        notes=(
            "Widefield event-based ETA workflow for a longitudinal 10-day dataset. "
            "Includes pooled 10-day averages and day-wise longitudinal visualizations."
        ),
    )

# %%