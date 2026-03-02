# %%
from __future__ import annotations

from databench import Bench
from databench.analysis.eta import EtaByConditionAnalysis
from databench.config import resolve_dataset
from databench.features.treadmill import LocomotionBoutEventsExtractor
from databench.plotting.eta import EtaConditionPlotter


DATASET = resolve_dataset()
REGIONS = ("L_MOp", "R_MOp", "L_MOs", "R_MOs")
TASK = "task-spont"
BASELINE_S = (-5.0, 0.0)
WINDOW_S = (-1.0, 3.0)
STEP_S = 0.02

SESSION_TO_CONDITION = {
    "ses-01": "baseline",
    "ses-02": "saline",
    "ses-03": "ethanol_low",
    "ses-04": "ethanol_high",
}

bench = Bench()
bench.setup(
    DATASET,
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name="event-triggered-average",
    tag="MOp-MOs_spont",
)

bout_feature = bench.get_feature("locomotion_bouts_n")

analysis = EtaByConditionAnalysis(
    roi_cols=REGIONS,
    task=TASK,
    event_types=("onset", "offset"),
    window=WINDOW_S,
    dt=STEP_S,
    baseline=BASELINE_S,
    bout_events_extractor=LocomotionBoutEventsExtractor.from_feature(bout_feature),
)

plotters = {
    event_type: EtaConditionPlotter(
        rois=REGIONS,
        task=analysis.task,
        event_type=event_type,
        baseline=analysis.baseline,
    )
    for event_type in analysis.event_types
}

with bench.run("spont-mop-eta") as run:
    frame = (
        run.build_long(
            sources=[
                ("mesomap", REGIONS),
                ("pupil", ["pupil_diameter_mm"]),
                ("treadmill", ["speed_mm"]),
            ],
            tol=0.25,
            time_column="time_elapsed_s",
            reference_source="mesomap",
        )
        .label_conditions(SESSION_TO_CONDITION)
    )

    result = run.analyze(analysis, df=frame)

    for event_type, plotter in plotters.items():
        run.plot(plotter, result, save=f"eta_{event_type}_rois.png")

    run.save_tables(result, prefix="eta")
    run.save_run_summary(
        notes="Event-based ETA workflow using registered analysis/plotter."
    )

