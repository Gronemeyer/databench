"""
Canonical event-triggered average script.

Computes locomotion-bout-triggered ETA for mesomap ROIs across sessions.
Supports both:
    1. Condition-aware analysis (map session to condition labels)
    2. Condition-agnostic analysis (all events pooled)

Usage:
    python Scripts/event-triggered/event-based.py
"""
from databench.project import Project
from databench.analysis.eta import EtaAnalysis
from databench.analysis.locomotion import locomotion_events
from databench.config import resolve_dataset
from databench.plotting import set_theme

set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET = resolve_dataset("etoh")
ROI_COLUMNS = ("L_MOp", "R_MOp", "L_MOs", "R_MOs")
TASK = "task-spont"
WINDOW = (-1.0, 3.0)
BASELINE = (-5.0, 0.0)

# Optional: set to a tuple like ("onset", "offset") to force specific plots.
# Leave as None to plot all event types present in the result.
PLOT_EVENT_TYPES = None

SESSION_TO_CONDITION = {
    "ses-01": "baseline",
    "ses-02": "saline",
    "ses-03": "ethanol_low",
    "ses-04": "ethanol_high",
}

# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name="MOp-MOs_spont",
    tag="dev",
)

group = proj.sessions(task=TASK)
print(f"Selected {len(group)} sessions for task={TASK}")

# ─── Detect locomotion events ────────────────────────────────────────────

events = locomotion_events(
    group,
    speed_source="treadmill",
    speed_column="speed_mm",
    speed_scale_to_cms=10.0,
    min_speed_cms=0.5,
    min_duration_s=1.0,
    merge_gap_s=0.5,
)
if SESSION_TO_CONDITION:
    events["Condition"] = events["Session"].map(SESSION_TO_CONDITION)
print(f"Detected {len(events)} locomotion events across {events['Subject'].nunique()} subjects")

# ─── Configure and run ETA ───────────────────────────────────────────────

eta = EtaAnalysis(
    roi_columns=ROI_COLUMNS,
    window=WINDOW,
    baseline=BASELINE,
    source="mesomap",
    reference_source="mesomap",
    alignment_tolerance_s=0.25,
)

result = eta.run(group, events)

# ─── Plot & save ─────────────────────────────────────────────────────────

event_types = PLOT_EVENT_TYPES if PLOT_EVENT_TYPES is not None else result.event_types
for event_type in event_types:
    result.plot(event=event_type, rois=list(ROI_COLUMNS)).save(f"eta_{event_type}_rois.svg")

result.save_tables(prefix="eta")
result.save_summary("eta_summary.json")

report_path = proj.save_report(
    result,
    notes="Event-based ETA for mesomap ROIs aligned to locomotion onset/offset across conditions.",
)

print(f"Done — {len(result.events)} events across {result.events['Subject'].nunique()} subjects.")
print(f"Report: {report_path}")
print(f"Outputs in: {proj._context.run_dir}")

