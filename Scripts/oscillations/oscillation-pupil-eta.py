"""
Oscillation-triggered pupil ETA — two-stage provenance chain.

Stage 1: Detect 2–4 Hz oscillation bursts in a detection ROI using
         Hilbert envelope thresholding.
Stage 2: Event-triggered average of a comparison ROI and pupil diameter
         aligned to burst onset/offset.

Usage:
    python Scripts/oscillation-pupil-eta.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from databench.project import Project
from databench.analysis.eta import EtaAnalysis
from databench.analysis.oscillation import OscillationDetector
from databench.signal.epoching import make_events
from databench.config import resolve_dataset
from databench.plotting import set_theme
set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET = resolve_dataset("etoh-hfsa")

ROI_SOURCE = "mesomap"
DETECT_ROI = "L_VISp"
ETA_ROI = "L_VISp"
PUPIL_SOURCE = "pupil"
PUPIL_KEY = "pupil_diameter_mm"
TASK = "task-widefield"

ETA_WINDOW = (-2.0, 3.0)
ETA_DT = 0.02
ETA_BASELINE = (-2.0, -0.5)
EDGE_PAD_S = max(abs(ETA_WINDOW[0]), abs(ETA_WINDOW[1]))

OSC_FS = 50.0
OSC_BAND = (2.0, 4.0)
OSC_ORDER = 4
OSC_THRESHOLD = 0.02
OSC_MIN_DURATION = 1.0
OSC_MERGE_GAP = 0.5

BAND_LABEL = f"{OSC_BAND[0]}–{OSC_BAND[1]} Hz"

# ─── Project setup ───────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    run_name="osc-eta-detect-compare",
    tag="L_VISp",
).filter(include={"session": "ses-00"})

group = proj.sessions(task=TASK)
print(f"Selected {len(group)} sessions for task={TASK}")

# ─── Stage 1: Oscillation burst detection ────────────────────────────────

detector = OscillationDetector(
    source=ROI_SOURCE,
    signal=DETECT_ROI,
    fs=OSC_FS,
    band_hz=OSC_BAND,
    filter_order=OSC_ORDER,
    threshold=OSC_THRESHOLD,
    min_duration_s=OSC_MIN_DURATION,
    merge_gap_s=OSC_MERGE_GAP,
)

osc_results: list[tuple] = []  # (OscillationResult, Session)
all_events: list[pd.DataFrame] = []

for sess in group:
    result = detector.run(sess)
    osc_results.append((result, sess))

    if result.events.empty:
        continue

    # Edge-exclude: drop events whose ETA window extends beyond recording
    t_max = float(result.time[-1])
    valid = result.events[
        (result.events["start_s"] >= EDGE_PAD_S)
        & (result.events["end_s"] <= t_max - EDGE_PAD_S)
    ]
    if valid.empty:
        continue

    ev = make_events(
        {"onset": valid["start_s"].values, "offset": valid["end_s"].values},
        subject=sess.subject,
        session=sess.session,
        task=sess.task,
    )
    all_events.append(ev)

if all_events:
    events = pd.concat(all_events, ignore_index=True)
else:
    events = pd.DataFrame(columns=["Subject", "Session", "Task", "EventType", "event_time"])

n_bursts = len(events) // 2
print(f"Detected {n_bursts} valid bursts → {len(events)} events")

# Save burst events table
proj.io.table(events, "oscillation_burst_events.csv")

# ─── Stage 2: ETA at oscillation events ──────────────────────────────────

# Pre-align multi-source data for ETA (mesomap + pupil)
aligned_list = []
for sess in group:
    ad = sess.align(
        {ROI_SOURCE: [DETECT_ROI, ETA_ROI], PUPIL_SOURCE: [PUPIL_KEY]},
        reference=ROI_SOURCE,
        tolerance_s=0.25,
    )
    aligned_list.append(ad)

eta = EtaAnalysis(
    roi_columns=(ETA_ROI, PUPIL_KEY),
    window=ETA_WINDOW,
    dt=ETA_DT,
    baseline=ETA_BASELINE,
    source=ROI_SOURCE,
    reference_source=ROI_SOURCE,
)

eta_result = eta.run(group, events, aligned=aligned_list)

# ─── Plot ETA ────────────────────────────────────────────────────────────

for event_type in eta_result.event_types:
    fig = eta_result.plot(
        event=event_type,
        rois=[ETA_ROI, PUPIL_KEY],
    )
    proj.io.figure(fig, f"osc_eta_{event_type}.svg")

for key, df in eta_result.tables.items():
    proj.io.table(df, f"osc_eta_detect_compare_{key}.csv")

# ─── Stage 3: Oscillation detection PDF report ──────────────────────────

with proj.io.pdf("oscillation_detection_report.pdf") as pdf:
    for osc_result, sess in osc_results:
        if osc_result.events.empty:
            continue
        ad = sess.align(
            {ROI_SOURCE: [DETECT_ROI], PUPIL_SOURCE: [PUPIL_KEY]},
            reference=ROI_SOURCE,
            tolerance_s=0.25,
        )
        fig = osc_result.plot_overview(aligned=ad, pupil=PUPIL_KEY)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print("PDF report saved.")

# ─── Report ──────────────────────────────────────────────────────────────

proj.io.report(
    eta_result,
    notes=(
        f"Two-stage: (1) oscillation detection ({BAND_LABEL} Hilbert envelope, "
        f"threshold={OSC_THRESHOLD}) in {ROI_SOURCE}/{DETECT_ROI}, "
        f"(2) ETA of {ETA_ROI} and {PUPIL_KEY} at burst onset/offset. "
        f"Edge events within {EDGE_PAD_S}s excluded. "
        f"Aggregation: per-event → per-session mean → group mean ± SEM."
    ),
)

n_subj = eta_result.events["Subject"].nunique() if not eta_result.events.empty else 0
print(f"Done — {len(eta_result.events)} ETA events across {n_subj} subjects.")
