"""
Event-triggered averages for 2p ROI traces aligned to locomotion onset/offset.

Computes mean ΔF/F across suite2p ROIs, then runs ETA at locomotion bout
onset/offset, split by injection condition.

Usage:
    python Scripts/2p-event-triggered-eta.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from databench.project import Project
from databench.analysis.eta import EtaAnalysis
from databench.analysis.locomotion import locomotion_events
from databench.config import resolve_dataset
from databench.plotting import set_theme

set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET = resolve_dataset("acutevis")
TASK = "task-gratings"
ROI_FEATURE = "deltaf_f"
MEAN_FEATURE = f"{ROI_FEATURE}_mean"

INJECTION_MAP = {
    "baseline": "baseline",
    "saline": "saline",
    "ethanol_low": "low",
    "ethanol_high": "high",
    "low": "low",
    "high": "high",
}
CONDITION_ORDER = ("baseline", "saline", "low", "high")
CONDITION_COLORS = {
    "baseline": "#bbabab",
    "saline": "#4289e6",
    "low": "#ffa251",
    "high": "#ce1818",
}

# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name="2p-locomotion-eta",
    tag="ACUTEVIS",
)

# ─── Compute mean suite2p trace if needed ─────────────────────────────────

raw = proj._df
mean_col = ("suite2p", MEAN_FEATURE)
if mean_col not in raw.columns:
    def _mean_trace(row: pd.Series) -> np.ndarray | float:
        arr = np.asarray(row.get(("suite2p", ROI_FEATURE)))
        if arr.ndim == 2:
            return np.nanmean(arr, axis=0)
        return arr if arr.ndim == 1 else np.nan

    raw[mean_col] = raw.apply(_mean_trace, axis=1)

# ─── Build condition map from session_config ──────────────────────────────

injection_col = ("session_config", "injection")
condition_map: dict[str, str] = {}
if injection_col in raw.columns:
    for idx in raw.index:
        subj, ses, task = idx
        inj = str(raw.loc[idx, injection_col]).strip().lower()
        cond = INJECTION_MAP.get(inj, inj)
        condition_map[f"{subj},{ses},{task}"] = cond

# ─── Select sessions and detect events ────────────────────────────────────

group = proj.sessions(task=TASK)
print(f"Selected {len(group)} sessions for task={TASK}")

events = locomotion_events(
    group,
    speed_source="encoder",
    speed_column="speed_mm",
    speed_scale_to_cms=10.0,
    min_speed_cms=0.05,
    min_duration_s=1.0,
    merge_gap_s=0.5,
)

# Add condition from injection map
events["Condition"] = events.apply(
    lambda r: condition_map.get(f"{r['Subject']},{r['Session']},{r['Task']}", "unknown"),
    axis=1,
)
events["Condition"] = pd.Categorical(
    events["Condition"], categories=list(CONDITION_ORDER), ordered=True,
)
print(f"Detected {len(events)} locomotion events")

# ─── Pre-align data (suite2p + encoder) ──────────────────────────────────

aligned_list = []
for sess in group:
    ad = sess.align(
        {"suite2p": [MEAN_FEATURE], "encoder": ["speed_mm"]},
        reference="suite2p",
        tolerance_s=0.25,
    )
    aligned_list.append(ad)

# ─── Run ETA ─────────────────────────────────────────────────────────────

eta = EtaAnalysis(
    roi_columns=(MEAN_FEATURE,),
    window=(-1.0, 3.0),
    dt=0.02,
    baseline=(-5.0, 0.0),
    source="suite2p",
    reference_source="suite2p",
)

result = eta.run(group, events, aligned=aligned_list)

# ─── Plot & save ─────────────────────────────────────────────────────────

for event_type in result.event_types:
    result.plot(
        event=event_type,
        rois=[MEAN_FEATURE],
        condition_colors=CONDITION_COLORS,
        conditions=list(CONDITION_ORDER),
    ).save(f"eta_locomotion_{event_type}.svg")

result.save_tables(prefix="eta_locomotion")

report_path = proj.save_report(
    result,
    notes="Event-triggered ETA for 2p ROI traces aligned to locomotion onset/offset by injection condition.",
)

print(f"Done — {len(result.events)} events across {result.events['Subject'].nunique()} subjects.")
print(f"Report: {report_path}")
