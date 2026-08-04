"""Interactive single-session trace plotter (+ optional OscillationDetector test).

Run as cells in VS Code using ``#%%`` blocks, or run as a normal script.
"""

#%% Imports
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from databench import Project, resolve_dataset, set_theme
from databench.analysis.oscillation import OscillationDetector

set_theme()


#%% Select one session and one ROI

DATASET = resolve_dataset("etoh-hfsa")
# DATASET = r"/Volumes/Untitled/Projects/data/260513_ETOH-HFSA.pkl"
project = Project(DATASET)

#%%
SUBJECT = "GS29"
SESSION = "ses-04"
TASK = "task-widefield"
SOURCE = "mesomap"
ROI = "R_MOp"

# Optional detector settings for quick single-trace testing.
FS = 50.0
BAND_HZ = (2.0, 4.0)
FILTER_ORDER = 4
THRESHOLD = None
THRESHOLD_K = 4.0
MIN_DURATION_S = 1.0
MERGE_GAP_S = 0.5

selected_session = project.session(subject=SUBJECT, session=SESSION, task=TASK)

#%% Load the selected session and trace

time_seconds = selected_session.time(SOURCE)
roi_trace = selected_session.signal(SOURCE, ROI)

sample_count = min(len(time_seconds), len(roi_trace))
time_seconds = time_seconds[:sample_count]
roi_trace = roi_trace[:sample_count]

print(f"Loaded {SUBJECT} | {SESSION} | {TASK} | {SOURCE}/{ROI} (n={sample_count})")


# Plot raw trace
figure, axis = plt.subplots(figsize=(11, 4))
axis.plot(time_seconds, roi_trace, lw=0.9, color="C0")
axis.set_xlabel("Time (s)")
axis.set_ylabel("dF/F")
axis.set_title(f"Raw trace: {SUBJECT} | {SESSION} | {TASK} | {SOURCE}/{ROI}")


#%% Align and plot ROI, pupil, and treadmill speed
aligned = selected_session.align(
    {
        SOURCE: [ROI],
        "pupil": ["pupil_diameter_mm"],
        "treadmill": ["speed_mm"],
    },
    reference=SOURCE,
    tolerance_s=0.25,
)

aligned_df = aligned.df

figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

axes[0].plot(aligned_df[aligned.time_column], aligned_df[ROI], lw=0.9, color="C0")
axes[0].set_ylabel("dF/F")
axes[0].set_title(f"Aligned traces: {SUBJECT} | {SESSION} | {TASK}")

axes[1].plot(
    aligned_df[aligned.time_column],
    aligned_df["pupil_diameter_mm"],
    lw=0.9,
    color="C2",
)
axes[1].set_ylabel("Pupil (mm)")

axes[2].plot(
    aligned_df[aligned.time_column],
    aligned_df["speed_mm"],
    lw=0.9,
    color="C1",
)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Speed (mm/s)")


#%% Run OscillationDetector on the same trace (optional)
detector = OscillationDetector(
    source=SOURCE,
    signal=ROI,
    fs=FS,
    band_hz=BAND_HZ,
    filter_order=FILTER_ORDER,
    threshold=THRESHOLD,
    threshold_k=THRESHOLD_K,
    min_duration_s=MIN_DURATION_S,
    merge_gap_s=MERGE_GAP_S,
)

result = detector.run(selected_session)
print(
    f"Bursts: {len(result.bursts)} | "
    f"threshold={result.threshold_value:.5f} | "
    f"band={BAND_HZ[0]}-{BAND_HZ[1]} Hz"
)


# Plot detector outputs (filtered + envelope + burst windows)
figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

ax1.plot(result.time, result.raw_signal, lw=0.7, color="0.6", label="raw")
ax1.plot(result.time, result.filtered_signal, lw=0.9, color="C3", label="filtered")
ax1.set_ylabel("dF/F")
ax1.set_title(f"OscillationDetector test: {SUBJECT} | {SESSION} | {SOURCE}/{ROI}")
ax1.legend(frameon=False)

ax2.plot(result.time, result.envelope, lw=1.0, color="C2", label="envelope")
ax2.axhline(result.threshold_value, ls="--", lw=1.0, color="C1", label="threshold")
for start_idx, end_idx in result.bursts:
    ax2.axvspan(result.time[start_idx], result.time[end_idx], color="C1", alpha=0.2)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Envelope")
ax2.legend(frameon=False)


#%% Show first burst rows as a quick table
if result.events.empty:
    print("No bursts detected.")
else:
    display_cols = ["start_s", "end_s", "duration_s", "peak_env"]
    preview = result.events[display_cols].head(10)
    print(preview.to_string(index=False))

# %%
