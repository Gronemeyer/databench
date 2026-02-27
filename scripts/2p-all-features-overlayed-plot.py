
#%%
from databench import Bench
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%

bench = Bench()

bench.setup(
	input_path=Path(r'/Users/jakegronemeyer/Desktop/4jake/260212_ACUTEVIS_dataset.pkl'),
	output_root=Path(__file__).resolve().parents[1] / "outputs",
	analyst="Jacob Gronemeyer",
	lab="Sipe Lab",
	run_name="260216",
	tag="event-based-analysis"
)

bench.load()
df = bench.df

#%%
def plot_psychopy_roi_overview(
	dataset: pd.DataFrame,
	*,
	source: str = "suite2p",
	trace_key: str = "deltaf_f",
	time_key: str = "time_elapsed_s",
	pupil_source: str = "pupil",
	pupil_trace_key: str = "pupil_diameter_mm",
	pupil_time_key: str = "time_elapsed_s",
	encoder_source: str = "encoder",
	encoder_trace_key: str = "speed_mm",
	encoder_time_key: str = "time_elapsed_s",
	roi_index: int | None = None,
	task_filter: str = "task-gratings",
	title: str = "Psychopy ROI Overview",
	max_plots: int | None = None,
) -> None:
	"""Plot traces for every row matching the task filter."""
 
	task_mask = dataset.index.get_level_values("Task").astype(str) == task_filter
	rows = dataset.loc[task_mask]

	if max_plots:
		rows = rows.head(max_plots)

	if rows.empty:
		return

	n_rows = len(rows)
	n_axes = n_rows * 2
	fig, axes = plt.subplots(n_axes, 1, figsize=(12, max(4, 2.2 * n_axes)), sharex=False)
	if n_axes == 1:
		axes = [axes]

	trace_col = (source, trace_key)
	time_col = (source, time_key)
	pupil_col = (pupil_source, pupil_trace_key)
	pupil_time_col = (pupil_source, pupil_time_key)
	encoder_col = (encoder_source, encoder_trace_key)
	encoder_time_col = (encoder_source, encoder_time_key)

	for row_idx, (index, row) in enumerate(rows.iterrows()):
		ax_suite2p = axes[row_idx * 2]
		ax_overlay = axes[row_idx * 2 + 1]

		suite2p_trace = np.asarray(row[trace_col], dtype=float)
		suite2p_time = np.asarray(row[time_col], dtype=float)
		if suite2p_trace.ndim > 1:
			suite2p_trace = (
				np.nanmean(suite2p_trace, axis=0) if roi_index is None else suite2p_trace[roi_index]
			)
		suite2p_time = suite2p_time[:len(suite2p_trace)]
		ax_suite2p.plot(suite2p_time, suite2p_trace, color="#1f77b4", linewidth=1.0, label=trace_key)
      
		grating_starts = np.asarray(row[("psychopy", "gratings_window_grating_start")], dtype=float)
		grating_ends = np.asarray(row[("psychopy", "gratings_window_grating_stop")], dtype=float)

		for idx, (start, stop) in enumerate(zip(grating_starts, grating_ends)):
			ax_suite2p.axvspan(
				start,
				stop,
				color="#ff7f0e",
				alpha=0.2,
				label="grating" if idx == 0 else None,
			)
			ax_overlay.axvspan(start, stop, color="#ff7f0e", alpha=0.2)

		gray_starts = np.asarray(row[("psychopy", "gratings_window_gray_start")], dtype=float)
		gray_ends = np.asarray(row[("psychopy", "gratings_window_gray_stop")], dtype=float)

		for idx, (start, stop) in enumerate(zip(gray_starts, gray_ends)):
			ax_suite2p.axvspan(
				start,
				stop,
				color="#7f7f7f",
				alpha=0.15,
				label="gray" if idx == 0 else None,
			)
			ax_overlay.axvspan(start, stop, color="#7f7f7f", alpha=0.15)

		pupil_trace = np.asarray(row[pupil_col], dtype=float)
		pupil_time = np.asarray(row[pupil_time_col], dtype=float)
		pupil_time = pupil_time[:len(pupil_trace)]
		ax_overlay.plot(pupil_time, pupil_trace, color="#2ca02c", linewidth=1.0, label=pupil_trace_key)

		encoder_trace = np.asarray(row[encoder_col], dtype=float)
		encoder_time = np.asarray(row[encoder_time_col], dtype=float)
		encoder_time = encoder_time[:len(encoder_trace)]
		ax_overlay.plot(encoder_time, encoder_trace, color="#d62728", linewidth=1.0, label=encoder_trace_key)

		label = " | ".join(str(item) for item in index) if isinstance(index, tuple) else str(index)
		ax_suite2p.set_title(label)
		ax_suite2p.set_ylabel(trace_key)
		ax_overlay.set_ylabel("Pupil/Speed")
		ax_suite2p.grid(True, alpha=0.3)
		ax_overlay.grid(True, alpha=0.3)
		ax_overlay.set_xlabel("Time (s)")

	fig.suptitle(title)
	axes[0].legend(loc="upper right")
	if len(axes) > 1:
		axes[1].legend(loc="upper right")
	fig.tight_layout()
	plt.show()


plot_psychopy_roi_overview(
	df,
	source="suite2p",
	trace_key="deltaf_f",
	time_key="time_elapsed_s",
	roi_index=4,
	task_filter="task-gratings",
	title="2p ROI overview",
	max_plots=6,
)

bench.save_provenance()

# %%
