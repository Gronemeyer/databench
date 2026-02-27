
#%%
from databench import Bench
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Sequence
#%%

bench = Bench()

(bench
	.setup(
		input_path=Path(r'/Users/jakegronemeyer/Desktop/4jake/260212_ACUTEVIS_dataset.pkl'),
		output_root=Path(__file__).resolve().parents[1] / "outputs",
		analyst="Jacob Gronemeyer",
		lab="Sipe Lab",
		run_name="260216",
		tag="event-based-analysis",
	)
	.load())

df = bench.df

#%%
def plot_psychopy_roi_overview(
	dataset: pd.DataFrame,
	*,
	source: str = "suite2p",
	trace_key: str = "deltaf_f",
	time_key: str = "time_elapsed_s",
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

	fig, axes = plt.subplots(len(rows), 1, figsize=(12, max(3, 2.5 * len(rows))), sharex=False)

	trace_col = (source, trace_key)
	time_col = (source, time_key)

	for ax, (index, row) in zip(axes, rows.iterrows()):
		trace = np.asarray(row[trace_col], dtype=float)
		time = np.asarray(row[time_col], dtype=float)
		trace = trace[roi_index]
		time = time[:len(trace)]

		ax.plot(time, trace, color="#1f77b4", linewidth=1.0)
      
		grating_starts = np.asarray(row[("psychopy", "gratings_window_grating_start")], dtype=float)
		grating_ends = np.asarray(row[("psychopy", "gratings_window_grating_stop")], dtype=float)

		for idx, (start, stop) in enumerate(zip(grating_starts, grating_ends)):
			ax.axvspan(
				start,
				stop,
				color="#ff7f0e",
				alpha=0.2,
				label="grating" if idx == 0 else None,
			)

		gray_starts = np.asarray(row[("psychopy", "gratings_window_gray_start")], dtype=float)
		gray_ends = np.asarray(row[("psychopy", "gratings_window_gray_stop")], dtype=float)

		for idx, (start, stop) in enumerate(zip(gray_starts, gray_ends)):
			ax.axvspan(
				start,
				stop,
				color="#7f7f7f",
				alpha=0.15,
				label="gray" if idx == 0 else None,
			)

		label = " | ".join(str(item) for item in index) if isinstance(index, tuple) else str(index)
		ax.set_title(label)
		ax.set_ylabel(trace_key)
		ax.grid(True, alpha=0.3)

	axes[-1].set_xlabel("Time (s)")
	fig.suptitle(title)
	axes[0].legend(loc="upper right")
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
