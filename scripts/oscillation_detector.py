from __future__ import annotations

from pathlib import Path
from typing import Optional

from databench import Bench
from databench.analysis import (
    OscillationDetectorAnalysis,
    OscillationDetectorConfig,
    context_from_index,
)
from databench.debug import get_row
from databench.utils import as_1d


DATASET = Path(r"/Volumes/ake.bin/Projects/RO1_ETOH/260120_dataset_mvp.pkl")
SUBJECT = "GS27"
SESSION = "10"
TASK = "widefield"
ROI_SOURCE = "mesomap"
ROI_NAME = "R_VISp"
PUPIL_SOURCE = "pupil"
PUPIL_KEY = "pupil_diameter_mm"

FS: float = 50.0
BAND_LO: float = 3.1
BAND_HI: float = 4.3
ORDER: int = 4
K: float = 0.5
MIN_DURATION: float = 1
MERGE_GAP: float = 5

TIME_KEY: Optional[str] = None
NO_PLOT: bool = False



def main():
    bench = Bench()
    bench.setup(input_path=DATASET, run_name='databench', tag='dev')
    df = bench.load()
    #df = bench.filter_data(df)

    if bench.output_paths is None:
        raise ValueError("Output paths not initialized. Call setup() first.")
    paths = bench.output_paths

    idx, row = get_row(df, subject=SUBJECT, session=SESSION, task=TASK)
    
    cfg = OscillationDetectorConfig(
        fs=FS,
        band=(BAND_LO, BAND_HI),
        order=ORDER,
        k=K,
        min_duration_s=MIN_DURATION,
        merge_gap_s=MERGE_GAP,
    )

    ctx = context_from_index(idx, source=ROI_SOURCE, signal_key=ROI_NAME, time_key=TIME_KEY)
    analysis = OscillationDetectorAnalysis()
    result = bench.analyze(
        analysis,
        row,
        cfg,
        source=ROI_SOURCE,
        signal_key=ROI_NAME,
        time_key=TIME_KEY,
        context=ctx,
    )
    if result.data is None:
        raise ValueError("No signal data found for oscillation detector.")

    pupil = as_1d(row.get((PUPIL_SOURCE, PUPIL_KEY)))
    overlay = (pupil, "pupil_diameter_mm") if pupil is not None else None
    bench.save(
        result,
        stats_dir=paths.stats,
        plots_dir=paths.plots,
        overlay=overlay,
        overlay_subplot=True,
        time_window=[300, 500],
        save_plot=not NO_PLOT,
    )


if __name__ == "__main__":
    main()
