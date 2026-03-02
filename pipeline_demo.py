from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from databench import Bench
from databench.analysis import export_hilbert_envelopes
from databench.analysis.longitudinal import LongitudinalAnalysis
from databench.analysis.mesomap_hilbert import MesomapHilbertAnalysis
from databench.config import resolve_dataset
from databench.plotting import plot_spectrogram_panel


DATASET = resolve_dataset()

bench = Bench()
(bench
    .setup(DATASET, analyst="Jacob Gronemeyer", lab="Sipe Lab")
    .filter(drop_rows=(
        ("GS29", "ses-04", "task-movies"),
        ("GS26", "ses-00", "task-widefield"),
        ("GS27", "ses-11", "task-violet"),
    )))

features = ["speed_mean_cms", "pupil_mean_mm"]
speed_mean = bench.get_feature("speed_mean_cms")
pupil_mean = bench.get_feature("pupil_mean_mm")

hilbert_analysis = MesomapHilbertAnalysis()

with bench.run("pipeline-demo") as run:
    run.build_session_table(features=[speed_mean, pupil_mean])
    result = run.analyze(LongitudinalAnalysis(ycols=tuple(features)), run.session_table)

    run.save_table(result.data.reset_index(), "session_table.csv")

    # Optional: mesomap Hilbert analysis for every session (if mesomap exists)
    df = bench.df
    if isinstance(df.columns, pd.MultiIndex) and "mesomap" in df.columns.get_level_values(0):
        hilbert_plot_dir = run.output_paths.plots / "mesomap_hilbert"
        hilbert_plot_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in df.iterrows():
            h_result = run.analyze(hilbert_analysis, df=row)
            if not h_result.data:
                continue

            out = h_result.data
            subject, session, task = idx
            subject = str(subject).removeprefix("sub-")
            session = str(session).removeprefix("ses-")
            task = str(task).removeprefix("task-")

            keys = list(out["resolved"].keys())
            if "GLOBAL" in out["specs"]:
                keys.append("GLOBAL")

            run.plot(
                hilbert_analysis, h_result,
                save=str(hilbert_plot_dir / f"sub-{subject}_ses-{session}_task-{task}_hilbert_envelopes.png"),
            )

            meta = h_result.meta or {}
            fig_spec = plot_spectrogram_panel(
                out["specs"], keys,
                fmax=meta.get("fmax", 12.0),
                win_s=meta.get("win_s", 4.0),
                overlap_frac=meta.get("overlap_frac", 0.95),
            )
            run.save_and_close(
                fig_spec,
                str(hilbert_plot_dir / f"sub-{subject}_ses-{session}_task-{task}_hilbert_spectrograms.png"),
            )

        export_hilbert_envelopes(df, run.output_paths.stats / "mesomap_hilbert", source="mesomap")

    run.save_run_summary(
        notes="Pipeline demo: longitudinal session table + optional mesomap Hilbert envelopes/spectrograms.",
    )
