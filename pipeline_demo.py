from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from databench.project import Project
from databench.analysis.longitudinal import LongitudinalAnalysis, build_session_table
from databench.analysis.mesomap_hilbert import MesomapHilbertAnalysis, export_hilbert_envelopes
from databench.config import resolve_dataset
from databench.analysis.locomotion import MeanSpeedCMS
from databench.analysis.features import MeanPupilMM
from databench.plotting.mesomap import plot_spectrogram_panel


DATASET = resolve_dataset()

proj = Project(
    dataset=DATASET,
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name="pipeline-demo",
).filter(drop_rows=(
    ("GS29", "ses-04", "task-movies"),
    ("GS26", "ses-00", "task-widefield"),
    ("GS27", "ses-11", "task-violet"),
))

speed_mean = MeanSpeedCMS()
pupil_mean = MeanPupilMM()
feature_names = [speed_mean.name, pupil_mean.name]

hilbert_analysis = MesomapHilbertAnalysis()
longitudinal = LongitudinalAnalysis(ycols=tuple(feature_names))

# Build session feature table and run longitudinal analysis
session_table = build_session_table(proj.df, features=[speed_mean, pupil_mean])
result = longitudinal.run(session_table)

proj.save_table(result.data.reset_index(), "session_table.csv")

# Optional: mesomap Hilbert analysis for every session (if mesomap exists)
df = proj.df
if isinstance(df.columns, pd.MultiIndex) and "mesomap" in df.columns.get_level_values(0):
    hilbert_plot_dir = proj._context.plots_dir / "mesomap_hilbert"
    hilbert_plot_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        h_result = hilbert_analysis.run(row)
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

        fig_env = hilbert_analysis.plot(h_result)
        if fig_env is not None:
            fig = fig_env[0] if isinstance(fig_env, tuple) else fig_env
            proj.save_and_close(
                fig,
                str(hilbert_plot_dir / f"sub-{subject}_ses-{session}_task-{task}_hilbert_envelopes.png"),
            )

        meta = h_result.meta or {}
        fig_spec = plot_spectrogram_panel(
            out["specs"], keys,
            fmax=meta.get("fmax", 12.0),
            win_s=meta.get("win_s", 4.0),
            overlap_frac=meta.get("overlap_frac", 0.95),
        )
        proj.save_and_close(
            fig_spec,
            str(hilbert_plot_dir / f"sub-{subject}_ses-{session}_task-{task}_hilbert_spectrograms.png"),
        )

    export_hilbert_envelopes(df, proj._context.stats_dir / "mesomap_hilbert", source="mesomap")

print("Pipeline demo complete.")
