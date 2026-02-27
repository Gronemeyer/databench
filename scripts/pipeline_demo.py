from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from databench import Bench
from databench.analysis import MesomapHilbertConfig, export_hilbert_envelopes
from databench.analysis.longitudinal import LongitudinalAnalysis
from databench.analysis.mesomap_hilbert import MesomapHilbertAnalysis


def main():
    bench = Bench()
    (bench
        .setup(Path(r"D:\Projects\RO1_ETOH\260120_dataset_mvp.pkl"), analyst="Jacob Gronemeyer", lab="Sipe Lab")
        .load()
        .filter(drop_rows=(
            ("GS29", "ses-04", "task-movies"),
            ("GS26", "ses-00", "task-widefield"),
            ("GS27", "ses-11", "task-violet"),
        )))

    features = ["speed_mean_cms", "pupil_mean_mm"]
    speed_mean = bench.get_feature("speed_mean_cms")
    pupil_mean = bench.get_feature("pupil_mean_mm")

    (bench
        .build_session_table(features=[speed_mean, pupil_mean])
        .analyze(LongitudinalAnalysis(), bench.session_table, ycols=features))

    bench.save_table(bench.result.data.reset_index(), "session_table.csv")

    # Optional: mesomap Hilbert analysis for every session (if mesomap exists)
    df = bench.df
    if isinstance(df.columns, pd.MultiIndex) and "mesomap" in df.columns.get_level_values(0):
        cfg_h = MesomapHilbertConfig()
        hilbert_analysis = MesomapHilbertAnalysis()
        hilbert_plot_dir = bench.output_paths.plots / "mesomap_hilbert"
        hilbert_plot_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in df.iterrows():
            bench.analyze(hilbert_analysis, row, cfg_h, source="mesomap")
            if not bench.result.data:
                continue

            out = bench.result.data
            subject, session, task = idx
            subject = str(subject).removeprefix("sub-")
            session = str(session).removeprefix("ses-")
            task = str(task).removeprefix("task-")

            keys = list(out["resolved"].keys())
            if "GLOBAL" in out["specs"]:
                keys.append("GLOBAL")

            bench.plot(
                hilbert_analysis, bench.result,
                kind="envelopes", keys=keys,
                save=str(hilbert_plot_dir / f"sub-{subject}_ses-{session}_task-{task}_hilbert_envelopes.png"),
            )
            bench.plot(
                hilbert_analysis, bench.result,
                kind="spectrograms", keys=keys,
                save=str(hilbert_plot_dir / f"sub-{subject}_ses-{session}_task-{task}_hilbert_spectrograms.png"),
            )

        # Batch export envelopes for all rows
        export_hilbert_envelopes(df, cfg_h, bench.output_paths.stats / "mesomap_hilbert", source="mesomap")

    bench.save_provenance()


if __name__ == "__main__":
    main()
