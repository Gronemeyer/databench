from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from databench import Bench
from databench.analysis import MesomapHilbertConfig, export_hilbert_envelopes
from databench.plotting import plot_two_panel_longitudinal


def main():
    bench = Bench()
    bench.setup(Path(r"D:\Projects\RO1_ETOH\260120_dataset_mvp.pkl"))
    paths = bench.output_paths
    bench.set_filters(drop_rows=(
        ("GS29", "ses-04", "task-movies"),
        ("GS26", "ses-00", "task-widefield"),
        ("GS27", "ses-11", "task-violet"),
    ))
    df = bench.load()
    df = bench.filter_data(df)

    features = ["speed_mean_cms", "pupil_mean_mm"]
    speed_mean = bench.get_feature("speed_mean_cms")
    pupil_mean = bench.get_feature("pupil_mean_mm")
    feature_fns = [speed_mean, pupil_mean]

    session_table = bench.build_session_table(df, feature_fns)
    summary = bench.analyze("longitudinal_summary", session_table, ycols=features)
    session_table = summary.data

    bench.save_table(session_table.reset_index(), "session_table.csv")

    # fig, _ = plot_two_panel_longitudinal(
    #     session_table,
    #     y1=speed_mean.name,
    #     y2=pupil_mean.name,
    #     titles=(f"{speed_mean.label} across sessions", f"{pupil_mean.label} across sessions"),
    #     y_labels=(speed_mean.label, pupil_mean.label),
    #     x_label="Session (days)",
    # )
    # fig.savefig(paths.plots / "longitudinal_speed_pupil.png", dpi=300, bbox_inches="tight")
    # plt.close(fig)

    # fig, _ = bench.plot("feature", session_table, feature=speed_mean, x_label="Session (days)")
    # fig.savefig(paths.plots / "longitudinal_speed.png", dpi=300, bbox_inches="tight")
    # plt.close(fig)

    # Example: export a single feature report (wide CSV)
    #bench.export_feature_report(df, "pupil", "pupil_diameter_mm", name="report_pupil_diameter.csv")

    # Optional: mesomap Hilbert analysis for every session (if mesomap exists)
    if isinstance(df.columns, pd.MultiIndex) and "mesomap" in df.columns.get_level_values(0):
        cfg_h = MesomapHilbertConfig()
        hilbert_plot_dir = paths.plots / "mesomap_hilbert"
        hilbert_plot_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in df.iterrows():
            result = bench.analyze("mesomap_hilbert", row, cfg_h, source="mesomap")
            if not result.data:
                continue

            out = result.data

            subject, session, task = idx
            subject = str(subject).removeprefix("sub-")
            session = str(session).removeprefix("ses-")
            task = str(task).removeprefix("task-")

            keys = list(out["resolved"].keys())
            if "GLOBAL" in out["specs"]:
                keys.append("GLOBAL")

            fig = bench.plot("mesomap_hilbert", result, kind="envelopes", keys=keys)
            fig.savefig(
                hilbert_plot_dir / f"sub-{subject}_ses-{session}_task-{task}_hilbert_envelopes.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

            fig = bench.plot("mesomap_hilbert", result, kind="spectrograms", keys=keys)
            fig.savefig(
                hilbert_plot_dir / f"sub-{subject}_ses-{session}_task-{task}_hilbert_spectrograms.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        # Batch export envelopes for all rows
        export_hilbert_envelopes(df, cfg_h, paths.stats / "mesomap_hilbert", source="mesomap")


if __name__ == "__main__":
    main()
