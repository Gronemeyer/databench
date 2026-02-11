from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from databench import Bench
from databench.features.treadmill import detect_locomotion_bouts
from databench.plotting import plot_locomotion_bouts
from databench.utils import clean_xy, get_first, strip_prefix


DATASET = Path(r"C:\dev\datakit\260129_HFSA-full.pkl")
RUN_NAME = "new-plots"
EXPORT_SVG = True


def main() -> None:
    bench = Bench()
    bench.setup(input_path=DATASET, run_name=RUN_NAME, tag="HFSA_locomotion_bouts-5-seconds")
    df = bench.load()
    df = bench.filter_data(df, drop_rows_list=[
		("STREHAB07", "ses-11", "task-widefield"),])

    if bench.output_paths is None:
        raise ValueError("Output paths not initialized. Call setup() first.")
    paths = bench.output_paths

    bouts_feature = bench.get_feature("locomotion_bouts_n")
    features = [
        bench.get_feature("speed_mean_cms"),
        bench.get_feature("distance_m"),
        bouts_feature,
        bench.get_feature("locomotion_bout_speed_mean_cms"),
        bench.get_feature("locomotion_bout_distance_m"),
        bench.get_feature("locomotion_bout_duration_s"),
    ]

    session_table = bench.build_session_table(df, features)
    bench.save_table(session_table.reset_index(), "locomotion_bouts_session_table.csv")

    for feat in features:
        fig, _ = bench.plot(
            "feature",
            session_table,
            feature=feat,
            x_label="Session (days)",
        )
        base = f"{feat.name}_boxplot"
        bench.save_figure(fig, f"{base}.png", folder="plots", dpi=300, bbox_inches="tight")
        if EXPORT_SVG:
            bench.save_figure(fig, f"{base}.svg", folder="plots", dpi=300, bbox_inches="tight")
        plt.close(fig)

    report_path = paths.reports / "locomotion_bouts_report.pdf"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(report_path) as pdf:
        for idx, row in df.iterrows():
            if not isinstance(idx, tuple) or len(idx) < 3:
                continue
            subject, session, task = idx[:3]
            subject_id = strip_prefix(subject, "sub-")
            session_id = strip_prefix(session, "ses-")
            task_id = strip_prefix(task, "task-")

            t, spd_mm = clean_xy(
                get_first(row, [("treadmill", "time_elapsed_s"), ("encoder", "time_elapsed_s")]),
                get_first(row, [("treadmill", "speed_mm"), ("encoder", "speed")]),
            )
            if t is None:
                continue
            speed_cms = spd_mm / 10.0
            bouts = detect_locomotion_bouts(
                t,
                speed_cms,
                min_speed_cms=bouts_feature.min_speed_cms,
                min_duration_s=bouts_feature.min_duration_s,
                merge_gap_s=bouts_feature.merge_gap_s,
            )

            title = f"Subject={subject_id} | Session={session_id} | Task={task_id}"
            fig, _ = plot_locomotion_bouts(t, speed_cms, bouts, title=title)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    bench.save_provenance()


if __name__ == "__main__":
    main()
