from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from databench import Bench
from databench.features.treadmill import locomotion_bout_events
from databench.plotting import FeaturePlotter, plot_locomotion_bouts
from databench.utils import clean_xy, get_first, strip_prefix


DATASET = Path(r"/Users/jakegronemeyer/Desktop/4jake/260212_ETOH-HFSA_dataset.pkl")
RUN_NAME = "260217_test-bouts"
EXPORT_SVG = True
TASK_FILTER = "task-widefield"


def main() -> None:
    bench = Bench()
    bench.setup(input_path=DATASET, run_name=RUN_NAME, tag="ETOH_locomotion_bouts-5-seconds")
    df = bench.load()
    df = df[df.index.get_level_values("Task") == TASK_FILTER]

    paths = bench.output_paths
    feature_plotter = FeaturePlotter()

    features = [
        bench.get_feature("speed_mean_cms"),
        bench.get_feature("speed_std_cms"),
        bench.get_feature("distance_m"),
        bench.get_feature("locomotion_bouts_n"),
        bench.get_feature("locomotion_bout_speed_mean_cms"),
        bench.get_feature("locomotion_bout_distance_m"),
        bench.get_feature("locomotion_bout_duration_s"),
    ]

    session_table = bench.build_session_table(df, features)
    bench.save_table(session_table.reset_index(), "locomotion_bouts_session_table.csv")

    for feat in features:
        fig, _ = bench.plot(
            feature_plotter,
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
        all_bout_speeds = []
        all_bout_durations = []
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
            bouts, _, _, _, _ = locomotion_bout_events(
                t,
                speed_cms,
                min_speed_cms=bench.get_feature("locomotion_bouts_n").min_speed_cms,
                min_duration_s=bench.get_feature("locomotion_bouts_n").min_duration_s,
                merge_gap_s=bench.get_feature("locomotion_bouts_n").merge_gap_s,
            )
            if bouts:
                dt_med = float(np.nanmedian(np.diff(t))) if t.size > 1 else 0.0
                for s, e in bouts:
                    if e < s:
                        continue
                    duration = float(t[e] - t[s] + dt_med)
                    all_bout_durations.append(duration)
                    all_bout_speeds.append(float(np.nanmean(np.abs(speed_cms[s : e + 1]))))

            title = f"Subject={subject_id} | Session={session_id} | Task={task_id}"
            fig, _ = plot_locomotion_bouts(t, speed_cms, bouts, title=title)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if all_bout_durations:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(all_bout_durations, bins=30, color="#4c72b0", edgecolor="white")
            ax.set_title("Bout duration distribution")
            ax.set_xlabel("Duration (s)")
            ax.set_ylabel("Count")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if all_bout_speeds:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(all_bout_speeds, bins=30, color="#55a868", edgecolor="white")
            ax.set_title("Bout mean speed distribution")
            ax.set_xlabel("Speed (cm/s)")
            ax.set_ylabel("Count")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    bench.save_provenance()


if __name__ == "__main__":
    main()
