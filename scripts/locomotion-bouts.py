from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from databench import Bench
from databench.features.treadmill import locomotion_bout_events
from databench.plotting import FeaturePlotter, plot_locomotion_bouts
from databench.utils import clean_xy, get_first, strip_prefix

from databench.config import resolve_dataset
DATASET = resolve_dataset()

EXPORT_SVG = True

bench = Bench()
(bench
    .setup(input_path=DATASET,
            analyst="Jacob Gronemeyer",
            lab="Sipe Lab",
            run_name="locomotion_bouts",
            tag="ETOH_5-seconds")
    .filter(Task="task-widefield"))

df = bench.df
paths = bench.output_paths
features = [
    bench.get_feature("speed_mean_cms"),
    bench.get_feature("speed_std_cms"),
    bench.get_feature("distance_m"),
    bench.get_feature("locomotion_bouts_n"),
    bench.get_feature("locomotion_bout_speed_mean_cms"),
    bench.get_feature("locomotion_bout_distance_m"),
    bench.get_feature("locomotion_bout_duration_s"),
]

with bench.run("locomotion-bouts", notes="First-order locomotion bout features for ET0H R01 10-day pre-condition dataset.") as run:
    run.build_session_table(features=features)
    run.save_table(run.session_table.reset_index(), "locomotion_bouts_session_table.csv")

    for feat in features:
        plotter = FeaturePlotter(feature=feat, x_label="Session (days)")
        fig, _ = run.plot(plotter, run.session_table)
        base = f"{feat.name}_boxplot"
        run.save_feature_plot(fig, f"{base}.png", feature_name=feat.name, folder="plots", dpi=300, bbox_inches="tight")
        if EXPORT_SVG:
            run.save_figure(fig, f"{base}.svg", folder="plots", dpi=300, bbox_inches="tight")
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

    run.save_run_summary(
        notes="First-order locomotion bout features for ET0H R01 10-day pre-condition dataset.",
    )
