from __future__ import annotations

"""
blessed_workflow.py

A procedural, novice-friendly example showing how to:
1) load/filter data
2) compute first-order features
3) derive second-order features via a custom analysis
4) plot both first- and second-order features
5) define custom analysis/plotter using decorators

Tip:
- Keep this script simple and explicit.
- Move stable custom classes into databench/analysis or databench/plotting later.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from databench import Bench
from databench.analysis.base import Analysis, AnalysisResult
from databench.plotting.base import Plotter
from databench.plotting.core import FeaturePlotter, plot_feature_boxplot
from databench.registry import register_analysis, register_plotter


@register_analysis
@dataclass(frozen=True)
class DeltaFromBaseline(Analysis):
    """Simple second-order analysis: subtract baseline session mean per subject."""

    name: str = "delta_from_baseline"
    y: str = "speed_mean_cms"
    baseline_session_n: int = 1
    output_col: str = "d_speed_mean_cms"

    def run(self, wide: pd.DataFrame) -> AnalysisResult:
        y = self.y
        if y not in wide.columns:
            raise ValueError(f"Column {y!r} not found in table.")
        if "Subject" not in wide.columns:
            raise ValueError("Expected 'Subject' column in table.")
        if "session_n" not in wide.columns:
            raise ValueError("Expected 'session_n' column in table.")

        out_col = self.output_col or f"d_{y}"
        table = wide.copy()

        baseline = (
            table.loc[table["session_n"] == self.baseline_session_n, ["Subject", y]]
            .groupby("Subject", as_index=False)[y]
            .mean()
            .rename(columns={y: "_baseline"})
        )
        table = table.merge(baseline, on="Subject", how="left")
        table[out_col] = table[y] - table["_baseline"]
        table = table.drop(columns=["_baseline"])

        return AnalysisResult(name=self.name, data=table)


@register_plotter
@dataclass(frozen=True)
class DerivedBoxplotPlotter(Plotter):
    """Custom plotter for derived/second-order columns."""

    name: str = "derived_boxplot"
    y: str = "d_speed_mean_cms"
    x_label: str = "Session"

    def plot(self, wide, **kwargs):
        return plot_feature_boxplot(wide, y=self.y, x_label=self.x_label, **kwargs)


def main() -> None:
    # 1) Setup, load, filter
    pickle_path = Path(r"D:\4jake\260211_ETOH_dataset.pkl")

    bench = Bench()
    (bench
        .setup(pickle_path, analyst="Jacob Gronemeyer", lab="Sipe Lab", run_name="sandbox", tag="blessed")
        .load()
        .filter(drop_rows=(("GS27", "ses-02", "task-spont"),)))

    # 2) Build first-order feature table
    feature_names = ["speed_mean_cms", "pupil_mean_mm"]
    feature_fns = [bench.get_feature(name) for name in feature_names]
    bench.build_session_table(features=feature_fns)

    # 3) Second-order analysis
    analysis = DeltaFromBaseline(y="speed_mean_cms", baseline_session_n=1, output_col="d_speed_mean_cms")
    derived_plotter = DerivedBoxplotPlotter(y="d_speed_mean_cms", x_label="Session")
    feature_plotter = FeaturePlotter()

    bench.analyze(analysis, bench.session_table.reset_index())
    table2 = bench.result.data

    # Register metadata so second-order column can be plotted like a feature
    bench.register_derived_column(
        "d_speed_mean_cms",
        label="Δ Speed from baseline (cm/s)",
        color="#9467bd",
        plotter="longitudinal",
    )

    # 4) Plot first-order feature
    bench.plot(feature_plotter, table2, feature="speed_mean_cms", x_label="Session", save="first_order_speed.png")

    # 5) Plot second-order derived feature using same API
    bench.plot(feature_plotter, table2, feature="d_speed_mean_cms", x_label="Session", save="second_order_delta_speed.png")

    # 6) Plot second-order with custom plotter defined in this script
    bench.plot(derived_plotter, table2, save="second_order_delta_speed_boxplot.png")

    bench.save_table(table2, "blessed_session_table.csv")
    bench.save_run_summary(
        notes="Blessed procedural workflow with custom decorators for analysis + plotter.",
    )
    bench.save_provenance()


if __name__ == "__main__":
    main()
