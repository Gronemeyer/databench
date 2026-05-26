#!/usr/bin/env python3
"""Correlate event timing and magnitudes from event metrics tables.

Reads pupil and mesofield event metrics from a single event-detection output,
filters to ses-01..ses-10, and computes Pearson/Spearman correlations between
onset time and selected magnitude columns.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from databench import Project
from databench.config import resolve_dataset

# Parameters
DATASET = resolve_dataset("hfsa")
TAG = "hfsa"
SOURCE_STATS_DIR = Path("outputs/hfsa/event-detection/260509_hfsa/stats")
SESSIONS = [f"ses-{index:02d}" for index in range(1, 11)]

TIMING_COLUMN = "onset_s"
MAGNITUDE_COLUMNS = [
    "peak_z_detrended",
    "peak_z",
    "peak_raw_detrended",
    "duration_s",
]

METRIC_FILES = {
    "pupil": "pupil_event_metrics.csv",
    "mesofield": "mesofield_event_metrics.csv",
}


def load_metrics_table(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {csv_path}")

    table = pd.read_csv(csv_path)
    table = table[table["Session"].isin(SESSIONS)].copy()

    for column in [TIMING_COLUMN, *MAGNITUDE_COLUMNS]:
        table[column] = pd.to_numeric(table[column], errors="coerce")

    return table


def correlation_rows(
    events: pd.DataFrame,
    *,
    signal: str,
    session: str,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []

    for magnitude_column in MAGNITUDE_COLUMNS:
        pair = events[[TIMING_COLUMN, magnitude_column]].dropna()
        event_count = len(pair)

        if event_count < 3:
            rows.append(
                {
                    "signal": signal,
                    "Session": session,
                    "timing_column": TIMING_COLUMN,
                    "magnitude_column": magnitude_column,
                    "n_events": event_count,
                    "pearson_r": float("nan"),
                    "pearson_p": float("nan"),
                    "spearman_rho": float("nan"),
                    "spearman_p": float("nan"),
                }
            )
            continue

        pearson_r, pearson_p = pearsonr(pair[TIMING_COLUMN], pair[magnitude_column])
        spearman_rho, spearman_p = spearmanr(
            pair[TIMING_COLUMN],
            pair[magnitude_column],
        )

        rows.append(
            {
                "signal": signal,
                "Session": session,
                "timing_column": TIMING_COLUMN,
                "magnitude_column": magnitude_column,
                "n_events": event_count,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
            }
        )

    return rows


def pooled_scatter_plot(
    pooled_events_by_signal: dict[str, pd.DataFrame],
) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    signal_order = ["pupil", "mesofield"]
    colors = {
        "pupil": "#1f77b4",
        "mesofield": "#d62728",
    }

    for axis, signal in zip(axes, signal_order):
        events = pooled_events_by_signal[signal]
        pair = events[[TIMING_COLUMN, "peak_z_detrended"]].dropna()

        axis.scatter(
            pair[TIMING_COLUMN],
            pair["peak_z_detrended"],
            s=11,
            alpha=0.35,
            color=colors[signal],
            edgecolors="none",
        )

        if len(pair) >= 2:
            slope, intercept = np.polyfit(
                pair[TIMING_COLUMN],
                pair["peak_z_detrended"],
                1,
            )
            fit_x = np.array([pair[TIMING_COLUMN].min(), pair[TIMING_COLUMN].max()])
            fit_y = (slope * fit_x) + intercept
            axis.plot(fit_x, fit_y, color="black", linewidth=1.8)

        if len(pair) >= 3:
            pearson_r, pearson_p = pearsonr(
                pair[TIMING_COLUMN],
                pair["peak_z_detrended"],
            )
            axis.set_title(f"{signal}: r={pearson_r:+.3f}, p={pearson_p:.2e}")
        else:
            axis.set_title(signal)

        axis.set_xlabel("onset_s")
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("peak_z_detrended")
    figure.suptitle("Event timing vs magnitude (ses-01..ses-10)")
    return figure


def main() -> None:
    proj = Project(dataset=DATASET, analyst="databench")
    run = proj.run(name="timing-magnitude-correlation", tag=TAG)

    all_rows: list[dict[str, float | str | int]] = []
    pooled_events_by_signal: dict[str, pd.DataFrame] = {}

    for signal, filename in METRIC_FILES.items():
        metrics_table = load_metrics_table(SOURCE_STATS_DIR / filename)
        pooled_events_by_signal[signal] = metrics_table

        all_rows.extend(
            correlation_rows(
                metrics_table,
                signal=signal,
                session="all",
            )
        )

        for session in SESSIONS:
            session_table = metrics_table[metrics_table["Session"] == session]
            all_rows.extend(
                correlation_rows(
                    session_table,
                    signal=signal,
                    session=session,
                )
            )

    correlations = pd.DataFrame(all_rows)
    correlations = correlations.sort_values(
        ["signal", "Session", "magnitude_column"],
        kind="stable",
    ).reset_index(drop=True)

    pooled = correlations[correlations["Session"] == "all"].reset_index(drop=True)

    run.save_table(correlations, "timing_magnitude_correlations.csv")
    run.save_table(pooled, "timing_magnitude_correlations_pooled.csv")

    pooled_figure = pooled_scatter_plot(pooled_events_by_signal)
    run.save_figure(pooled_figure, "timing_vs_peak_z_detrended.png")

    print("\nPooled correlations (all ses-01..ses-10 events):")
    print(pooled.to_string(index=False))

    run.finish(
        notes=(
            "Computed event timing-vs-magnitude correlations from pupil and "
            "mesofield event metrics for ses-01..ses-10."
        )
    )
    print("Done.")


if __name__ == "__main__":
    main()
