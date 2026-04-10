#!/usr/bin/env python3
"""
Event-level trajectory analysis across longitudinal sessions.

For each signal (mesofield and pupil), plots six metrics across sessions:
  - n_events           : event count
  - mean_duration_s    : mean event duration (s)
  - mean_peak_raw      : mean peak amplitude (non-detrended absolute units)
  - mean_peak_z        : mean peak amplitude (z-scored on non-detrended trace)
  - mean_mean_z        : mean event amplitude (z-scored on non-detrended trace)
  - session_baseline    : session baseline mean (rolling-quantile, for QC)

Amplitude metrics (peak_raw, peak_z, mean_z) are measured on the non-
detrended cleaned trace produced by event-detection.py.  Detrending is
used only for event boundary detection; amplitudes preserve absolute
scale for hierarchical within-animal / between-animal comparisons.

Each panel draws one line per animal and a heavier group mean ± SEM.
A stats figure shows LMM fixed-effect slopes with 95% CIs.

Usage:
    python event-trajectories.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from databench.config import resolve_dataset
from databench.plotting import set_theme

set_theme()

# ─── Locate latest event-detection output ────────────────────────────────────

DATASET = resolve_dataset("hfsa")
OUTPUT_ROOT = Path(r'C:\Users\cakei\OneDrive\Desktop\databench\outputs\etoh-hfsa\260408\event-detection_hfsa')

stats_dirs = [OUTPUT_ROOT / "stats"]
if not stats_dirs:
    stats_dirs = [path for path in OUTPUT_ROOT.glob("*/stats") if path.is_dir()]
if not stats_dirs:
    raise FileNotFoundError(f"No stats directories found under {OUTPUT_ROOT}")

STATS_DIR = max(stats_dirs, key=lambda path: path.stat().st_mtime)
PLOT_DIR = STATS_DIR.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

print(f"Dataset: {DATASET.name}")
print(f"Reading from: {STATS_DIR}")

# ─── Signal definitions ───────────────────────────────────────────────────────

SIGNALS = [
    # {
    #     "name": "mesofield",
    #     "summary_csv": "mesofield_event_summary.csv",
    #     "metrics_csv": "mesofield_event_metrics.csv",
    #     "title": "Mesofield",
    # },
    {
        "name": "pupil",
        "summary_csv": "pupil_event_summary.csv",
        "metrics_csv": "pupil_event_metrics.csv",
        "title": "Pupil",
    },
]

METRICS = [
    "n_events", "mean_duration_s",
    "mean_peak_raw", "mean_peak_z", "mean_mean_z",
    "session_baseline",
]

METRIC_LABELS = {
    "n_events": "Event count",
    "mean_duration_s": "Mean duration (s)",
    "mean_peak_raw": "Mean peak amplitude (raw)",
    "mean_peak_z": "Mean peak amplitude (z)",
    "mean_mean_z": "Mean event amplitude (z)",
    "session_baseline": "Session baseline mean",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────


def build_session_table(summary_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-event metrics to one row per (Subject, Session, Task)."""
    # Amplitude columns are measured on the non-detrended cleaned trace
    session_amplitudes = (
        metrics_df
        .groupby(["Subject", "Session", "Task"], as_index=False)
        .agg(
            mean_peak_raw=("peak_raw", "mean"),
            mean_peak_z=("peak_z", "mean"),
            mean_mean_z=("mean_z", "mean"),
            session_baseline=("session_baseline_mean", "first"),
        )
    )

    merged = summary_df[
        ["Subject", "Session", "Task", "n_events", "mean_duration_s"]
    ].merge(session_amplitudes, on=["Subject", "Session", "Task"], how="left")

    merged["session_num"] = pd.to_numeric(
        merged["Session"].str.extract(r"ses-(\d+)", expand=False),
        errors="coerce",
    )
    merged = merged.dropna(subset=["session_num"]).copy()
    merged["session_num"] = merged["session_num"].astype(int)
    return merged


def make_subject_colors(subjects: list[str]) -> dict[str, tuple]:
    colormap = plt.get_cmap("tab10")
    return {subject: colormap(i % 10) for i, subject in enumerate(sorted(subjects))}


def style_axis(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─── Mixed-effects models ────────────────────────────────────────────────────

_LMM_OPTIMIZERS = ["lbfgs", "powell", "cg", "nm"]


def fit_lmm(df: pd.DataFrame, metric: str) -> dict | None:
    """
    Fit metric ~ session_num with random intercept + slope per Subject.
    Tries several optimizers to improve convergence.
    """
    sub = df[["Subject", "session_num", metric]].dropna()
    if sub["Subject"].nunique() < 2 or len(sub) < 5:
        return None

    formula = f"{metric} ~ session_num"
    result = None

    for opt in _LMM_OPTIMIZERS:
        try:
            model = smf.mixedlm(
                formula,
                data=sub,
                groups=sub["Subject"],
                re_formula="~session_num",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(reml=True, method=opt)
            if result.converged:
                break
        except Exception:
            continue

    if result is None:
        print(f"    LMM failed for {metric}: all optimizers failed")
        return None

    if not result.converged:
        print(f"    LMM warning: {metric} did not fully converge (best attempt kept)")

    fe = result.fe_params
    pvals = result.pvalues
    ci = result.conf_int()

    return {
        "metric": metric,
        "intercept": fe["Intercept"],
        "slope": fe["session_num"],
        "slope_se": result.bse["session_num"],
        "slope_p": pvals["session_num"],
        "slope_ci_lo": ci.loc["session_num", 0],
        "slope_ci_hi": ci.loc["session_num", 1],
        "re_intercept_var": result.cov_re.iloc[0, 0],
        "re_slope_var": result.cov_re.iloc[1, 1] if result.cov_re.shape[0] > 1 else np.nan,
        "n_obs": int(result.nobs),
        "n_groups": int(result.k_re),
        "aic": result.aic,
        "bic": result.bic,
        "converged": result.converged,
        "_result": result,
    }


def format_p(p: float) -> str:
    if p < 0.001:
        return "p < .001"
    if p < 0.01:
        return f"p = {p:.3f}"
    if p < 0.05:
        return f"p = {p:.2f}"
    return f"p = {p:.2f}"


def annotate_lmm(ax: Axes, lmm: dict | None) -> None:
    if lmm is None:
        return
    slope = lmm["slope"]
    p_str = format_p(lmm["slope_p"])
    sig = "*" if lmm["slope_p"] < 0.05 else ""
    txt = f"β = {slope:+.3f}, {p_str}{sig}"
    ax.text(
        0.02, 0.96, txt,
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="0.7", alpha=0.85),
    )


# ─── Stats figure ────────────────────────────────────────────────────────────


def plot_lmm_summary(
    lmm_results: dict[str, dict | None],
    signal_title: str,
) -> Figure | None:
    """
    Dedicated stats figure: horizontal forest plot of fixed-effect slopes
    with 95% CIs, plus a coefficient summary table.
    """
    valid = [(m, r) for m, r in lmm_results.items() if r is not None]
    if not valid:
        return None

    metrics = [m for m, _ in valid]
    slopes = np.array([r["slope"] for _, r in valid])
    ci_lo = np.array([r["slope_ci_lo"] for _, r in valid])
    ci_hi = np.array([r["slope_ci_hi"] for _, r in valid])
    pvals = np.array([r["slope_p"] for _, r in valid])
    converged = [r["converged"] for _, r in valid]

    y_pos = np.arange(len(metrics))

    fig, (ax_forest, ax_table) = plt.subplots(
        1, 2, figsize=(12, max(0.9 * len(metrics) + 2.4, 4.5)),
        gridspec_kw={"width_ratios": [3, 2]},
        facecolor="white",
    )

    # ── Forest plot ──
    colours = ["#2C73D2" if p < 0.05 else "#888888" for p in pvals]
    for i in range(len(metrics)):
        ax_forest.plot(
            [ci_lo[i], ci_hi[i]], [y_pos[i], y_pos[i]],
            color=colours[i], linewidth=2.5, solid_capstyle="round",
        )
        ax_forest.plot(
            slopes[i], y_pos[i], "o",
            color=colours[i], markersize=8,
            markeredgecolor="white", markeredgewidth=1.0, zorder=5,
        )
        if not converged[i]:
            ax_forest.plot(
                slopes[i], y_pos[i], "x",
                color="red", markersize=10, markeredgewidth=1.5, zorder=6,
            )

    ax_forest.axvline(0, color="0.4", linewidth=0.8, linestyle="--", zorder=0)
    ax_forest.set_yticks(y_pos)
    ax_forest.set_yticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=10)
    ax_forest.set_xlabel("Fixed-effect slope (β per session)", fontsize=10)
    ax_forest.invert_yaxis()
    style_axis(ax_forest)
    ax_forest.set_title("LMM fixed effects", fontsize=11, fontweight="bold")

    # ── Summary table ──
    ax_table.axis("off")
    col_labels = ["β", "SE", "95% CI", "p", "Sig", "Conv"]
    table_data = []
    for i, (m, r) in enumerate(valid):
        sig_str = "*" if pvals[i] < 0.05 else ("†" if pvals[i] < 0.10 else "")
        conv_str = "✓" if converged[i] else "✗"
        table_data.append([
            f"{slopes[i]:+.4f}",
            f"{r['slope_se']:.4f}",
            f"[{ci_lo[i]:+.3f}, {ci_hi[i]:+.3f}]",
            format_p(pvals[i]),
            sig_str,
            conv_str,
        ])

    row_labels = [METRIC_LABELS.get(m, m) for m in metrics]
    tbl = ax_table.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)

    # Colour significant rows
    for i in range(len(metrics)):
        if pvals[i] < 0.05:
            for j in range(len(col_labels)):
                tbl[i + 1, j].set_facecolor("#D6EDFF")

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#E8E8E8")
        tbl[0, j].set_text_props(fontweight="bold")

    ax_table.set_title("Model coefficients", fontsize=11, fontweight="bold", pad=12)

    fig.suptitle(
        f"{signal_title} — hierarchical LMM summary\n"
        f"metric ~ session  |  random intercept + slope per animal",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ─── Trajectory plot ─────────────────────────────────────────────────────────


def plot_event_basics(
    dataframe: pd.DataFrame,
    subject_colors: dict[str, tuple],
    signal_title: str,
    lmm_results: dict[str, dict | None] | None = None,
) -> Figure:
    """Trajectory plot: one panel per metric, lines per animal + group mean."""
    n_metrics = len(METRICS)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5.5 * nrows),
                             facecolor="white", squeeze=False)
    flat_axes = axes.flatten()
    session_ticks = sorted(dataframe["session_num"].unique())

    for panel_index, metric in enumerate(METRICS):
        ax = flat_axes[panel_index]

        for subject, group in dataframe.groupby("Subject"):
            ordered = group.sort_values("session_num")
            ax.plot(
                ordered["session_num"],
                ordered[metric],
                color=subject_colors[str(subject)],
                linewidth=1.0,
                alpha=0.4,
                marker="o",
                markersize=4,
                markeredgecolor="white",
                markeredgewidth=0.4,
                label=str(subject),
            )

        group_mean = dataframe.groupby("session_num")[metric].mean()
        group_sem = dataframe.groupby("session_num")[metric].sem()
        x = group_mean.index.to_numpy()
        y_mean = group_mean.to_numpy()
        y_sem = group_sem.to_numpy()

        ax.fill_between(x, y_mean - y_sem, y_mean + y_sem, color="#1A1A2E", alpha=0.12)
        ax.plot(
            x,
            y_mean,
            color="#1A1A2E",
            linewidth=2.2,
            marker="s",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor="#1A1A2E",
            markeredgewidth=1.2,
            label="Group mean ± SEM",
            zorder=5,
        )

        ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
        ax.set_xlabel("Session / day", fontsize=10)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_xticks(session_ticks)
        style_axis(ax)

        if lmm_results:
            annotate_lmm(ax, lmm_results.get(metric))

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 10),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.suptitle(
        f"{signal_title} — event trajectories across sessions",
        fontsize=13,
        fontweight="bold",
    )
    # Hide unused axes if grid has extra slots
    for i in range(n_metrics, len(flat_axes)):
        flat_axes[i].set_visible(False)
    fig.tight_layout()
    return fig


# ─── Run ──────────────────────────────────────────────────────────────────────

for signal in SIGNALS:
    summary_path = STATS_DIR / signal["summary_csv"]
    metrics_path = STATS_DIR / signal["metrics_csv"]

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing: {summary_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing: {metrics_path}")

    summary_df = pd.read_csv(summary_path)
    metrics_df = pd.read_csv(metrics_path)

    session_df = build_session_table(summary_df, metrics_df)
    n_subjects = session_df["Subject"].nunique()
    n_sessions = session_df["session_num"].nunique()
    print(f"\n{signal['title']}: {n_subjects} subjects × {n_sessions} sessions")

    subject_colors = make_subject_colors(sorted(session_df["Subject"].astype(str).unique()))

    # ── Fit hierarchical mixed-effects models ──
    print(f"  Fitting LMMs (day=fixed, animal=random intercept+slope)...")
    lmm_results: dict[str, dict | None] = {}
    lmm_rows: list[dict] = []
    for metric in METRICS:
        lmm = fit_lmm(session_df, metric)
        lmm_results[metric] = lmm
        if lmm is not None:
            slope = lmm["slope"]
            p = lmm["slope_p"]
            sig = "*" if p < 0.05 else ""
            print(f"    {metric:20s}  β={slope:+.4f}  {format_p(p)}{sig}")
            lmm_rows.append({k: v for k, v in lmm.items() if k != "_result"})
        else:
            print(f"    {metric:20s}  (model did not converge or insufficient data)")

    # Save LMM summary table
    if lmm_rows:
        lmm_df = pd.DataFrame(lmm_rows)
        lmm_df.insert(0, "signal", signal["name"])
        lmm_path = STATS_DIR / f"{signal['name']}_lmm_summary.csv"
        lmm_df.to_csv(lmm_path, index=False)
        print(f"    LMM table → {lmm_path.name}")

    # ── Trajectory plot ──
    fig = plot_event_basics(session_df, subject_colors, signal["title"],
                            lmm_results=lmm_results)

    png_path = PLOT_DIR / f"{signal['name']}_event_basics.png"
    pdf_path = PLOT_DIR / f"{signal['name']}_event_basics.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {png_path.name}")

    # ── Stats figure ──
    stats_fig = plot_lmm_summary(lmm_results, signal["title"])
    if stats_fig is not None:
        stats_png = PLOT_DIR / f"{signal['name']}_lmm_forest.png"
        stats_pdf = PLOT_DIR / f"{signal['name']}_lmm_forest.pdf"
        stats_fig.savefig(stats_png, dpi=220, bbox_inches="tight")
        with PdfPages(stats_pdf) as pdf:
            pdf.savefig(stats_fig, bbox_inches="tight")
        plt.close(stats_fig)
        print(f"  Saved {stats_png.name}")

print(f"\nPlots in {PLOT_DIR}")

# ─── Methodology report ──────────────────────────────────────────────────

report_lines = [
    "# tl;dr",
    "",
    "Longitudinal trajectories of event metrics (count, duration, amplitude)",
    "were plotted per-animal across sessions and tested with hierarchical",
    "linear mixed-effects models (LMM: metric ~ session, random intercept",
    "+ slope per animal).  Amplitude metrics use the non-detrended signal",
    "so cross-session and cross-animal comparisons are on a common scale.",
    "",
    "# Methods — Longitudinal Event Trajectory Analysis",
    "",
    "## Input Data",
    "",
    "Per-event metrics were produced by `event-detection.py` (see that",
    "report for detection methodology).  Each event row contains amplitudes",
    "measured on the non-detrended cleaned trace, preserving absolute",
    "signal scale for longitudinal comparison.",
    "",
    "## Session-Level Aggregation",
    "",
    "Per-event metrics were aggregated to one row per (Subject × Session)",
    "by computing the mean of each amplitude metric across events within",
    "that session.  Metrics tracked:",
    "",
    "| Metric | Description |",
    "|--------|-------------|",
    "| n_events | Number of detected events in the session |",
    "| mean_duration_s | Mean event duration (seconds) |",
    "| mean_peak_raw | Mean peak amplitude (non-detrended, absolute units) |",
    "| mean_peak_z | Mean peak amplitude (z-scored on non-detrended trace) |",
    "| mean_mean_z | Mean event amplitude (z-scored on non-detrended trace) |",
    "| session_baseline | Session baseline mean (rolling-quantile level) |",
    "",
    "Z-scores use the session-level mean and SD of the non-detrended",
    "cleaned+smoothed trace.  `mean_peak_raw` is in the original signal",
    "units (ΔF/F for mesofield, mm for pupil) and is the recommended",
    "response variable for hierarchical models.",
    "",
    "## Visualization",
    "",
    "Trajectory plots show one thin line per animal and a heavier group",
    "mean ± SEM overlay, with session/day on the x-axis.  This reveals",
    "both individual variability and population-level trends.",
    "",
    "## Statistical Modelling",
    "",
    "Each metric was fit with a hierarchical linear mixed-effects model",
    "(LMM) using REML estimation:",
    "",
    "    metric ~ session_num  (fixed effect: linear day trend)",
    "    random: ~session_num | Subject  (random intercept + slope per animal)",
    "",
    "This accounts for:",
    "  - Repeated measures within animals (non-independence)",
    "  - Between-animal differences in baseline level (random intercept)",
    "  - Between-animal differences in rate of change (random slope)",
    "",
    "Multiple optimizers (L-BFGS, Powell, CG, Nelder-Mead) were tried",
    "sequentially to improve convergence.  Results are presented as",
    "fixed-effect slopes (β) with 95% CIs in a forest plot.",
    "",
    "## Interpretation Notes",
    "",
    "  - A significant positive β for `mean_peak_raw` indicates that event",
    "    amplitude increases across sessions (e.g., sensitization).",
    "  - `session_baseline` tracks gross signal level drift; a trend here",
    "    may indicate optical/hardware changes rather than biology.",
    "  - `n_events` and `mean_duration_s` are unaffected by the detrending",
    "    choice (events are detected identically regardless of which trace",
    "    amplitudes are measured on).",
    "  - For publication, report the fixed-effect slope, 95% CI, and p-value",
    "    from the LMM summary table.  Also report the number of subjects,",
    "    sessions, and total events.",
    "",
    "## Output Files",
    "",
    "  - `{signal}_event_basics.png/pdf` : trajectory plots (6 panels)",
    "  - `{signal}_lmm_forest.png/pdf`   : forest plot + coefficient table",
    "  - `{signal}_lmm_summary.csv`       : full LMM results table",
    "",
]

report_path = STATS_DIR.parent / "reports"
report_path.mkdir(exist_ok=True)
report_file = report_path / "event-trajectories_methods.md"
report_file.write_text("\n".join(report_lines), encoding="utf-8")
print(f"Report → {report_file}")
