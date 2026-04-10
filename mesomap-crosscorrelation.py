"""
mesomap-crosscorrelation.py
===========================
Cross-correlation matrices of common mesomap regions across all
subject/sessions in the **etoh-hfsa** dataset, split by Day 1, Day 5,
and Day 10.

For each day the script:
  1. Finds regions present in every session.
  2. Computes per-session pairwise Pearson correlation matrices.
  3. Averages the matrices across subjects.
  4. Plots a Day 1 / Day 5 / Day 10 triptych heatmap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from databench.project import Project
from databench.config import resolve_dataset
from databench.analysis._signal.preproc import detrend_zscore_1d

# ── Config ────────────────────────────────────────────────────────────────

DATASET = "etoh-hfsa"
SOURCE = "mesomap"
DAY_SESSIONS = {
    "Day 1":  "ses-01",
    "Day 5":  "ses-05",
    "Day 10": "ses-10",
}

# ── Project setup ─────────────────────────────────────────────────────────

proj = Project(
    dataset=resolve_dataset(DATASET),
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name="mesomap-xcorr",
)

# ── Discover common regions across all requested sessions ─────────────────

all_sessions = proj.sessions(
    sessions=list(DAY_SESSIONS.values()),
)

common_regions: set[str] | None = None
for sess in all_sessions:
    regions = set(sess._available_signals(SOURCE))
    common_regions = regions if common_regions is None else common_regions & regions

# Keep only regions that appear in both hemispheres; exclude "frame" and noisy ROIs
EXCLUDE_REGIONS = {"frame", "VISal", "VISl", "SSp-n", "SSp-m"}
_strip_hemi = lambda r: r.replace("L_", "").replace("R_", "")
bare_names = {_strip_hemi(r) for r in (common_regions or set())}
bilateral = {b for b in bare_names if f"L_{b}" in common_regions and f"R_{b}" in common_regions}
common_regions = {
    r for r in (common_regions or set())
    if _strip_hemi(r) in bilateral and _strip_hemi(r) not in EXCLUDE_REGIONS
}

common_regions_sorted = sorted(common_regions or [])
n_regions = len(common_regions_sorted)
print(f"Common bilateral regions ({n_regions}): {common_regions_sorted}")

# ── Compute per-day average cross-correlation ─────────────────────────────

day_corr: dict[str, np.ndarray] = {}

for day_label, ses_label in DAY_SESSIONS.items():
    day_group = proj.sessions(sessions=[ses_label])
    matrices = []
    for sess in day_group:
        signals = np.column_stack([
            detrend_zscore_1d(sess.signal(SOURCE, r)) for r in common_regions_sorted
        ])
        matrices.append(np.corrcoef(signals, rowvar=False))
    day_corr[day_label] = np.nanmean(matrices, axis=0)

# ── Plot ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

# Short display labels (strip hemisphere prefix)
labels = [r.replace("L_", "").replace("R_", "") for r in common_regions_sorted]

vmin, vmax = -1, 1
cmap = "RdBu_r"

for ax, (day_label, corr) in zip(axes, day_corr.items()):
    im = ax.imshow(corr, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_xticks(range(n_regions))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_regions))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(day_label, fontweight="bold")

fig.colorbar(im, ax=axes, shrink=0.8, label="Pearson r")
fig.suptitle(
    f"Mesomap region cross-correlation ({DATASET})",
    fontsize=13, fontweight="bold",
)

out = proj.plots_dir / "mesomap-crosscorrelation.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")

# ── Per-subject plots ─────────────────────────────────────────────────────

subjects = sorted({sess.subject for sess in all_sessions})

for subj in subjects:
    subj_sessions = proj.sessions(subject=subj, sessions=list(DAY_SESSIONS.values()))
    available_days = {}
    for sess in subj_sessions:
        for day_label, ses_label in DAY_SESSIONS.items():
            if sess.session == ses_label:
                signals = np.column_stack([
                    detrend_zscore_1d(sess.signal(SOURCE, r)) for r in common_regions_sorted
                ])
                available_days[day_label] = np.corrcoef(signals, rowvar=False)

    if not available_days:
        continue

    n_days = len(available_days)
    fig_s, axes_s = plt.subplots(1, n_days, figsize=(6 * n_days, 5.5), constrained_layout=True)
    if n_days == 1:
        axes_s = [axes_s]

    for ax, (day_label, corr) in zip(axes_s, available_days.items()):
        im = ax.imshow(corr, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
        ax.set_xticks(range(n_regions))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n_regions))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(day_label, fontweight="bold")

    fig_s.colorbar(im, ax=list(axes_s), shrink=0.8, label="Pearson r")
    fig_s.suptitle(
        f"Mesomap cross-correlation — {subj}",
        fontsize=13, fontweight="bold",
    )

    out_s = proj.plots_dir / f"mesomap-crosscorrelation_{subj}.png"
    fig_s.savefig(out_s, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_s}")
plt.show()
