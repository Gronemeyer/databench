"""Plot session-average speed, distance, and pupil across sessions.

Per-session feature extraction goes through ``group.to_frame``; the
longitudinal-by-subject lines + group mean/SEM go through
``plot_metric_by_session``.  Pupil is normalised both as a per-subject
z-score and as a percent-change from baseline (session 1).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench import Project, resolve_dataset, set_theme
from databench.analysis.longitudinal import longitudinal_summary
from databench.plotting import plot_metric_by_session
from databench.plotting.core import _theme_color
from databench.session import SignalNotFoundError

set_theme()

GROUP_BY_SEX = True
SEX_COLORS   = {
    "M": _theme_color("accent")  or "#1f77b4",
    "F": _theme_color("primary") or "#d62728",
}


# ─── Project setup ───────────────────────────────────────────────────────

proj = Project(
    dataset=resolve_dataset("etoh-hfsa"),
    run_name="session-averages",
).filter(exclude={"session": ["ses-11", "ses-00"]})

group = proj.sessions()
print(f"Loaded {len(group)} sessions")


# ─── Per-session feature extraction ──────────────────────────────────────

def session_features(sess):
    for source in ("treadmill", "encoder"):
        time_s   = sess.time(source)
        speed_mm = sess.signal(source, "speed_mm")
        if time_s is not None and speed_mm is not None:
            break
    else:
        return None

    time_valid       = np.isfinite(time_s) & np.isfinite(speed_mm)
    time_s, speed_mm = time_s[time_valid], speed_mm[time_valid]
    if time_s.size < 3:
        return None

    speed_cms = np.abs(speed_mm) / 10.0
    distance_m = float(np.nansum(speed_cms[1:] * np.diff(time_s))) / 100.0

    pupil = sess.signal("pupil", "pupil_diameter_mm")
    pupil_mean = (
        float(np.nanmean(pupil))
        if pupil is not None and np.isfinite(pupil).any()
        else np.nan
    )

    try:
        sex = str(sess.signal("session_config", "sex")[0])
    except (SignalNotFoundError, IndexError):
        sex = "unknown"

    yield {
        "day":             sess.day,
        "sex":             sex,
        "speed_mean_cms":  float(np.nanmean(speed_cms)),
        "distance_m":      distance_m,
        "pupil_mean_mm":   pupil_mean,
    }

session_table = group.to_frame(session_features)
session_table = session_table.set_index(["Subject", "Session", "Task"])
print(f"Computed features for {len(session_table)} sessions")


# ─── Pupil normalisation ─────────────────────────────────────────────────

session_table["pupil_z"] = (
    session_table.groupby(level="Subject")["pupil_mean_mm"]
    .transform(lambda s: (s - s.mean()) / s.std())
)
session_table, _ = longitudinal_summary(session_table, ycols=["pupil_mean_mm"])


# ─── Plot helpers ────────────────────────────────────────────────────────

def plot_panel(ax: plt.Axes, metric: str, title: str, ylabel: str, color: str) -> None:
    if GROUP_BY_SEX:
        for sex, sex_rows in session_table.reset_index().groupby("sex"):
            plot_metric_by_session(
                sex_rows, x="day", y=metric, ax=ax,
                subject_colors={s: SEX_COLORS.get(str(sex), color)
                                for s in sex_rows["Subject"].unique()},
                group_color=SEX_COLORS.get(str(sex), color),
                group_label=str(sex),
            )
    else:
        plot_metric_by_session(session_table.reset_index(), x="day", y=metric, ax=ax,
                               group_color=color)
    ax.set_xlabel("Session (day)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


# ─── Locomotion figure ───────────────────────────────────────────────────

fig_loco, (ax_speed, ax_dist) = plt.subplots(1, 2, figsize=(8, 3), sharey=False)
plot_panel(ax_speed, "speed_mean_cms", "Mean Speed",     "Speed (cm/s)",  _theme_color("accent"))
plot_panel(ax_dist,  "distance_m",     "Total Distance", "Distance (m)",  _theme_color("secondary"))
fig_loco.tight_layout()
proj.io.figure(fig_loco, "session_avg_locomotion.png")


# ─── Pupil figure ────────────────────────────────────────────────────────

fig_pupil, (ax_z, ax_pct) = plt.subplots(1, 2, figsize=(8, 3), sharey=False)
plot_panel(ax_z,   "pupil_z",            "Pupil Diameter (Z-scored)",                 "Z-score",  _theme_color("primary"))
plot_panel(ax_pct, "pct_pupil_mean_mm",  "Pupil Diameter (% Change from Baseline)",   "% Change", _theme_color("primary"))
fig_pupil.tight_layout()
proj.io.figure(fig_pupil, "session_avg_pupil.png")

print("Done.")
