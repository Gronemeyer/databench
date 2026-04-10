"""Plot session-average speed and distance across all sessions."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench.project import Project
from databench.config import resolve_dataset
from databench.session import SaveableFigure
from databench.plotting import set_theme
from databench.plotting.core import plot_mean_sem, _theme_color
from databench.analysis.longitudinal import longitudinal_summary

set_theme()

proj = Project(
    dataset=resolve_dataset('etoh-hfsa'),
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name="session-averages",
).filter(exclude={"session": ["ses-11", "ses-00"]})

group = proj.sessions()
print(f"Loaded {len(group)} sessions")

rows = []
for sess in group:
    for src in ("treadmill", "encoder"):
        t = sess.time(src)
        spd = sess.signal(src, "speed_mm")
        if t is not None and spd is not None:
            break
    else:
        continue

    valid = np.isfinite(t) & np.isfinite(spd)
    t, spd = t[valid], spd[valid]
    if t.size < 3:
        continue

    speed_cms = np.abs(spd) / 10.0
    dt = np.diff(t)
    distance_m = float(np.nansum(speed_cms[1:] * dt)) / 100.0

    pupil = sess.signal("pupil", "pupil_diameter_mm")
    pupil_mean = float(np.nanmean(pupil)) if pupil is not None and np.isfinite(pupil).any() else np.nan

    rows.append({
        "Subject": sess.subject,
        "Session": sess.session,
        "Task": sess.task,
        "speed_mean_cms": float(np.nanmean(speed_cms)),
        "distance_m": distance_m,
        "pupil_mean_mm": pupil_mean,
    })

session_table = pd.DataFrame(rows)
session_table = proj.tabler.add_session_number(session_table)
session_table = session_table.set_index(["Subject", "Session", "Task"])
print(f"Computed features for {len(session_table)} sessions")

# Normalize pupil: z-score per subject
session_table["pupil_z"] = session_table.groupby("Subject")["pupil_mean_mm"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Normalize pupil: percent change from baseline (session 1)
session_table, _ = longitudinal_summary(session_table, ycols=["pupil_mean_mm"])

def _plot_panels(panels, figsize=(8, 3)):
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, (y, title, ylabel, color) in zip(axes, panels):
        _, stats = longitudinal_summary(session_table, y=y, x="session_n")
        plot_mean_sem(stats, x="session_n", y_label=ylabel, title=title, color=color, ax=ax)
        ax.set_xlabel("Session")
    fig.tight_layout()
    return fig

# Locomotion figure (1×2 horizontal layout)
loco_panels = [
    ("speed_mean_cms", "Mean Speed", "Speed (cm/s)", _theme_color("accent")),
    ("distance_m", "Total Distance", "Distance (m)", _theme_color("secondary")),
]
fig_loco = _plot_panels(loco_panels)
SaveableFigure(fig_loco, proj._context).save("session_avg_locomotion.png")
plt.close(fig_loco)

# Pupil figure (1×2 horizontal layout)
pupil_panels = [
    ("pupil_z", "Pupil Diameter (Z-scored)", "Z-score", _theme_color("primary")),
    ("pct_pupil_mean_mm", "Pupil Diameter (% Change from Baseline)", "% Change", _theme_color("primary")),
]
fig_pupil = _plot_panels(pupil_panels)
SaveableFigure(fig_pupil, proj._context).save("session_avg_pupil.png")
plt.close(fig_pupil)
print("Done.")
