"""Template databench analysis script.

Conventions
-----------
* All analysis parameters live at the top as UPPER_CASE module globals.
  ``proj.io.report(...)`` automatically snapshots them into
  ``<run_dir>/config/params.json`` for reproducibility.
* All persistence goes through ``proj.io.*`` (figure / table / json /
  report / params).  Never call ``fig.savefig`` or ``df.to_csv`` directly.
* Lead every script with ``proj.describe()`` — it prints the dataset
  shape, the declared sources/columns (when a schema is configured in
  ``datasets.toml``), and where outputs will land.  That single call
  tells you what to type next.

Outputs land in
``outputs/<dataset_alias>/<script_name>/<YYMMDD_HHMMSS>[_<TAG>]/`` with
``plots/``, ``stats/``, ``reports/``, ``config/``, and a top-level
``provenance.json`` (databench version, git hash + GitHub permalink,
env versions).  Each run gets a unique timestamped dir so repeated
runs in the same day no longer overwrite each other.  Use
``proj.list_runs()`` to navigate run history.
"""
from __future__ import annotations

from databench import Project, set_theme

set_theme()

# ─── Parameters (all UPPER_CASE, captured by proj.io.report) ─────────────

DATASET = "hfsa"            # alias from datasets.toml (or a path string)
TAG     = ""                # optional run-dir suffix
SUBJECT = "STREHAB02"
SESSION = "ses-01"
TASK    = "task-widefield"

WINDOW   = (-2.0, 5.0)
BASELINE = (-2.0, -1.0)

# ─── Project setup ───────────────────────────────────────────────────────

proj = Project(DATASET, tag=TAG)
proj.describe()             # uncomment-free: schema-aware data summary

# Pick a single session or a group:
# sess  = proj.session(subject=SUBJECT, session=SESSION, task=TASK)
# group = proj.sessions(task=TASK)

# ─── Run an analysis ─────────────────────────────────────────────────────

# from databench.analysis.locomotion import locomotion_bout_events
# bouts = locomotion_bout_events(
#     sess.time("treadmill"),
#     sess.signal("treadmill", "speed_mm") / 10.0,   # mm/s → cm/s
# )

# ─── Plot ────────────────────────────────────────────────────────────────

# Default approach — plain matplotlib + proj.io.figure:
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(bouts["start_s"], bouts["mean_speed_cms"], "o")
# proj.io.figure(fig, "bouts.svg")

# Reusable plotters in databench.plotting (see its module docstring for
# the full inventory): set_theme, new_figure, style_axes,
# plot_metric_by_session, quickplot, quickplot_group.

# Advanced: some analyses (oscillation, eta) expose a Plotter factory
# with a JSON recipe sidecar.  See databench/analysis/oscillation.py and
# databench/analysis/eta.py for reference implementations.

# ─── Tables ──────────────────────────────────────────────────────────────

# proj.io.table(bouts, "bouts.csv")

# ─── PDF report ──────────────────────────────────────────────────────────

# with proj.io.pdf("session_overview.pdf") as pdf:
#     for s in group:
#         fig, ax = plt.subplots()
#         ax.plot(s.time("treadmill"), s.signal("treadmill", "speed_mm"))
#         pdf.savefig(fig, bbox_inches="tight")

# ─── Report (writes provenance + params + markdown) ──────────────────────

# proj.io.report(notes="Free-form description of this run.")
