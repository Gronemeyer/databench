"""One-line description of this analysis.

Flow: params -> setup -> analyze -> save -> finish.

* UPPER_CASE module constants are the knobs. ``run.finish()`` snapshots
  them into ``provenance.json`` so the run is reproducible.
* All outputs go through ``run.save_figure / save_table / save_json``.
* ``run.finish()`` writes ``provenance.json`` (dataset hash, git commit,
  params, env) and a human-readable ``report.md``.

Run:
    python Scripts/_template.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from databench import Project, set_theme

set_theme()

# ── Parameters (captured into provenance.json) ──────────────────────────────
DATASET = "hfsa"            # datasets.toml alias or a path
TASK    = "task-widefield"
SUBJECT = "STREHAB02"
SESSION = "ses-01"

# ── Setup ───────────────────────────────────────────────────────────────────
proj = Project(DATASET)
proj.describe()                                  # what's in here?

run = proj.run(name="template", tag="")          # versioned output directory

# ── Analyze ─────────────────────────────────────────────────────────────────
sess = proj.session(subject=SUBJECT, session=SESSION, task=TASK)
t = sess.time("treadmill")
speed = sess.signal("treadmill", "speed_mm") / 10.0   # mm/s -> cm/s

# ── Save ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots()
ax.plot(t, speed, lw=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Speed (cm/s)")
ax.set_title(sess.label)
run.save_figure(fig, "speed.svg")

# ── Finish (provenance.json + report.md) ────────────────────────────────────
run.finish(notes="Template run.")
