"""Template databench analysis script.

Conventions
-----------
* All analysis parameters live at the top of the script as UPPER_CASE
  module globals.  ``proj.io.report(...)`` automatically snapshots them
  into ``<run_dir>/config/params.json`` for reproducibility.
* All persistence goes through ``proj.io.*`` (figure / table / json /
  report / params).  Never call ``fig.savefig`` or ``df.to_csv`` directly.
* Plotters are reusable: build via ``result.<analysis>_plotter(...)``,
  then ``proj.io.figure(fig, name, sidecar=plotter.recipe(result))`` to
  drop a JSON recipe next to the saved figure.

Outputs land in
``outputs/<dataset_alias>/<script_name>/<YYMMDD>[_<TAG>]/`` with
``plots/``, ``stats/``, ``reports/``, ``config/``, and a top-level
``provenance.json`` (databench version, git hash + GitHub permalink,
env versions).
"""
from __future__ import annotations

from databench import Project, resolve_dataset, set_theme

set_theme()

# ─── Parameters (all UPPER_CASE, captured by proj.io.report) ─────────────

DATASET = "my_dataset"      # alias from datasets.toml
TAG     = ""                # optional run suffix → 250101_<TAG>
SUBJECT = "M01"
SESSION = "S01"
TASK    = "open_field"

# Analysis knobs
WINDOW  = (-2.0, 5.0)
BASELINE = (-2.0, -1.0)

# ─── Pipeline ────────────────────────────────────────────────────────────

dataset_path = resolve_dataset(DATASET)
proj = Project(dataset_path, tag=TAG, dataset_alias=DATASET)

# … run your detector / analysis …
# result = MyDetector(...).run(proj.session(SUBJECT, SESSION, TASK))

# ─── Plot via reusable Plotter ───────────────────────────────────────────

# plotter = result.overview_plotter(window=WINDOW)
# fig = plotter(result)
# proj.io.figure(fig, "overview.svg", sidecar=plotter.recipe(result),
#                suptitle="My Run", tight=True, formats=("png",))

# ─── Tables ──────────────────────────────────────────────────────────────

# proj.io.table(result.events, "events.csv")
# proj.io.tables({"summary": summary_df, "details": details_df}, prefix="run_")
# proj.io.dict_table(stats_dict, "stats.csv")

# ─── PDF report ──────────────────────────────────────────────────────────

# with proj.io.pdf("session_overview.pdf") as pdf:
#     for sess in group:
#         fig = sess.plot()
#         pdf.savefig(fig, bbox_inches="tight")

# ─── Report (writes provenance + params + markdown) ──────────────────────

# proj.io.report(result, notes="Free-form description of this run.")
