"""Tour of the databench non-analysis API.

A single runnable script that exercises every user-facing surface for
exploring, accessing, filtering, aligning, persisting, and plotting —
without invoking any specific analysis (locomotion, eta, oscillation,
etc.).  Read top-to-bottom; each section is self-contained.

Run:
    python Scripts/databench-tour.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import loguru
loguru.logger.enable("rich")  # pretty tracebacks for demo purposes
# ─── Top-level re-exports from databench/__init__.py ──────────────────────
from databench import (
    Project,                # the main entry point
    resolve_dataset,        # still exported for advanced cases
    dataset_params,         # raw [datasets.<alias>] dict
    set_theme,              # plotting theme
)
from databench.utils import session_to_int as parse_session_day   # "ses-07" -> 7
from databench.plotting import (
    get_theme, new_figure, style_axes, style_figure,
    plot_metric_by_session, quickplot, quickplot_group,
)
from databench.utils import (
    clean_xy, drop_rows, get_first, label_conditions,
    session_to_int, strip_prefix, time_mask,
)


def section(title: str) -> None:
    print(f"\n{'─' * 4} {title} {'─' * max(2, 72 - len(title))}")


set_theme()                       # global plot style; theme = get_theme()


# ═════════════════════════════════════════════════════════════════════════
# 1. Project construction — alias string OR path
# ═════════════════════════════════════════════════════════════════════════
section("Project construction")

proj = Project("hfsa")
run = proj.run(name="databench-tour", tag="api-demo")
print(repr(proj))
print(f"run dir    : {run.dir}")
print(f"plots_dir  : {run.plots_dir}")
print(f"tables_dir : {run.tables_dir}")

# Path form (advanced — most scripts don't need this):
# proj_from_path = Project(resolve_dataset("hfsa"))

# Per-dataset extras come back as a plain dict:
print(f"dataset_params keys: {sorted(dataset_params('hfsa'))}")


# ═════════════════════════════════════════════════════════════════════════
# 2. Schema-first discovery
# ═════════════════════════════════════════════════════════════════════════
section("Project.describe + schema introspection")

proj.describe()                   # the canonical "what's in here?" call

schema = proj.schema
print(f"\nalias map        : {dict(schema.sources)}")
print(f"declared sources : {schema.declared_sources()}")
print(f"role(pupil/pupil_diameter_mm) = {schema.role_of('pupil', 'pupil_diameter_mm')}")
print(f"unit(pupil/pupil_diameter_mm) = {schema.unit_of('pupil', 'pupil_diameter_mm')}")


# ═════════════════════════════════════════════════════════════════════════
# 3. Filtering at the Project level
# ═════════════════════════════════════════════════════════════════════════
section("Project.filter (chainable, in-place)")

# include/exclude/drop_rows accept dicts, tuples, lists, or mixed collections:
proj.filter(exclude={"session": ["ses-01", "ses-10"]})
proj.filter(include={"task": "task-widefield"})
proj.filter(drop_rows=[("STREHAB07", "ses-08")])    # drop a single (subject, session)

print(f"after filter: subjects={proj.subjects}")
print(f"after filter: sessions={proj.all_sessions}")
print(f"after filter: tasks={proj.tasks}")


# ═════════════════════════════════════════════════════════════════════════
# 4. Single-session and multi-session selection
# ═════════════════════════════════════════════════════════════════════════
section("Project.session / Project.sessions / first_session")

# One row (raises if zero or multiple match):
sess = proj.session(subject="STREHAB02", session="ses-02", task="task-widefield")
print(f"single: {sess.label}  day={sess.day}")

# A group:
group = proj.sessions(task="task-widefield")
print(f"group : {group!r}")
print(f"  subjects       : {group.subjects}")
print(f"  session labels : {group.session_labels}")

# The "any session — what's in this dataset?" shortcut:
sample = proj.first_session()
print(f"first_session().label = {sample.label}")


# ═════════════════════════════════════════════════════════════════════════
# 5. Session introspection (no analysis yet)
# ═════════════════════════════════════════════════════════════════════════
section("Session: sources / signals / describe")

print(f"sess.sources                  = {sess.sources}")
print(f"sess.signals('pupil')         = {sess.signals('pupil')}    # alias works")
print(f"sess.signals('treadmill')[:4] = {sess.signals('treadmill')[:4]}")
#sess.describe()                   # multi-line summary


# ═════════════════════════════════════════════════════════════════════════
# 6. Raw signal/time access (alias-aware)
# ═════════════════════════════════════════════════════════════════════════
section("Session.signal / Session.time")

t_pupil  = sess.time("pupil")                       # aliased to pupil_dlc
pup      = sess.signal("pupil", "pupil_diameter_mm")
t_tread  = sess.time("treadmill")
speed_mm = sess.signal("treadmill", "speed_mm")
print(f"pupil:    t.shape={t_pupil.shape}, y.shape={pup.shape}, "
      f"t range = {t_pupil[0]:.1f}–{t_pupil[-1]:.1f}s")
print(f"treadmill: t.shape={t_tread.shape}, y.shape={speed_mm.shape}")

# Fuzzy-suggest example — uncomment to see the error message:
# sess.signal("pupli", "pupil_diameter_mm")
# → SignalNotFoundError: ... Did you mean 'pupil_dlc', 'pupil_metadata'?


# ═════════════════════════════════════════════════════════════════════════
# 7. Cross-source alignment with merge_asof
# ═════════════════════════════════════════════════════════════════════════
section("Session.align -> AlignedData")

ad = sess.align(
    {"treadmill": ["speed_mm"], "pupil": ["pupil_diameter_mm"]},
    reference="treadmill",
    tolerance_s=0.25,
)
print(f"AlignedData : {ad!r}")
print(f"  reference : {ad.reference}")
print(f"  time col  : {ad.time_column}")
print(f"  columns   : {ad.columns}")
print(ad.df.head(3).to_string())

# Sessiongroup alignment — concatenates aligned frames from every member:
group_ad = group.align(
    {"treadmill": ["speed_mm"], "pupil": ["pupil_diameter_mm"]},
    reference="treadmill",
)
print(f"\ngroup.align(...).df.shape = {group_ad.df.shape}")


# ═════════════════════════════════════════════════════════════════════════
# 8. SessionGroup.to_frame — long-form extractor
# ═════════════════════════════════════════════════════════════════════════
section("SessionGroup.to_frame (custom extractor)")


def per_session_features(s):
    """Yield a single row of scalar features per session."""
    try:
        spd_mm = s.signal("treadmill", "speed_mm")
    except Exception:
        return
    spd_cms = spd_mm / 10.0
    yield {
        "day": s.day,
        "mean_speed_cms": float(np.nanmean(np.abs(spd_cms))),
        "max_speed_cms":  float(np.nanmax(np.abs(spd_cms))),
        "n_samples":      int(spd_cms.size),
    }


features = group.to_frame(per_session_features)
print(features.head(5).to_string(index=False))


# ═════════════════════════════════════════════════════════════════════════
# 9. databench.utils — array & label helpers
# ═════════════════════════════════════════════════════════════════════════
section("databench.utils helpers")

t_clean, y_clean = clean_xy(t_pupil, pup)          # drop NaN, sort by t
print(f"clean_xy: {t_pupil.size} -> {t_clean.size} valid samples")

mask, sl = time_mask(t_clean, window=(60.0, 120.0))
print(f"time_mask(60–120s): mask.sum()={int(mask.sum())}, slice={sl}")

print(f"session_to_int('ses-07') = {session_to_int('ses-07')}")
print(f"parse_session_day('ses-07') = {parse_session_day('ses-07')}")
print(f"strip_prefix('ses-07', 'ses-') = {strip_prefix('ses-07', 'ses-')!r}")

# get_first: pick the first present alternative from a row
sample_row = proj.df.iloc[0]
val = get_first(sample_row, [("treadmill", "speed_mm"), ("encoder", "speed")])
print(f"get_first(...): array len = {np.asarray(val).size}")

# label_conditions: map Session -> Condition on any long table
cond_map = {"ses-01": "early", "ses-02": "early", "ses-09": "late", "ses-10": "late"}
labelled = label_conditions(features.copy(), cond_map)
print(f"label_conditions: counts = {labelled['Condition'].value_counts(dropna=False).to_dict()}")

# drop_rows: MultiIndex-aware DataFrame filter (used internally by .filter)
small = drop_rows(proj.df, {"session": "ses-02"})
print(f"drop_rows({{'session':'ses-02'}}): {len(proj.df)} -> {len(small)} rows")


# ═════════════════════════════════════════════════════════════════════════
# 10. DataTabler (lower level — usually accessed via Project.filter)
# ═════════════════════════════════════════════════════════════════════════
section("Project.tabler — index-aware DataFrame helper")

tabler = proj.tabler
print(f"  type   : {type(tabler).__name__}")
print(f"  rows   : {len(tabler.df)}")
print(f"  index  : {tabler.df.index.names}")


# ═════════════════════════════════════════════════════════════════════════
# 11. Plotting — themes, figures, reusable plotters
# ═════════════════════════════════════════════════════════════════════════
section("Plotting surface: themes, factories, plot_metric_by_session")

theme = get_theme()
print(f"active theme: {theme.__class__.__name__}")

fig, ax = new_figure(width=7.0, height_per_row=3.2)
ax.plot(t_pupil[:5000], pup[:5000], lw=0.8)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Pupil (mm)")
style_axes(ax)
style_figure(fig)
run.save_figure(fig, "01_new_figure.png")


# Schema-aware one-liners
fig = quickplot(sess, source="pupil")            # role -> timeseries trace
run.save_figure(fig, "02_quickplot_single.png")

fig = quickplot_group(group, source="pupil", reducer="median")
run.save_figure(fig, "03_quickplot_group.png")

# Per-subject longitudinal directly from a feature table
fig, ax = new_figure()
plot_metric_by_session(features, x="day", y="mean_speed_cms", ax=ax)
ax.set_xlabel("Session day"); ax.set_ylabel("Mean speed (cm/s)")
ax.legend(fontsize=8, ncol=2, frameon=False)
run.save_figure(fig, "04_plot_metric_by_session.png")


# ═════════════════════════════════════════════════════════════════════════
# 12. Run — unified persistence (save_figure / save_table / save_json / pdf / finish)
# ═════════════════════════════════════════════════════════════════════════
section("run.* persistence surface")

# Tables (suffix decides format: csv / parquet / json)
run.save_table(features, "features.csv")
run.save_table(labelled, "block_labelled.csv")

# Raw JSON (e.g. for parameters/sidecars)
run.save_json({"hello": "world", "n_sessions": len(group)}, "demo.json")

# Multi-page PDF report
with run.pdf("session_overview.pdf") as pdf:
    for s in list(group)[:3]:
        fig, ax = new_figure()
        try:
            ax.plot(s.time("treadmill"), s.signal("treadmill", "speed_mm") / 10.0, lw=0.6)
        except Exception:
            continue
        ax.set_title(s.label)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (cm/s)")
        style_axes(ax)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

# Full provenance snapshot (writes provenance.json + report.md)
run.finish(notes="Tour of databench non-analysis APIs.")


print("\nDone — explore the run directory below to see everything that was written:")
print(f"  {run.dir}")
