# databench

Reproducible analysis toolkit for multiindex neuroscience datasets.

## Setup

```bash
conda env create -f environment.yml
conda activate databench
pip install -e .
```

Copy the dataset config and fill in your local paths:

```bash
cp datasets.toml.example datasets.toml
```

```toml
# datasets.toml
default = "etoh-hfsa"

[datasets]
etoh      = "/path/to/260211_ETOH_dataset.pkl"
etoh-hfsa = "/path/to/260212_ETOH-HFSA_dataset.pkl"
acutevis  = "/path/to/260212_ACUTEVIS_dataset.pkl"
```

Optionally copy `databench.toml.example` to `databench.toml` (or `~/.databench.toml`)
to record your analyst/lab name in every run's provenance:

```toml
# databench.toml
[user]
analyst = "Your Name"
lab     = "Your Lab"
```

Select a dataset at runtime via the `DATABENCH_DATASET` env var, or pass an
alias to `Project(...)` / `resolve_dataset()`:

```bash
DATABENCH_DATASET=etoh python analysis_scripts/oscillations/oscillation-detector.py
```

### Lazy dataset archives (`.npz`)

A dataset pickle is read in full on every run. Convert it once to an `.npz`
dataset archive and a run reads only what it uses:

```bash
python analysis_scripts/Tools/convert-dataset.py etoh2      # alias or path; writes beside the .pkl
```

Then point the alias's `path` at the `.npz`. Scripts need no change:
`Project` reads only a small index frame on open, and `sess.signal(...)`
/ `sess.align(...)` load each array as a zero-copy view of the file. The
converter compares every cell with the pickle before it reports success.
Cells in `proj.df` that live in the archive appear as `LazyCell` objects;
`np.asarray(cell)` loads one. Provenance records a fingerprint of the
archive's member table plus the SHA-256 of the pickle it came from. The
archive is a plain zip of `.npy` files, so
`np.load(path, allow_pickle=True)["GS26/ses-01/task-spont/mesomap/L_VISp"]`
works without databench. Format details: `databench/store.py`.

---

## Core Concepts

The API follows a linear pipeline:

```
Project → session(s) → align() → Detector.run() → Result → plot → run.save_*() → run.finish()
```

| Object | Role |
|---|---|
| `Project` | Opens a dataset (read-only), selects sessions, exposes the schema |
| `Run` | One output directory + its provenance — created by `proj.run(...)` |
| `Session` | One experimental session — provides `signal()`, `time()`, `align()` |
| `SessionGroup` | Iterable collection of sessions (from `proj.sessions(...)`) |
| `AlignedData` | Time-aligned DataFrame produced by `session.align()` |
| `OscillationDetector` | Detects oscillatory bursts via Hilbert envelope thresholding |
| `EtaAnalysis` | Computes event-triggered averages across sessions/conditions |
| `Plotter` | Frozen-dataclass plot recipe with `.recipe(result) → dict` sidecar |

`Project` is read-only data; a `Run` is where results go. All
detector/analysis/plotter objects are **frozen dataclasses** — immutable
configs that produce a result (or figure) when you call `.run()` or call them.

A new analysis script begins from `analysis_scripts/_template.py`. All knobs are
UPPER_CASE module globals at the top — `run.finish()` snapshots them into the
run's `provenance.json` so the run is reproducible. `analysis_scripts/how-to.py` is a
runnable tour of the full API.

---

## Discovering Your Data

After opening a project, inspect what's available:

```python
from databench import Project

proj = Project("etoh")     # datasets.toml alias (or a path to the .pkl)
proj.describe()            # the canonical "what's in here?" call

proj.subjects       # ['GS26', 'GS27', 'GS28', 'GS29']
proj.tasks          # ['task-movies', 'task-spont']
proj.all_sessions   # ['ses-01', 'ses-02', 'ses-03', 'ses-04']
proj.schema         # declared sources, columns, units, roles
```

Error messages always list valid options. Pass a wrong signal name and the
error will show every available signal for that source:

```
SignalNotFoundError: Signal 'FAKE' not found in source 'mesomap'.
  Available signals for 'mesomap': ['L_MOp', 'L_MOs', 'L_VISp', ...]
  Available sources: ['mesomap', 'pupil', 'treadmill']
```

---

## Analysis Scripts

Runnable analyses live in the separate `analysis_scripts` git repository,
checked out at `./analysis_scripts/` (git ignores it here). Each script's
header docstring is a self-contained usage guide; read it before you run
the script.

```bash
python analysis_scripts/event-triggered/event-based.py
```

| Script | Description |
|---|---|
| `_template.py` | Minimal starting point: setup → analyze → save → finish |
| `how-to.py` | Runnable tour of the Project / Run / Session API |
| `event-triggered/event-based.py` | Event-triggered averages across conditions (locomotion-bout ETA) |
| `event-triggered/event-based-widefield-longitudinal.py` | Longitudinal ETA for widefield sessions |
| `event-triggered/2p-event-triggered-eta.py` | Event-triggered averages for 2p sessions |
| `locomotion/locomotion-bouts.py` | Locomotion bout detection, speed/distance stats, PDF report |
| `locomotion/locomotion-condition-task-compare.py` | Paired boxplots: conditions × tasks |
| `pupil/pupil-at-running-scatter.py` | Pupil diameter at bout onset/offset scatter |
| `pupil/pupil-baseline-normalize.py` | Pupil baseline normalization |
| `oscillations/oscillation-detector.py` | Single-session oscillation burst detection |
| `oscillations/oscillation-descriptive-stats.py` | Oscillation descriptive statistics |
| `Tools/extract-features-csv.py` | Batch feature extraction to long-format CSV |

---

## Output Structure

Every run produces a versioned, fully-provenanced output folder:

```
outputs/<dataset_alias>/<script_name>/<YYMMDD_HHMMSS>[_<tag>]/
├── overview.svg            # figures (SVG/PNG, multi-page PDFs) — flat
├── bursts.csv              # tables (CSV / parquet / JSON) — flat
├── provenance.json        # databench version, git hash, env versions, paths
└── report.md              # human-readable summary with clickable git permalink
```

Everything goes through the `Run`:

```python
proj = Project("etoh")
run  = proj.run(name="oscillation-detector", tag="L_VISp-spont")

run.save_figure(fig, "overview.svg")        # → run dir
run.save_table(result.events, "bursts.csv") # → run dir (suffix-driven)
run.save_json({"n": 42}, "summary.json")    # → run dir
run.finish(notes="...")                      # writes provenance.json + report.md
```

`provenance.json` is the single source of truth: it includes a clickable
GitHub permalink (`https://github.com/<owner>/<repo>/blob/<full-hash>/<script>`)
plus a `Reproduce` block in the rendered report (`git checkout <hash> && python
<script>`). Reports use relative image links (`figure.svg`) so they
render correctly on GitHub and in local Markdown viewers.

### Browsing past runs

```bash
python -m databench.viewer          # or: databench-view
```

This opens a local page that finds every run under `outputs/`. It rescans
each time you reload, so it needs no index. You can search by script, tag,
dataset, notes or file name, and filter by dataset. Each run shows its
figures in a gallery, previews of its tables and documents, and its
provenance. The page also has buttons to copy the reproduce command and to
reveal the folder in Finder. Options: `--root PATH`, `--port N`,
`--no-browser`. The viewer uses only the standard library, and it binds to
`127.0.0.1` by default.

### Sharing without Python

Every `run.finish()` writes two pages:

- `<run dir>/report.html` — that one run, fully self-contained. Figures are
  embedded in the file, so you can email it by itself.
- `outputs/index.html` — every run, but figures are linked by relative path,
  so keep it with the `outputs/` folder.

To build one file for a group of runs, embed their figures and scope it:

```bash
python -m databench.viewer --standalone --alias etoh2
python -m databench.viewer --standalone --script fig1-cortical-state-correlation
python -m databench.viewer --standalone --since 260801 --out ~/Desktop/august.html
```

Filters: `--alias`, `--script`, `--tag`, `--since` (`260801` or `2026-08-01`),
`--run <outputs-relative path>`. `--export` writes the linked-figure
`index.html` instead.

Each page holds the run list, provenance, notes, the first 100 rows of each
table, and text files. Standalone pages embed one image per figure: the SVG
when there is one, because it stays sharp at any zoom, or the PNG when the
SVG is much larger. PDFs are too large to embed, so those links need the
folder and the page dims them.

Standalone pages also contain a plain-HTML copy of each run. Phones usually
open a local `.html` file in a preview that doesn't run JavaScript: iOS Quick
Look in Files, Mail and Messages, and mail or chat apps on Android. That copy
is what such previews show. In a real browser, the interactive viewer replaces
it.

Browsers treat each local file as its own origin and ask permission before a
page reads files beside it. Standalone pages avoid that prompt; `index.html`
does not, because its figures are separate files.

These pages include the absolute paths recorded in provenance (dataset
location, repo path). Check them before you send anything outside your lab.

`dataset_alias` comes from `DATABENCH_DATASET` (or `DATASET`) when set,
otherwise from the `default` entry in `datasets.toml`. `script_name` is the
executing script stem; `tag` is appended after the timestamp when supplied.

## Time-column registry

`session.align()` falls back through `databench.config.TIME_COLUMNS` when a
requested `(source, column)` time vector is missing. Override at the top of a
script to teach databench about a new dataset's clocks:

```python
from databench import config
config.TIME_COLUMNS = [
    ("dataqueue", "time_elapsed_s"),
    ("time",      "master_elapsed_s"),
]
```

---

## API Reference

### `Project`

```python
proj = Project(
    "etoh",                  # datasets.toml alias OR a path to the .pkl dataset
    output_root="outputs",   # root output directory
    analyst="Name",          # recorded in provenance (falls back to databench.toml)
    lab="Lab Name",          # recorded in provenance (falls back to databench.toml)
)
```

| Method | Returns | Description |
|---|---|---|
| `run(name=, tag=)` | `Run` | Create a versioned output directory |
| `session(subject=, session=, task=)` | `Session` | Select one session |
| `sessions(task=, subject=, ...)` | `SessionGroup` | Select multiple sessions |
| `first_session()` | `Session` | First session in the current selection |
| `filter(drop_rows=, include=, exclude=, **levels)` | `Project` | In-place dataset filter (chainable) |
| `describe()` | `str` | Print/return the "what's in here?" summary |
| `subjects` / `all_sessions` / `tasks` | `list[str]` | Index level values |
| `df` / `schema` / `tabler` | — | Underlying DataFrame, schema, and DataTabler |

### `Run`

| Method | Returns | Description |
|---|---|---|
| `save_figure(fig, name, *, formats=None, close=True)` | `Path` | Save figure to the run dir (closes fig) |
| `save_table(df, name, *, index=False)` | `Path` | Save table to the run dir (suffix-driven) |
| `save_json(payload, name)` | `Path` | Save JSON to the run dir |
| `pdf(name)` | context manager | Multi-page PDF in the run dir |
| `finish(notes=)` | `Path` | Write `provenance.json` + `report.md`; return the run dir |

### `Session`

| Method | Returns | Description |
|---|---|---|
| `signal(source, name)` | `np.ndarray` | Extract a 1-D signal |
| `time(source, column)` | `np.ndarray` | Extract the time array |
| `align(sources, reference=, tolerance_s=)` | `AlignedData` | Time-align signals via `merge_asof` |

### Analysis APIs

Analysis-specific parameter tables and usage examples live in module
docstrings and script headers:

- **OscillationDetector / OscillationResult** — see `databench/analysis/oscillation.py`
  and `analysis_scripts/oscillations/oscillation-detector.py`
- **EtaAnalysis / EtaResult** — see `databench/analysis/eta.py`
  and `analysis_scripts/event-triggered/event-based.py`
- **Event generation** — `locomotion_events` in `databench/analysis/locomotion.py`,
  `make_events` in `databench/signal/epoching.py`

---

## Trace Config System

Trace rendering (how treadmill speed, pupil, etc. are processed and drawn) is
defined centrally in `databench.plotting.style.trace_config`:

```python
from databench.plotting.style.trace_config import get_style, TRACE_STYLES

tread = get_style("treadmill")  # step_previous remap, steps-post drawstyle, #00CC96
pupil = get_style("pupil", smooth_savgol_s=1.0)  # override smoothing window
```

Pre-defined styles: `treadmill`, `pupil`, `roi`, `filtered`, `envelope`.

Each `TraceStyle` controls: remap method, gap-filling, smoothing, outlier
removal, colour, line width, opacity, drawstyle, and y-axis label. Analysis
plotting modules use these internally — adding a new trace type is a one-line
entry in `TRACE_STYLES`.

---

## Versioning

The version comes from git tags (setuptools-scm). The repository does not
store the version in a file. A tagged commit is that version (`v0.3.0` →
`0.3.0`); later commits are development versions (`0.3.1.devN`).
`databench.__version__` and every run's `provenance.json` read the
version from `git describe` in a checkout, so an editable install never
reports a stale version.

```bash
make version            # print the current version
make release V=0.3.1    # tag v0.3.1 on a clean tree and push it
```

When you push a `v*` tag, `.github/workflows/release.yml` builds the
distributions and attaches them to a GitHub release.

---

## Project Layout

```
databench/
├── __init__.py              # Public exports
├── project.py               # Project: open dataset, select sessions (read-only)
├── run.py                   # Run: output dir + save_figure/table/json + finish()
├── provenance.py            # Provenance capture + report.md rendering
├── session.py               # Session / SessionGroup: signal(), time(), align()
├── config.py                # resolve_dataset, dataset_params, TIME_COLUMNS
├── tabler.py                # DataTabler — DataFrame selection/filtering
├── types.py                 # Shared type aliases
├── analysis/
│   ├── eta.py               # EtaAnalysis, EtaResult (primary API)
│   ├── oscillation.py       # OscillationDetector, OscillationResult (primary API)
│   ├── locomotion.py        # locomotion_events + bout analysis
│   ├── longitudinal.py      # Longitudinal / multi-session analysis
│   └── pupil_baseline.py    # Pupil baseline normalization
├── signal/
│   ├── bandpass.py          # Bandpass + Hilbert envelope, robust_threshold
│   ├── epoching.py          # detect_epochs, make_events, extract_epoch_*
│   ├── preproc.py           # Preprocessing helpers
│   └── remap.py             # Time-base remapping
├── plotting/
│   ├── base.py              # Plotter base (frozen-dataclass recipes)
│   ├── core.py              # Core plotter infrastructure
│   ├── traces.py            # Trace drawing
│   ├── oscillation.py       # Oscillation plotters
│   ├── mesomap.py           # Mesomap spatial overlay plots
│   ├── overview.py          # Session overview figures
│   ├── treadmill.py         # Treadmill-specific plots
│   └── style/
│       ├── trace_config.py  # get_style, TRACE_STYLES
│       └── cold_field_v5.py # Lab theme
└── utils/
    ├── arrays.py            # as_1d, clean_xy, drop_rows, time_mask, ...
    ├── labels.py            # Session-label parsing / condition labeling
    └── logger.py            # Logging configuration

analysis_scripts/             # Separate git repo of analysis scripts (ignored here)
datasets.toml                 # Local dataset path aliases
databench.toml                # Local analyst/lab config (optional)
```
