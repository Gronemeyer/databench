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
DATABENCH_DATASET=etoh python Scripts/oscillations/oscillation-detector.py
```

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

A new analysis script begins from `Scripts/_template.py`. All knobs are
UPPER_CASE module globals at the top — `run.finish()` snapshots them into the
run's `provenance.json` so the run is reproducible. `Scripts/how-to.py` is a
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

## Example Scripts

Runnable analyses live under `Scripts/`. Each script's header docstring is a
self-contained usage guide — read it before running.

```bash
python Scripts/event-triggered/event-based.py
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
| `oscillations/oscillation-pupil-eta.py` | Pupil ETA around oscillation bursts |
| `Tools/extract-features-csv.py` | Batch feature extraction to long-format CSV |

---

## Output Structure

Every run produces a versioned, fully-provenanced output folder:

```
outputs/<dataset_alias>/<script_name>/<YYMMDD_HHMMSS>[_<tag>]/
├── plots/                 # SVG/PNG figures (+ multi-page PDFs)
├── tables/                # CSV / parquet / JSON tables
├── provenance.json        # databench version, git hash, env versions, paths
└── report.md              # human-readable summary with clickable git permalink
```

Everything goes through the `Run`:

```python
proj = Project("etoh")
run  = proj.run(name="oscillation-detector", tag="L_VISp-spont")

run.save_figure(fig, "overview.svg")        # → plots/
run.save_table(result.events, "bursts.csv") # → tables/ (suffix-driven)
run.save_json({"n": 42}, "summary.json")    # → tables/
run.finish(notes="...")                      # writes provenance.json + report.md
```

`provenance.json` is the single source of truth: it includes a clickable
GitHub permalink (`https://github.com/<owner>/<repo>/blob/<full-hash>/<script>`)
plus a `Reproduce` block in the rendered report (`git checkout <hash> && python
<script>`). Reports use relative image links (`plots/figure.svg`) so they
render correctly on GitHub and in local Markdown viewers.

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
| `save_figure(fig, name, *, formats=None, close=True)` | `Path` | Save figure to `plots/` (closes fig) |
| `save_table(df, name, *, index=False)` | `Path` | Save table to `tables/` (suffix-driven) |
| `save_json(payload, name)` | `Path` | Save JSON to `tables/` |
| `pdf(name)` | context manager | Multi-page PDF rooted at `plots/` |
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
  and `Scripts/oscillations/oscillation-detector.py`
- **EtaAnalysis / EtaResult** — see `databench/analysis/eta.py`
  and `Scripts/event-triggered/event-based.py`
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

Scripts/_template.py          # Minimal analysis-script starting point
Scripts/how-to.py             # Runnable API tour
datasets.toml                 # Local dataset path aliases
databench.toml                # Local analyst/lab config (optional)
```
