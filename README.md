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

Select a dataset at runtime via the `DATABENCH_DATASET` env var, or pass an
alias to `resolve_dataset()`:

```bash
DATABENCH_DATASET=etoh python Scripts/oscillation-detector.py
```

---

## Core Concepts

The API follows a linear pipeline:

```
resolve_dataset() → Project → Session → align() → Detector.run() → Result → Plotter → proj.io.* → report
```

| Object | Role |
|---|---|
| `Project` | Opens a dataset, creates the output directory, selects sessions |
| `Project.io` | Single save surface: `figure`, `table`, `json`, `report`, `params` |
| `Session` | One experimental session — provides `signal()`, `time()`, `align()` |
| `SessionGroup` | Iterable collection of sessions (from `project.sessions()`) |
| `AlignedData` | Time-aligned DataFrame produced by `session.align()` |
| `OscillationDetector` | Detects oscillatory bursts via Hilbert envelope thresholding |
| `EtaAnalysis` | Computes event-triggered averages across sessions/conditions |
| `Plotter` | Frozen-dataclass plot recipe with `.recipe(result) → dict` sidecar |

All detector/analysis/plotter objects are **frozen dataclasses** — immutable
configs that produce a result (or figure) when you call `.run()` or call them.

A new analysis script begins from `SCRIPT_TEMPLATE.py` at the repo root.
All knobs are UPPER_CASE module globals at the top — they are auto-snapshotted
into `<run_dir>/config/params.json` on `proj.io.report(...)`.

---

## Discovering Your Data

After opening a project, inspect what's available:

```python
from databench import Project, resolve_dataset

proj = Project(dataset=resolve_dataset("etoh"), run_name="explore", tag="test")

proj.subjects       # ['GS26', 'GS27', 'GS28', 'GS29']
proj.tasks          # ['task-movies', 'task-spont']
proj.all_sessions   # ['ses-01', 'ses-02', 'ses-03', 'ses-04']
```

Error messages always list valid options.  Pass a wrong signal name and the
error will show every available signal for that source:

```
SignalNotFoundError: Signal 'FAKE' not found in source 'mesomap'.
  Available signals for 'mesomap': ['L_MOp', 'L_MOs', 'L_VISp', ...]
  Available sources: ['mesomap', 'pupil', 'treadmill']
```

---

## Example Scripts

Runnable analyses live in `Scripts/scriptings/`.  Each script's header
docstring is a self-contained usage guide — read it before running.

```bash
python Scripts/scriptings/event-based.py
```

| Script | Description |
|---|---|
| `blessed-workflow.py` | Novice-friendly procedural example: load → features → delta → plot → save |
| `event-based.py` | Event-triggered averages across conditions (locomotion-bout ETA) |
| `event-based-widefield-longitudinal.py` | Longitudinal ETA for widefield sessions |
| `extract-features-csv.py` | Batch feature extraction to long-format CSV |
| `locomotion-bouts.py` | Locomotion bout detection, speed/distance stats, PDF report |
| `locomotion-across-conditions.py` | Per-condition bout statistics with boxplots |
| `locomotion-condition-task-compare.py` | Paired boxplots: conditions × tasks |
| `pupil-at-running-scatter.py` | Pupil diameter at bout onset/offset scatter |
| `habituation-oscillation-coupling.py` | Oscillation habituation across 10 days |
| `oscillations/oscillation-detector.py` | Single-session oscillation burst detection |
| `oscillations/oscillation-descriptive-stats.py` | Oscillation descriptive statistics |
| `oscillations/oscillation-timing-analysis.py` | Burst timing and inter-burst intervals |
| `oscillations/oscillation-locomotion-prediction.py` | Oscillation–locomotion coupling |
| `oscillations/oscillation-pupil-eta.py` | Pupil ETA around oscillation bursts |
| `oscillations/quiescent-oscillation-probability.py` | Quiescent-state oscillation probability |

---

## Output Structure

Every script run produces a versioned, fully-provenanced output folder:

```
outputs/<dataset_alias>/<script_name>/<YYMMDD>[_<tag>]/
├── plots/                 # SVG/PNG figures (+ optional <name>.recipe.json sidecars)
├── stats/                 # CSV / parquet / JSON tables
├── reports/               # Markdown summary with clickable git permalink
├── config/
│   └── params.json        # snapshot of all UPPER_CASE script globals
└── provenance.json        # databench version, git hash, env versions, paths
```

Everything goes through `proj.io`:

```python
proj.io.figure(fig, "overview.svg", sidecar=plotter.recipe(result))
proj.io.table(result.events, "bursts.csv")
proj.io.json({"n": 42}, "summary.json")
proj.io.report(result, notes="...")   # writes provenance + params + report.md
```

`provenance.json` includes a clickable GitHub permalink
(`https://github.com/<owner>/<repo>/blob/<full-hash>/<script>`) plus a
`Reproduce` block in the rendered report (`git checkout <hash> && python
<script>`).  Reports use relative image links (`../plots/figure.svg`) so they
render correctly on GitHub and in local Markdown viewers.

`dataset_alias` comes from `DATABENCH_DATASET` (or `DATASET`) when set,
otherwise from the `default` entry in `datasets.toml`.  `script_name` is the
executing script stem; `tag` is appended after the date when supplied.

## Time-column registry

`session.align()` falls back through `databench.config.TIME_COLUMNS` when a
requested `(source, column)` time vector is missing.  Override at the top of a
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
    dataset=Path(...),       # path to the .pkl dataset
    output_root="outputs",   # root output directory
    analyst="Name",          # recorded in reports
    lab="Lab Name",          # recorded in reports
    run_name="my-analysis",  # report/provenance label
    tag="description",       # appended to run folder name
)
```

| Method | Returns | Description |
|---|---|---|
| `session(subject=, session=, task=)` | `Session` | Select one session |
| `sessions(task=, subject=, ...)` | `SessionGroup` | Select multiple sessions |
| `filter(drop_rows=, include=, exclude=, **levels)` | `Project` | Single-call dataset filter |
| `io.figure(fig, name, *, sidecar=None)` | `Path` | Save figure to `plots/` (closes fig) |
| `io.table(df, name, *, index=False)` | `Path` | Save table to `stats/` (suffix-driven) |
| `io.json(payload, name)` | `Path` | Save JSON to `stats/` |
| `io.params()` | `Path` | Snapshot UPPER_CASE script globals to `config/params.json` |
| `io.report(*results, notes=)` | `Path` | Provenance + params + Markdown report |

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
  and `Scripts/scriptings/oscillations/oscillation-detector.py`
- **EtaAnalysis / EtaResult** — see `databench/analysis/eta.py`
  and `Scripts/scriptings/event-based.py`
- **Event generation** (`locomotion_events`, `make_events`) — see
  `databench/analysis/_signal/epoching.py`

---

## Trace Config System

Trace rendering (how treadmill speed, pupil, etc. are processed and drawn) is
defined centrally in `databench.plotting.trace_config`:

```python
from databench.plotting.trace_config import get_style, TRACE_STYLES

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
├── project.py               # Project entry point, output directory management
├── session.py               # Session API + alignment helpers (build_long)
├── config.py                # OutputContext, resolve_dataset, condition colours
├── registry.py              # Internal registration decorators
├── debug.py                 # Diagnostic helpers
├── _reporting.py            # Markdown report writer
├── _utils/
│   ├── __init__.py
│   ├── _logger.py           # Logging configuration
│   └── utils.py             # Shared utility helpers + condition labeling
├── analysis/
│   ├── __init__.py
│   ├── base.py              # Analysis base utilities
│   ├── oscillation.py       # OscillationDetector, OscillationResult (primary API)
│   ├── oscillation_detector.py  # Legacy oscillation detector with plotting/saving
│   ├── eta.py               # EtaAnalysis, EtaResult (primary API)
│   ├── longitudinal.py      # Longitudinal / multi-session analysis
│   ├── mesomap_hilbert.py   # Mesomap Hilbert envelope analysis (spectrograms)
│   └── _signal/             # Canonical signal/epoching primitives
└── plotting/
    ├── __init__.py
    ├── base.py              # Shared plotting base utilities
    ├── core.py              # Core plotter infrastructure
    ├── axes.py              # Axes layout and formatting helpers
    ├── oscillation.py       # Registered oscillation plotters (Overview, BurstDetail, etc.)
    ├── eta.py               # Registered ETA plotters
    ├── mesomap.py           # Mesomap spatial overlay plots
    ├── alignment.py         # Alignment / multi-trace overlay plots
    ├── overview.py          # Session overview figures
    └── treadmill.py         # Treadmill-specific plots

pipeline_demo.py              # Runnable analysis script example
datasets.toml                 # Local dataset path aliases
```
