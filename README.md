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
resolve_dataset() → Project → Session → align() → Detector.run() → Result → plot/save → report
```

| Object | Role |
|---|---|
| `Project` | Opens a dataset, creates the output directory, selects sessions |
| `Session` | One experimental session — provides `signal()`, `time()`, `align()` |
| `SessionGroup` | Iterable collection of sessions (from `project.sessions()`) |
| `AlignedData` | Time-aligned DataFrame produced by `session.align()` |
| `OscillationDetector` | Detects oscillatory bursts via Hilbert envelope thresholding |
| `EtaAnalysis` | Computes event-triggered averages across sessions/conditions |
| `SaveableFigure` | Wraps a matplotlib Figure with `.save()` / `.show()` |

All detector/analysis objects are **frozen dataclasses** — immutable configs
that produce a result when you call `.run()`.

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

Every script run produces a timestamped output folder:

```
outputs/<dataset_alias>/<script_name>/<YYMMDD>/<tag>/
├── plots/          # SVG/PNG figures
├── stats/          # CSV tables
├── reports/        # Markdown summary
└── summary.json    # Machine-readable run metadata
```

`dataset_alias` comes from `DATABENCH_DATASET` (or `DATASET`) when set,
otherwise from the `default` entry in `datasets.toml`.
`script_name` is the executing script stem, and `tag` defaults to `untagged`
when no tag is provided.

`SaveableFigure.save()` writes to `plots/`, `result.save_events()` writes to
`stats/`, and `proj.save_report()` writes to `reports/`. Reports use relative
image links (`../plots/figure.svg`) so they render correctly on GitHub and in
local Markdown viewers.

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
| `save_report(*results, notes=)` | `Path` | Generate Markdown report |

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
  `databench/_signal/events.py`

---

## Trace Config System

Trace rendering (how treadmill speed, pupil, etc. are processed and drawn) is
defined centrally in `databench._plotting.trace_config`:

```python
from databench._plotting.trace_config import get_style, TRACE_STYLES

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
├── bench.py                 # Internal coordinator (not part of the script-facing API)
├── project.py               # Project entry point, output directory management
├── session.py               # Session, SessionGroup, AlignedData, SaveableFigure
├── config.py                # OutputContext, resolve_dataset, condition colours
├── registry.py              # Internal registration decorators
├── provenance.py            # Run metadata and provenance tracking
├── utils.py                 # Shared array utilities (as_1d, clean_xy, get_first, etc.)
├── debug.py                 # Diagnostic helpers
├── _reporting.py            # Markdown report writer
├── _io/
│   ├── __init__.py
│   └── loader.py            # Dataset loading (pickle / HDF5)
├── _utils/
│   ├── __init__.py
│   └── _logger.py           # Logging configuration
├── _signal/
│   ├── __init__.py
│   ├── bandpass.py           # Butterworth bandpass filter + Hilbert envelope
│   ├── bouts.py             # Locomotion bout detection from speed traces
│   ├── epochs.py            # Peri-event epoch extraction and interpolation
│   ├── eta_core.py          # Event-triggered average computation core
│   ├── events.py            # Event detection (locomotion_events, etc.)
│   └── segments.py          # Contiguous-segment detection, merging, filtering
├── analysis/
│   ├── __init__.py
│   ├── base.py              # Analysis base utilities
│   ├── oscillation.py       # OscillationDetector, OscillationResult (primary API)
│   ├── oscillation_detector.py  # Legacy oscillation detector with plotting/saving
│   ├── eta.py               # EtaAnalysis, EtaResult (primary API)
│   ├── longitudinal.py      # Longitudinal / multi-session analysis
│   └── mesomap_hilbert.py   # Mesomap Hilbert envelope analysis (spectrograms)
├── features/
│   ├── __init__.py
│   ├── base.py              # FeatureFn base class, @register_feature decorator
│   ├── meso.py              # Mesoscale imaging features
│   ├── pupil.py             # Pupil diameter features
│   └── treadmill.py         # Treadmill / locomotion features and bout stats
├── _plotting/
│   ├── __init__.py
│   ├── trace_config.py      # TraceStyle, TRACE_STYLES, get_style
│   ├── traces.py            # Signal conditioning & trace plotting helpers
│   ├── oscillation.py       # Oscillation-specific multi-panel plots (internal)
│   └── eta.py               # ETA plotting helpers (internal)
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
