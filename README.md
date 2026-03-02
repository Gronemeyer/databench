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

## Quickstart — Oscillation Detection

```python
from databench import Project, OscillationDetector
from databench.config import resolve_dataset

# ─── Setup ────────────────────────────────────────────────────────────────
proj = Project(
    dataset=resolve_dataset("etoh"),
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name="oscillations",
    tag="L_VISp-spont",
)

session = proj.session(subject="GS28", session="ses-01", task="task-spont")

# ─── Align auxiliary traces onto ROI timebase ─────────────────────────────
aligned = session.align(
    {"mesomap": ["L_VISp"], "pupil": ["pupil_diameter_mm"], "treadmill": ["speed_mm"]},
    reference="mesomap",
    tolerance_s=0.25,
)

# ─── Configure and run ───────────────────────────────────────────────────
detector = OscillationDetector(
    source="mesomap",
    signal="L_VISp",
    fs=50.0,
    band_hz=(2.0, 4.0),
    filter_order=4,
    threshold=0.02,
    min_duration_s=1.0,
    merge_gap_s=0.5,
)
result = detector.run(session)

# ─── Plot, save, report ──────────────────────────────────────────────────
result.plot_overview(
    aligned=aligned,
    pupil="pupil_diameter_mm",
    speed="speed_mm",
).save("overview.svg")

result.plot_bursts(aligned=aligned, pupil="pupil_diameter_mm", speed="speed_mm")
result.save_events("bursts.csv")
result.save_summary()

proj.save_report(result, notes=f"Detected {len(result.bursts)} bursts.")
```

Raw treadmill arrays are extracted **internally** from the session — scripts
never need to manually call `session.time("treadmill")` /
`session.signal("treadmill", ...)` for plotting.

---

## Output Structure

Every script run produces a timestamped output folder:

```
outputs/<run_name>_<tag>/<YYMMDD_HHMMSS>/
├── plots/          # SVG/PNG figures
├── stats/          # CSV tables
├── reports/        # Markdown summary
└── summary.json    # Machine-readable run metadata
```

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
    run_name="my-analysis",  # output subfolder name
    tag="description",       # appended to run_name
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

### `OscillationDetector`

| Parameter | Default | Description |
|---|---|---|
| `source` | *(required)* | Data source (e.g. `"mesomap"`) |
| `signal` | *(required)* | Signal name (e.g. `"L_VISp"`) |
| `fs` | `50.0` | Sampling rate (Hz) |
| `band_hz` | `(3.1, 4.3)` | Bandpass frequency range |
| `filter_order` | `4` | Butterworth filter order |
| `threshold` | `None` | Fixed threshold (`None` = adaptive) |
| `threshold_k` | `4.0` | Adaptive threshold multiplier |
| `min_duration_s` | `0.5` | Minimum burst duration |
| `merge_gap_s` | `0.25` | Merge bursts closer than this |

`detector.run(session) → OscillationResult`

### `OscillationResult`

| Method | Returns | Description |
|---|---|---|
| `plot_overview(aligned=, pupil=, speed=, window=)` | `SaveableFigure` | Full-session multi-panel plot |
| `plot_bursts(aligned=, max_examples=12, pad_s=5.0, ...)` | `list[SaveableFigure]` | Zoomed per-burst detail plots |
| `save_events(name)` | `Path` | Burst table as CSV |
| `save_summary(name)` | `Path` | JSON run metadata |

### `EtaAnalysis`

```python
eta = EtaAnalysis(
    roi_columns=("L_VISp", "R_VISp"),
    window=(-2.0, 5.0),
    dt=0.05,
    baseline=(-2.0, -1.0),
    source="mesomap",
)
result = eta.run(sessions, events, condition_map={"baseline": [...], ...})
result.plot(event="onset", conditions=["baseline", "ethanol_low"]).save("eta.svg")
result.save_tables()
result.save_summary()
```

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
├── project.py               # Project entry point
├── session.py               # Session, SessionGroup, AlignedData, SaveableFigure
├── config.py                # OutputContext, resolve_dataset, condition colours
├── _reporting.py            # Markdown report writer
├── analysis/
│   ├── oscillation.py       # OscillationDetector, OscillationResult
│   └── eta.py               # EtaAnalysis, EtaResult
├── _plotting/
│   ├── trace_config.py      # TraceStyle, TRACE_STYLES, get_style
│   ├── traces.py            # Signal conditioning & trace plotting helpers
│   └── oscillation.py       # Oscillation-specific multi-panel plots
├── _signal/
│   ├── bandpass.py           # Bandpass filter + Hilbert envelope
│   ├── events.py             # Event detection (locomotion_events, etc.)
│   └── segments.py           # Burst segment utils
├── features/                 # Feature extraction modules
└── plotting/                 # Legacy plotter modules
Scripts/                      # Runnable analysis scripts
datasets.toml                 # Local dataset path aliases
```
