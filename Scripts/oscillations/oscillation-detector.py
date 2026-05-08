"""
Oscillation detector — single-session analysis with overview + burst detail plots.

Detects oscillatory bursts via Hilbert envelope thresholding and generates:
    1. Full-session overview (ROI, filtered, envelope, pupil, speed)
    2. Per-burst zoomed detail windows
    3. Burst events CSV table

Usage:
    python Scripts/oscillation-detector.py
"""
from databench.project import Project
from databench.analysis.oscillation import OscillationDetector
from databench.config import resolve_dataset
from databench.plotting import set_theme

set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET = resolve_dataset("etoh")
SUBJECT = "GS29"
SESSION = "ses-04"
TASK = "task-spont"
ROI_SOURCE = "mesomap"
ROI_NAME = "L_VISp"
PUPIL_KEY = "pupil_diameter_mm"
SPEED_KEY = "speed_mm"

FS = 50.0
BAND = (2.0, 4.0)
ORDER = 4
THRESHOLD = 0.02
MIN_DURATION = 1.0
MERGE_GAP = 0.5

MAX_BURST_EXAMPLES = 12
BURST_PAD_S = 5.0

# ─── Project + session ───────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    run_name=f"{SUBJECT}-{SESSION}-osc",
    tag="L_VISp-spont",
)

session = proj.session(subject=SUBJECT, session=SESSION, task=TASK)

# ─── Align auxiliary traces ──────────────────────────────────────────────

aligned = session.align(
    {ROI_SOURCE: [ROI_NAME], "pupil": [PUPIL_KEY], "treadmill": [SPEED_KEY]},
    reference=ROI_SOURCE,
    tolerance_s=0.25,
)

# ─── Detect oscillations ─────────────────────────────────────────────────

detector = OscillationDetector(
    source=ROI_SOURCE,
    signal=ROI_NAME,
    fs=FS,
    band_hz=BAND,
    filter_order=ORDER,
    threshold=THRESHOLD,
    min_duration_s=MIN_DURATION,
    merge_gap_s=MERGE_GAP,
)

result = detector.run(session)

# ─── Plot overview ───────────────────────────────────────────────────────

slug = f"{SUBJECT}_{SESSION}_{TASK}"
overview = result.overview_plotter(
    aligned=aligned,
    pupil=PUPIL_KEY,
    speed=SPEED_KEY,
    smooth_pupil_s=0.5,
    smooth_speed_s=0.2,
)
overview_fig = overview(result)
proj.io.figure(
    overview_fig,
    f"{slug}_{ROI_NAME}_overview.svg",
    sidecar=overview.recipe(result),
)

# ─── Plot burst details ─────────────────────────────────────────────────

bursts = result.burst_plotter(
    aligned=aligned,
    max_examples=MAX_BURST_EXAMPLES,
    pad_s=BURST_PAD_S,
    pupil=PUPIL_KEY,
    speed=SPEED_KEY,
)
burst_figs = bursts(result)
burst_recipe = bursts.recipe(result)
for rank, fig in enumerate(burst_figs, start=1):
    proj.io.figure(
        fig,
        f"{slug}_{ROI_NAME}_burst_{rank:02d}.svg",
        sidecar={**burst_recipe, "rank": rank},
    )

# ─── Save events & report ───────────────────────────────────────────────

proj.io.table(result.events, f"{slug}_{ROI_NAME}_bursts.csv")

report_path = proj.io.report(
    result,
    notes=(
        f"Oscillation detection ({BAND[0]}–{BAND[1]} Hz Hilbert envelope, "
        f"threshold={THRESHOLD}) in {ROI_SOURCE}/{ROI_NAME} for "
        f"{SUBJECT}/{SESSION}/{TASK}. "
        f"Detected {len(result.bursts)} bursts "
        f"(min_dur={MIN_DURATION}s, merge_gap={MERGE_GAP}s)."
    ),
)

print(f"Done — {len(result.bursts)} bursts detected.")
print(f"Report: {report_path}")
