"""
mesomap-crosscorrelation-movie.py
=================================
Animated sliding-window cross-correlation matrix of mesomap regions
for a single session, saved as an MP4 movie.

Steps:
  1. Load a single session's mesomap signals.
  2. Discover bilateral regions (same filtering as the static script).
  3. Slide a 1 s window across all region signals in 0.1 s steps,
     computing the pairwise Pearson correlation matrix at each step.
  4. Render each matrix as a heatmap frame and encode to MP4.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from databench.project import Project
from databench.config import resolve_dataset
from databench.analysis import detrend_zscore_1d

# ── Config ────────────────────────────────────────────────────────────────

DATASET = "etoh-hfsa"
SOURCE = "mesomap"
SUBJECT = None          # None → first available subject
SESSION = "ses-01"
TASK = None             # None → first available task

WINDOW_S = 1.0          # correlation window duration (seconds)
STEP_S = 0.02           # step between successive windows (seconds)
FS = 50.0               # mesomap sampling rate (Hz)
FPS = 50                # output video frame rate
CLIP_S = 30.0           # only use the first N seconds (None → full recording)

EXCLUDE_REGIONS = {"frame", "VISal", "VISl", "SSp-n", "SSp-m", "SSs"}

# ── Project setup ─────────────────────────────────────────────────────────

proj = Project(
    dataset=resolve_dataset(DATASET),
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
)
run = proj.run(name="mesomap-xcorr-movie")

# ── Select single session ────────────────────────────────────────────────

sessions = proj.sessions(
    subject=SUBJECT,
    sessions=[SESSION] if SESSION else None,
    task=TASK,
)
sess = next(iter(sessions))
print(f"Session: {sess.subject} / {sess.session}")

# ── Discover bilateral regions ────────────────────────────────────────────

strip_hemi = lambda r: r.replace("L_", "").replace("R_", "")
available = set(sess.signals(SOURCE))
bare_names = {strip_hemi(r) for r in available}
bilateral = {
    b for b in bare_names
    if f"L_{b}" in available and f"R_{b}" in available
}
regions_sorted = sorted(
    r for r in available
    if strip_hemi(r) in bilateral and strip_hemi(r) not in EXCLUDE_REGIONS
)
n_regions = len(regions_sorted)
print(f"Bilateral regions ({n_regions}): {regions_sorted}")

# ── Extract and preprocess signals ────────────────────────────────────────

signals = np.column_stack([
    detrend_zscore_1d(sess.signal(SOURCE, r)) for r in regions_sorted
])
n_samples = signals.shape[0]
if CLIP_S is not None:
    n_samples = min(n_samples, int(CLIP_S * FS))
    signals = signals[:n_samples]
total_duration_s = n_samples / FS
print(f"Signal length: {n_samples} samples ({total_duration_s:.1f} s at {FS} Hz)")

# ── Compute sliding-window correlation matrices ──────────────────────────

window_samples = int(WINDOW_S * FS)
step_samples = max(1, int(STEP_S * FS))
n_frames = (n_samples - window_samples) // step_samples + 1

print(f"Window: {window_samples} samples, step: {step_samples} samples, frames: {n_frames}")
print("Computing correlation matrices …")

corr_stack = np.empty((n_frames, n_regions, n_regions), dtype=np.float32)

for frame_index in range(n_frames):
    start = frame_index * step_samples
    end = start + window_samples
    corr_stack[frame_index] = np.corrcoef(signals[start:end], rowvar=False)
    if frame_index % 500 == 0:
        print(f"  frame {frame_index}/{n_frames}")

print("Correlation matrices done.")

# ── Animate ───────────────────────────────────────────────────────────────

labels = [r.replace("L_", "").replace("R_", "") for r in regions_sorted]

fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
im = ax.imshow(corr_stack[0], vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
ax.set_xticks(range(n_regions))
ax.set_xticklabels(labels, rotation=90, fontsize=7)
ax.set_yticks(range(n_regions))
ax.set_yticklabels(labels, fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

window_center_offset_s = (WINDOW_S / 2)
time_text = ax.set_title("", fontweight="bold", fontsize=11)


def update(frame_index):
    center_s = frame_index * STEP_S + window_center_offset_s
    im.set_data(corr_stack[frame_index])
    time_text.set_text(
        f"{sess.subject} {sess.session} — t = {center_s:.1f} s"
    )
    return [im, time_text]


anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // FPS, blit=True)

# ── Save ──────────────────────────────────────────────────────────────────

out_path = run.plots_dir / f"{sess.subject}_{sess.session}_mesomap-ccr.mp4"

try:
    writer = FFMpegWriter(fps=FPS, metadata={"title": "mesomap xcorr"})
    print(f"Saving MP4 ({n_frames} frames at {FPS} fps) …")
    anim.save(str(out_path), writer=writer)
    print(f"Saved → {out_path}")
except (FileNotFoundError, RuntimeError):
    # ffmpeg not available — fall back to GIF via Pillow
    out_path = out_path.with_suffix(".gif")
    print("ffmpeg not found, falling back to GIF …")
    writer = PillowWriter(fps=FPS)
    anim.save(str(out_path), writer=writer)
    print(f"Saved → {out_path}")

plt.close(fig)
