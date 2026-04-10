"""
Spatial phase structure analysis of mesoscope widefield oscillatory activity.

Two-pass spatial-tiling approach (RAM-friendly, GPU-accelerated via CuPy):
  Pass 1 — Combined FFT bandpass + Hilbert per tile on GPU; accumulate global
            analytic signal (mean over cortex pixels).
  Pass 2 — Compute amplitude-gated relative-phase summaries per tile on GPU
            using angle(analytic · conj(global_analytic)).

Cortex masking via intensity threshold on the temporal-mean frame.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import cupy as cp
import tifffile
import pandas as pd
import re
from scipy.signal import butter, sosfreqz
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap
import imageio.v3 as iio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from PIL import Image, ImageDraw, ImageFont

# ═══════════════════════════════════════════════════════════════════════════════
# Global Parameters — Analysis
# ═══════════════════════════════════════════════════════════════════════════════

# Path to the source mesoscope widefield TIFF stack
TIFF_PATH = Path(
r"F:\251215_ETOH_RO1\data\sub-GS27\ses-01\func\20251215_151240_sub-GS27_ses-01_task-spont_mesoscope.ome.tiff"
)

FS = 50.0                 # Acquisition rate in Hz
BAND = (3, 4)             # Bandpass frequency range (Hz) around the oscillation peak
FILTER_ORDER = 4          # Butterworth filter order (applied as zero-phase via filtfilt)
DOWNSAMPLE_SPATIAL = 2    # Spatial bin factor (1 = no binning)
CORTEX_THRESH = 0.05      # Intensity threshold for cortex mask (fraction of normalised range)
AMP_PERCENTILE = 20       # Percentile for amplitude gating (exclude low-amplitude frames)

MOVIE_DUR = 60.0          # Duration of the output movie clip in seconds (max; may be shorter)
N_SNAP = 100              # Number of consecutive snapshot frames for the grid figure
SNAP_START_S = None        # Start time for snapshots (seconds); None = same as movie start

# ── Locomotion masking ──────────────────────────────────────────────────────
# Locomotion mask table (columns: Subject, Session, Task, state, onset_idx, offset_idx, …)
LOCO_MASK_PATH = Path(
    r"C:\dev\databench\outputs\etoh\locomotion-bouts"
    r"\260323\ETOH_5-seconds\stats\locomotion_mask_table.csv"
)
MIN_QUIESCENCE_S = 5.0     # Minimum duration (seconds) for a quiescent epoch to be used

# Contour levels and RGB colours drawn on the relative-phase movie panel
CONTOUR_LEVELS = [-np.pi / 2, 0.0, np.pi / 2]
CONTOUR_COLORS_RGB = [
    np.array([0, 255, 255], dtype=np.uint8),    # cyan  at -π/2
    np.array([255, 255, 255], dtype=np.uint8),   # white at 0
    np.array([255, 255, 0], dtype=np.uint8),     # yellow at +π/2
]

# Colormap names / objects for each movie panel
CMAP_DFF = LinearSegmentedColormap.from_list(
    "green_fluorescence",
    [(0.0, "#000000"), (0.5, "#003300"), (0.75, "#00aa00"), (1.0, "#66ff66")],
)
CMAP_BP = "RdBu_r"        # Colormap for the bandpassed panel
CMAP_PHASE = "twilight"    # Colormap for relative- and absolute-phase panels

MIN_PANEL_HEIGHT_PX = 500  # Minimum pixel height of each movie panel (drives upscale factor)

# Codec priority list for MP4 encoding (first successful codec wins)
VIDEO_CODECS = ("libx264", "h264", "mpeg4")

# ═══════════════════════════════════════════════════════════════════════════════
# Global Parameters — Performance Tuning
# ═══════════════════════════════════════════════════════════════════════════════

TILE_SIZE = 16             # Base spatial tile size in downsampled pixels (auto-scaled up by VRAM)
PRELOAD_TO_RAM = True      # If True, copy full TIFF into contiguous RAM before processing
NUM_IO_WORKERS = 4         # CPU threads for prefetching tiles ahead of GPU
VRAM_BUDGET_FRAC = 0.60    # Fraction of free GPU VRAM to budget for tile buffers
VRAM_BUFFERS_PER_PX = 5    # Estimated number of complex64 buffers per pixel during FFT
N_MEAN_FRAMES = 1000       # Max frames sampled for computing the temporal mean image


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions — Image Processing
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_image(arr):
    """Min-max normalise an array to [0, 1]. Returns zeros if range is degenerate."""
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or np.isclose(mn, mx):
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def build_cortex_mask(mean_frame, threshold):
    """Binary cortex mask: normalise → threshold → fill holes → keep largest component."""
    img = normalize_image(mean_frame)
    mask = binary_fill_holes(img > threshold).astype(bool)
    labelled = label(mask)
    if labelled.max() > 0:
        props = regionprops(labelled)
        main = max(props, key=lambda p: p.area)
        mask = labelled == main.label
    return mask


def build_bp_hilbert_kernel(n_frames, sos, fs):
    """Build a combined bandpass + analytic-signal kernel in the frequency domain.

    Returns a CuPy complex64 array of length n_frames that, when multiplied with
    the FFT of a signal, simultaneously applies zero-phase bandpass filtering
    and the Hilbert transform.
    """
    _, H = sosfreqz(sos, worN=n_frames, fs=fs, whole=True)
    H_filt = np.abs(H) ** 2  # Zero-phase (filtfilt) magnitude response

    h = np.zeros(n_frames)
    h[0] = 1.0
    if n_frames % 2 == 0:
        h[n_frames // 2] = 1.0
        h[1 : n_frames // 2] = 2.0
    else:
        h[1 : (n_frames + 1) // 2] = 2.0

    return cp.asarray((H_filt * h).astype(np.complex64))


def make_colormap_lut(cmap):
    """Create a 256-entry uint8 RGB lookup table from a matplotlib colormap."""
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    return (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def apply_lut(data, lut, vmin, vmax):
    """Map a float array to RGB via a 256-entry LUT. NaNs map to mid-range."""
    safe = np.nan_to_num(data, nan=(vmin + vmax) * 0.5)
    idx = np.clip((safe - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    return lut[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions — Performance / IO
# ═══════════════════════════════════════════════════════════════════════════════

def auto_tile_size(n_frames, ny_ds, nx_ds, base_tile):
    """Scale tile size up to fill available GPU VRAM (within budget fraction)."""
    free_gpu, _ = cp.cuda.Device().mem_info
    bytes_per_px = 8 * VRAM_BUFFERS_PER_PX * n_frames  # complex64 × buffers
    max_px = int(free_gpu * VRAM_BUDGET_FRAC) // max(bytes_per_px, 1)
    auto = max(base_tile, min(int(np.sqrt(max(max_px, 1))), ny_ds, nx_ds))
    return auto


def tile_ranges(total, tile):
    """Yield (start, end) pairs that tile an axis of length *total*."""
    for s in range(0, total, tile):
        yield s, min(s + tile, total)


def read_tile(tif, r0, r1, c0, c1, ds, n_frames):
    """Read a spatial tile from the data array, spatially bin, and mean-subtract."""
    raw = tif[:, r0 * ds : r1 * ds, c0 * ds : c1 * ds].astype(np.float32)
    if ds > 1:
        th, tw = r1 - r0, c1 - c0
        raw = raw[:, : th * ds, : tw * ds].reshape(
            n_frames, th, ds, tw, ds
        ).mean(axis=(2, 4))
    raw -= raw.mean(axis=0, keepdims=True)
    return raw


def gpu_fft_filter(raw, kernel):
    """Transfer tile to GPU → FFT → apply kernel → IFFT. Returns CuPy complex64."""
    d_raw = cp.asarray(raw)
    D = cp.fft.fft(d_raw, axis=0)
    del d_raw
    D *= kernel[:, None, None]
    result = cp.fft.ifft(D, axis=0).astype(cp.complex64)
    del D
    return result


def prefetch_tiles(tile_list, tif, ds, n_frames, n_workers, lookahead):
    """Generator that yields (index, tile_coords, cpu_tile) with IO prefetch.

    Uses a thread pool to read the next *lookahead* tiles on CPU while the
    caller processes the current one on GPU.
    """
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for j in range(min(lookahead, len(tile_list))):
            futures[j] = pool.submit(read_tile, tif, *tile_list[j], ds, n_frames)
        for i, coords in enumerate(tile_list):
            raw = futures.pop(i).result()
            nxt = i + lookahead
            if nxt < len(tile_list):
                futures[nxt] = pool.submit(read_tile, tif, *tile_list[nxt], ds, n_frames)
            yield i, coords, raw


def draw_contours_gpu(phase_buf, cortex_mask, levels, colors, dest_rgb):
    """Draw iso-phase contours onto *dest_rgb* using GPU level-crossing detection.

    For each level, detects sign changes between adjacent pixels (horizontal and
    vertical) on the GPU, then stamps the corresponding colour onto the RGB array.
    """
    d_phase = cp.asarray(np.nan_to_num(phase_buf, nan=999.0))
    d_mask = cp.asarray(cortex_mask)

    for level, color in zip(levels, colors):
        shifted = d_phase - level
        h_cross = (shifted[:, :, :-1] * shifted[:, :, 1:]) < 0
        v_cross = (shifted[:, :-1, :] * shifted[:, 1:, :]) < 0
        edges = cp.zeros(d_phase.shape, dtype=cp.bool_)
        edges[:, :, :-1] |= h_cross
        edges[:, :, 1:] |= h_cross
        edges[:, :-1, :] |= v_cross
        edges[:, 1:, :] |= v_cross
        edges &= d_mask[None, :, :]
        edges_cpu = edges.get()
        dest_rgb[edges_cpu] = color
        del shifted, h_cross, v_cross, edges, edges_cpu

    del d_phase, d_mask
    cp.get_default_memory_pool().free_all_blocks()


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions — Video Compositing
# ═══════════════════════════════════════════════════════════════════════════════

def load_fonts(render_scale):
    """Try to load scaled TrueType fonts; fall back to PIL default."""
    sizes = {
        "title": max(14, 14 * render_scale),
        "time":  max(12, 13 * render_scale),
        "foot":  max(9,  10 * render_scale),
    }
    try:
        font_title = ImageFont.truetype("arialbd.ttf", sizes["title"])
        font_time  = ImageFont.truetype("consola.ttf", sizes["time"])
        font_foot  = ImageFont.truetype("arial.ttf",   sizes["foot"])
    except Exception:
        try:
            font_title = ImageFont.truetype("arial.ttf",   sizes["title"])
            font_time  = ImageFont.truetype("consola.ttf", sizes["time"])
            font_foot  = ImageFont.truetype("arial.ttf",   sizes["foot"])
        except Exception:
            font_title = ImageFont.load_default()
            font_time  = font_title
            font_foot  = font_title
    return font_title, font_time, font_foot


def compose_movie_frames(
    panels, render_scale, panel_titles, param_str,
    movie_start, n_frames_total, fs, font_title, font_time, font_foot,
):
    """Add header (titles + timestamp) and footer (params) to the panel array.

    Parameters
    ----------
    panels : ndarray (T, H, W_total, 3) uint8 — horizontally concatenated panel frames.
    render_scale : int — upscale factor applied to panels.

    Returns
    -------
    rgb_frames : ndarray (T, H_out, W_out, 3) uint8 — composited frames ready for encoding.
    """
    n_movie, ph, pw_total, _ = panels.shape
    header_px = 35 * render_scale
    footer_px = 25 * render_scale
    frame_h = header_px + ph + footer_px
    # x264 requires even dimensions
    if frame_h % 2:
        footer_px += 1
        frame_h += 1
    if pw_total % 2:
        pw_total += 1
        panels = np.pad(panels, ((0, 0), (0, 0), (0, 1), (0, 0)))
    panel_w = pw_total // 4

    # Pre-render static overlay (titles + footer) as RGBA template
    template = Image.new("RGBA", (pw_total, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(template)
    for pi, title in enumerate(panel_titles):
        draw.text(
            (panel_w * pi + panel_w // 2, header_px - 4 * render_scale),
            title, fill=(255, 255, 255, 255), anchor="mb", font=font_title,
        )
    draw.text(
        (pw_total // 2, frame_h - 3 * render_scale),
        param_str, fill=(128, 128, 128, 255), anchor="mb", font=font_foot,
    )
    static_overlay = np.array(template)
    static_mask = static_overlay[:, :, 3] > 0
    static_rgb = static_overlay[:, :, :3]
    del template, draw

    # Embed panels into frames with header/footer rows
    rgb_frames = np.zeros((n_movie, frame_h, pw_total, 3), dtype=np.uint8)
    rgb_frames[:, header_px : header_px + ph, :, :] = panels

    # Stamp static overlay across all frames (vectorised)
    rgb_frames[:, static_mask] = static_rgb[static_mask]

    # Per-frame timestamp
    fsz_time = max(12, 13 * render_scale)
    ts_row_h = 4 + fsz_time + 4
    for fi in range(n_movie):
        abs_idx = movie_start + fi
        ts_img = Image.new("RGBA", (pw_total, ts_row_h), (0, 0, 0, 0))
        ts_draw = ImageDraw.Draw(ts_img)
        ts_draw.text(
            (pw_total // 2, 2),
            f"t = {abs_idx / fs:.2f} s    frame {abs_idx}/{n_frames_total}",
            fill=(255, 255, 0, 255), anchor="mt", font=font_time,
        )
        ts_arr = np.array(ts_img)
        ts_mask = ts_arr[:, :, 3] > 0
        rgb_frames[fi, 2 : 2 + ts_row_h, :, :][ts_mask] = ts_arr[:, :, :3][ts_mask]
        if fi % 500 == 0:
            print(f"    frame {fi}/{n_movie}")

    return rgb_frames


def encode_video(frames, path, fps, codecs):
    """Encode RGB frames to an MP4, trying each codec in order."""
    print(f"  encoding {len(frames)} frames ({frames.shape[2]}x{frames.shape[1]}) …")
    for codec in codecs:
        try:
            iio.imwrite(path, frames, fps=fps, codec=codec, plugin="pyav")
            print(f"  saved → {path}  (codec={codec})")
            return
        except Exception as e:
            print(f"  codec {codec} failed: {e}")
    print("  WARNING: all codecs failed; saving as uncompressed AVI")
    avi_path = path.with_suffix(".avi")
    iio.imwrite(avi_path, frames, fps=fps, plugin="pyav")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 1. Load TIFF data ───────────────────────────────────────────────────
    print(f"Opening {TIFF_PATH.name} …")
    tif = tifffile.memmap(str(TIFF_PATH), mode="r")
    n_frames, ny, nx = tif.shape[:3]
    print(f"  frames={n_frames}  size={ny}x{nx}  fs={FS} Hz")

    ds = DOWNSAMPLE_SPATIAL
    if ds > 1:
        ny_ds, nx_ds = ny // ds, nx // ds
        print(f"  spatial bin {ds}x → {ny_ds}x{nx_ds}")
    else:
        ny_ds, nx_ds = ny, nx

    if PRELOAD_TO_RAM:
        data_gb = n_frames * ny * nx * tif.dtype.itemsize / 1e9
        print(f"  preloading {data_gb:.1f} GB into RAM …")
        tif = np.array(tif)
        print(f"  done — dtype={tif.dtype}, contiguous={tif.flags['C_CONTIGUOUS']}")

    # Auto-scale GPU tile size based on available VRAM
    tile_sz = auto_tile_size(n_frames, ny_ds, nx_ds, TILE_SIZE)
    if tile_sz > TILE_SIZE:
        free_gpu, _ = cp.cuda.Device().mem_info
        print(f"  GPU VRAM: {free_gpu / 1e9:.1f} GB free → tile size {TILE_SIZE} → {tile_sz}")
    else:
        tile_sz = TILE_SIZE

    # ── 1b. Build quiescence mask from locomotion mask table ──────────
    m = re.search(r'sub-([A-Za-z0-9]+).*?(ses-\d+).*?(task-[A-Za-z0-9]+)', TIFF_PATH.stem)
    if m is None:
        raise ValueError(f"Cannot parse subject/session/task from {TIFF_PATH.name}")
    subject, session, task = m.group(1), m.group(2), m.group(3)
    print(f"  subject={subject}  session={session}  task={task}")

    mask_df = pd.read_csv(LOCO_MASK_PATH)
    sess_mask = mask_df[
        (mask_df["Subject"] == subject)
        & (mask_df["Session"] == session)
        & (mask_df["Task"] == task)
    ]
    quiet_rows = sess_mask[sess_mask["state"] == "quiescent"].copy()
    quiet_rows = quiet_rows[quiet_rows["duration_s"] >= MIN_QUIESCENCE_S]
    quiet_rows = quiet_rows.sort_values("onset_frame").reset_index(drop=True)

    quiescence_mask = np.zeros(n_frames, dtype=bool)
    for _, row in quiet_rows.iterrows():
        f0 = max(0, int(row["onset_frame"]))
        f1 = min(n_frames, int(row["offset_frame"]) + 1)  # offset_frame is inclusive
        quiescence_mask[f0:f1] = True
    n_quiescent = int(quiescence_mask.sum())
    n_running = int((sess_mask["state"] == "running").sum())
    print(f"  running epochs: {n_running}  "
          f"quiescent epochs (>={MIN_QUIESCENCE_S}s): {len(quiet_rows)}  "
          f"quiescent frames: {n_quiescent}/{n_frames} "
          f"({100 * n_quiescent / n_frames:.1f}%)")
    for qi, (_, qr) in enumerate(quiet_rows.iterrows()):
        print(f"    epoch {qi}: {qr['onset_t']:.2f} – {qr['offset_t']:.2f} s  "
              f"({qr['duration_s']:.1f} s, frames {int(qr['onset_frame'])}–{int(qr['offset_frame'])})")
    quiescence_mask_gpu = cp.asarray(quiescence_mask)

    # ── 2. Cortex mask ──────────────────────────────────────────────────────
    print("Building cortex mask …")
    n_mean = min(N_MEAN_FRAMES, n_frames)
    mean_idx = np.linspace(0, n_frames - 1, n_mean, dtype=int)
    mean_frame = np.zeros((ny, nx), dtype=np.float64)
    for i in mean_idx:
        mean_frame += tif[i].astype(np.float64)
    mean_frame /= n_mean

    if ds > 1:
        mean_frame = (
            mean_frame[: ny_ds * ds, : nx_ds * ds]
            .reshape(ny_ds, ds, nx_ds, ds)
            .mean(axis=(1, 3))
        )

    cortex_mask = build_cortex_mask(mean_frame, CORTEX_THRESH)
    n_cortex = int(cortex_mask.sum())
    print(f"  cortex pixels: {n_cortex} / {ny_ds * nx_ds}")

    # ── 3. Build frequency-domain kernel ────────────────────────────────────
    sos = butter(FILTER_ORDER, BAND, btype="bandpass", fs=FS, output="sos")
    bp_hilbert_kernel = build_bp_hilbert_kernel(n_frames, sos, FS)
    print("  bandpass+Hilbert kernel built on GPU")

    # ── 4. Pass 1 — Global analytic signal ──────────────────────────────────
    print("Pass 1: computing global analytic signal …")
    global_analytic = cp.zeros(n_frames, dtype=cp.complex128)
    tile_list = [
        (r0, r1, c0, c1)
        for r0, r1 in tile_ranges(ny_ds, tile_sz)
        for c0, c1 in tile_ranges(nx_ds, tile_sz)
    ]
    print(f"  {len(tile_list)} tiles, tile_size={tile_sz}")

    lookahead = NUM_IO_WORKERS + 1
    for i, (r0, r1, c0, c1), raw in prefetch_tiles(
        tile_list, tif, ds, n_frames, NUM_IO_WORKERS, lookahead
    ):
        atile = gpu_fft_filter(raw, bp_hilbert_kernel)
        del raw
        mask_tile_gpu = cp.asarray(cortex_mask[r0:r1, c0:c1])
        global_analytic += (atile * mask_tile_gpu[None, :, :]).sum(axis=(1, 2))
        del atile, mask_tile_gpu
        if (i + 1) % max(1, len(tile_list) // 10) == 0 or i == len(tile_list) - 1:
            print(f"  tile {i + 1}/{len(tile_list)}")

    global_analytic /= n_cortex

    # Zero out locomotion frames so the reference is quiescence-only
    global_analytic[~quiescence_mask_gpu] = 0.0
    ref = cp.conj(global_analytic).astype(cp.complex64)

    global_amp = cp.abs(global_analytic)
    # Amplitude gate computed only over quiescent frames
    quiescent_amps = global_amp[quiescence_mask_gpu]
    amp_thresh_val = cp.percentile(quiescent_amps, AMP_PERCENTILE)
    global_good = (global_amp > amp_thresh_val) & quiescence_mask_gpu
    n_good = int(global_good.sum())
    print(f"  global amplitude gate: {n_good}/{n_quiescent} quiescent frames pass")

    # ── 5. Pass 2 — Amplitude-gated phase summaries ────────────────────────
    print("Pass 2: computing amplitude-gated phase summaries …")

    sum_vec = cp.zeros((ny_ds, nx_ds), dtype=cp.complex128)
    sum_w = cp.zeros((ny_ds, nx_ds), dtype=cp.float64)

    # Find the largest contiguous quiescence period for the movie
    _changes = np.diff(quiescence_mask.astype(np.int8), prepend=0, append=0)
    _starts = np.where(_changes == 1)[0]
    _stops = np.where(_changes == -1)[0]
    _lengths = _stops - _starts
    _best = int(np.argmax(_lengths))
    quiet_start, quiet_stop = int(_starts[_best]), int(_stops[_best])
    # Clip to MOVIE_DUR if the quiescence period is longer
    max_movie_frames = int(MOVIE_DUR * FS)
    if (quiet_stop - quiet_start) > max_movie_frames:
        mid_q = (quiet_start + quiet_stop) // 2
        quiet_start = max(0, mid_q - max_movie_frames // 2)
        quiet_stop = min(n_frames, quiet_start + max_movie_frames)
    movie_start = quiet_start
    movie_end = quiet_stop
    print(f"  movie window: frames {movie_start}–{movie_end} "
          f"({(movie_end - movie_start) / FS:.1f} s, largest quiescence)")

    if SNAP_START_S is not None:
        snap_origin = int(SNAP_START_S * FS)
    else:
        snap_origin = movie_start
    snap_indices = snap_origin + np.arange(N_SNAP)
    snap_indices = snap_indices[snap_indices < n_frames]
    snap_frames = np.full((len(snap_indices), ny_ds, nx_ds), np.nan, dtype=np.float32)
    snap_times = snap_indices / FS

    n_movie = movie_end - movie_start
    movie_t0 = movie_start / FS
    movie_buf = np.full((n_movie, ny_ds, nx_ds), np.nan, dtype=np.float32)
    movie_bp_buf = np.full((n_movie, ny_ds, nx_ds), np.nan, dtype=np.float32)
    movie_absphase_buf = np.full((n_movie, ny_ds, nx_ds), np.nan, dtype=np.float32)
    movie_dff_buf = np.full((n_movie, ny_ds, nx_ds), np.nan, dtype=np.float32)

    n_tiles = 0
    for idx, (r0, r1, c0, c1), raw in prefetch_tiles(
        tile_list, tif, ds, n_frames, NUM_IO_WORKERS, lookahead
    ):
        n_tiles += 1
        atile = gpu_fft_filter(raw, bp_hilbert_kernel)
        del raw
        mask_tile_gpu = cp.asarray(cortex_mask[r0:r1, c0:c1])

        # Relative phase via conjugate product
        rel = atile * ref[:, None, None]
        amp = cp.abs(atile)
        del atile

        # Amplitude gate: local per-pixel + global timewise + cortex mask
        amp_thresh = cp.percentile(amp, AMP_PERCENTILE, axis=0)
        gate = (
            (amp > amp_thresh)
            & mask_tile_gpu[None, :, :]
            & global_good[:, None, None]
        )
        del amp, amp_thresh

        abs_rel = cp.maximum(cp.abs(rel), 1e-30)
        unit_rel = rel / abs_rel
        del abs_rel

        sum_vec[r0:r1, c0:c1] += (gate * unit_rel).sum(axis=0)
        sum_w[r0:r1, c0:c1] += gate.astype(cp.float64).sum(axis=0)
        del unit_rel, gate

        # Transfer relative phase to CPU for snapshots & movie
        rel_phase = cp.angle(rel).get()
        rel_movie_gpu = rel[movie_start:movie_end]
        ref_movie_gpu = ref[movie_start:movie_end, None, None]
        analytic_movie_gpu = rel_movie_gpu / ref_movie_gpu
        bp_real = cp.real(analytic_movie_gpu).get()
        abs_phase = cp.angle(analytic_movie_gpu).get()
        del analytic_movie_gpu, rel_movie_gpu, ref_movie_gpu
        del rel, mask_tile_gpu

        mask_tile_cpu = cortex_mask[r0:r1, c0:c1]
        for i, si in enumerate(snap_indices):
            vals = rel_phase[si].copy()
            vals[~mask_tile_cpu] = np.nan
            snap_frames[i, r0:r1, c0:c1] = vals

        movie_tile = rel_phase[movie_start:movie_end]
        movie_tile[:, ~mask_tile_cpu] = np.nan
        movie_buf[:, r0:r1, c0:c1] = movie_tile

        bp_real[:, ~mask_tile_cpu] = np.nan
        movie_bp_buf[:, r0:r1, c0:c1] = bp_real

        abs_phase[:, ~mask_tile_cpu] = np.nan
        movie_absphase_buf[:, r0:r1, c0:c1] = abs_phase

        # ΔF/F from raw fluorescence for the movie window
        raw_movie = tif[
            movie_start:movie_end,
            r0 * ds : r1 * ds,
            c0 * ds : c1 * ds,
        ].astype(np.float32)
        if ds > 1:
            th, tw = r1 - r0, c1 - c0
            raw_movie = raw_movie[:, : th * ds, : tw * ds].reshape(
                n_movie, th, ds, tw, ds
            ).mean(axis=(2, 4))
        f0_tile = mean_frame[r0:r1, c0:c1].astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            dff_tile = (raw_movie - f0_tile[None, :, :]) / f0_tile[None, :, :]
        dff_tile[:, ~mask_tile_cpu] = np.nan
        movie_dff_buf[:, r0:r1, c0:c1] = dff_tile
        del raw_movie, f0_tile, dff_tile, rel_phase, movie_tile, bp_real, abs_phase

        if n_tiles % max(1, len(tile_list) // 10) == 0 or idx == len(tile_list) - 1:
            print(f"  tile {n_tiles}/{len(tile_list)}")

    # ── 6. Final summary maps ──────────────────────────────────────────────
    safe_w = cp.where(sum_w > 0, sum_w, 1.0)
    mean_rel_phase = cp.angle(sum_vec / safe_w).get()
    plv = cp.abs(sum_vec / safe_w).get()
    del sum_vec, sum_w, safe_w

    mean_rel_phase[~cortex_mask] = np.nan
    plv[~cortex_mask] = np.nan

    # ── 7. Write 4-panel movie ─────────────────────────────────────────────
    out_dir = TIFF_PATH.parent
    print("Generating 4-panel phase movie (vectorised) …")

    lut_dff = make_colormap_lut(CMAP_DFF)
    lut_bp = make_colormap_lut(CMAP_BP)
    lut_phase = make_colormap_lut(CMAP_PHASE)

    bp_finite = movie_bp_buf[np.isfinite(movie_bp_buf)]
    bp_vlim = np.percentile(np.abs(bp_finite), 99.5)
    del bp_finite
    dff_finite = movie_dff_buf[np.isfinite(movie_dff_buf)]
    dff_vlim = np.percentile(np.abs(dff_finite), 99.5)
    del dff_finite

    print("  applying colourmaps to all frames …")
    dff_rgb = apply_lut(movie_dff_buf, lut_dff, -dff_vlim, dff_vlim)
    dff_rgb[:, ~cortex_mask] = 0
    del movie_dff_buf

    bp_rgb = apply_lut(movie_bp_buf, lut_bp, -bp_vlim, bp_vlim)
    bp_rgb[:, ~cortex_mask] = 0
    del movie_bp_buf

    rel_rgb = apply_lut(movie_buf, lut_phase, -np.pi, np.pi)
    rel_rgb[:, ~cortex_mask] = 0

    abs_rgb = apply_lut(movie_absphase_buf, lut_phase, -np.pi, np.pi)
    abs_rgb[:, ~cortex_mask] = 0
    del movie_absphase_buf

    print("  drawing contours on GPU (all frames vectorised) …")
    draw_contours_gpu(movie_buf, cortex_mask, CONTOUR_LEVELS, CONTOUR_COLORS_RGB, rel_rgb)
    del movie_buf

    print("  compositing panels …")
    panels = np.concatenate([dff_rgb, bp_rgb, rel_rgb, abs_rgb], axis=2)
    del dff_rgb, bp_rgb, rel_rgb, abs_rgb

    render_scale = max(1, int(np.ceil(MIN_PANEL_HEIGHT_PX / ny_ds)))
    if render_scale > 1:
        print(f"  upscaling panels {render_scale}x for crisp text …")
        panels = np.repeat(np.repeat(panels, render_scale, axis=1), render_scale, axis=2)

    panel_titles = ["\u0394F/F", "Bandpassed", "Relative Phase", "Absolute Phase"]
    param_str = (
        f"Band: {BAND[0]}-{BAND[1]} Hz  |  Fs: {FS:.0f} Hz  |  "
        f"Spatial bin: {ds}x  |  Amp gate: P{AMP_PERCENTILE}  |  "
        f"Filter order: {FILTER_ORDER}  |  TIFF: {TIFF_PATH.name}"
    )

    font_title, font_time, font_foot = load_fonts(render_scale)
    print("  compositing frames with text …")
    rgb_frames = compose_movie_frames(
        panels, render_scale, panel_titles, param_str,
        movie_start, n_frames, FS, font_title, font_time, font_foot,
    )
    del panels

    movie_path = out_dir / "phase_movie.mp4"
    encode_video(rgb_frames, movie_path, int(FS), VIDEO_CODECS)
    del rgb_frames

    # ── 8. Save summary data ───────────────────────────────────────────────
    data_path = out_dir / "spatial_phase_data.npz"
    print(f"Saving data → {data_path}")
    np.savez_compressed(
        data_path,
        mean_rel_phase=mean_rel_phase,
        plv=plv,
        cortex_mask=cortex_mask,
        quiescence_mask=quiescence_mask,
        mean_frame=mean_frame,
        snap_frames=snap_frames,
        snap_times=snap_times,
        band=np.array(BAND),
        fs=FS,
        downsample_spatial=ds,
        amp_percentile=AMP_PERCENTILE,
        min_quiescence_s=MIN_QUIESCENCE_S,
        tiff_name=TIFF_PATH.name,
    )

    # ── 9. Summary plots ───────────────────────────────────────────────────
    print("Plotting …")
    dat = np.load(data_path, allow_pickle=True)
    mean_rel_phase = dat["mean_rel_phase"]
    plv = dat["plv"]
    cortex_mask = dat["cortex_mask"]
    snap_frames = dat["snap_frames"]
    snap_times = dat["snap_times"]
    band = dat["band"]
    tiff_name = str(dat["tiff_name"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(mean_rel_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0].contour(cortex_mask, levels=[0.5], colors="w", linewidths=0.5)
    axes[0].set_title("Mean relative phase (pixel − global)")
    plt.colorbar(im0, ax=axes[0], label="phase (rad)")

    im1 = axes[1].imshow(plv, cmap="inferno", vmin=0, vmax=1)
    axes[1].contour(cortex_mask, levels=[0.5], colors="w", linewidths=0.5)
    axes[1].set_title("Phase-locking value (PLV)")
    plt.colorbar(im1, ax=axes[1], label="PLV")

    phase_safe = np.where(np.isnan(mean_rel_phase), 0.0, mean_rel_phase)
    plv_safe = np.where(np.isnan(plv), 0.0, plv)
    hsv = np.zeros((*phase_safe.shape, 3), dtype=np.float64)
    hsv[..., 0] = (phase_safe + np.pi) / (2 * np.pi)
    hsv[..., 1] = plv_safe
    hsv[..., 2] = plv_safe
    rgb = hsv_to_rgb(hsv)
    axes[2].imshow(rgb)
    axes[2].contour(cortex_mask, levels=[0.5], colors="w", linewidths=0.5)
    axes[2].set_title("HSV composite (hue=phase, sat/val=PLV)")

    for ax in axes:
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

    fig.suptitle(
        f"Spatial phase structure — {band[0]}–{band[1]} Hz band\n{tiff_name}",
        fontsize=12,
    )
    fig.tight_layout()
    plt.savefig(out_dir / "spatial_phase_structure.png", dpi=200)
    plt.show()

    # ── 10. Snapshot grid ──────────────────────────────────────────────────
    n_snaps = len(snap_times)
    n_cols = 10
    n_rows = (n_snaps + n_cols - 1) // n_cols
    cell = 1.2
    fig2, axes2 = plt.subplots(
        n_rows, n_cols, figsize=(cell * n_cols, cell * n_rows),
        gridspec_kw={"wspace": 0.02, "hspace": 0.25},
    )
    axes2_flat = axes2.ravel()
    t0_snap = snap_times[0]
    for idx in range(len(axes2_flat)):
        ax = axes2_flat[idx]
        if idx < n_snaps:
            ax.imshow(snap_frames[idx], cmap="twilight", vmin=-np.pi, vmax=np.pi)
            with np.errstate(invalid="ignore"):
                ax.contour(
                    snap_frames[idx],
                    levels=[-np.pi / 2, 0, np.pi / 2],
                    colors=["cyan", "white", "yellow"],
                    linewidths=0.3,
                )
            dt_ms = (snap_times[idx] - t0_snap) * 1000
            ax.set_title(f"{dt_ms:.0f} ms", fontsize=5, pad=1)
        else:
            ax.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig2.suptitle(
        f"Relative phase snapshots — 20 ms steps (t₀ = {t0_snap:.2f} s)",
        fontsize=10,
    )
    plt.savefig(out_dir / "phase_snapshots.png", dpi=250, bbox_inches="tight")
    plt.show()

    print("Done.")
