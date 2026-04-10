"""
mesofield_sync_analysis.py
==========================
Synchronization analysis for mesofield dataqueue CSVs and camera frame
metadata JSONs. Characterizes timing relationships between Dhyana (sCMOS,
50 Hz), ThorCam (pupil, 20 Hz), and treadmill (sparse quadrature encoder).

Produces an 8-panel matplotlib figure covering:
  A. Dhyana ElapsedTime IFI distribution
  B. ThorCam ElapsedTime IFI distribution (integer-ms quantized)
  C. Dhyana–ThorCam cross-camera temporal residual
  D. Camera buffer backpressure over session
  E. Treadmill speed profile with locomotion bouts
  F. Treadmill encoder HW clock drift vs system clock
  G. Treadmill phase uniformity relative to Dhyana frames
  H. Treadmill HW IFI distribution (within-bout)

Usage:
    python mesofield_sync_analysis.py

Expects paired dataqueue CSVs and metadata JSONs in the same directory,
or pass files via constants below.
"""

import json
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
SUBJECT_DIR = Path(r"D:\Projects\RO1_ETOH\sub-GS27")
DATAQUEUE_GLOB = "*dataqueue*.csv"
MESO_JSON_GLOB = "*mesoscope*frame_metadata.json"
PUPIL_JSON_GLOB = "*pupil*frame_metadata.json"
OUTPUT_PATH = Path("mesofield_sync_analysis.png")
OUTPUT_DPI = 200
DHYANA_NOMINAL_HZ = 50
THORCAM_NOMINAL_HZ = 20
TREADMILL_BOUT_GAP_MS = 200
SPEED_BIN_WIDTH_S = 5
CROSS_CAMERA_SAMPLE_N = 500
EXCLUDE_SESSIONS = {"ses-06"}   # session labels to skip

# ---------------------------------------------------------------------------
# Style — light mode
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#ffffff",
    "surface": "#f7f8fa",
    "border": "#d0d3dc",
    "text": "#2e3038",
    "text_bright": "#111318",
    "text_dim": "#7a7f92",
    "dhyana": "#2563eb",
    "thorcam": "#16a34a",
    "treadmill": "#7c3aed",
    "accent_red": "#dc2626",
    "accent_amber": "#d97706",
    "grid": "#e5e7eb",
}

SESSION_COLORS = ["#2563eb", "#16a34a", "#d97706"]


def setup_style():
    mpl.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["surface"],
        "axes.edgecolor": COLORS["border"],
        "axes.labelcolor": COLORS["text"],
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.6,
        "grid.linewidth": 0.5,
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text_dim"],
        "ytick.color": COLORS["text_dim"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.fontsize": 7,
        "legend.facecolor": COLORS["surface"],
        "legend.edgecolor": COLORS["border"],
        "legend.framealpha": 0.9,
        "font.family": "sans-serif",
        "font.size": 8,
        "savefig.facecolor": COLORS["bg"],
        "savefig.edgecolor": COLORS["bg"],
        "savefig.dpi": OUTPUT_DPI,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def discover_sessions(subject_dir):
    """Glob for dataqueue CSVs under *subject_dir* and pair with metadata JSONs.

    Handles layouts where dataqueue lives in ``ses-XX/beh/`` while camera
    metadata JSONs live in ``ses-XX/func/``.  The search for companion JSONs
    starts from the *session directory* (the deepest ancestor whose name
    matches ``ses-*``), falling back to the dataqueue's parent.

    Returns dict  {session_label: {"dataqueue": Path, "meso_json": Path, "pupil_json": Path}}.
    """
    subject_dir = Path(subject_dir)
    dq_files = sorted(subject_dir.rglob(DATAQUEUE_GLOB))
    if not dq_files:
        raise FileNotFoundError(f"No dataqueue CSVs matched '{DATAQUEUE_GLOB}' under {subject_dir}")

    sessions = {}
    for dq_path in dq_files:
        # Derive session label
        ses_match = re.search(r"(ses-\d+)", str(dq_path))
        label = ses_match.group(1) if ses_match else dq_path.parent.name

        # Find the session-level directory (e.g. sub-GS27/ses-09/)
        ses_dir = dq_path.parent
        for p in dq_path.parents:
            if re.match(r"ses-\d+", p.name):
                ses_dir = p
                break

        # Skip excluded sessions
        if label in EXCLUDE_SESSIONS:
            print(f"  [excluded] {label}")
            continue

        # Search recursively under the session directory
        meso_hits = sorted(ses_dir.rglob(MESO_JSON_GLOB))
        pupil_hits = sorted(ses_dir.rglob(PUPIL_JSON_GLOB))
        if not meso_hits or not pupil_hits:
            print(f"  [skip] {label}: missing metadata JSON under {ses_dir}")
            continue

        sessions[label] = {
            "dataqueue": dq_path,
            "meso_json": meso_hits[0],
            "pupil_json": pupil_hits[0],
        }
    return sessions


def load_json_meta(filepath):
    """Load frame metadata JSON -> list of frame dicts."""
    with open(filepath) as f:
        data = json.load(f)
    return data["p0"]


def load_dataqueue(filepath):
    """Load dataqueue CSV with parsed timestamps."""
    df = pd.read_csv(filepath, low_memory=False)
    df["device_ts"] = pd.to_datetime(df["device_ts"], format="mixed")
    df["packet_ts"] = pd.to_datetime(df["packet_ts"], format="mixed")
    return df


def extract_camera_arrays(frames):
    """
    Extract timing arrays from camera JSON metadata.

    Returns dict with:
        elapsed_ms   : camera ElapsedTime (camera's own clock)
        runner_ms    : runner_time_ms (reconstructed = elapsed + constant)
        device_ts    : TimeReceivedByCore as datetime64
        buffer_remaining : images_remaining_in_buffer
        image_number : frame index
        runner_offset_ms : constant offset (runner - elapsed)
    """
    elapsed = np.array([float(f["camera_metadata"]["ElapsedTime-ms"]) for f in frames])
    runner = np.array([f["runner_time_ms"] for f in frames])
    device_ts = pd.to_datetime([f["camera_metadata"]["TimeReceivedByCore"] for f in frames])
    buf = np.array([f["images_remaining_in_buffer"] for f in frames])
    img_num = np.array([int(f["camera_metadata"]["ImageNumber"]) for f in frames])
    offset = runner[0] - elapsed[0]

    return {
        "elapsed_ms": elapsed,
        "runner_ms": runner,
        "device_ts": device_ts,
        "buffer_remaining": buf,
        "image_number": img_num,
        "runner_offset_ms": offset,
        "n_frames": len(frames),
        "camera": frames[0]["camera_device"],
        "exposure_ms": frames[0]["exposure_ms"],
    }


def extract_treadmill(df):
    """
    Parse treadmill stream from dataqueue.

    Returns dict with HW timestamps (us), speeds, distances, system timestamps.
    """
    tread = df[df["device_id"] == "treadmill"].copy()
    if len(tread) == 0:
        return None

    hw_ts, speeds, distances = [], [], []
    for _, row in tread.iterrows():
        payload = str(row["payload"])
        m_ts = re.search(r"timestamp=(\d+)", payload)
        m_sp = re.search(r"speed=([\d.]+)", payload)
        m_d = re.search(r"distance=([\d.]+)", payload)
        hw_ts.append(int(m_ts.group(1)) if m_ts else np.nan)
        speeds.append(float(m_sp.group(1)) if m_sp else np.nan)
        distances.append(float(m_d.group(1)) if m_d else np.nan)

    return {
        "hw_ts_us": np.array(hw_ts, dtype=float),
        "speed_mm_s": np.array(speeds),
        "distance_mm": np.array(distances),
        "device_ts": tread["device_ts"].values,
        "n_packets": len(tread),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_session(ses_label, files):
    """Run full sync analysis for one session. Returns analysis dict."""
    # Load data
    dq = load_dataqueue(files["dataqueue"])
    meso_frames = load_json_meta(files["meso_json"])
    pupil_frames = load_json_meta(files["pupil_json"])

    meso = extract_camera_arrays(meso_frames)
    pupil = extract_camera_arrays(pupil_frames)
    tread = extract_treadmill(dq)

    # --- Camera IFI ---
    meso_ifi = np.diff(meso["elapsed_ms"])
    pupil_ifi = np.diff(pupil["elapsed_ms"])
    meso_runner_ifi = np.diff(meso["runner_ms"])
    pupil_runner_ifi = np.diff(pupil["runner_ms"])

    # --- Cross-camera alignment ---
    # For each ThorCam frame, find nearest Dhyana frame by device_ts
    meso_ts_ns = meso["device_ts"].values.astype(np.int64)
    pupil_ts_ns = pupil["device_ts"].values.astype(np.int64)

    # Full residual array via searchsorted — O(N log M), all ThorCam frames
    ins = np.searchsorted(meso_ts_ns, pupil_ts_ns)
    ins_c = np.clip(ins, 1, len(meso_ts_ns) - 1)
    left_dist = pupil_ts_ns - meso_ts_ns[ins_c - 1]
    right_dist = meso_ts_ns[ins_c] - pupil_ts_ns
    nearest_idx_full = np.where(left_dist <= right_dist, ins_c - 1, ins_c)
    cross_residuals_full_ms = (pupil_ts_ns - meso_ts_ns[nearest_idx_full]) / 1e6

    # Rolling mean (window=75 ThorCam frames ≈ 3.75 s)
    cross_rolling_mean_ms = (
        pd.Series(cross_residuals_full_ms)
        .rolling(window=75, center=True, min_periods=1)
        .mean()
        .values
    )
    cross_elapsed_s_full = (pupil["elapsed_ms"] - pupil["elapsed_ms"][0]) / 1000

    # Subsampled scatter for display
    sample_step = max(1, len(pupil_ts_ns) // CROSS_CAMERA_SAMPLE_N)
    sample_idx = np.arange(0, len(pupil_ts_ns), sample_step)
    cross_residuals_ms = cross_residuals_full_ms[sample_idx]
    cross_meso_idx = nearest_idx_full[sample_idx]
    cross_elapsed_s = cross_elapsed_s_full[sample_idx]

    # --- ThorCam cumulative drift from nominal period ---
    pupil_nominal_ms = 1000 / THORCAM_NOMINAL_HZ
    pupil_expected = np.arange(len(pupil["elapsed_ms"])) * pupil_nominal_ms
    pupil_cum_drift = (pupil["elapsed_ms"] - pupil["elapsed_ms"][0]) - pupil_expected

    # --- Treadmill analysis ---
    tread_analysis = None
    if tread is not None and tread["n_packets"] > 10:
        hw_elapsed_s = (tread["hw_ts_us"] - tread["hw_ts_us"][0]) / 1e6
        sys_elapsed_s = (tread["device_ts"] - tread["device_ts"][0]) / np.timedelta64(1, "s")

        # Clock drift
        clock_diff_ms = (sys_elapsed_s - hw_elapsed_s) * 1000
        drift_total_ms = clock_diff_ms[-1] - clock_diff_ms[0]
        drift_ppm = drift_total_ms / (sys_elapsed_s[-1] * 1000) * 1e6 if sys_elapsed_s[-1] > 0 else 0

        # HW IFI
        hw_ifi_ms = np.diff(tread["hw_ts_us"]) / 1e3  # us to ms

        # Bout detection
        bout_gaps = np.where(hw_ifi_ms > TREADMILL_BOUT_GAP_MS)[0]
        n_bouts = len(bout_gaps) + 1
        active_ifi = hw_ifi_ms[hw_ifi_ms < TREADMILL_BOUT_GAP_MS]

        # Phase relative to Dhyana frames
        meso_sys_elapsed = (meso["device_ts"].values - meso["device_ts"].values[0]) / np.timedelta64(1, "s")
        tread_sys_from_meso0 = (tread["device_ts"] - meso["device_ts"].values[0]) / np.timedelta64(1, "s")
        tread_frame_idx = np.interp(tread_sys_from_meso0, meso_sys_elapsed, np.arange(len(meso_sys_elapsed)))
        tread_phase = tread_frame_idx % 1

        # Speed binned over time
        time_bins = np.arange(0, hw_elapsed_s[-1] + SPEED_BIN_WIDTH_S, SPEED_BIN_WIDTH_S)
        speed_binned = np.zeros(len(time_bins) - 1)
        for ib in range(len(time_bins) - 1):
            mask = (hw_elapsed_s >= time_bins[ib]) & (hw_elapsed_s < time_bins[ib + 1])
            if mask.any():
                speed_binned[ib] = np.nanmean(tread["speed_mm_s"][mask])
        speed_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

        tread_analysis = {
            "hw_elapsed_s": hw_elapsed_s,
            "sys_elapsed_s": sys_elapsed_s,
            "clock_diff_ms": clock_diff_ms,
            "drift_total_ms": drift_total_ms,
            "drift_ppm": drift_ppm,
            "hw_ifi_ms": hw_ifi_ms,
            "active_ifi_ms": active_ifi,
            "n_bouts": n_bouts,
            "phase": tread_phase,
            "speed_bin_centers": speed_bin_centers,
            "speed_binned": speed_binned,
            "speeds": tread["speed_mm_s"],
        }

    return {
        "label": ses_label,
        "meso": meso,
        "pupil": pupil,
        "meso_ifi": meso_ifi,
        "pupil_ifi": pupil_ifi,
        "meso_runner_ifi": meso_runner_ifi,
        "pupil_runner_ifi": pupil_runner_ifi,
        "cross_residuals_ms": cross_residuals_ms,
        "cross_elapsed_s": cross_elapsed_s,
        "cross_meso_idx": cross_meso_idx,
        "cross_residuals_full_ms": cross_residuals_full_ms,
        "cross_rolling_mean_ms": cross_rolling_mean_ms,
        "cross_elapsed_s_full": cross_elapsed_s_full,
        "pupil_cum_drift_ms": pupil_cum_drift,
        "tread": tread_analysis,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _annotate(ax, text, loc="bl"):
    """Add a dim italic annotation to a panel corner."""
    x, y, ha, va = {
        "bl": (0.02, 0.05, "left", "bottom"),
        "br": (0.98, 0.05, "right", "bottom"),
        "tl": (0.02, 0.95, "left", "top"),
        "tr": (0.98, 0.95, "right", "top"),
    }[loc]
    ax.text(x, y, text, fontsize=6.5, color=COLORS["text_dim"],
            transform=ax.transAxes, va=va, ha=ha, fontstyle="italic",
            bbox=dict(facecolor=COLORS["surface"], edgecolor="none", alpha=0.85, pad=3))


def plot_sync(sessions, output_path="mesofield_sync_analysis.png"):
    """Build the 8-panel figure."""
    setup_style()
    n = len(sessions)
    sc = SESSION_COLORS[:n] if n <= 3 else [
        mpl.colors.to_hex(plt.cm.viridis(i / max(1, n - 1))) for i in range(n)
    ]

    fig = plt.figure(figsize=(14, 20))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.30,
                           left=0.07, right=0.96, top=0.93, bottom=0.04)

    # Suptitle
    labels = ", ".join(s["label"] for s in sessions)
    fig.text(0.07, 0.975, "Mesofield Dataqueue \u2014 Synchronization Analysis",
             fontsize=15, fontweight="bold", color=COLORS["text_bright"], va="top")
    fig.text(0.07, 0.955,
             f"{SUBJECT_DIR.name}  \u00b7  {labels}  \u00b7  Dhyana {DHYANA_NOMINAL_HZ} Hz (ElapsedTime) "
             f"\u00b7 ThorCam {THORCAM_NOMINAL_HZ} Hz (ElapsedTime) \u00b7 Treadmill (HW \u00b5s clock)",
             fontsize=7.5, color=COLORS["text_dim"], va="top")

    # ==========================================================
    # A: Dhyana IFI — ElapsedTime (foreground) + runner_time (underlay)
    # ==========================================================
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A   Dhyana IFI \u2014 ElapsedTime / runner_time (50 Hz \u2192 20 ms)", loc="left")
    bins_meso = np.arange(18.5, 22.0, 0.02)
    for i, s in enumerate(sessions):
        ax.hist(s["meso_runner_ifi"], bins=bins_meso,
                alpha=0.2, color=sc[i], edgecolor="none", zorder=2)
        ax.hist(s["meso_ifi"], bins=bins_meso,
                alpha=0.55, color=sc[i], edgecolor="none", zorder=3,
                label=f'{s["label"]} (\u03c3={s["meso_ifi"].std():.3f} ms)')
    ax.axvline(20.0, color=COLORS["accent_red"], ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("IFI (ms)")
    ax.set_ylabel("count")
    ax.legend(loc="upper right")
    _annotate(ax, f'mean bias: +{sessions[0]["meso_ifi"].mean() - 20:.3f} ms\n'
                  f'faint = runner_time_ms  \u00b7  solid = ElapsedTime-ms')

    # ==========================================================
    # B: ThorCam IFI — ElapsedTime (foreground) + runner_time (underlay)
    # ==========================================================
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("B   ThorCam IFI \u2014 ElapsedTime / runner_time (20 Hz \u2192 50 ms)", loc="left")
    bins_pupil = np.arange(47.5, 53.5, 0.1)
    for i, s in enumerate(sessions):
        ax.hist(s["pupil_runner_ifi"], bins=bins_pupil,
                alpha=0.2, color=sc[i], edgecolor="none", zorder=2)
        ax.hist(s["pupil_ifi"], bins=bins_pupil,
                alpha=0.55, color=sc[i], edgecolor="none", zorder=3,
                label=f'{s["label"]}')
    ax.axvline(50.0, color=COLORS["accent_red"], ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("IFI (ms)")
    ax.set_ylabel("count")
    ax.set_xlim(47.5, 53.5)
    ax.legend(loc="upper right")
    _annotate(ax, "faint = runner_time_ms (continuous float)\n"
                  "solid = ElapsedTime-ms (integer-quantized)\n"
                  "49/50/51 ms spikes = SDK rounding of ~49.975 ms")

    # ==========================================================
    # C: Cross-camera residual — rolling mean (trend detection)
    # ==========================================================
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("C   ThorCam\u2013Dhyana residual \u2014 rolling mean (75 frames \u2248 3.75 s)", loc="left")
    for i, s in enumerate(sessions):
        ax.plot(s["cross_elapsed_s_full"], s["cross_rolling_mean_ms"],
                color=sc[i], lw=0.9, alpha=0.85, label=s["label"])
    ax.axhline(0, color=COLORS["accent_red"], ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("time into session (s)")
    ax.set_ylabel("rolling mean residual (ms)")
    ax.set_ylim(-2, 2)
    ax.legend(loc="upper right")
    _annotate(ax, "trends reveal clock skew, CPU throttling, thermal drift\n"
                  "flat \u2248 0 \u2192 no systematic offset between cameras")

    # ==========================================================
    # D: Cross-camera residual — per-frame scatter
    # ==========================================================
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("D   ThorCam\u2013Dhyana residual \u2014 per-frame scatter (device_ts)", loc="left")
    for i, s in enumerate(sessions):
        ax.scatter(s["cross_elapsed_s"], s["cross_residuals_ms"],
                   s=1.5, alpha=0.3, color=sc[i], label=s["label"], zorder=3)
    ax.axhline(0, color=COLORS["accent_red"], ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("time into session (s)")
    ax.set_ylabel("ThorCam \u2212 nearest Dhyana (ms)")
    ax.set_ylim(-12, 12)
    ax.legend(loc="upper right")
    _annotate(ax, "bounded by \u00b1\u00bd Dhyana period (\u00b110 ms)\n"
                  "uniform scatter \u2192 asynchronous independent clocks")

    # ==========================================================
    # E: Treadmill speed profile
    # ==========================================================
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title("E   Treadmill speed (5s bins) \u2014 locomotion bouts", loc="left")
    for i, s in enumerate(sessions):
        t = s["tread"]
        if t is None:
            continue
        ax.fill_between(t["speed_bin_centers"], t["speed_binned"],
                        alpha=0.3, color=sc[i], step="mid")
        ax.step(t["speed_bin_centers"], t["speed_binned"],
                where="mid", color=sc[i], lw=0.7, alpha=0.85,
                label=f'{s["label"]} ({t["n_bouts"]} bouts, '
                      f'{(t["speeds"] > 0).sum()} active pkts)')
    ax.set_xlabel("time into session (s)")
    ax.set_ylabel("speed (mm/s)")
    ax.legend(loc="upper right")

    # ==========================================================
    # F: Treadmill HW clock drift
    # ==========================================================
    ax = fig.add_subplot(gs[2, 1])
    ax.set_title("F   Treadmill HW clock drift (system \u2212 encoder \u00b5s clock)", loc="left")
    for i, s in enumerate(sessions):
        t = s["tread"]
        if t is None:
            continue
        # Subsample for plotting
        step = max(1, len(t["hw_elapsed_s"]) // 2000)
        ax.plot(t["hw_elapsed_s"][::step], t["clock_diff_ms"][::step],
                color=sc[i], lw=0.8, alpha=0.8,
                label=f'{s["label"]} ({t["drift_ppm"]:.0f} ppm, '
                      f'{t["drift_total_ms"]:.1f} ms total)')
    ax.set_xlabel("time into session (s)")
    ax.set_ylabel("system \u2212 HW clock (ms)")
    ax.legend(loc="upper right")
    _annotate(ax, "20\u201355 ppm drift is typical for independent\n"
                  "crystal oscillators (Teensy vs PC)", loc="br")

    # ==========================================================
    # G: Treadmill phase relative to Dhyana frames
    # ==========================================================
    ax = fig.add_subplot(gs[3, 0])
    ax.set_title("G   Treadmill phase within Dhyana frame period", loc="left")
    for i, s in enumerate(sessions):
        t = s["tread"]
        if t is None:
            continue
        ax.hist(t["phase"], bins=50, range=(0, 1),
                alpha=0.4, color=sc[i], edgecolor="none",
                label=s["label"], density=True)
    ax.axhline(1.0, color=COLORS["accent_red"], ls="--", lw=0.8, alpha=0.5,
               label="uniform expectation")
    ax.set_xlabel("phase within Dhyana frame [0, 1)")
    ax.set_ylabel("density")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    _annotate(ax, "flat = treadmill is asynchronous to camera\n"
                  "\u00b5\u22480.50, \u03c3\u22480.29 matches U(0,1) exactly\n"
                  "\u2192 independent clocks, no shared trigger")

    # ==========================================================
    # H: Treadmill within-bout HW IFI
    # ==========================================================
    ax = fig.add_subplot(gs[3, 1])
    ax.set_title("H   Treadmill encoder IFI (within-bout, HW clock)", loc="left")
    for i, s in enumerate(sessions):
        t = s["tread"]
        if t is None:
            continue
        active = t["active_ifi_ms"]
        # Focus on the physiologically meaningful range
        active_filt = active[(active > 1) & (active < 150)]
        ax.hist(active_filt, bins=np.arange(0, 120, 2),
                alpha=0.4, color=sc[i], edgecolor="none",
                label=f'{s["label"]} (med={np.median(active_filt):.1f} ms)')
    ax.set_xlabel("IFI (ms)")
    ax.set_ylabel("count")
    ax.legend(loc="upper right")
    _annotate(ax, "encoder update rate ~20\u201325 ms during movement\n"
                  "rate depends on speed (faster = shorter IFI)", loc="br")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Discovering sessions under {SUBJECT_DIR} ...")
    session_files = discover_sessions(SUBJECT_DIR)
    if not session_files:
        print("No sessions found.")
        return
    print(f"  Found {len(session_files)} session(s): {', '.join(session_files)}")

    print("\nLoading sessions...")
    sessions = []
    for ses_label, files in session_files.items():
        print(f"  \u2192 {ses_label}")
        try:
            result = analyze_session(ses_label, files)
            sessions.append(result)
            m = result["meso"]
            p = result["pupil"]
            t = result["tread"]
            print(f"    Dhyana: {m['n_frames']} frames, "
                  f"IFI={result['meso_ifi'].mean():.3f}\u00b1{result['meso_ifi'].std():.3f} ms")
            print(f"    ThorCam: {p['n_frames']} frames, "
                  f"IFI={result['pupil_ifi'].mean():.3f}\u00b1{result['pupil_ifi'].std():.3f} ms")
            print(f"    runner_time = ElapsedTime + {m['runner_offset_ms']:.2f} ms (Dhyana)")
            print(f"    runner_time = ElapsedTime + {p['runner_offset_ms']:.2f} ms (ThorCam)")
            print(f"    Cross-camera residual: "
                  f"mean={result['cross_residuals_ms'].mean():.2f} ms, "
                  f"std={result['cross_residuals_ms'].std():.2f} ms")
            if t:
                print(f"    Treadmill: {t['n_bouts']} bouts, "
                      f"drift={t['drift_ppm']:.0f} ppm, "
                      f"phase \u03c3={t['phase'].std():.3f} (uniform=0.289)")
        except FileNotFoundError as e:
            print(f"    SKIPPED: {e}")

    if not sessions:
        print("No sessions loaded.")
        return

    print(f"\nGenerating figure \u2192 {OUTPUT_PATH}")
    fig = plot_sync(sessions, str(OUTPUT_PATH))
    fig.savefig(str(OUTPUT_PATH), dpi=OUTPUT_DPI, bbox_inches="tight")
    print("Done.")


if __name__ == "__main__":
    main()