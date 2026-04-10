"""
Habituation-dependent oscillation-behavior coupling.

Core question: Does the oscillation-locomotion relationship drift across the
10-day HFSA period, and if so, how?

Tracks across days (binned into phase = early / middle / late):
    1. Oscillation properties — P(osc | quiescence), onset latency, frequency,
       amplitude, duration.
    2. Timing-vigor coupling — β(osc_latency → locomotion_vigor) per phase.
    3. Arousal modulation — β_interaction(latency × pupil) across phases.
    4. State occupancy — fraction in quiescence vs locomotion, quiescent bout
       duration distribution, fraction of bouts containing oscillation.

Model:
    locomotion_vigor ~ osc_latency * phase * pupil_z + (1 | animal)
    (GEE with exchangeable correlation, clustered by animal)

Outputs:
    Figure 1: Oscillation probability and latency across days (per animal + group).
    Figure 2: β(latency → vigor) per phase, with CI.
    Figure 3: Arousal interaction term across phases.
    Table:   Model comparison — does adding day/phase interactions improve fit?

Usage:
    python Scripts/scriptings/habituation-oscillation-coupling.py
"""
from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import savgol_filter, hilbert
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian, Binomial
from statsmodels.genmod.cov_struct import Exchangeable

from databench import Project, OscillationDetector
from databench.analysis._signal.bouts import _locomotion_bouts
from databench.config import resolve_dataset
from databench.session import SaveableFigure
from databench.plotting import set_theme

set_theme()

# ─── Parameters ───────────────────────────────────────────────────────────

DATASET = resolve_dataset("etoh-hfsa")
TASK = "task-widefield"
ROI_SOURCE = "mesomap"
ROI_NAME = "L_VISp"

# Oscillation detector settings (match sister scripts)
FS = 50.0
BAND = (2.0, 4.0)
ORDER = 4
THRESHOLD = 0.02
THRESHOLD_K = 4.0
MIN_DURATION_OSC = 2.0
MERGE_GAP_OSC = 0.0

# Locomotion bout settings
MIN_SPEED_CMS = 0.5
MIN_LOCO_DURATION_S = 2.0
MERGE_LOCO_GAP_S = 0.5

# Quiescent bout minimum
MIN_QUIESCENT_S = 1.0

# Pupil
PUPIL_SMOOTH_S = 0.5
PUPIL_WINDOW_S = 5.0

# Phase bins
PHASE_MAP = {
    1: "early", 2: "early", 3: "early",
    4: "middle", 5: "middle", 6: "middle", 7: "middle",
    8: "late", 9: "late", 10: "late",
}
PHASE_ORDER = ["early", "middle", "late"]


# ─── Helpers ──────────────────────────────────────────────────────────────

def day_from_session(session_label: str) -> int:
    """Extract day number from session label, e.g. 'ses-04' → 4."""
    return int(session_label.split("-")[1])


def quiescent_bouts(
    t: np.ndarray,
    loco_bouts: list[Tuple[int, int]],
    min_duration_s: float = 0.0,
) -> list[Tuple[int, int]]:
    """Return index pairs for non-locomotion periods between locomotion bouts."""
    n = len(t)
    if not loco_bouts:
        return [(0, n - 1)]
    quiet: list[Tuple[int, int]] = []
    first_start = loco_bouts[0][0]
    if first_start > 0:
        quiet.append((0, first_start - 1))
    for i in range(len(loco_bouts) - 1):
        gap_start = loco_bouts[i][1] + 1
        gap_end = loco_bouts[i + 1][0] - 1
        if gap_end >= gap_start:
            quiet.append((gap_start, gap_end))
    last_end = loco_bouts[-1][1]
    if last_end < n - 1:
        quiet.append((last_end + 1, n - 1))
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 0.02
    return [
        (s, e) for s, e in quiet
        if float(t[e] - t[s] + dt) >= min_duration_s
    ]


def oscillation_features_in_bout(
    q_start_s: float,
    q_end_s: float,
    osc_events: pd.DataFrame,
) -> dict:
    """Extract oscillation features within a quiescent bout."""
    result = {
        "has_oscillation": False,
        "osc_onset_latency_s": None,
        "osc_total_duration_s": 0.0,
        "osc_mean_peak_env": None,
        "osc_n_events": 0,
    }
    if osc_events.empty:
        return result
    overlapping = osc_events[
        (osc_events["end_s"] > q_start_s) & (osc_events["start_s"] < q_end_s)
    ]
    if overlapping.empty:
        return result
    result["has_oscillation"] = True
    result["osc_n_events"] = len(overlapping)
    earliest_start = max(overlapping["start_s"].min(), q_start_s)
    result["osc_onset_latency_s"] = earliest_start - q_start_s
    total_dur = 0.0
    for _, ev in overlapping.iterrows():
        ov_start = max(q_start_s, ev["start_s"])
        ov_end = min(q_end_s, ev["end_s"])
        if ov_end > ov_start:
            total_dur += ov_end - ov_start
    result["osc_total_duration_s"] = total_dur
    if "peak_env" in overlapping.columns:
        result["osc_mean_peak_env"] = float(overlapping["peak_env"].mean())
    return result


def locomotion_bout_features(
    t: np.ndarray,
    speed_cms: np.ndarray,
    bout_start_idx: int,
    bout_end_idx: int,
) -> dict:
    """Extract features of a single locomotion bout."""
    s, e = int(bout_start_idx), int(bout_end_idx)
    dt_med = float(np.nanmedian(np.diff(t))) if t.size > 1 else 0.02
    bout_t = t[s : e + 1]
    bout_speed = np.abs(speed_cms[s : e + 1])
    duration = float(bout_t[-1] - bout_t[0] + dt_med)
    peak_speed = float(np.nanmax(bout_speed))
    mean_speed = float(np.nanmean(bout_speed))
    peak_idx = int(np.nanargmax(bout_speed))
    latency_to_peak = float(bout_t[peak_idx] - bout_t[0])
    return {
        "loco_duration_s": round(duration, 3),
        "loco_peak_speed_cms": round(peak_speed, 3),
        "loco_mean_speed_cms": round(mean_speed, 3),
        "latency_to_peak_speed_s": round(latency_to_peak, 3),
    }


def _smooth_pupil(raw: np.ndarray, fs: float, window_s: float = 0.5) -> np.ndarray:
    """Savitzky-Golay smooth pupil with NaN handling."""
    if raw is None or len(raw) == 0:
        return raw
    wl = int(round(window_s * fs))
    if wl % 2 == 0:
        wl += 1
    wl = max(3, wl)
    if len(raw) <= wl:
        return raw
    nan_mask = np.isnan(raw)
    if nan_mask.all():
        return raw
    if nan_mask.any():
        filled = raw.copy()
        good = np.flatnonzero(~nan_mask)
        filled[nan_mask] = np.interp(np.flatnonzero(nan_mask), good, raw[good])
        out = savgol_filter(filled, window_length=wl, polyorder=3)
        out[nan_mask] = np.nan
        return out
    return savgol_filter(raw, window_length=wl, polyorder=3)


def _get_pupil_arrays(sess) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Try to load pupil time + diameter from a session."""
    for sig_name in ("pupil_diameter_mm", "diameter_mm"):
        try:
            pup_t = sess.time("pupil", "time_elapsed_s")
            pup_raw = sess.signal("pupil", sig_name)
            return pup_t, pup_raw
        except Exception:
            continue
    return None, None


def pupil_features_for_transition(
    pup_t: np.ndarray,
    pup_smooth: np.ndarray,
    q_start_s: float,
    q_end_s: float,
    osc_onset_s: float | None,
    window_s: float = 5.0,
) -> dict:
    """Extract pupil features around a quiescence → locomotion transition."""
    result: dict = {
        "pupil_at_osc_onset_mm": None,
        "pupil_early_q_mm": None,
        "pupil_late_q_mm": None,
        "pupil_delta_q_mm": None,
        "pupil_mean_q_mm": None,
    }
    if pup_t is None or pup_smooth is None:
        return result
    q_mask = (pup_t >= q_start_s) & (pup_t <= q_end_s)
    if q_mask.sum() < 3:
        return result
    q_pupil = pup_smooth[q_mask]
    result["pupil_mean_q_mm"] = round(float(np.nanmean(q_pupil)), 4)
    if osc_onset_s is not None:
        onset_mask = (pup_t >= osc_onset_s - 0.5) & (pup_t <= osc_onset_s + 0.5)
        if onset_mask.sum() > 0:
            result["pupil_at_osc_onset_mm"] = round(
                float(np.nanmean(pup_smooth[onset_mask])), 4
            )
    early_end = min(q_start_s + window_s, q_end_s)
    early_mask = (pup_t >= q_start_s) & (pup_t <= early_end)
    if early_mask.sum() > 0:
        result["pupil_early_q_mm"] = round(
            float(np.nanmean(pup_smooth[early_mask])), 4
        )
    late_start = max(q_end_s - window_s, q_start_s)
    late_mask = (pup_t >= late_start) & (pup_t <= q_end_s)
    if late_mask.sum() > 0:
        result["pupil_late_q_mm"] = round(
            float(np.nanmean(pup_smooth[late_mask])), 4
        )
    if result["pupil_early_q_mm"] is not None and result["pupil_late_q_mm"] is not None:
        result["pupil_delta_q_mm"] = round(
            result["pupil_late_q_mm"] - result["pupil_early_q_mm"], 4
        )
    return result


def instantaneous_frequency(filtered: np.ndarray, fs: float,
                            start_idx: int, end_idx: int) -> float | None:
    """Mean instantaneous frequency within a burst segment."""
    seg = filtered[start_idx : end_idx + 1]
    if len(seg) < 4:
        return None
    analytic = hilbert(seg)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * fs
    # Clamp to passband to reject edge artefacts
    valid = inst_freq[(inst_freq > 1.0) & (inst_freq < 8.0)]
    if len(valid) == 0:
        return None
    return float(np.nanmean(valid))


def fit_gee(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    group_col: str = "subj_id",
    family=None,
    min_n: int = 10,
) -> dict | None:
    """Fit a GEE model. Returns coef/se/p/ci dict or None."""
    if family is None:
        family = Gaussian()
    cols = [outcome] + predictors + [group_col]
    sub = data[cols].dropna().sort_values(group_col)
    if len(sub) < min_n:
        return None
    try:
        fit = GEE(
            endog=sub[outcome],
            exog=sm.add_constant(sub[predictors]),
            groups=sub[group_col],
            family=family,
            cov_struct=Exchangeable(),
        ).fit()
    except Exception:
        return None
    result: dict = {"intercept": fit.params["const"], "n": len(sub)}
    for pred in predictors:
        ci = fit.conf_int().loc[pred]
        result[f"coef_{pred}"] = fit.params[pred]
        result[f"se_{pred}"] = fit.bse[pred]
        result[f"p_{pred}"] = fit.pvalues[pred]
        result[f"ci_lo_{pred}"] = ci[0]
        result[f"ci_hi_{pred}"] = ci[1]
    if len(predictors) == 1:
        p0 = predictors[0]
        for short in ("coef", "se", "p", "ci_lo", "ci_hi"):
            result[short] = result[f"{short}_{p0}"]
    return result


def save_fig(fig, ctx, name, suptitle=""):
    """Suptitle + tight_layout + save via SaveableFigure."""
    if suptitle:
        fig.suptitle(suptitle, y=1.03)
    fig.tight_layout()
    SaveableFigure(fig, ctx).save(name)


# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name="habituation-oscillation-coupling",
    tag=f"{ROI_NAME}-{TASK}",
)

group = proj.sessions(task=TASK)
print(f"Selected {len(group)} sessions for task={TASK!r}")
print(f"  Subjects: {group.subjects}")
print(f"  Sessions: {group.session_labels}")

detector = OscillationDetector(
    source=ROI_SOURCE,
    signal=ROI_NAME,
    fs=FS,
    band_hz=BAND,
    filter_order=ORDER,
    threshold=THRESHOLD,
    threshold_k=THRESHOLD_K,
    min_duration_s=MIN_DURATION_OSC,
    merge_gap_s=MERGE_GAP_OSC,
)

# ═════════════════════════════════════════════════════════════════════════
# PASS 1 — Per-session summary (oscillation properties + state occupancy)
# ═════════════════════════════════════════════════════════════════════════

session_rows: list[dict] = []
transition_rows: list[dict] = []

for sess in group:
    day = day_from_session(sess.session)
    phase = PHASE_MAP.get(day, "unknown")

    # ── Treadmill / locomotion ───────────────────────────────────────────
    try:
        speed_t = sess.time("treadmill", "time_elapsed_s")
        speed_v = sess.signal("treadmill", "speed_mm")
    except Exception as e:
        print(f"  SKIP {sess.subject}/{sess.session} (no treadmill): {e}")
        continue
    speed_cms = speed_v / 10.0
    total_time_s = float(speed_t[-1] - speed_t[0])
    dt = float(np.nanmedian(np.diff(speed_t))) if len(speed_t) > 1 else 0.02

    loco = _locomotion_bouts(
        speed_t, speed_cms,
        min_speed_cms=MIN_SPEED_CMS,
        min_duration_s=MIN_LOCO_DURATION_S,
        merge_gap_s=MERGE_LOCO_GAP_S,
    )
    quiet = quiescent_bouts(speed_t, loco, min_duration_s=MIN_QUIESCENT_S)

    # State occupancy
    loco_time_s = sum(
        float(speed_t[e] - speed_t[s] + dt) for s, e in loco
    )
    quiet_time_s = sum(
        float(speed_t[e] - speed_t[s] + dt) for s, e in quiet
    )
    frac_loco = loco_time_s / total_time_s if total_time_s > 0 else 0.0
    frac_quiet = quiet_time_s / total_time_s if total_time_s > 0 else 0.0
    quiet_durations = [float(speed_t[e] - speed_t[s] + dt) for s, e in quiet]
    mean_quiet_dur = float(np.mean(quiet_durations)) if quiet_durations else 0.0
    median_quiet_dur = float(np.median(quiet_durations)) if quiet_durations else 0.0

    # ── Oscillation detection ────────────────────────────────────────────
    try:
        osc_result = detector.run(sess)
    except Exception as e:
        print(f"  SKIP {sess.subject}/{sess.session} (oscillation detection): {e}")
        continue
    osc_events = osc_result.events

    # Oscillation probability: fraction of quiescent bouts containing oscillation
    n_quiet_with_osc = 0
    osc_latencies = []
    for q_s, q_e in quiet:
        q_start_s = float(speed_t[q_s])
        q_end_s = float(speed_t[q_e])
        overlapping = osc_events[
            (osc_events["end_s"] > q_start_s) & (osc_events["start_s"] < q_end_s)
        ] if not osc_events.empty else pd.DataFrame()
        if len(overlapping) > 0:
            n_quiet_with_osc += 1
            earliest = max(overlapping["start_s"].min(), q_start_s)
            osc_latencies.append(earliest - q_start_s)

    p_osc_given_quiet = n_quiet_with_osc / len(quiet) if quiet else np.nan
    mean_osc_latency = float(np.mean(osc_latencies)) if osc_latencies else np.nan
    median_osc_latency = float(np.median(osc_latencies)) if osc_latencies else np.nan

    # Oscillation frequency: instantaneous frequency within each burst
    burst_freqs = []
    for start_idx, end_idx in osc_result.bursts:
        f_inst = instantaneous_frequency(
            osc_result.filtered_signal, FS, start_idx, end_idx,
        )
        if f_inst is not None:
            burst_freqs.append(f_inst)
    mean_osc_freq = float(np.mean(burst_freqs)) if burst_freqs else np.nan

    # Oscillation amplitude and duration (from events table)
    mean_osc_amp = float(osc_events["peak_env"].mean()) if len(osc_events) > 0 else np.nan
    mean_osc_dur = float(osc_events["duration_s"].mean()) if len(osc_events) > 0 else np.nan

    session_rows.append({
        "Subject": sess.subject,
        "Session": sess.session,
        "Day": day,
        "Phase": phase,
        # State occupancy
        "total_time_s": round(total_time_s, 1),
        "frac_locomotion": round(frac_loco, 4),
        "frac_quiescence": round(frac_quiet, 4),
        "n_loco_bouts": len(loco),
        "n_quiet_bouts": len(quiet),
        "mean_quiet_duration_s": round(mean_quiet_dur, 2),
        "median_quiet_duration_s": round(median_quiet_dur, 2),
        # Oscillation properties
        "n_osc_events": len(osc_events),
        "n_quiet_with_osc": n_quiet_with_osc,
        "p_osc_given_quiet": round(p_osc_given_quiet, 4),
        "frac_osc_bouts": round(n_quiet_with_osc / len(quiet), 4) if quiet else np.nan,
        "mean_osc_latency_s": round(mean_osc_latency, 3) if not np.isnan(mean_osc_latency) else np.nan,
        "median_osc_latency_s": round(median_osc_latency, 3) if not np.isnan(median_osc_latency) else np.nan,
        "mean_osc_freq_hz": round(mean_osc_freq, 3) if not np.isnan(mean_osc_freq) else np.nan,
        "mean_osc_amplitude": round(mean_osc_amp, 5) if not np.isnan(mean_osc_amp) else np.nan,
        "mean_osc_duration_s": round(mean_osc_dur, 3) if not np.isnan(mean_osc_dur) else np.nan,
    })

    # ── Pupil for this session ───────────────────────────────────────────
    pup_t, pup_raw = _get_pupil_arrays(sess)
    if pup_t is not None:
        pup_fs = 1.0 / float(np.nanmedian(np.diff(pup_t))) if len(pup_t) > 1 else 30.0
        pup_smooth = _smooth_pupil(pup_raw.astype(float), pup_fs, PUPIL_SMOOTH_S)
    else:
        pup_smooth = None

    # ── Quiescence → locomotion transition rows ──────────────────────────
    loco_starts = np.array([b[0] for b in loco])

    n_paired = 0
    for q_s, q_e in quiet:
        q_start_s = float(speed_t[q_s])
        q_end_s = float(speed_t[q_e])
        q_dur = q_end_s - q_start_s

        candidates = np.where(loco_starts >= q_e)[0]
        if len(candidates) == 0:
            continue
        next_loco_idx = candidates[0]
        next_loco = loco[next_loco_idx]

        osc_feats = oscillation_features_in_bout(q_start_s, q_end_s, osc_events)
        loco_feats = locomotion_bout_features(
            speed_t, speed_cms, next_loco[0], next_loco[1],
        )

        osc_onset_abs = None
        if osc_feats["has_oscillation"] and osc_feats["osc_onset_latency_s"] is not None:
            osc_onset_abs = q_start_s + osc_feats["osc_onset_latency_s"]
        pup_feats = pupil_features_for_transition(
            pup_t, pup_smooth, q_start_s, q_end_s, osc_onset_abs,
            window_s=PUPIL_WINDOW_S,
        )

        transition_rows.append({
            "Subject": sess.subject,
            "Session": sess.session,
            "Day": day,
            "Phase": phase,
            "q_start_s": round(q_start_s, 3),
            "q_end_s": round(q_end_s, 3),
            "q_duration_s": round(q_dur, 3),
            **osc_feats,
            **pup_feats,
            **loco_feats,
        })
        n_paired += 1

    print(
        f"  {sess.subject}/{sess.session} (Day {day}, {phase}): "
        f"{len(quiet)} quiet, {len(loco)} loco, {n_paired} paired, "
        f"P(osc)={p_osc_given_quiet:.2f}, freq={mean_osc_freq:.2f} Hz"
        if not np.isnan(mean_osc_freq) else
        f"  {sess.subject}/{sess.session} (Day {day}, {phase}): "
        f"{len(quiet)} quiet, {len(loco)} loco, {n_paired} paired, "
        f"P(osc)={p_osc_given_quiet:.2f}"
    )

# ═════════════════════════════════════════════════════════════════════════
# Build DataFrames
# ═════════════════════════════════════════════════════════════════════════

sess_df = pd.DataFrame(session_rows)
sess_df["Phase"] = pd.Categorical(sess_df["Phase"], categories=PHASE_ORDER, ordered=True)

trans_df = pd.DataFrame(transition_rows)
trans_df["Phase"] = pd.Categorical(trans_df["Phase"], categories=PHASE_ORDER, ordered=True)
trans_df["has_osc_int"] = trans_df["has_oscillation"].astype(int)

subjects = sorted(trans_df["Subject"].unique())
n_subjects = len(subjects)
subj_map = {s: i for i, s in enumerate(subjects)}
trans_df["subj_id"] = trans_df["Subject"].map(subj_map)

COLORS = plt.cm.tab10(np.linspace(0, 1, max(n_subjects, 1)))
subject_colors = {s: COLORS[i] for i, s in enumerate(subjects)}
PHASE_COLORS = {"early": "#4C72B0", "middle": "#55A868", "late": "#DD8452"}

# Z-score pupil within each subject
for col in ["pupil_mean_q_mm", "pupil_at_osc_onset_mm",
            "pupil_early_q_mm", "pupil_late_q_mm"]:
    zcol = col.replace("_mm", "_z")
    trans_df[zcol] = trans_df.groupby("Subject")[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
    )

osc_trans = trans_df[trans_df["has_oscillation"]].copy()

# Phase indicator dummies for GEE (reference = early)
trans_df["phase_middle"] = (trans_df["Phase"] == "middle").astype(float)
trans_df["phase_late"] = (trans_df["Phase"] == "late").astype(float)
osc_trans["phase_middle"] = (osc_trans["Phase"] == "middle").astype(float)
osc_trans["phase_late"] = (osc_trans["Phase"] == "late").astype(float)

print(f"\n── Summary ──")
print(f"Sessions: {len(sess_df)} | Subjects: {n_subjects}")
print(f"Transitions: {len(trans_df)} (osc: {len(osc_trans)})")
print(f"Phase counts (sessions): {sess_df['Phase'].value_counts().to_dict()}")
print(f"Phase counts (transitions): {trans_df['Phase'].value_counts().to_dict()}")

# ═════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Oscillation properties across days
# ═════════════════════════════════════════════════════════════════════════

print("\n══ Oscillation properties by phase ══")
for phase in PHASE_ORDER:
    sub = sess_df[sess_df["Phase"] == phase]
    print(f"  {phase}: "
          f"P(osc)={sub['p_osc_given_quiet'].mean():.3f} ± {sub['p_osc_given_quiet'].std():.3f}, "
          f"latency={sub['mean_osc_latency_s'].mean():.2f}s, "
          f"freq={sub['mean_osc_freq_hz'].mean():.2f} Hz, "
          f"amp={sub['mean_osc_amplitude'].mean():.5f}, "
          f"dur={sub['mean_osc_duration_s'].mean():.2f}s")

# ═════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Timing-vigor coupling per phase (β per phase)
# ═════════════════════════════════════════════════════════════════════════

print("\n══ β(osc_latency → vigor) by phase ══")

VIGOR_FEATURES = {
    "loco_peak_speed_cms": "Peak speed (cm/s)",
    "loco_mean_speed_cms": "Mean speed (cm/s)",
    "loco_duration_s": "Bout duration (s)",
    "latency_to_peak_speed_s": "Latency to peak speed (s)",
}

phase_betas: dict[str, list[dict]] = {feat: [] for feat in VIGOR_FEATURES}

for phase in PHASE_ORDER:
    phase_osc = osc_trans[osc_trans["Phase"] == phase].copy()
    phase_osc["subj_id"] = phase_osc["Subject"].map(subj_map)
    phase_osc = phase_osc.sort_values("subj_id")

    for feat, label in VIGOR_FEATURES.items():
        r = fit_gee(phase_osc, feat, ["osc_onset_latency_s"])
        if r is None:
            phase_betas[feat].append({
                "phase": phase, "beta": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "p": np.nan, "n": 0,
            })
            print(f"  {label:<30} [{phase}]: insufficient")
            continue
        phase_betas[feat].append({
            "phase": phase,
            "beta": r["coef"],
            "se": r["se"],
            "ci_lo": r["ci_lo"],
            "ci_hi": r["ci_hi"],
            "p": r["p"],
            "n": r["n"],
        })
        sig = "*" if r["p"] < 0.05 else ""
        print(f"  {label:<30} [{phase}]: β={r['coef']:+.4f} "
              f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] p={r['p']:.2e} {sig}")

# ═════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Full interaction model: vigor ~ latency * phase * pupil
# ═════════════════════════════════════════════════════════════════════════

print("\n══ Full model: vigor ~ osc_latency * phase * pupil_z ══")

# Build interaction terms
osc_full = osc_trans.copy()
osc_full["subj_id"] = osc_full["Subject"].map(subj_map)
osc_full = osc_full.sort_values("subj_id")

osc_full["lat_x_mid"] = osc_full["osc_onset_latency_s"] * osc_full["phase_middle"]
osc_full["lat_x_late"] = osc_full["osc_onset_latency_s"] * osc_full["phase_late"]
osc_full["lat_x_pupil"] = osc_full["osc_onset_latency_s"] * osc_full["pupil_mean_q_z"]
osc_full["mid_x_pupil"] = osc_full["phase_middle"] * osc_full["pupil_mean_q_z"]
osc_full["late_x_pupil"] = osc_full["phase_late"] * osc_full["pupil_mean_q_z"]
osc_full["lat_x_mid_x_pupil"] = (
    osc_full["osc_onset_latency_s"] * osc_full["phase_middle"] * osc_full["pupil_mean_q_z"]
)
osc_full["lat_x_late_x_pupil"] = (
    osc_full["osc_onset_latency_s"] * osc_full["phase_late"] * osc_full["pupil_mean_q_z"]
)

model_comparison: dict[str, dict] = {}

for feat, label in VIGOR_FEATURES.items():
    # Model 1: latency only
    m1 = fit_gee(osc_full, feat, ["osc_onset_latency_s"])
    # Model 2: + phase main effects
    m2 = fit_gee(osc_full, feat,
                 ["osc_onset_latency_s", "phase_middle", "phase_late"])
    # Model 3: + latency × phase interaction
    m3 = fit_gee(osc_full, feat,
                 ["osc_onset_latency_s", "phase_middle", "phase_late",
                  "lat_x_mid", "lat_x_late"])
    # Model 4: + pupil + latency × pupil
    m4 = fit_gee(osc_full, feat,
                 ["osc_onset_latency_s", "phase_middle", "phase_late",
                  "lat_x_mid", "lat_x_late", "pupil_mean_q_z", "lat_x_pupil"])
    # Model 5: full three-way (*phase *pupil)
    m5 = fit_gee(osc_full, feat,
                 ["osc_onset_latency_s", "phase_middle", "phase_late",
                  "pupil_mean_q_z",
                  "lat_x_mid", "lat_x_late", "lat_x_pupil",
                  "mid_x_pupil", "late_x_pupil",
                  "lat_x_mid_x_pupil", "lat_x_late_x_pupil"])

    row = {"outcome": label}
    for name, model in [("M1_latency", m1), ("M2_+phase", m2),
                        ("M3_+latXphase", m3), ("M4_+pupil", m4),
                        ("M5_full", m5)]:
        if model is not None:
            row[f"n_{name}"] = model["n"]
            # Report the latency coefficient for all models
            if "coef_osc_onset_latency_s" in model:
                row[f"beta_lat_{name}"] = model["coef_osc_onset_latency_s"]
                row[f"p_lat_{name}"] = model["p_osc_onset_latency_s"]
            elif "coef" in model:
                row[f"beta_lat_{name}"] = model["coef"]
                row[f"p_lat_{name}"] = model["p"]
            # Report interaction terms where available
            if "coef_lat_x_mid" in model:
                row[f"beta_latXmid_{name}"] = model["coef_lat_x_mid"]
                row[f"p_latXmid_{name}"] = model["p_lat_x_mid"]
            if "coef_lat_x_late" in model:
                row[f"beta_latXlate_{name}"] = model["coef_lat_x_late"]
                row[f"p_latXlate_{name}"] = model["p_lat_x_late"]
            if "coef_lat_x_pupil" in model:
                row[f"beta_latXpupil_{name}"] = model["coef_lat_x_pupil"]
                row[f"p_latXpupil_{name}"] = model["p_lat_x_pupil"]
        else:
            row[f"n_{name}"] = 0
    model_comparison[feat] = row

    print(f"\n  {label}:")
    for name, model in [("M1", m1), ("M2", m2), ("M3", m3), ("M4", m4), ("M5", m5)]:
        if model is not None:
            lat_key = "coef_osc_onset_latency_s" if "coef_osc_onset_latency_s" in model else "coef"
            lat_p_key = "p_osc_onset_latency_s" if "p_osc_onset_latency_s" in model else "p"
            print(f"    {name}: β_lat={model[lat_key]:+.4f} p={model[lat_p_key]:.2e} (n={model['n']})")

# ═════════════════════════════════════════════════════════════════════════
# ANALYSIS 4 — Arousal interaction across phases
# ═════════════════════════════════════════════════════════════════════════

print("\n══ Arousal interaction (latency × pupil) per phase ══")

arousal_phase_results: dict[str, list[dict]] = {feat: [] for feat in VIGOR_FEATURES}

for phase in PHASE_ORDER:
    phase_data = osc_full[osc_full["Phase"] == phase].copy()
    phase_data["subj_id"] = phase_data["Subject"].map(subj_map)
    phase_data = phase_data.sort_values("subj_id")
    phase_data["lat_x_pupil_local"] = (
        phase_data["osc_onset_latency_s"] * phase_data["pupil_mean_q_z"]
    )

    for feat, label in VIGOR_FEATURES.items():
        r = fit_gee(
            phase_data, feat,
            ["osc_onset_latency_s", "pupil_mean_q_z", "lat_x_pupil_local"],
        )
        if r is None:
            arousal_phase_results[feat].append({
                "phase": phase, "beta_interaction": np.nan, "p_interaction": np.nan,
                "beta_latency": np.nan, "beta_pupil": np.nan, "n": 0,
            })
            continue
        arousal_phase_results[feat].append({
            "phase": phase,
            "beta_interaction": r["coef_lat_x_pupil_local"],
            "p_interaction": r["p_lat_x_pupil_local"],
            "beta_latency": r["coef_osc_onset_latency_s"],
            "beta_pupil": r["coef_pupil_mean_q_z"],
            "n": r["n"],
        })
        sig = "*" if r["p_lat_x_pupil_local"] < 0.05 else ""
        print(f"  {label:<28} [{phase}]: β_int={r['coef_lat_x_pupil_local']:+.4f} "
              f"p={r['p_lat_x_pupil_local']:.2e} {sig}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Oscillation probability and latency across days
# ═════════════════════════════════════════════════════════════════════════

fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))

# 1a: P(oscillation | quiescence) by day, per animal
ax = axes1[0, 0]
for subj in subjects:
    sub = sess_df[sess_df["Subject"] == subj].sort_values("Day")
    ax.plot(sub["Day"], sub["p_osc_given_quiet"], "o-", ms=5,
            color=subject_colors[subj], label=subj, alpha=0.7)
# Group mean ± SEM
grp = sess_df.groupby("Day")["p_osc_given_quiet"]
ax.errorbar(grp.mean().index, grp.mean().values,
            yerr=grp.sem().values, fmt="s-", color="black", ms=7,
            lw=2, capsize=4, label="mean ± SEM", zorder=10)
ax.set_xlabel("Day")
ax.set_ylabel("P(oscillation | quiescence)")
ax.set_title("A. Oscillation probability")
ax.legend(frameon=False)
ax.set_xticks(range(1, 11))
ax.grid(alpha=0.3)

# 1b: Mean oscillation latency by day, per animal
ax = axes1[0, 1]
for subj in subjects:
    sub = sess_df[sess_df["Subject"] == subj].sort_values("Day")
    ax.plot(sub["Day"], sub["mean_osc_latency_s"], "o-", ms=5,
            color=subject_colors[subj], label=subj, alpha=0.7)
grp = sess_df.groupby("Day")["mean_osc_latency_s"]
ax.errorbar(grp.mean().index, grp.mean().values,
            yerr=grp.sem().values, fmt="s-", color="black", ms=7,
            lw=2, capsize=4, label="mean ± SEM", zorder=10)
ax.set_xlabel("Day")
ax.set_ylabel("Mean onset latency (s)")
ax.set_title("B. Oscillation onset latency")
ax.set_xticks(range(1, 11))
ax.grid(alpha=0.3)

# 1c: Mean oscillation frequency by day
ax = axes1[0, 2]
for subj in subjects:
    sub = sess_df[sess_df["Subject"] == subj].sort_values("Day")
    ax.plot(sub["Day"], sub["mean_osc_freq_hz"], "o-", ms=5,
            color=subject_colors[subj], label=subj, alpha=0.7)
grp = sess_df.groupby("Day")["mean_osc_freq_hz"]
ax.errorbar(grp.mean().index, grp.mean().values,
            yerr=grp.sem().values, fmt="s-", color="black", ms=7,
            lw=2, capsize=4, label="mean ± SEM", zorder=10)
ax.axhline(3.7, color="red", ls=":", lw=1, alpha=0.6, label="3.7 Hz reference")
ax.set_xlabel("Day")
ax.set_ylabel("Mean frequency (Hz)")
ax.set_title("C. Oscillation frequency stability")
ax.set_xticks(range(1, 11))
ax.grid(alpha=0.3)
ax.legend(frameon=False)

# 1d: Mean oscillation amplitude by day
ax = axes1[1, 0]
for subj in subjects:
    sub = sess_df[sess_df["Subject"] == subj].sort_values("Day")
    ax.plot(sub["Day"], sub["mean_osc_amplitude"], "o-", ms=5,
            color=subject_colors[subj], label=subj, alpha=0.7)
grp = sess_df.groupby("Day")["mean_osc_amplitude"]
ax.errorbar(grp.mean().index, grp.mean().values,
            yerr=grp.sem().values, fmt="s-", color="black", ms=7,
            lw=2, capsize=4, label="mean ± SEM", zorder=10)
ax.set_xlabel("Day")
ax.set_ylabel("Mean peak envelope (ΔF/F)")
ax.set_title("D. Oscillation amplitude")
ax.set_xticks(range(1, 11))
ax.grid(alpha=0.3)

# 1e: Mean oscillation duration by day
ax = axes1[1, 1]
for subj in subjects:
    sub = sess_df[sess_df["Subject"] == subj].sort_values("Day")
    ax.plot(sub["Day"], sub["mean_osc_duration_s"], "o-", ms=5,
            color=subject_colors[subj], label=subj, alpha=0.7)
grp = sess_df.groupby("Day")["mean_osc_duration_s"]
ax.errorbar(grp.mean().index, grp.mean().values,
            yerr=grp.sem().values, fmt="s-", color="black", ms=7,
            lw=2, capsize=4, label="mean ± SEM", zorder=10)
ax.set_xlabel("Day")
ax.set_ylabel("Mean burst duration (s)")
ax.set_title("E. Oscillation duration")
ax.set_xticks(range(1, 11))
ax.grid(alpha=0.3)

# 1f: State occupancy by day (frac quiescence)
ax = axes1[1, 2]
for subj in subjects:
    sub = sess_df[sess_df["Subject"] == subj].sort_values("Day")
    ax.plot(sub["Day"], sub["frac_quiescence"], "o-", ms=5,
            color=subject_colors[subj], label=subj, alpha=0.7)
grp = sess_df.groupby("Day")["frac_quiescence"]
ax.errorbar(grp.mean().index, grp.mean().values,
            yerr=grp.sem().values, fmt="s-", color="black", ms=7,
            lw=2, capsize=4, label="mean ± SEM", zorder=10)
ax.set_xlabel("Day")
ax.set_ylabel("Fraction of session quiescent")
ax.set_title("F. State occupancy (quiescence)")
ax.set_xticks(range(1, 11))
ax.grid(alpha=0.3)

save_fig(fig1, proj._context, "fig1_oscillation_properties_across_days.svg",
         f"Oscillation properties across 10-day HFSA\n"
         f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK} | N = {n_subjects}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 2 — β(latency → vigor) by phase, with CI
# ═════════════════════════════════════════════════════════════════════════

n_feats = len(VIGOR_FEATURES)
fig2, axes2 = plt.subplots(1, n_feats, figsize=(4 * n_feats, 5))
if n_feats == 1:
    axes2 = [axes2]

for ax, (feat, label) in zip(axes2, VIGOR_FEATURES.items()):
    betas = phase_betas[feat]
    x_pos = np.arange(len(PHASE_ORDER))
    beta_vals = [b["beta"] for b in betas]
    ci_lo = [b["ci_lo"] for b in betas]
    ci_hi = [b["ci_hi"] for b in betas]
    ps = [b["p"] for b in betas]
    ns = [b["n"] for b in betas]

    for i, phase in enumerate(PHASE_ORDER):
        if np.isnan(beta_vals[i]):
            continue
        err_lo = beta_vals[i] - ci_lo[i]
        err_hi = ci_hi[i] - beta_vals[i]
        color = PHASE_COLORS[phase]
        sig = "filled" if ps[i] < 0.05 else "none"
        ax.errorbar(i, beta_vals[i], yerr=[[err_lo], [err_hi]],
                    fmt="o", color=color, ms=10, capsize=6, lw=2,
                    markerfacecolor=color if sig == "filled" else "white",
                    markeredgecolor=color, markeredgewidth=2)
        ax.annotate(f"p={ps[i]:.2e}\nn={ns[i]}",
                    (i, ci_hi[i] + 0.002), ha="center", fontsize=7)

    ax.axhline(0, color="red", ls=":", lw=1, alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(PHASE_ORDER)
    ax.set_ylabel(f"β (osc_latency → {label})")
    ax.set_title(label)
    ax.grid(axis="y", alpha=0.3)

save_fig(fig2, proj._context, "fig2_latency_vigor_coupling_by_phase.svg",
         f"Timing-vigor coupling across habituation phases\n"
         f"β(osc_latency → vigor) | {ROI_NAME} | N = {n_subjects}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Arousal interaction term across phases
# ═════════════════════════════════════════════════════════════════════════

fig3, axes3 = plt.subplots(1, n_feats, figsize=(4 * n_feats, 5))
if n_feats == 1:
    axes3 = [axes3]

for ax, (feat, label) in zip(axes3, VIGOR_FEATURES.items()):
    results = arousal_phase_results[feat]
    x_pos = np.arange(len(PHASE_ORDER))

    for i, phase in enumerate(PHASE_ORDER):
        r = results[i]
        beta = r["beta_interaction"]
        if np.isnan(beta):
            continue
        color = PHASE_COLORS[phase]
        sig = r["p_interaction"] < 0.05
        ax.bar(i, beta, color=color, alpha=0.7 if sig else 0.3,
               edgecolor=color, lw=2, width=0.6)
        ax.annotate(f"p={r['p_interaction']:.2e}\nn={r['n']}",
                    (i, beta + 0.002 * np.sign(beta) if beta != 0 else 0.002),
                    ha="center", fontsize=7)

    ax.axhline(0, color="red", ls=":", lw=1, alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(PHASE_ORDER)
    ax.set_ylabel(f"β (latency × pupil → {label})")
    ax.set_title(label)
    ax.grid(axis="y", alpha=0.3)

save_fig(fig3, proj._context, "fig3_arousal_interaction_across_phases.svg",
         f"Arousal modulation of oscillation-vigor coupling\n"
         f"β(latency × pupil_z) | {ROI_NAME} | N = {n_subjects}")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 4 (bonus) — State occupancy + quiescent bout duration by phase
# ═════════════════════════════════════════════════════════════════════════

fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))

# 4a: Quiescent bout duration distribution by phase
ax = axes4[0]
for phase in PHASE_ORDER:
    sub = trans_df[trans_df["Phase"] == phase]["q_duration_s"]
    if sub.empty:
        continue
    ax.hist(sub, bins=30, alpha=0.45, color=PHASE_COLORS[phase],
            label=f"{phase} (n={len(sub)})", edgecolor="white", lw=0.5,
            density=True)
ax.set_xlabel("Quiescent bout duration (s)")
ax.set_ylabel("Density")
ax.set_title("A. Quiescent bout duration by phase")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)

# 4b: Fraction of osc-containing bouts by phase and animal
ax = axes4[1]
bout_frac = sess_df.groupby(["Phase", "Subject"], observed=True)["frac_osc_bouts"].mean().unstack()
for subj in subjects:
    if subj in bout_frac.columns:
        vals = bout_frac[subj]
        ax.plot(range(len(PHASE_ORDER)), vals.values, "o-", ms=6,
                color=subject_colors[subj], label=subj, alpha=0.7)
# Group mean
grp_frac = sess_df.groupby("Phase", observed=True)["frac_osc_bouts"]
ax.errorbar(range(len(PHASE_ORDER)), grp_frac.mean().values,
            yerr=grp_frac.sem().values, fmt="s-", color="black", ms=8,
            lw=2, capsize=4, zorder=10, label="mean ± SEM")
ax.set_xticks(range(len(PHASE_ORDER)))
ax.set_xticklabels(PHASE_ORDER)
ax.set_ylabel("Fraction bouts with oscillation")
ax.set_title("B. Oscillation-containing bouts")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)

# 4c: Median quiescent bout duration by phase and animal
ax = axes4[2]
dur_by_phase = sess_df.groupby(["Phase", "Subject"], observed=True)["median_quiet_duration_s"].mean().unstack()
for subj in subjects:
    if subj in dur_by_phase.columns:
        vals = dur_by_phase[subj]
        ax.plot(range(len(PHASE_ORDER)), vals.values, "o-", ms=6,
                color=subject_colors[subj], label=subj, alpha=0.7)
grp_dur = sess_df.groupby("Phase", observed=True)["median_quiet_duration_s"]
ax.errorbar(range(len(PHASE_ORDER)), grp_dur.mean().values,
            yerr=grp_dur.sem().values, fmt="s-", color="black", ms=8,
            lw=2, capsize=4, zorder=10, label="mean ± SEM")
ax.set_xticks(range(len(PHASE_ORDER)))
ax.set_xticklabels(PHASE_ORDER)
ax.set_ylabel("Median quiescent bout duration (s)")
ax.set_title("C. Quiescent bout duration")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)

save_fig(fig4, proj._context, "fig4_state_occupancy_by_phase.svg",
         f"State occupancy across habituation phases\n"
         f"{TASK} | N = {n_subjects}")

# ═════════════════════════════════════════════════════════════════════════
# Save tables
# ═════════════════════════════════════════════════════════════════════════

stats_dir = proj._context.stats_dir
stats_dir.mkdir(parents=True, exist_ok=True)

sess_df.to_csv(stats_dir / "session_summary.csv", index=False)
trans_df.to_csv(stats_dir / "transitions.csv", index=False)
pd.DataFrame(model_comparison).T.reset_index().rename(
    columns={"index": "feature"}
).to_csv(stats_dir / "model_comparison.csv", index=False)

# Phase beta tables
for feat, label in VIGOR_FEATURES.items():
    betas = phase_betas[feat]
    pd.DataFrame(betas).to_csv(
        stats_dir / f"phase_betas_{feat}.csv", index=False
    )
    arousal_res = arousal_phase_results[feat]
    pd.DataFrame(arousal_res).to_csv(
        stats_dir / f"arousal_interaction_{feat}.csv", index=False
    )

# ═════════════════════════════════════════════════════════════════════════
# Markdown report
# ═════════════════════════════════════════════════════════════════════════

# Session summary table
sess_table_lines = []
for _, row in sess_df.sort_values(["Subject", "Day"]).iterrows():
    sess_table_lines.append(
        f"| {row['Subject']} | {row['Session']} | {row['Day']} | {row['Phase']} | "
        f"{row['p_osc_given_quiet']:.3f} | {row.get('mean_osc_latency_s', 'N/A')} | "
        f"{row.get('mean_osc_freq_hz', 'N/A')} | {row.get('mean_osc_amplitude', 'N/A')} | "
        f"{row['frac_quiescence']:.3f} |"
    )
sess_table = "\n".join(sess_table_lines)

# Phase beta summary
beta_table_lines = []
for feat, label in VIGOR_FEATURES.items():
    for b in phase_betas[feat]:
        if np.isnan(b["beta"]):
            beta_table_lines.append(f"| {label} | {b['phase']} | — | — | — | 0 |")
        else:
            sig = "**" if b["p"] < 0.05 else ""
            beta_table_lines.append(
                f"| {label} | {b['phase']} | {b['beta']:+.4f} | "
                f"[{b['ci_lo']:.4f}, {b['ci_hi']:.4f}] | "
                f"{sig}{b['p']:.2e}{sig} | {b['n']} |"
            )
beta_table = "\n".join(beta_table_lines)

# Model comparison table
mc_table_lines = []
for feat, row in model_comparison.items():
    label = row["outcome"]
    parts = [f"| {label}"]
    for mname in ["M1_latency", "M2_+phase", "M3_+latXphase", "M4_+pupil", "M5_full"]:
        bk = f"beta_lat_{mname}"
        pk = f"p_lat_{mname}"
        if bk in row and not (isinstance(row[bk], float) and np.isnan(row[bk])):
            parts.append(f"{row[bk]:+.4f} (p={row[pk]:.2e})")
        else:
            parts.append("—")
    mc_table_lines.append(" | ".join(parts) + " |")
mc_table = "\n".join(mc_table_lines)

# Arousal interaction table
arousal_table_lines = []
for feat, label in VIGOR_FEATURES.items():
    for r in arousal_phase_results[feat]:
        if np.isnan(r["beta_interaction"]):
            arousal_table_lines.append(f"| {label} | {r['phase']} | — | — | 0 |")
        else:
            sig = "**" if r["p_interaction"] < 0.05 else ""
            arousal_table_lines.append(
                f"| {label} | {r['phase']} | "
                f"{r['beta_interaction']:+.4f} | "
                f"{sig}{r['p_interaction']:.2e}{sig} | {r['n']} |"
            )
arousal_table = "\n".join(arousal_table_lines)

# Interpretation patterns table
patterns_table = (
    "| Pattern | Interpretation |\n"
    "|---------|----------------|\n"
    "| Coupling strengthens with habituation | Oscillation becomes more 'meaningful' as environment becomes predictable |\n"
    "| Coupling weakens | Early oscillations reflect orienting/vigilance; later ones are true idle activity |\n"
    "| Arousal modulation disappears | Animal enters stable low-arousal state; oscillation timing carries most information |\n"
    "| Frequency drifts | Possibly argues against fixed thalamocortical generator, or for state-dependent tuning |"
)

report_path = proj.save_report(
    notes=(
        f"## Habituation-dependent oscillation-behavior coupling\n\n"
        f"**Core question:** Does the oscillation-locomotion relationship drift "
        f"across the 10-day HFSA period, and if so, how?\n\n"
        f"**Dataset:** ETOH-HFSA | **Task:** {TASK}\n\n"
        f"**ROI:** {ROI_NAME} | **Band:** {BAND[0]}–{BAND[1]} Hz\n\n"
        f"**Phase binning:** early (Days 1–3), middle (Days 4–7), late (Days 8–10)\n\n"
        f"**N animals:** {n_subjects} | **N sessions:** {len(sess_df)} | "
        f"**N transitions:** {len(trans_df)} (with osc: {len(osc_trans)})\n\n"
        f"---\n\n"
        f"### 1. Oscillation properties across days\n\n"
        f"| Subject | Session | Day | Phase | P(osc) | Latency (s) | Freq (Hz) | Amplitude | Frac quiescent |\n"
        f"|---------|---------|----:|:-----:|-------:|------------:|----------:|----------:|---------------:|\n"
        f"{sess_table}\n\n"
        f"---\n\n"
        f"### 2. Timing-vigor coupling by phase\n\n"
        f"β(osc_onset_latency → locomotion feature), GEE clustered by animal, per phase.\n\n"
        f"| Outcome | Phase | β | 95% CI | p | N |\n"
        f"|---------|:-----:|--:|-------:|--:|--:|\n"
        f"{beta_table}\n\n"
        f"---\n\n"
        f"### 3. Model comparison\n\n"
        f"β_latency across progressively complex models. Does adding phase "
        f"interactions or pupil improve the latency coefficient?\n\n"
        f"| Outcome | M1 (lat) | M2 (+phase) | M3 (+lat×phase) | M4 (+pupil) | M5 (full) |\n"
        f"|---------|:--------:|:-----------:|:---------------:|:-----------:|:---------:|\n"
        f"{mc_table}\n\n"
        f"---\n\n"
        f"### 4. Arousal interaction across phases\n\n"
        f"β(osc_latency × pupil_z) per phase. Does the pupil moderation "
        f"of oscillation-vigor coupling change with habituation?\n\n"
        f"| Outcome | Phase | β_interaction | p | N |\n"
        f"|---------|:-----:|-------------:|--:|--:|\n"
        f"{arousal_table}\n\n"
        f"---\n\n"
        f"### Interpretation guide\n\n"
        f"{patterns_table}\n\n"
        f"---\n\n"
        f"### Caveats\n\n"
        f"- 10 sessions / {n_subjects} animals means day-by-day estimates are noisy; "
        f"binning into phases is used to stabilize.\n"
        f"- If behavioral structure changes (longer quiescent bouts on later days), "
        f"comparison becomes apples-to-oranges unless bout duration is controlled.\n"
        f"- This is correlational — habituation and oscillation changes could be "
        f"driven by a third variable (fatigue, circadian drift, etc.).\n\n"
        f"**Animals:** {', '.join(subjects)}\n"
    ),
)

print(f"\nReport: {report_path}")
print(f"Outputs saved to: {proj._context.run_dir}")
