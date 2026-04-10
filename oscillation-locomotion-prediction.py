"""
Oscillation → locomotion prediction (with pupil arousal control).

For each quiescent bout that terminates in locomotion, extract:
    1. Did oscillation occur during quiescence?
    2. Oscillation onset latency (from quiescence start)
    3. Total oscillation duration and mean peak envelope during quiescence
    4. Pupil diameter at oscillation onset, during early/late quiescence,
       and at the quiescence-locomotion transition.

Then correlate with features of the *next* locomotion bout:
    - Bout duration (sustained vs brief movement)
    - Peak speed (vigor)
    - Mean speed (vigor)
    - Latency to peak speed (ramp-up dynamics: explosive vs gradual)

Key question: Does oscillation timing predict locomotion vigor beyond what
pupil state (arousal) already explains?

Analyses:
    1. Pupil at oscillation onset — does early osc coincide with dilation?
    2. Mediation — does pupil explain away the osc-latency → locomotion link?
    3. Pupil trajectory — early vs late quiescence pupil by osc timing group
    4. Interaction — does osc timing mean different things in alert vs drowsy?

Statistical approach:
    - GEE linear models clustered by animal
    - Mediation check: compare osc-latency β with and without pupil covariate

Usage:
    python Scripts/scriptings/oscillation-locomotion-prediction.py
"""
from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import savgol_filter
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian
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

# Quiescent bout minimum duration
MIN_QUIESCENT_S = 1.0

# Pupil settings
PUPIL_SIGNAL = "pupil_diameter_mm"   # fallback: "diameter_mm"
PUPIL_SMOOTH_S = 0.5                 # Savitzky-Golay smoothing window
PUPIL_WINDOW_S = 5.0                 # seconds for early/late quiescence pupil


# ─── Helpers ──────────────────────────────────────────────────────────────

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
    """Extract oscillation features within a quiescent bout.

    Returns dict with:
        has_oscillation, osc_onset_latency_s, osc_total_duration_s,
        osc_mean_peak_env, osc_n_events
    """
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

    # Onset latency: time from quiescence start to first oscillation
    earliest_start = max(overlapping["start_s"].min(), q_start_s)
    result["osc_onset_latency_s"] = earliest_start - q_start_s

    # Total oscillation duration within the bout (clamp to bout boundaries)
    total_dur = 0.0
    for _, ev in overlapping.iterrows():
        ov_start = max(q_start_s, ev["start_s"])
        ov_end = min(q_end_s, ev["end_s"])
        if ov_end > ov_start:
            total_dur += ov_end - ov_start
    result["osc_total_duration_s"] = total_dur

    # Mean peak envelope across overlapping events
    if "peak_env" in overlapping.columns:
        result["osc_mean_peak_env"] = float(overlapping["peak_env"].mean())

    return result


def locomotion_bout_features(
    t: np.ndarray,
    speed_cms: np.ndarray,
    bout_start_idx: int,
    bout_end_idx: int,
) -> dict:
    """Extract features of a single locomotion bout.

    Returns dict with:
        loco_duration_s, loco_peak_speed_cms, loco_mean_speed_cms,
        latency_to_peak_speed_s
    """
    s, e = int(bout_start_idx), int(bout_end_idx)
    dt_med = float(np.nanmedian(np.diff(t))) if t.size > 1 else 0.02

    bout_t = t[s : e + 1]
    bout_speed = np.abs(speed_cms[s : e + 1])

    duration = float(bout_t[-1] - bout_t[0] + dt_med)
    peak_speed = float(np.nanmax(bout_speed))
    mean_speed = float(np.nanmean(bout_speed))

    # Latency to peak speed: time from bout onset to first peak-speed sample
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
    """Try to load pupil time + diameter from a session. Returns (t, pupil) or (None, None)."""
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
    """Extract pupil features around a quiescence → locomotion transition.

    Returns dict with:
        pupil_at_osc_onset_mm, pupil_early_q_mm, pupil_late_q_mm,
        pupil_delta_q_mm (late - early), pupil_mean_q_mm
    """
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

    # Pupil at oscillation onset: mean in ±0.5s window around onset
    if osc_onset_s is not None:
        onset_mask = (pup_t >= osc_onset_s - 0.5) & (pup_t <= osc_onset_s + 0.5)
        if onset_mask.sum() > 0:
            result["pupil_at_osc_onset_mm"] = round(
                float(np.nanmean(pup_smooth[onset_mask])), 4
            )

    # Early quiescence: first `window_s` seconds
    q_dur = q_end_s - q_start_s
    early_end = min(q_start_s + window_s, q_end_s)
    early_mask = (pup_t >= q_start_s) & (pup_t <= early_end)
    if early_mask.sum() > 0:
        result["pupil_early_q_mm"] = round(
            float(np.nanmean(pup_smooth[early_mask])), 4
        )

    # Late quiescence: last `window_s` seconds before locomotion
    late_start = max(q_end_s - window_s, q_start_s)
    late_mask = (pup_t >= late_start) & (pup_t <= q_end_s)
    if late_mask.sum() > 0:
        result["pupil_late_q_mm"] = round(
            float(np.nanmean(pup_smooth[late_mask])), 4
        )

    # Delta: late minus early (positive = dilation over quiescence)
    if result["pupil_early_q_mm"] is not None and result["pupil_late_q_mm"] is not None:
        result["pupil_delta_q_mm"] = round(
            result["pupil_late_q_mm"] - result["pupil_early_q_mm"], 4
        )

    return result


# ─── Reusable analysis / plotting helpers ────────────────────────────────

def fit_gee(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    group_col: str = "subj_id",
    min_n: int = 10,
) -> dict | None:
    """Fit a GEE Gaussian model.  Returns coef/se/p/ci dict or None.

    Single-predictor: flat keys ``coef, se, p, ci_lo, ci_hi``.
    Multi-predictor: keys prefixed per predictor, e.g. ``coef_<pred>``.
    Always includes ``intercept`` and ``n``.
    """
    cols = [outcome] + predictors + [group_col]
    sub = data[cols].dropna().sort_values(group_col)
    if len(sub) < min_n:
        return None
    try:
        fit = GEE(
            endog=sub[outcome],
            exog=sm.add_constant(sub[predictors]),
            groups=sub[group_col],
            family=Gaussian(),
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


def feature_panel(n_cols: int, height: float = 5.0):
    """Create a 1×n subplot row; always returns (fig, list[Axes])."""
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, height))
    if n_cols == 1:
        axes = [axes]
    return fig, list(axes)


def scatter_by_subject(ax, data, x, y, subjects, colors, **kw):
    """Scatter *x* vs *y*, one colour per subject."""
    kw.setdefault("s", 18)
    kw.setdefault("alpha", 0.5)
    kw.setdefault("edgecolors", "none")
    for subj in subjects:
        sub = data[data["Subject"] == subj]
        if sub.empty:
            continue
        ax.scatter(sub[x], sub[y], color=colors[subj], label=subj, **kw)


def violin_two_groups(ax, vals_a, vals_b, label_a, label_b,
                      color_a="#4C72B0", color_b="#DD8452"):
    """Side-by-side violin at positions 0 and 1."""
    parts = ax.violinplot([vals_a, vals_b], positions=[0, 1],
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.4)
    parts["bodies"][0].set_facecolor(color_a)
    parts["bodies"][1].set_facecolor(color_b)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_a, label_b])


def save_fig(fig, ctx, name, suptitle=""):
    """Suptitle + tight_layout + save via SaveableFigure."""
    if suptitle:
        fig.suptitle(suptitle, y=1.03)
    fig.tight_layout()
    SaveableFigure(fig, ctx).save(name)


def save_dict_csv(results, path):
    """Save ``{key: dict}`` mapping as CSV (keys → ``feature`` column)."""
    if not results:
        return
    pd.DataFrame(results).T.reset_index().rename(
        columns={"index": "feature"}
    ).to_csv(path, index=False)


# ─── Project setup ────────────────────────────────────────────────────────

proj = Project(
    dataset=DATASET,
    analyst="Jacob Gronemeyer",
    lab="Sipe Lab",
    run_name="oscillation-locomotion-prediction",
    tag=f"{ROI_NAME}-{TASK}",
)

group = proj.sessions(task=TASK)
print(f"Selected {len(group)} sessions for task={TASK!r}")
print(f"  Subjects: {group.subjects}")

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

# ─── Collect per-transition data ─────────────────────────────────────────
# For each quiescent bout that is followed by a locomotion bout, pair them.

rows: list[dict] = []

for sess in group:
    try:
        speed_t = sess.time("treadmill", "time_elapsed_s")
        speed_v = sess.signal("treadmill", "speed_mm")
    except Exception as e:
        print(f"  SKIP {sess.subject}/{sess.session} (no treadmill): {e}")
        continue

    speed_cms = speed_v / 10.0

    loco = _locomotion_bouts(
        speed_t, speed_cms,
        min_speed_cms=MIN_SPEED_CMS,
        min_duration_s=MIN_LOCO_DURATION_S,
        merge_gap_s=MERGE_LOCO_GAP_S,
    )
    quiet = quiescent_bouts(speed_t, loco, min_duration_s=MIN_QUIESCENT_S)

    if not loco:
        print(f"  SKIP {sess.subject}/{sess.session}: no locomotion bouts")
        continue

    try:
        osc_result = detector.run(sess)
    except Exception as e:
        print(f"  SKIP {sess.subject}/{sess.session} (oscillation detection): {e}")
        continue

    osc_events = osc_result.events

    # Load pupil data for this session
    pup_t, pup_raw = _get_pupil_arrays(sess)
    if pup_t is not None:
        pup_fs = 1.0 / float(np.nanmedian(np.diff(pup_t))) if len(pup_t) > 1 else 30.0
        pup_smooth = _smooth_pupil(pup_raw.astype(float), pup_fs, PUPIL_SMOOTH_S)
    else:
        pup_smooth = None

    # Build a lookup: for each quiescent bout, find the *next* locomotion bout
    # A quiescent bout (q_s, q_e) is followed by the locomotion bout whose
    # start index == q_e + 1 (or the nearest loco bout starting after q_e).
    loco_starts = np.array([b[0] for b in loco])

    n_paired = 0
    for qi, (q_s, q_e) in enumerate(quiet):
        q_start_s = float(speed_t[q_s])
        q_end_s = float(speed_t[q_e])
        q_dur = q_end_s - q_start_s

        # Find the next locomotion bout starting at or after quiescent bout end
        candidates = np.where(loco_starts >= q_e)[0]
        if len(candidates) == 0:
            continue  # quiescent bout at end of session, no following loco
        next_loco_idx = candidates[0]
        next_loco = loco[next_loco_idx]

        # Compute inter-bout interval: time gap between quiescence end and loco start
        # (should be near zero for adjacent bouts, but captures any transition gap)
        ibi_s = float(speed_t[next_loco[0]] - speed_t[q_e])

        # Also compute inter-bout interval to *next* quiescent period after this loco
        # (i.e., how long until the animal returns to quiescence)
        loco_end_idx = next_loco[1]
        next_quiet_candidates = [(qs, qe) for qs, qe in quiet if qs > loco_end_idx]
        if next_quiet_candidates:
            time_to_next_quiet_s = float(
                speed_t[next_quiet_candidates[0][0]] - speed_t[loco_end_idx]
            )
        else:
            time_to_next_quiet_s = None

        # Extract oscillation features during this quiescent bout
        osc_feats = oscillation_features_in_bout(q_start_s, q_end_s, osc_events)

        # Extract locomotion features of the next bout
        loco_feats = locomotion_bout_features(
            speed_t, speed_cms, next_loco[0], next_loco[1],
        )

        # Extract pupil features around this transition
        osc_onset_abs = None
        if osc_feats["has_oscillation"] and osc_feats["osc_onset_latency_s"] is not None:
            osc_onset_abs = q_start_s + osc_feats["osc_onset_latency_s"]
        pup_feats = pupil_features_for_transition(
            pup_t, pup_smooth, q_start_s, q_end_s, osc_onset_abs,
            window_s=PUPIL_WINDOW_S,
        )

        row = {
            "Subject": sess.subject,
            "Session": sess.session,
            # Quiescent bout context
            "q_start_s": round(q_start_s, 3),
            "q_end_s": round(q_end_s, 3),
            "q_duration_s": round(q_dur, 3),
            # Transition gap
            "transition_gap_s": round(ibi_s, 3),
            # Oscillation features during quiescence
            **osc_feats,
            # Pupil features
            **pup_feats,
            # Locomotion bout features
            **loco_feats,
            # Post-locomotion: time to next quiescence
            "time_to_next_quiet_s": (
                round(time_to_next_quiet_s, 3)
                if time_to_next_quiet_s is not None else None
            ),
        }
        rows.append(row)
        n_paired += 1

    print(
        f"  {sess.subject}/{sess.session}: "
        f"{len(quiet)} quiescent, {len(loco)} locomotion, "
        f"{n_paired} paired transitions"
    )

# ─── Build DataFrame ─────────────────────────────────────────────────────

df = pd.DataFrame(rows)
df["has_osc_int"] = df["has_oscillation"].astype(int)

subjects = sorted(df["Subject"].unique())
n_subjects = len(subjects)
COLORS = plt.cm.tab10(np.linspace(0, 1, max(n_subjects, 1)))
subject_colors = {s: COLORS[i] for i, s in enumerate(subjects)}

osc_df = df[df["has_oscillation"]].copy()
no_osc_df = df[~df["has_oscillation"]].copy()

# Z-score pupil within each subject for cross-animal comparisons
for col in ["pupil_mean_q_mm", "pupil_at_osc_onset_mm",
            "pupil_early_q_mm", "pupil_late_q_mm"]:
    zcol = col.replace("_mm", "_z")
    df[zcol] = df.groupby("Subject")[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
    )
osc_df = df[df["has_oscillation"]].copy()
no_osc_df = df[~df["has_oscillation"]].copy()

n_with_pupil = df["pupil_mean_q_mm"].notna().sum()

print(f"\nTotal paired transitions: {len(df)}")
print(f"  With prior oscillation: {len(osc_df)}")
print(f"  Without prior oscillation: {len(no_osc_df)}")
print(f"  With pupil data: {n_with_pupil}")

# ─── Locomotion features to compare ─────────────────────────────────────

LOCO_FEATURES = {
    "loco_duration_s": {
        "label": "Bout duration (s)",
        "question": "sustained vs brief movement",
    },
    "loco_peak_speed_cms": {
        "label": "Peak speed (cm/s)",
        "question": "vigor of subsequent locomotion",
    },
    "loco_mean_speed_cms": {
        "label": "Mean speed (cm/s)",
        "question": "vigor of subsequent locomotion",
    },
    "latency_to_peak_speed_s": {
        "label": "Latency to peak speed (s)",
        "question": "ramp-up dynamics (explosive vs gradual)",
    },
}

# ═════════════════════════════════════════════════════════════════════════
# Descriptive statistics: oscillation vs no-oscillation
# ═════════════════════════════════════════════════════════════════════════

print("\n── Locomotion features by prior oscillatory state ──")
print(f"{'Feature':<30} {'Osc (med [IQR])':<30} {'No-osc (med [IQR])':<30} {'MW U p':>10}")

descriptive_rows: list[dict] = []
for feat, info in LOCO_FEATURES.items():
    vals_osc = osc_df[feat].dropna()
    vals_no = no_osc_df[feat].dropna()

    med_osc = vals_osc.median()
    q1_osc, q3_osc = vals_osc.quantile(0.25), vals_osc.quantile(0.75)
    med_no = vals_no.median()
    q1_no, q3_no = vals_no.quantile(0.25), vals_no.quantile(0.75)

    if len(vals_osc) > 0 and len(vals_no) > 0:
        u_stat, u_p = sp_stats.mannwhitneyu(vals_osc, vals_no, alternative="two-sided")
    else:
        u_stat, u_p = np.nan, np.nan

    print(
        f"  {info['label']:<28} "
        f"{med_osc:6.2f} [{q1_osc:.2f}, {q3_osc:.2f}]  "
        f"{med_no:6.2f} [{q1_no:.2f}, {q3_no:.2f}]  "
        f"p = {u_p:.2e}"
    )

    descriptive_rows.append({
        "feature": feat,
        "label": info["label"],
        "osc_n": len(vals_osc),
        "osc_median": round(med_osc, 3),
        "osc_q1": round(q1_osc, 3),
        "osc_q3": round(q3_osc, 3),
        "no_osc_n": len(vals_no),
        "no_osc_median": round(med_no, 3),
        "no_osc_q1": round(q1_no, 3),
        "no_osc_q3": round(q3_no, 3),
        "mann_whitney_U": round(u_stat, 2) if not np.isnan(u_stat) else None,
        "mann_whitney_p": u_p,
    })

desc_df = pd.DataFrame(descriptive_rows)

# ═════════════════════════════════════════════════════════════════════════
# GEE linear models: loco_feature ~ has_oscillation, clustered by animal
# ═════════════════════════════════════════════════════════════════════════

print("\n── GEE linear models: locomotion feature ~ prior oscillation ──")

subj_map = {s: i for i, s in enumerate(subjects)}
df["subj_id"] = df["Subject"].map(subj_map)
df_sorted = df.sort_values("subj_id")

gee_results: dict[str, dict] = {}

for feat, info in LOCO_FEATURES.items():
    r = fit_gee(df_sorted, feat, ["has_osc_int"])
    if r is None:
        print(f"  {info['label']}: insufficient data or fit failure")
        continue
    gee_results[feat] = {"label": info["label"], **r}
    sig = "*" if r["p"] < 0.05 else ""
    print(
        f"  {info['label']:<30} β = {r['coef']:+.3f} "
        f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}], p = {r['p']:.2e} {sig}"
    )

# ═════════════════════════════════════════════════════════════════════════
# GEE with continuous predictors: osc_total_duration and osc_onset_latency
# (among bouts WITH oscillation only)
# ═════════════════════════════════════════════════════════════════════════

print("\n── GEE: locomotion features ~ oscillation duration (osc bouts only) ──")

osc_sorted = osc_df.copy()
osc_sorted["subj_id"] = osc_sorted["Subject"].map(subj_map)
osc_sorted = osc_sorted.sort_values("subj_id")

continuous_gee: dict[str, dict] = {}

for feat, info in LOCO_FEATURES.items():
    for predictor, pred_label in [
        ("osc_total_duration_s", "osc duration"),
        ("osc_onset_latency_s", "osc latency"),
    ]:
        r = fit_gee(osc_sorted, feat, [predictor])
        if r is None:
            continue
        key = f"{feat}__{predictor}"
        continuous_gee[key] = {"outcome": info["label"], "predictor": pred_label, **r}
        sig = "*" if r["p"] < 0.05 else ""
        print(
            f"  {info['label']:<28} ~ {pred_label:<14} "
            f"β = {r['coef']:+.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}], "
            f"p = {r['p']:.2e} {sig}"
        )

# ═════════════════════════════════════════════════════════════════════════
# PUPIL ANALYSIS 1: Pupil at oscillation onset — does early osc = dilated?
# ═════════════════════════════════════════════════════════════════════════

print("\n── Pupil at oscillation onset ──")

osc_with_pupil = osc_df[osc_df["pupil_at_osc_onset_mm"].notna()].copy()
if len(osc_with_pupil) > 10:
    rho_pup_lat, p_pup_lat = sp_stats.spearmanr(
        osc_with_pupil["osc_onset_latency_s"],
        osc_with_pupil["pupil_at_osc_onset_mm"],
    )
    print(f"  Spearman ρ(osc_latency, pupil_at_onset): {rho_pup_lat:.3f}, p = {p_pup_lat:.2e}")
else:
    rho_pup_lat, p_pup_lat = np.nan, np.nan
    print("  Insufficient pupil data at oscillation onset")

# Compare pupil between osc and no-osc transitions
pup_osc_vals = osc_df["pupil_mean_q_mm"].dropna()
pup_no_vals = no_osc_df["pupil_mean_q_mm"].dropna()
if len(pup_osc_vals) > 5 and len(pup_no_vals) > 5:
    u_pup, p_pup = sp_stats.mannwhitneyu(pup_osc_vals, pup_no_vals, alternative="two-sided")
    print(f"  Pupil during quiescence — osc vs no-osc: "
          f"med {pup_osc_vals.median():.3f} vs {pup_no_vals.median():.3f}, "
          f"MW p = {p_pup:.2e}")
else:
    u_pup, p_pup = np.nan, np.nan

# ═════════════════════════════════════════════════════════════════════════
# PUPIL ANALYSIS 2: Mediation — does pupil explain away osc-latency effect?
# osc_latency → locomotion (total effect, already computed)
# osc_latency → locomotion + pupil_mean_q (controlled effect)
# If β_latency shrinks when adding pupil, pupil mediates.
# ═════════════════════════════════════════════════════════════════════════

print("\n── Mediation: osc_latency → locomotion, controlling for pupil ──")

mediation_results: dict[str, dict] = {}

for feat, info in LOCO_FEATURES.items():
    # Model A: loco ~ latency only (total effect)
    r_a = fit_gee(osc_sorted, feat, ["osc_onset_latency_s"], min_n=15)
    if r_a is None:
        print(f"  {info['label']}: model A (latency only) insufficient or failed")
        continue
    # Model B: loco ~ latency + pupil (controlled effect)
    r_b = fit_gee(osc_sorted, feat, ["osc_onset_latency_s", "pupil_mean_q_z"], min_n=15)
    if r_b is None:
        print(f"  {info['label']}: model B (+ pupil) insufficient or failed")
        continue

    beta_total = r_a["coef"]
    beta_ctrl = r_b["coef_osc_onset_latency_s"]
    pct_change = (
        (1 - abs(beta_ctrl) / abs(beta_total)) * 100
        if abs(beta_total) > 1e-8 else 0.0
    )

    mediation_results[feat] = {
        "label": info["label"],
        "beta_total": beta_total,
        "p_total": r_a["p"],
        "beta_controlled": beta_ctrl,
        "p_controlled": r_b["p_osc_onset_latency_s"],
        "beta_pupil": r_b["coef_pupil_mean_q_z"],
        "p_pupil": r_b["p_pupil_mean_q_z"],
        "pct_attenuation": pct_change,
        "n": r_b["n"],
    }

    sig_t = "*" if r_a["p"] < 0.05 else ""
    sig_c = "*" if r_b["p_osc_onset_latency_s"] < 0.05 else ""
    print(
        f"  {info['label']:<28} "
        f"β_total={beta_total:+.4f}{sig_t}  "
        f"β_ctrl={beta_ctrl:+.4f}{sig_c}  "
        f"Δ={pct_change:+.1f}%  "
        f"β_pupil={r_b['coef_pupil_mean_q_z']:+.3f} (p={r_b['p_pupil_mean_q_z']:.2e})"
    )

# ═════════════════════════════════════════════════════════════════════════
# PUPIL ANALYSIS 3: Pupil trajectory — early vs late quiescence by osc group
# ═════════════════════════════════════════════════════════════════════════

print("\n── Pupil trajectory during quiescence ──")

for group_label, group_df in [("with osc", osc_df), ("no osc", no_osc_df)]:
    early = group_df["pupil_early_q_mm"].dropna()
    late = group_df["pupil_late_q_mm"].dropna()
    delta = group_df["pupil_delta_q_mm"].dropna()
    if len(delta) > 3:
        t_stat, t_p = sp_stats.ttest_1samp(delta, 0)
        print(
            f"  {group_label:<10}: early={early.median():.3f}, late={late.median():.3f}, "
            f"Δ={delta.median():.3f} (t={t_stat:.2f}, p={t_p:.2e})"
        )
    else:
        print(f"  {group_label}: insufficient pupil trajectory data")

# ═════════════════════════════════════════════════════════════════════════
# PUPIL ANALYSIS 4: Interaction — osc_latency × pupil_state → locomotion
# Does oscillation timing predict locomotion differently in alert vs drowsy?
# ═════════════════════════════════════════════════════════════════════════

print("\n── Interaction: osc_latency × pupil → locomotion ──")

interaction_results: dict[str, dict] = {}

osc_sorted["latency_x_pupil"] = (
    osc_sorted["osc_onset_latency_s"] * osc_sorted["pupil_mean_q_z"]
)

for feat, info in LOCO_FEATURES.items():
    r = fit_gee(osc_sorted, feat,
                ["osc_onset_latency_s", "pupil_mean_q_z", "latency_x_pupil"],
                min_n=20)
    if r is None:
        continue
    interaction_results[feat] = {
        "label": info["label"],
        "beta_interaction": r["coef_latency_x_pupil"],
        "p_interaction": r["p_latency_x_pupil"],
        "beta_latency": r["coef_osc_onset_latency_s"],
        "p_latency": r["p_osc_onset_latency_s"],
        "beta_pupil": r["coef_pupil_mean_q_z"],
        "p_pupil": r["p_pupil_mean_q_z"],
        "n": r["n"],
    }
    sig = "*" if r["p_latency_x_pupil"] < 0.05 else ""
    print(
        f"  {info['label']:<28} "
        f"β_interact={r['coef_latency_x_pupil']:+.4f} "
        f"(p={r['p_latency_x_pupil']:.2e}) {sig}"
    )

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 1: Violin/box plots — locomotion features by prior oscillation
# ═════════════════════════════════════════════════════════════════════════

n_feats = len(LOCO_FEATURES)
fig1, axes1 = feature_panel(n_feats)

for ax, (feat, info) in zip(axes1, LOCO_FEATURES.items()):
    data_osc = osc_df[feat].dropna()
    data_no = no_osc_df[feat].dropna()
    violin_two_groups(ax, data_no.values, data_osc.values,
                      f"No osc\n(n={len(data_no)})", f"Osc\n(n={len(data_osc)})")
    # Overlay individual animal medians
    for subj in subjects:
        for pos, sdf in [(0, no_osc_df), (1, osc_df)]:
            vals = sdf.loc[sdf["Subject"] == subj, feat].dropna()
            if len(vals) > 0:
                ax.scatter(pos, vals.median(), s=30, color=subject_colors[subj],
                           edgecolors="white", lw=0.5, zorder=5)
    r = gee_results.get(feat)
    title = f"{info['label']}\nβ = {r['coef']:+.2f}, p = {r['p']:.2e}" if r else info["label"]
    ax.set_title(title)
    ax.set_ylabel(info["label"])
    ax.grid(axis="y", alpha=0.3)

save_fig(fig1, proj._context, "locomotion_by_prior_oscillation.svg",
         f"Locomotion features by prior oscillatory state\n"
         f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK} | N = {n_subjects} animals")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 2: Scatter — oscillation duration vs locomotion features
# (among bouts with oscillation)
# ═════════════════════════════════════════════════════════════════════════

fig2, axes2 = feature_panel(n_feats)

for ax, (feat, info) in zip(axes2, LOCO_FEATURES.items()):
    scatter_by_subject(ax, osc_df, "osc_total_duration_s", feat, subjects, subject_colors)
    key = f"{feat}__osc_total_duration_s"
    r = continuous_gee.get(key)
    title = f"{info['label']}\nβ = {r['coef']:+.3f}, p = {r['p']:.2e}" if r else info["label"]
    ax.set_title(title)
    ax.set_xlabel("Oscillation duration (s)")
    ax.set_ylabel(info["label"])
    ax.grid(alpha=0.3)

axes2[0].legend(frameon=False, markerscale=1.5)
save_fig(fig2, proj._context, "osc_duration_vs_locomotion.svg",
         f"Oscillation duration → locomotion features (osc bouts only)\n"
         f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | N = {n_subjects} animals")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 3: GEE coefficient forest plot (binary predictor: osc vs no-osc)
# ═════════════════════════════════════════════════════════════════════════

if gee_results:
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    feat_keys = list(gee_results.keys())
    y_pos = np.arange(len(feat_keys))

    for i, feat in enumerate(feat_keys):
        r = gee_results[feat]
        # Standardize coefficient for comparable effect sizes
        feat_std = df[feat].std()
        if feat_std > 0:
            std_coef = r["coef"] / feat_std
            std_ci_lo = r["ci_lo"] / feat_std
            std_ci_hi = r["ci_hi"] / feat_std
        else:
            std_coef = r["coef"]
            std_ci_lo = r["ci_lo"]
            std_ci_hi = r["ci_hi"]

        color = "#DD8452" if r["p"] < 0.05 else "#4C72B0"
        ax3.errorbar(
            std_coef, i,
            xerr=[[std_coef - std_ci_lo], [std_ci_hi - std_coef]],
            fmt="o", color=color, ms=8, capsize=5, lw=1.5,
        )
        ax3.annotate(
            f"p = {r['p']:.2e}", (std_ci_hi + 0.02, i),
            fontsize=8, va="center",
        )

    ax3.axvline(0, color="red", ls=":", lw=1, alpha=0.6, label="no effect")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([gee_results[f]["label"] for f in feat_keys])
    ax3.set_xlabel("Standardized β (oscillation effect / feature SD)")
    ax3.set_title(
        f"GEE: locomotion feature ~ prior oscillation (clustered by animal)\n"
        f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | N = {n_subjects} animals",
    )
    ax3.legend(frameon=False)
    ax3.grid(axis="x", alpha=0.3)
    save_fig(fig3, proj._context, "gee_forest_plot.svg")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 4: Per-animal effect direction (consistency check)
# ═════════════════════════════════════════════════════════════════════════

fig4, axes4 = feature_panel(n_feats)

for ax, (feat, info) in zip(axes4, LOCO_FEATURES.items()):
    subj_effects: list[tuple] = []  # (subj, diff, n_osc, n_no)
    for subj in subjects:
        s_osc = osc_df.loc[osc_df["Subject"] == subj, feat].dropna()
        s_no = no_osc_df.loc[no_osc_df["Subject"] == subj, feat].dropna()
        if len(s_osc) >= 3 and len(s_no) >= 3:
            diff = s_osc.median() - s_no.median()
            subj_effects.append((subj, diff, len(s_osc), len(s_no)))

    if not subj_effects:
        ax.set_title(f"{info['label']}\ninsufficient per-animal data")
        continue

    y_pos = np.arange(len(subj_effects))
    for i, (subj, diff, n_o, n_n) in enumerate(subj_effects):
        ax.barh(i, diff, color=subject_colors[subj], height=0.6, alpha=0.7)
        ax.text(
            diff + (0.02 * np.sign(diff) if diff != 0 else 0.02), i,
            f"n={n_o}/{n_n}", fontsize=7, va="center",
        )

    ax.axvline(0, color="red", ls=":", lw=1, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[0] for s in subj_effects])
    ax.set_xlabel(f"Δ {info['label']} (osc − no-osc)")
    ax.grid(axis="x", alpha=0.3)

    signs = [np.sign(s[1]) for s in subj_effects if s[1] != 0]
    if signs:
        consistent = all(s == signs[0] for s in signs)
        direction = "positive" if signs[0] > 0 else "negative"
        ax.set_title(
            f"{info['label']}\nconsistent={consistent} ({direction})",
        )
    else:
        ax.set_title(info["label"])

save_fig(fig4, proj._context, "per_animal_effect_direction.svg",
         f"Per-animal median difference (osc − no-osc)\n"
         f"N = {n_subjects} animals")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 5: Pupil at oscillation onset vs latency + pupil osc vs no-osc
# ═════════════════════════════════════════════════════════════════════════

fig5, axes5 = plt.subplots(1, 3, figsize=(15, 5))

# 5a: Scatter — pupil at osc onset vs oscillation latency
ax = axes5[0]
scatter_by_subject(ax, osc_with_pupil, "osc_onset_latency_s", "pupil_at_osc_onset_mm",
                   subjects, subject_colors)
if not np.isnan(rho_pup_lat):
    ax.set_title(
        f"Pupil at oscillation onset\n"
        f"ρ = {rho_pup_lat:.3f}, p = {p_pup_lat:.2e}",
    )
else:
    ax.set_title("Pupil at oscillation onset")
ax.set_xlabel("Oscillation onset latency (s)")
ax.set_ylabel("Pupil diameter (mm)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)

# 5b: Violin — pupil during quiescence: osc vs no-osc
ax = axes5[1]
if len(pup_no_vals) > 0 and len(pup_osc_vals) > 0:
    violin_two_groups(ax, pup_no_vals.values, pup_osc_vals.values,
                      f"No osc\n(n={len(pup_no_vals)})",
                      f"Osc\n(n={len(pup_osc_vals)})")
    title_5b = f"Mean pupil during quiescence\nMW p = {p_pup:.2e}" if not np.isnan(p_pup) else "Mean pupil during quiescence"
    ax.set_title(title_5b)
else:
    ax.set_title("Mean pupil during quiescence\n(insufficient data)")
ax.set_ylabel("Pupil diameter (mm)")
ax.grid(axis="y", alpha=0.3)

# 5c: Pupil trajectory — early vs late quiescence by osc group
ax = axes5[2]
for i, (label, gdf, color) in enumerate([
    ("No osc", no_osc_df, "#4C72B0"),
    ("Osc", osc_df, "#DD8452"),
]):
    early = gdf["pupil_early_q_mm"].dropna()
    late = gdf["pupil_late_q_mm"].dropna()
    if len(early) > 3 and len(late) > 3:
        ax.errorbar(
            [i - 0.15, i + 0.15],
            [early.median(), late.median()],
            yerr=[
                [early.median() - early.quantile(0.25), late.median() - late.quantile(0.25)],
                [early.quantile(0.75) - early.median(), late.quantile(0.75) - late.median()],
            ],
            fmt="o-", color=color, ms=8, capsize=4, lw=2,
            label=f"{label} (n={len(early)})",
        )
ax.set_xticks([0, 1])
ax.set_xticklabels(["No osc", "Osc"])
# Add secondary xtick labels
ax.text(-0.15, ax.get_ylim()[0] - 0.02, "early", ha="center", fontsize=7, color="gray")
ax.text(0.15, ax.get_ylim()[0] - 0.02, "late", ha="center", fontsize=7, color="gray")
ax.text(0.85, ax.get_ylim()[0] - 0.02, "early", ha="center", fontsize=7, color="gray")
ax.text(1.15, ax.get_ylim()[0] - 0.02, "late", ha="center", fontsize=7, color="gray")
ax.set_ylabel("Pupil diameter (mm)")
ax.set_title(f"Pupil trajectory: first vs last {PUPIL_WINDOW_S:.0f}s of quiescence")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)

save_fig(fig5, proj._context, "pupil_arousal_measures.svg",
         f"Pupil arousal measures\n"
         f"{ROI_NAME} | {BAND[0]}–{BAND[1]} Hz | {TASK} | N = {n_subjects} animals")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 6: Mediation comparison — β attenuation when adding pupil
# ═════════════════════════════════════════════════════════════════════════

if mediation_results:
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    feat_keys_med = list(mediation_results.keys())
    y_pos = np.arange(len(feat_keys_med))
    bar_h = 0.3

    for i, feat in enumerate(feat_keys_med):
        r = mediation_results[feat]
        ax6.barh(
            i - bar_h / 2, r["beta_total"], height=bar_h,
            color="#4C72B0", alpha=0.7, label="Total" if i == 0 else "",
        )
        ax6.barh(
            i + bar_h / 2, r["beta_controlled"], height=bar_h,
            color="#DD8452", alpha=0.7, label="Controlled (+ pupil)" if i == 0 else "",
        )
        ax6.text(
            max(abs(r["beta_total"]), abs(r["beta_controlled"])) * 1.1 * np.sign(r["beta_total"]),
            i,
            f"Δ = {r['pct_attenuation']:.0f}%",
            fontsize=8, va="center",
        )

    ax6.axvline(0, color="red", ls=":", lw=1, alpha=0.6)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels([mediation_results[f]["label"] for f in feat_keys_med])
    ax6.set_xlabel("β (osc_latency effect)")
    ax6.set_title(
        f"Mediation check: osc_latency → locomotion\n"
        f"Total effect vs controlled for pupil (z-scored)",
    )
    ax6.legend(frameon=False)
    ax6.grid(axis="x", alpha=0.3)
    save_fig(fig6, proj._context, "mediation_pupil_check.svg")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 7: Interaction — osc latency effect split by pupil tertile
# ═════════════════════════════════════════════════════════════════════════

osc_with_pup_z = osc_df[osc_df["pupil_mean_q_z"].notna()].copy()
if len(osc_with_pup_z) > 30:
    pup_terciles = osc_with_pup_z["pupil_mean_q_z"].quantile([1/3, 2/3])
    osc_with_pup_z["pupil_tercile"] = pd.cut(
        osc_with_pup_z["pupil_mean_q_z"],
        bins=[-np.inf, pup_terciles.iloc[0], pup_terciles.iloc[1], np.inf],
        labels=["constricted\n(drowsy)", "mid", "dilated\n(alert)"],
    )

    fig7, axes7 = feature_panel(n_feats)

    terc_colors = ["#4C72B0", "#55A868", "#DD8452"]
    terc_labels = ["constricted\n(drowsy)", "mid", "dilated\n(alert)"]

    for ax, (feat, info) in zip(axes7, LOCO_FEATURES.items()):
        for j, terc in enumerate(terc_labels):
            sub = osc_with_pup_z[osc_with_pup_z["pupil_tercile"] == terc]
            if len(sub) < 5:
                continue
            ax.scatter(
                sub["osc_onset_latency_s"], sub[feat],
                s=15, alpha=0.4, color=terc_colors[j],
                label=f"{terc} (n={len(sub)})", edgecolors="none",
            )
            # Trend line
            valid = sub[["osc_onset_latency_s", feat]].dropna()
            if len(valid) > 5:
                z = np.polyfit(valid["osc_onset_latency_s"], valid[feat], 1)
                x_line = np.linspace(valid["osc_onset_latency_s"].min(),
                                     valid["osc_onset_latency_s"].max(), 50)
                ax.plot(x_line, np.polyval(z, x_line), color=terc_colors[j], lw=1.5, alpha=0.7)

        ax.set_xlabel("Oscillation onset latency (s)")
        ax.set_ylabel(info["label"])

        if feat in interaction_results:
            r = interaction_results[feat]
            ax.set_title(
                f"{info['label']}\ninteraction p = {r['p_interaction']:.2e}",
            )
        else:
            ax.set_title(info["label"])
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)

    save_fig(fig7, proj._context, "interaction_pupil_tercile.svg",
             f"Osc latency → locomotion, split by pupil arousal tercile\n"
             f"{ROI_NAME} | N = {n_subjects} animals")

# ═════════════════════════════════════════════════════════════════════════
# Save tables
# ═════════════════════════════════════════════════════════════════════════

stats_dir = proj._context.stats_dir
stats_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(stats_dir / "quiescent_to_locomotion_transitions.csv", index=False)
desc_df.to_csv(stats_dir / "descriptive_comparisons.csv", index=False)
save_dict_csv(gee_results, stats_dir / "gee_binary_results.csv")
save_dict_csv(continuous_gee, stats_dir / "gee_continuous_results.csv")
save_dict_csv(mediation_results, stats_dir / "mediation_results.csv")
save_dict_csv(interaction_results, stats_dir / "interaction_results.csv")

# ═════════════════════════════════════════════════════════════════════════
# Markdown report
# ═════════════════════════════════════════════════════════════════════════

# Build GEE table rows
gee_table_lines = []
for feat, r in gee_results.items():
    sig = "**" if r["p"] < 0.05 else ""
    gee_table_lines.append(
        f"| {r['label']} | {r['coef']:+.3f} | {r['se']:.3f} | "
        f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {sig}{r['p']:.2e}{sig} | {r['n']} |"
    )
gee_table = "\n".join(gee_table_lines) if gee_table_lines else "| (no results) |"

# Build continuous GEE table rows
cont_table_lines = []
for key, r in continuous_gee.items():
    sig = "**" if r["p"] < 0.05 else ""
    cont_table_lines.append(
        f"| {r['outcome']} | {r['predictor']} | {r['coef']:+.4f} | "
        f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {sig}{r['p']:.2e}{sig} | {r['n']} |"
    )
cont_table = "\n".join(cont_table_lines) if cont_table_lines else "| (no results) |"

# Descriptive table
desc_table_lines = []
for _, row in desc_df.iterrows():
    desc_table_lines.append(
        f"| {row['label']} | "
        f"{row['osc_median']:.2f} [{row['osc_q1']:.2f}, {row['osc_q3']:.2f}] | "
        f"{row['no_osc_median']:.2f} [{row['no_osc_q1']:.2f}, {row['no_osc_q3']:.2f}] | "
        f"{row['mann_whitney_p']:.2e} |"
    )
desc_table = "\n".join(desc_table_lines)

report_path = proj.save_report(
    notes=(
        f"## Oscillation → locomotion prediction\n\n"
        f"**Question:** Does prior oscillatory state during quiescence predict "
        f"features of the subsequent locomotion bout?\n\n"
        f"**Dataset:** ETOH-HFSA | **Task:** {TASK}\n\n"
        f"**ROI:** {ROI_NAME} | **Band:** {BAND[0]}–{BAND[1]} Hz\n\n"
        f"**Oscillation:** threshold={THRESHOLD} (ΔF/F envelope), "
        f"min_duration={MIN_DURATION_OSC}s\n\n"
        f"**Min quiescent bout:** {MIN_QUIESCENT_S}s\n\n"
        f"---\n\n"
        f"### Data summary\n\n"
        f"- **Paired transitions (quiescence → locomotion):** {len(df)}\n"
        f"- **With prior oscillation:** {len(osc_df)}\n"
        f"- **Without prior oscillation:** {len(no_osc_df)}\n"
        f"- **N animals:** {n_subjects}\n\n"
        f"---\n\n"
        f"### 1. Descriptive comparison (naive — treats bouts as independent)\n\n"
        f"| Feature | Osc median [IQR] | No-osc median [IQR] | MW U p |\n"
        f"|---------|----------------:|-------------------:|-------:|\n"
        f"{desc_table}\n\n"
        f"*Mann-Whitney p-values are anticonservative due to pseudoreplication. "
        f"Use GEE results for inference.*\n\n"
        f"---\n\n"
        f"### 2. GEE linear models: locomotion feature ~ prior oscillation\n\n"
        f"Clustered by animal (exchangeable correlation, robust SE). "
        f"β = difference in outcome for osc vs no-osc transitions.\n\n"
        f"| Outcome | β (osc effect) | Robust SE | 95% CI | p | N |\n"
        f"|---------|---------------:|----------:|-------:|--:|--:|\n"
        f"{gee_table}\n\n"
        f"---\n\n"
        f"### 3. Continuous predictors (osc bouts only)\n\n"
        f"Among transitions with prior oscillation: does oscillation duration "
        f"or onset latency predict locomotion vigor?\n\n"
        f"| Outcome | Predictor | β | 95% CI | p | N |\n"
        f"|---------|-----------|--:|-------:|--:|--:|\n"
        f"{cont_table}\n\n"
        f"---\n\n"
        f"### 4. Pupil arousal analysis\n\n"
        f"**Pupil data available for {n_with_pupil}/{len(df)} transitions.**\n\n"
        f"#### 4a. Pupil at oscillation onset\n\n"
        f"Spearman ρ(osc_latency, pupil_at_onset) = {rho_pup_lat:.3f} "
        f"(p = {p_pup_lat:.2e})\n\n"
        f"Mean pupil during quiescence — osc vs no-osc MW p = {p_pup:.2e}\n\n"
        + (
            f"#### 4b. Mediation: osc_latency → locomotion, controlling for pupil\n\n"
            f"If β attenuates substantially when adding pupil to the model, "
            f"the oscillation-timing effect is partly a proxy for arousal.\n\n"
            f"| Outcome | β_total (latency only) | β_controlled (+pupil) | Δ% | β_pupil | p_pupil |\n"
            f"|---------|----------------------:|---------------------:|---:|--------:|--------:|\n"
            + "\n".join(
                f"| {r['label']} | {r['beta_total']:+.4f} | {r['beta_controlled']:+.4f} | "
                f"{r['pct_attenuation']:+.0f}% | {r['beta_pupil']:+.3f} | {r['p_pupil']:.2e} |"
                for r in mediation_results.values()
            )
            + "\n\n"
            if mediation_results else ""
        )
        + (
            f"#### 4c. Interaction: osc_latency × pupil → locomotion\n\n"
            f"Does oscillation timing predict locomotion differently in alert vs drowsy states?\n\n"
            f"| Outcome | β_interaction | p_interaction |\n"
            f"|---------|-------------:|-------------:|\n"
            + "\n".join(
                f"| {r['label']} | {r['beta_interaction']:+.4f} | {r['p_interaction']:.2e} |"
                for r in interaction_results.values()
            )
            + "\n\n"
            if interaction_results else ""
        )
        + f"---\n\n"
        f"### Interpretation\n\n"
        f"- **Arousal proxy:** If mediation attenuation is large (>30%) and "
        f"β_pupil is significant, oscillation timing was mainly indexing arousal state.\n"
        f"- **Independent signal:** If β_latency persists after controlling for pupil, "
        f"oscillation timing carries information beyond arousal.\n"
        f"- **Context-dependent:** Significant interaction means oscillation timing "
        f"predicts locomotion differently in alert vs drowsy contexts.\n\n"
        f"**Animals:** {', '.join(subjects)}\n"
    ),
)
print(f"\nReport: {report_path}")
print(f"Outputs saved to: {proj._context.run_dir}")
