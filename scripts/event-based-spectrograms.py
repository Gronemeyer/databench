#%%
# event-based-spectrograms.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from databench import Bench
from databench.features.treadmill import locomotion_bout_events, extract_epoch_interpolated


#%%
#pickle = Path(r"/Volumes/ake.bin/4jake/260211_ETOH_dataset.pkl")
pickle = Path(r"D:\4jake\260211_ETOH_dataset.pkl")

bench = Bench()
(bench
    .setup(pickle, analyst="Jacob Gronemeyer", lab="Sipe Lab", run_name="sandbox", tag="event-based")
    .load()
    .filter(drop_rows=(("GS27", "ses-02", "task-spont"),)))

paths = bench.output_paths
df = bench.df

#%%

mesomap_features = [
    feature
    for source, feature in df.columns
    if source == 'mesomap' and feature != 'time_elapsed_s'
]
source_features = [
    ('mesomap', mesomap_features),
    ('pupil', ['pupil_diameter_mm']),
    ('treadmill', ['speed_mm']),
]
bench.build_long(
    df,
    source_features=source_features,
    tol=0.25,
    time_column="time_elapsed_s",
    reference_source="mesomap",
)
long = bench.long

ses_to_cond = {
    'ses-01': 'baseline',
    'ses-02': 'saline',
    'ses-03': 'ethanol_low',
    'ses-04': 'ethanol_high',
}

long['Condition'] = long['Session'].map(ses_to_cond)

long_subset = long[['Subject', 'Session', 'Task', 'time_elapsed_s', 'L_VISp', 'R_VISp', 'pupil_diameter_mm', 'speed_mm', 'Condition']]
#long_subset.to_csv(paths['output'] / 'long_subset.csv', index=False)



#%% Spectrogram (STFT) + pre-event baseline normalization
'''
Baseline: compute mean power during a pre-event interval (e.g. (-2, -1)), for each frequency, then express as dB change.
'''

def stft_spectrogram(y, fs, nperseg, noverlap):
    """
    Returns f, t, P where P is power (freq x time).
    Tries scipy.signal.stft; falls back to numpy.
    """
    y = np.asarray(y, float)

    # Replace NaNs (STFT can’t handle). Here: simple linear fill; could also drop events with many NaNs.
    if np.isnan(y).any():
        s = pd.Series(y).interpolate(limit_direction='both')
        y = s.to_numpy()

    try:
        from scipy.signal import stft
        f, tt, Z = stft(y, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
                        boundary=None, padded=False)
        P = (np.abs(Z) ** 2)
        return f, tt, P
    except Exception:
        # Numpy fallback: sliding FFT (less feature-rich but fine)
        step = nperseg - noverlap
        nwin = 1 + (len(y) - nperseg) // step
        if nwin <= 0:
            return None, None, None
        win = np.hanning(nperseg)
        f = np.fft.rfftfreq(nperseg, d=1/fs)
        P = np.empty((len(f), nwin), float)
        tt = np.empty(nwin, float)
        for i in range(nwin):
            start = i * step
            seg = y[start:start+nperseg] * win
            spec = np.fft.rfft(seg)
            P[:, i] = (np.abs(spec) ** 2)
            tt[i] = (start + nperseg/2) / fs
        return f, tt, P


def baseline_normalize_db(P, t_rel, baseline=(-2.0, -1.0), eps=1e-12):
    """
    P: power (freq x timebins), t_rel: STFT time in seconds relative to epoch start (0..window)
    baseline window is relative to event time; we’ll map to rel-time in plotting step.
    """
    bmask = (t_rel >= baseline[0]) & (t_rel <= baseline[1])
    if not bmask.any():
        return 10 * np.log10(P + eps)

    base = np.mean(P[:, bmask], axis=1, keepdims=True)  # freq x 1
    return 10 * np.log10((P + eps) / (base + eps))


#%% Compute condition-wise mean spectrogram across common ROIs
from collections import defaultdict
import numpy as np

def spectrogram_subject_roi_by_condition(
    long,
    events,
    roi_cols=None,
    task="task-movies",
    window=(-2.0, 1.0),
    dt=0.02,                 # 50 Hz sampling
    baseline=(-2.0, -1.0),
    event_time_col="onset_t",
    stft_win_s=0.5,
    stft_overlap=0.75,
    max_freq=20.0,
    min_events_per_session=1,
    require_common_grid=True
):
    """
    Compute event-locked spectrograms with hierarchical averaging while retaining ROI.

    Returns
    -------
    results : dict
        {
          "f": f,
          "t": t_rel,
          "subject_arrays": {cond: {roi: (subjects, f, t)}},
          "group_mean": {cond: {roi: (f, t, mean)}},
          "group_sem":  {cond: {roi: (f, t, sem)}},
          "n_subjects": {cond: {roi: int}},
          "n_sessions": {cond: {roi: int}},
        }
    """
    required_event_cols = {"Subject", "Session", "Task", event_time_col}
    missing_event_cols = required_event_cols.difference(events.columns)
    if missing_event_cols:
        missing = ", ".join(sorted(missing_event_cols))
        raise ValueError(f"events table missing required columns: {missing}")

    exclude = {
        "time_elapsed_s",
        "speed_mm",
        "distance_mm",
        "pupil_diameter_mm",
        "is_running",
        "Subject",
        "Session",
        "Task",
        "Condition",
    }
    all_roi_cols = [
        c for c in long.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(long[c])
    ]
    use_roi_cols = all_roi_cols if roi_cols is None else [c for c in roi_cols if c in all_roi_cols]
    if not use_roi_cols:
        raise ValueError("No ROI columns available after applying roi_cols filter.")

    fs = 1.0 / dt

    nperseg = int(round(stft_win_s * fs))       # e.g. 0.5 s @ 50 Hz => 25 samples
    nperseg = max(8, nperseg)                  # allow shorter windows if requested
    noverlap = int(round(stft_overlap * nperseg))
    noverlap = min(noverlap, nperseg - 1)

    events_use = events.loc[events["Task"] == task].copy()
    event_map = {
        key: grp[event_time_col].to_numpy()
        for key, grp in events_use.groupby(["Subject", "Session", "Task"], sort=False)
    }

    # session-level storage: (subj, cond, roi) -> list of session means
    sess_means = defaultdict(list)
    sess_counts = defaultdict(int)

    # reference grids per condition (optional strictness)
    f_ref = {}
    t_ref = {}

    for (subj, ses, task_name), g in long.groupby(["Subject", "Session", "Task"], sort=False):
        if task_name != task:
            continue

        cond = g["Condition"].iloc[0] if "Condition" in g.columns else ses

        g = g.sort_values("time_elapsed_s")
        t_raw = g["time_elapsed_s"].to_numpy()
        onsets = event_map.get((subj, ses, task_name), np.array([], dtype=float))
        if onsets.size == 0:
            continue

        for roi in use_roi_cols:
            P_events = []
            f_use = t_use = None

            for t0 in onsets:
                y_raw = g[roi].to_numpy()
                rel_t, y = extract_epoch_interpolated(
                    t_raw,
                    y_raw,
                    t0,
                    window=window,
                    dt=dt,
                )
                if y is None:
                    continue

                f, tt, P = stft_spectrogram(y, fs=fs, nperseg=nperseg, noverlap=noverlap)
                if P is None:
                    continue

                t_rel = window[0] + tt

                if max_freq is not None:
                    m = f <= max_freq
                    f, P = f[m], P[m, :]

                P_db = baseline_normalize_db(P, t_rel, baseline=baseline)
                if P_db is None:
                    continue

                P_events.append(P_db)
                f_use, t_use = f, t_rel

            if len(P_events) < min_events_per_session:
                continue

            P_mean_session = np.mean(np.stack(P_events, axis=0), axis=0)  # (f, t)

            # Enforce common TF grid if requested
            if require_common_grid:
                if cond not in f_ref:
                    f_ref[cond] = f_use
                    t_ref[cond] = t_use
                else:
                    if (len(f_use) != len(f_ref[cond])) or (len(t_use) != len(t_ref[cond])) \
                       or (np.max(np.abs(f_use - f_ref[cond])) > 1e-9) \
                       or (np.max(np.abs(t_use - t_ref[cond])) > 1e-9):
                        continue

            sess_means[(subj, cond, roi)].append(P_mean_session)
            sess_counts[(cond, roi)] += 1

    # subject-level averaging: (cond, roi) -> list of subject means (one per subject)
    subj_arrays = defaultdict(list)
    subj_ids = defaultdict(list)

    for (subj, cond, roi), arr_list in sess_means.items():
        P_subj = np.mean(np.stack(arr_list, axis=0), axis=0)  # average sessions within subject
        subj_arrays[(cond, roi)].append(P_subj)
        subj_ids[(cond, roi)].append(subj)

    # group summaries
    group_mean = defaultdict(dict)
    group_sem = defaultdict(dict)
    n_subjects = defaultdict(dict)
    n_sessions = defaultdict(dict)

    # choose a single grid (if multiple conds have different grids and require_common_grid=False,
    # you should interpolate; here we assume common grid per condition)
    for (cond, roi), mats in subj_arrays.items():
        X = np.stack(mats, axis=0)  # (n_subj, f, t)
        mu = np.mean(X, axis=0)
        sem = np.std(X, axis=0, ddof=1) / np.sqrt(X.shape[0]) if X.shape[0] > 1 else np.zeros_like(mu)

        f = f_ref.get(cond, None)
        t = t_ref.get(cond, None)

        group_mean[cond][roi] = (f, t, mu)
        group_sem[cond][roi] = (f, t, sem)
        n_subjects[cond][roi] = X.shape[0]
        n_sessions[cond][roi] = sess_counts.get((cond, roi), 0)

    # package subject arrays with ids for downstream stats
    subject_arrays = defaultdict(dict)
    for (cond, roi), mats in subj_arrays.items():
        X = np.stack(mats, axis=0)
        subject_arrays[cond][roi] = (subj_ids[(cond, roi)], f_ref.get(cond, None), t_ref.get(cond, None), X)

    return {
        "subject_arrays": dict(subject_arrays),
        "group_mean": dict(group_mean),
        "group_sem": dict(group_sem),
        "n_subjects": dict(n_subjects),
        "n_sessions": dict(n_sessions),
    }

bouts_feature = bench.get_feature("locomotion_bouts_n")
events = locomotion_bout_events(
    long,
    min_speed_cms=bouts_feature.min_speed_cms,
    min_duration_s=bouts_feature.min_duration_s,
    merge_gap_s=bouts_feature.merge_gap_s,
    as_table=True,
    group_cols=("Subject", "Session", "Task"),
    time_col="time_elapsed_s",
    speed_col="speed_mm",
    speed_scale_to_cms=10.0,
)

res = spectrogram_subject_roi_by_condition(
    long,
    events,
    task="task-spont",
    window=(-2.0, 5.0),
    dt=0.02,                 # 50 Hz
    baseline=(-2.0, -1.0),
    stft_win_s=1,
    stft_overlap=0.75,
    max_freq=20.0,
    min_events_per_session=2
)

#%% import matplotlib.pyplot as plt


def plot_roi_condition_grid(
    res,
    roi_order=None,
    cond_order=None,
    vmin=-3,
    vmax=3,
    cmap="RdBu_r",
    baseline=(-2.0, -1.0),
    bouts_feature=None,
):
    group_mean = res["group_mean"]
    n_subjects = res["n_subjects"]

    conds = cond_order or sorted(group_mean.keys())
    rois = roi_order or sorted({r for c in conds for r in group_mean[c].keys()})

    fig, axes = plt.subplots(len(rois), len(conds), figsize=(4*len(conds), 2.8*len(rois)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    im = None
    for i, roi in enumerate(rois):
        for j, cond in enumerate(conds):
            ax = axes[i, j]
            if roi not in group_mean.get(cond, {}):
                ax.axis("off")
                continue

            f, t, S = group_mean[cond][roi]
            im = ax.imshow(
                S, aspect="auto", origin="lower",
                extent=[t.min(), t.max(), f.min(), f.max()],
                vmin=vmin, vmax=vmax, cmap=cmap
            )
            ax.axvline(0, color="k", lw=1)

            if i == 0:
                ax.set_title(f"{cond}\n(n={n_subjects.get(cond, {}).get(roi, 0)})")
            if j == 0:
                ax.set_ylabel(f"{roi}\nFreq (Hz)")
            if i == len(rois) - 1:
                ax.set_xlabel("Time relative to bout onset (s)")

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, label="dB vs baseline")
    if bouts_feature is not None:
        bout_text = (
            f"min_speed={bouts_feature.min_speed_cms:g} cm/s, "
            f"min_dur={bouts_feature.min_duration_s:g} s, "
            f"merge_gap={bouts_feature.merge_gap_s:g} s"
        )
    else:
        bout_text = ""
    base_text = f"baseline [{baseline[0]:g}, {baseline[1]:g}] s"
    title_text = "Locomotion bout-locked spectrograms"
    if bout_text:
        title_text = f"{title_text}\n{base_text}; {bout_text}"
    else:
        title_text = f"{title_text}\n{base_text}"
    fig.suptitle(title_text, y=0.995)
    plt.plot(rect=[0, 0, 1, 0.97])
    return fig


plot_roi_condition_grid(
    res,
    baseline=(-2.0, -1.0),
    bouts_feature=bouts_feature,
)

bench.save_provenance()
# %%
