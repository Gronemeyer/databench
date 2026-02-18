#%%
import numpy as np
import pandas as pd

# Global params (edit in place)
SOURCE_FEATURES = [
    ("suite2p", ["deltaf_f"], [0, 1, 2]),
    ("encoder", ["speed_mm"]),
    ("pupil", ["pupil_diameter_mm"]),
    #("psychopy", [''])
]
TOL = 0.25
TIME_COLUMN = "time_elapsed_s"
REFERENCE_SOURCE = None


def build_long(df, source_features, tol=TOL, time_column=TIME_COLUMN, reference_source=REFERENCE_SOURCE):
    # Normalize source_features to (source, features, indices)
    source_features_list = []
    for entry in source_features:
        source = entry[0]
        features = list(entry[1])
        indices = entry[2] if len(entry) > 2 else None
        source_features_list.append((source, features, indices))

    # Resolve reference source
    ref_idx = 0
    if reference_source is not None:
        for i, (source, _, _) in enumerate(source_features_list):
            if source == reference_source:
                ref_idx = i
                break

    ref_source, ref_features, ref_indices = source_features_list[ref_idx]
    merge_sources = [entry for i, entry in enumerate(source_features_list) if i != ref_idx]

    # Normalize indices
    if ref_indices is None:
        ref_index_list = []
    elif isinstance(ref_indices, (list, tuple, np.ndarray)):
        ref_index_list = list(ref_indices)
    else:
        ref_index_list = [ref_indices]

    frames = []

    for idx, row in df.iterrows():
        # Build reference frame
        if ref_indices is None:
            t = row.get((ref_source, time_column))
            if t is None:
                continue
            t_arr = np.atleast_1d(np.asarray(t)).astype(float, copy=False)
            data = {time_column: t_arr}
            for feature_name in ref_features:
                x = row.get((ref_source, feature_name))
                data[feature_name] = np.atleast_1d(np.asarray(x))
            out = pd.DataFrame(data)
        else:
            roi_frames = []
            base_time = None
            for ref_index in ref_index_list:
                t = row.get((ref_source, time_column))
                if t is None:
                    continue
                t_arr = np.atleast_1d(np.asarray(t)).astype(float, copy=False)
                if np.asarray(t_arr).ndim > 1:
                    t_arr = t_arr[ref_index]

                data = {time_column: t_arr}
                for feature_name in ref_features:
                    x = row.get((ref_source, feature_name))
                    arr = np.asarray(x)
                    if arr.ndim > 1:
                        arr = arr[ref_index]
                    data[feature_name] = np.atleast_1d(arr)

                roi_df = pd.DataFrame(data)

                if base_time is None:
                    base_time = roi_df[time_column].to_numpy()
                elif not np.array_equal(roi_df[time_column].to_numpy(), base_time):
                    raise ValueError(
                        f"Source {ref_source!r} ROI timebases differ; cannot align per-ROI columns."
                    )

                rename = {feature_name: f"{feature_name}_roi{ref_index}" for feature_name in ref_features}
                roi_frames.append(roi_df.rename(columns=rename))

            if base_time is None:
                continue

            out = pd.concat(
                [roi_frames[0][[time_column]]] + [frame.drop(columns=[time_column]) for frame in roi_frames],
                axis=1,
            )

        out = out.sort_values(time_column)

        # Merge other sources
        for source, features, _ in merge_sources:
            t = row.get((source, time_column))
            if t is None:
                for feature_name in features:
                    out[feature_name] = np.nan
                continue

            t_arr = np.atleast_1d(np.asarray(t)).astype(float, copy=False)
            data = {time_column: t_arr}
            for feature_name in features:
                x = row.get((source, feature_name))
                data[feature_name] = np.atleast_1d(np.asarray(x))

            ts = pd.DataFrame(data).dropna(subset=[time_column])
            out = pd.merge_asof(
                out,
                ts.sort_values(time_column),
                on=time_column,
                direction="nearest",
                tolerance=tol,
            )

        subj, ses, task = idx
        out.insert(0, "Task", task)
        out.insert(0, "Session", ses)
        out.insert(0, "Subject", subj)

        frames.append(out)

    return pd.concat(frames, ignore_index=True)



# Example:
df = pd.read_pickle("/Users/jakegronemeyer/Desktop/4jake/260212_ACUTEVIS_dataset.pkl")
long_df = build_long(df, SOURCE_FEATURES)

# Add Boolean mask for grating vs gray epochs

long_df["is_gratings"] = False
long_df["is_gray"] = False

for (subject, session, task), _row in df.iterrows():
    if task != "task-gratings":
        continue

    key = (subject, session, task)
    gratings = df.psychopy.gratings_gratings_windows[key]
    gray = df.psychopy.gratings_gray_windows[key]

    mask = (
        (long_df["Subject"] == subject)
        & (long_df["Session"] == session)
        & (long_df["Task"] == task)
    )
    t = long_df.loc[mask, "time_elapsed_s"].to_numpy()

    gratings_intervals = pd.IntervalIndex.from_tuples(gratings, closed="left")
    gratings_idx = gratings_intervals.get_indexer(t)
    long_df.loc[mask, "is_gratings"] = gratings_idx >= 0

    gray_intervals = pd.IntervalIndex.from_tuples(gray, closed="left")
    gray_idx = gray_intervals.get_indexer(t)
    long_df.loc[mask, "is_gray"] = gray_idx >= 0
# %%
