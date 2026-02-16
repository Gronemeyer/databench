#%%
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from databench import Bench


#%%
pickle = Path(r"/Volumes/ake.bin/4jake/260211_ETOH_dataset.pkl")

bench = Bench()
bench.setup(pickle, run_name="sandbox", tag="groups")

paths = bench.output_paths
df = bench.load()
# %%
print("MultiIndex Levels:")
print(df.index.names)
print("\nMultiIndex Values:")
print(df.index.levels)
print("\nColumn MultiIndex:")
print(df.columns.names)
print(df.columns.levels)
print("\nColumn Data Types:")
print(df.dtypes)
print("\nInferred Array Types:")
for col in df.columns:
	sample = df[col].iloc[0]
	if isinstance(sample, np.ndarray):
		print(f"{col}: ndarray with dtype {sample.dtype}")
# %%
df.index.names
df.columns.names

df.shape
df.index.to_frame(index=False).value_counts(["Subject","Session","Task"])

# %%
def col(df, source, feature):
    return df[(source, feature)]

t_meso = col(df, "mesomap", "time_elapsed_s")
t_pupil = col(df, "pupil", "time_elapsed_s")
pupil = col(df, "pupil", "pupil_diameter_mm")

speed = col(df, "treadmill", "speed_mm").dropna()
t_tm = col(df, "treadmill", "time_elapsed_s").dropna()

roi = col(df, "mesomap", "L_VISp")

len_check = (
	df.assign(
		n_meso=t_meso.map(len),
		n_pupil=t_pupil.map(len),
		n_tm=t_tm.map(len),
	)[["n_meso", "n_pupil", "n_tm"]]
)
len_check.describe()

# %%

# Try to identify any systematic timing offsets between sources by comparing their time_elapsed_s arrays
def arr_len(x):
    if isinstance(x, np.ndarray):
        return len(x)
    return np.nan

# example: pupil
pupil_t_len = df[('pupil','time_elapsed_s')].map(arr_len)
pupil_y_len = df[('pupil','pupil_diameter_mm')].map(arr_len)
(pupil_t_len - pupil_y_len).value_counts(dropna=False)

#%%

idx = ('GS26', 'ses-01', 'task-spont')
row = df.loc[idx]

def source_timeseries(row, source, features):
    """
    row: a Series corresponding to one (Subject, Session, Task)
    source: e.g. 'pupil', 'treadmill', 'mesomap'
    features: list of feature names (strings) to include
    returns: DataFrame with 'time_elapsed_s' + features columns
    """
    t = row[(source, 'time_elapsed_s')]
    if not isinstance(t, np.ndarray):
        return None

    out = pd.DataFrame({'time_elapsed_s': t})
    for f in features:
        x = row.get((source, f), None)
        if isinstance(x, np.ndarray):
            out[f] = x
        else:
            out[f] = np.nan
    return out

# mesomap: choose a subset of ROIs to start
rois = ['L_MOp','R_MOp','L_VISp','R_VISp']
meso = source_timeseries(row, 'mesomap', rois)  # includes time_elapsed_s + ROI columns

pupil = source_timeseries(row, 'pupil', ['pupil_diameter_mm'])
tread = source_timeseries(row, 'treadmill', ['speed_mm', 'distance_mm'])

meso = meso.sort_values('time_elapsed_s')

aligned = meso
if pupil is not None:
    aligned = pd.merge_asof(
        aligned, pupil.sort_values('time_elapsed_s'),
        on='time_elapsed_s', direction='nearest', tolerance=0.25  # seconds; tune
    )

if tread is not None:
    aligned = pd.merge_asof(
        aligned, tread.sort_values('time_elapsed_s'),
        on='time_elapsed_s', direction='nearest', tolerance=0.25
    )

aligned.head()
# %%

rois = ['L_MOp','R_MOp','L_VISp','R_VISp','L_RSPagl','R_RSPagl']
source_features = [
    ('mesomap', rois),
    ('pupil', ['pupil_diameter_mm']),
    ('treadmill', ['speed_mm']),
]
long = bench.build_long(df, source_features=source_features, tol=0.25)
long
# %%

ses_to_cond = {
    'ses-01': 'baseline',
    'ses-02': 'saline',
    'ses-03': 'ethanol_low',
    'ses-04': 'ethanol_high',
}

long['Condition'] = long['Session'].map(ses_to_cond)
# %%
