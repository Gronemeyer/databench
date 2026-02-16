
#%%
from databench import Bench
from pathlib import Path

bench = Bench()

bench.setup(
	input_path=Path(r'/Users/jakegronemeyer/Desktop/4jake/260212_ACUTEVIS_dataset.pkl'),
	output_root=Path(__file__).resolve().parents[1] / "outputs",
	run_name="260216",
	tag="event-based-analysis"
)

df = bench.load()
print(df.columns)
df.suite2p.time_elapsed_s[0][3]
source_features = [
    ("suite2p", ["deltaf_f", "roi_fluorescence"], [4]),
    #("suite2p", ["roi_fluorescence"], [4]),
    ("pupil", ["pupil_diameter_mm"]),
    ("encoder", ["speed_mm"]),
]

long = bench.build_long(
	df,
	source_features=source_features,
	tol=0.25,
	time_column="time_elapsed_s",
	reference_source="suite2p",
)

# %%
