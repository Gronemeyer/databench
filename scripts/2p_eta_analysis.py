
#%%
from databench import Bench
from pathlib import Path

bench = Bench()

bench.setup(
	input_path=Path('/Volumes/Sinbas_Stuf/Projects/ACUTEVIS/picklejar/260211_ACUTEVIS_dataset.pkl'),
	output_root=Path(__file__).resolve().parents[1] / "outputs",
	run_name="260215",
	tag="event-based-analysis"
)

df = bench.load()
print(df.columns)

source_features = [
    ("suite2p", ["deltaf_f"], [3, 15, 45]),
    ("pupil", ["pupil_diameter_mm"]),
    ("encoder", ["speed_mm"]),
]

long = bench.build_long(df, source_features=source_features, tol=0.25)

# %%
