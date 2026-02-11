# databench

Minimal, reproducible analysis/plotting toolkit for multiindex datasets.

## Setup (Conda)

```bash
conda env create -f environment.yml
conda activate databench
pip install -e .
```

## Quickstart

```bash
python pipeline_demo.py
```

Or in Python:

```python
from databench import Bench
from databench.analysis import longitudinal_summary
from databench import plotting

bench = Bench()
bench.setup("/path/to/input.pkl")
paths = bench.output_paths
df = bench.load()
bench.set_filters(drop_rows=(
    ("GS29", "ses-04", "task-movies"),
))
df = bench.filter_data(df)

features = bench.feature_names
feature_fns = bench.data

# Example: summarize longitudinal stats
session_table = bench.build_session_table(df, feature_fns)
summary = bench.analyze("longitudinal_summary", session_table, y=features[0])
session_table = summary.data

# Example: plot using plotting submodule
fig, _ = bench.plot("feature", session_table, feature=feature_fns[0])
```
