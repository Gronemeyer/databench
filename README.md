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

## Adding a new analysis/plotter/feature

Registration is explicit and local to the class via decorators. Define your class once and import it before you create a `Bench` instance.

```python
from dataclasses import dataclass

from databench.registry import register_analysis
from databench.analysis.base import Analysis, AnalysisResult
from databench import Bench


@register_analysis
@dataclass(frozen=True)
class MyAnalysis(Analysis):
    name: str = "my_analysis"

    def run(self, wide, **kwargs) -> AnalysisResult:
        result = wide.describe()
        return AnalysisResult(name=self.name, data=result)


# Make sure this module is imported before Bench() so registration runs.
bench = Bench()
```

Same idea for plotters and features:

```python
from dataclasses import dataclass

from databench.registry import register_plotter, register_feature
from databench.plotting.base import Plotter
from databench.features.base import FeatureFn


@register_plotter
@dataclass(frozen=True)
class MyPlotter(Plotter):
    name: str = "my_plotter"

    def plot(self, *args, **kwargs):
        ...


@register_feature
@dataclass(frozen=True)
class MyFeature(FeatureFn):
    name: str = "my_feature"
    label: str = "My Feature"

    def _run_impl(self, row) -> float:
        ...
```
