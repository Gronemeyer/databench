from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type, Union
import json
from dataclasses import asdict
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

from databench.analysis import Analysis, AnalysisResult
from databench.config import FilterConfig, IOConfig, OutputPaths
from databench.features import FeatureFn
from databench.plotting import Plotter
from databench.registry import ANALYSIS_CLASSES, FEATURE_CLASSES, PLOTTER_CLASSES
from databench.utils import drop_rows, session_to_int
from databench.debug import get_row, log_context


class Bench:
    """User-facing entry point for loading data, running analyses, and making plots.

    Typical flow:
    1) Create a `Bench()` instance.
    2) Call `setup()` to define input/output paths.
    3) `load()` your dataset.
    4) Build feature tables with `build_session_table()`.
    5) Use `analyze()` and `plot()` to generate results and figures.

    Features, analyses, and plotters are registered explicitly with decorators.
    Import any custom modules before creating `Bench()` so registration runs.
    """

    def __init__(
        self,
        features: Optional[Iterable[Union[FeatureFn, Type[FeatureFn]]]] = None,
        analyses: Optional[Iterable[Union[Analysis, Type[Analysis]]]] = None,
        plotters: Optional[Iterable[Union[Plotter, Type[Plotter]]]] = None,
    ) -> None:
        """Create a Bench and register default components.

        Example:
            bench = Bench()
        """
        self._features: Dict[str, FeatureFn] = {}
        self._analyses: Dict[str, Analysis] = {}
        self._plotters: Dict[str, Plotter] = {}
        self.io_config: Optional[IOConfig] = None
        self.filter_config: Optional[FilterConfig] = None
        self.output_paths: Optional[OutputPaths] = None
        self._usage: Dict[str, list] = {"features": [], "analyses": [], "plots": []}
        self._derived_column_meta: Dict[str, Dict[str, Any]] = {}
        self._saved_outputs: Dict[str, List[str]] = {
            "tables": [],
            "figures": [],
            "other": [],
        }

        for feat in (features or FEATURE_CLASSES):
            self.register_feature(feat)
        for analysis in (analyses or ANALYSIS_CLASSES):
            self.register_analysis(analysis)
        for plotter in (plotters or PLOTTER_CLASSES):
            self.register_plotter(plotter)

    def register_feature(self, feat: Union[FeatureFn, Type[FeatureFn]]) -> "Bench":
        """Register a feature class or instance.

        Example:
            bench.register_feature(MyFeature)
        """
        if isinstance(feat, type):
            if not issubclass(feat, FeatureFn):
                raise TypeError("Feature class must subclass FeatureFn")
            instance = feat()
        elif isinstance(feat, FeatureFn):
            instance = feat
        else:
            raise TypeError("Feature must be a FeatureFn or FeatureFn class")

        if instance.name in self._features:
            raise ValueError(f"Feature already registered: {instance.name!r}")
        self._features[instance.name] = instance
        return self

    def register_analysis(self, analysis: Union[Analysis, Type[Analysis]]) -> "Bench":
        """Register an analysis class or instance.

        Example:
            bench.register_analysis(MyAnalysis)
        """
        if isinstance(analysis, type):
            if not issubclass(analysis, Analysis):
                raise TypeError("Analysis class must subclass Analysis")
            instance = analysis()
        elif isinstance(analysis, Analysis):
            instance = analysis
        else:
            raise TypeError("Analysis must be an Analysis or Analysis class")

        if instance.name in self._analyses:
            raise ValueError(f"Analysis already registered: {instance.name!r}")
        if instance.name in self._plotters:
            raise ValueError(
                f"Analysis name conflicts with plotter: {instance.name!r}"
            )
        self._analyses[instance.name] = instance
        return self

    def register_plotter(self, plotter: Union[Plotter, Type[Plotter]]) -> "Bench":
        """Register a plotter class or instance.

        Example:
            bench.register_plotter(MyPlotter)
        """
        if isinstance(plotter, type):
            if not issubclass(plotter, Plotter):
                raise TypeError("Plotter class must subclass Plotter")
            instance = plotter()
        elif isinstance(plotter, Plotter):
            instance = plotter
        else:
            raise TypeError("Plotter must be a Plotter or Plotter class")

        if instance.name in self._plotters:
            raise ValueError(f"Plotter already registered: {instance.name!r}")
        if instance.name in self._analyses:
            raise ValueError(
                f"Plotter name conflicts with analysis: {instance.name!r}"
            )
        self._plotters[instance.name] = instance
        return self

    @property
    def data(self) -> List[FeatureFn]:
        """Return registered feature instances.

        Example:
            feature_fns = bench.data
        """
        return list(self._features.values())

    @property
    def feature_names(self) -> List[str]:
        """Return registered feature names.

        Example:
            names = bench.feature_names
        """
        return list(self._features.keys())

    def get_feature(self, name: str) -> FeatureFn:
        """Fetch a feature by name.

        Example:
            speed = bench.get_feature("speed_mean_cms")
        """
        try:
            return self._features[name]
        except KeyError as exc:
            raise KeyError(f"Unknown feature: {name!r}") from exc

    @staticmethod
    def _serialize_component(component: Any) -> Dict[str, Any]:
        try:
            return asdict(component)
        except Exception:
            return {"class": component.__class__.__name__}

    def _resolve_analysis(
        self,
        analysis: Union[Analysis, Type[Analysis]],
    ) -> Analysis:
        if isinstance(analysis, type):
            if not issubclass(analysis, Analysis):
                raise TypeError("analysis class must subclass Analysis")
            return analysis()
        if isinstance(analysis, Analysis):
            return analysis
        raise TypeError("analysis must be an Analysis instance or Analysis class")

    def _resolve_plotter_or_analysis(
        self,
        plotter: Union[Plotter, Type[Plotter], Analysis, Type[Analysis]],
    ) -> Union[Plotter, Analysis]:
        if isinstance(plotter, type):
            if issubclass(plotter, Plotter):
                return plotter()
            if issubclass(plotter, Analysis):
                return plotter()
            raise TypeError("plotter class must subclass Plotter or Analysis")
        if isinstance(plotter, (Plotter, Analysis)):
            return plotter
        raise TypeError("plotter must be a Plotter/Analysis instance or Plotter/Analysis class")

    def list_features(self) -> List[str]:
        """List registered first-order feature names."""
        return sorted(self._features.keys())

    def list_analyses(self) -> List[str]:
        """List registered analysis names."""
        return sorted(self._analyses.keys())

    def list_plotters(self) -> List[str]:
        """List registered plotter names."""
        return sorted(self._plotters.keys())

    def list_derived_columns(self) -> List[str]:
        """List registered second-order (derived) column names."""
        return sorted(self._derived_column_meta.keys())

    def register_derived_column(
        self,
        name: str,
        label: Optional[str] = None,
        color: Optional[str] = None,
        plotter: str = "longitudinal",
    ) -> Dict[str, Any]:
        """Register plotting metadata for a derived/second-order column."""
        spec = {
            "name": name,
            "label": label or name,
            "color": color,
            "plotter": plotter,
        }
        self._derived_column_meta[name] = spec
        return dict(spec)

    def get_column_plot_spec(self, name: str) -> Union[FeatureFn, Dict[str, Any]]:
        """Resolve first-order features or derived-column metadata for plotting."""
        if name in self._features:
            return self._features[name]
        if name in self._derived_column_meta:
            return dict(self._derived_column_meta[name])
        known = sorted(set(self.list_features()) | set(self.list_derived_columns()))
        raise KeyError(f"Unknown plottable column: {name!r}. Known: {', '.join(known)}")

    @staticmethod
    def _has_required_column(df: pd.DataFrame, col: str) -> bool:
        if col in df.columns:
            return True
        if isinstance(df.columns, pd.MultiIndex):
            for level in range(df.columns.nlevels):
                if col in df.columns.get_level_values(level):
                    return True
        if col in df.index.names:
            return True
        return False

    def preflight(
        self,
        *,
        analysis: Optional[Union[Analysis, Type[Analysis]]] = None,
        plotter: Optional[Union[Plotter, Type[Plotter], Analysis, Type[Analysis]]] = None,
        feature: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        required_columns: Optional[Iterable[str]] = None,
    ) -> None:
        """Validate names and required columns before expensive work."""
        if analysis is not None:
            self._resolve_analysis(analysis)
        if plotter is not None:
            self._resolve_plotter_or_analysis(plotter)

        if feature is not None:
            self.get_column_plot_spec(feature)

        if df is not None and required_columns is not None:
            missing = [c for c in required_columns if not self._has_required_column(df, c)]
            if missing:
                raise ValueError(f"Missing required columns: {', '.join(missing)}")

    def analyze(self, analysis: Union[Analysis, Type[Analysis]], *args, **kwargs) -> AnalysisResult:
        """Run a named analysis and return its result.

        Example:
            from databench.analysis import LongitudinalAnalysis
            result = bench.analyze(LongitudinalAnalysis(), table, y="speed_mean_cms")
        """
        analysis_obj = self._resolve_analysis(analysis)
        self._usage["analyses"].append(
            {
                "name": analysis_obj.name,
                "kwargs": kwargs,
                "config": self._serialize_component(analysis_obj),
            }
        )
        return analysis_obj.run(*args, **kwargs)

    def plot(self, plotter: Union[Plotter, Type[Plotter], Analysis, Type[Analysis]], *args, **kwargs):
        """Run a named plotter or analysis plot.

        Example:
            from databench.plotting import FeaturePlotter
            fig, _ = bench.plot(FeaturePlotter(), table, feature=bench.data[0])
        """
        plot_obj = self._resolve_plotter_or_analysis(plotter)
        if plot_obj.name == "feature":
            if "feature" in kwargs and isinstance(kwargs["feature"], str):
                kwargs["feature"] = self.get_column_plot_spec(kwargs["feature"])
            elif "feature" not in kwargs and "y" in kwargs and isinstance(kwargs["y"], str):
                kwargs["feature"] = self.get_column_plot_spec(kwargs.pop("y"))
        self._usage["plots"].append(
            {
                "name": plot_obj.name,
                "kwargs": kwargs,
                "config": self._serialize_component(plot_obj),
            }
        )
        return plot_obj.plot(*args, **kwargs)

    def save(self, result: AnalysisResult, **kwargs) -> list:
        """Save a result using the originating analysis.

        Example:
            bench.save(result, stats_dir=paths.stats, plots_dir=paths.plots)
        """
        try:
            analysis = self._analyses[result.name]
        except KeyError as exc:
            raise KeyError(f"Unknown analysis: {result.name!r}") from exc
        return analysis.save(result, **kwargs)

    def set_filters(self, drop_rows: tuple = ()) -> FilterConfig:
        """Set dataset filters for later use with `filter_data()`.

        Example:
            bench.set_filters(drop_rows=(("GS29", "ses-04", "task-movies"),))
        """
        cfg = FilterConfig(drop_rows=drop_rows)
        self.filter_config = cfg
        return cfg

    def setup(
        self,
        input_path: Path,
        output_root: Path = Path("outputs"),
        run_name: str = "databench",
        tag: Optional[str] = None,
    ) -> tuple[IOConfig, OutputPaths]:
        """Define input/output paths and create output folders.

        Example:
            io_cfg, paths = bench.setup("/path/to/input.pkl")
        """
        if tag:
            cfg = IOConfig(input_path=Path(input_path), output_root=output_root, run_name=run_name, tag=tag)
        else:
            cfg = IOConfig(input_path=Path(input_path), output_root=output_root, run_name=run_name)
        paths = self._make_output_paths(cfg)
        self.io_config = cfg
        self.output_paths = paths
        self._saved_outputs = {"tables": [], "figures": [], "other": []}
        return cfg, paths

    def _track_output(self, path: Path, category: str) -> None:
        key = category if category in self._saved_outputs else "other"
        as_str = str(path)
        if as_str not in self._saved_outputs[key]:
            self._saved_outputs[key].append(as_str)

    def load(self, input_path: Optional[Path] = None) -> pd.DataFrame:
        """Load the dataset from the configured input path.

        Example:
            df = bench.load()
        """
        if input_path is None:
            if self.io_config is None:
                raise ValueError("Call setup() or pass input_path before loading.")
            input_path = self.io_config.input_path
        return self._load_df(input_path)

    def build_session_table(
        self,
        df: pd.DataFrame,
        features: Optional[Iterable[FeatureFn]] = None,
    ) -> pd.DataFrame:
        """Compute a wide feature table from a raw dataset.

        Example:
            table = bench.build_session_table(df)
        """
        use_features = list(features) if features is not None else self.data
        self._usage["features"].append({"names": [f.name for f in use_features]})
        rows = []
        for _, row in df.iterrows():
            out = {}
            for feat in use_features:
                out[feat.name] = feat.run(row)
            rows.append(out)
        out = pd.DataFrame(rows, index=df.index)
        out["session_n"] = df.index.get_level_values("Session").map(session_to_int)
        return out.sort_values(["Subject", "session_n"])

    @staticmethod
    def _extract_trace(
        row: pd.Series,
        source: str,
        feature: str,
        index: Optional[int],
        label: str,
    ) -> Optional[np.ndarray]:
        x = row.get((source, feature), None)
        if not isinstance(x, np.ndarray):
            return None
        if x.ndim > 1:
            if index is None:
                raise ValueError(
                    f"{label} {source!r}.{feature!r} has nested arrays. "
                    "Provide an index to select a trace."
                )
            x = x[index]
        return x

    @staticmethod
    def _source_timeseries(
        row: pd.Series,
        source: str,
        features: Iterable[str],
        time_column: str,
        index: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        t = Bench._extract_trace(row, source, time_column, index, "Source")
        if t is None:
            return None
        data = {time_column: np.float64(t)}
        for feature_name in features:
            x = Bench._extract_trace(row, source, feature_name, index, "Feature")
            data[feature_name] = x if x is not None else np.nan
        return pd.DataFrame(data)

    def build_long(
        self,
        df: pd.DataFrame,
        source_features: Iterable[tuple[str, Iterable[str]]],
        tol: float = 0.25,
        time_column: str = "time_elapsed_s",
        reference_source: Optional[str] = None,
    ) -> pd.DataFrame:
        """Build a long table by aligning multiple source timeseries.

        Parameters
        ----------
        df : pd.DataFrame
            Input table with MultiIndex columns like (source, feature).
        source_features : iterable[tuple[str, iterable[str]]]
            Ordered list of (source, features) or (source, features, indices) pairs.
            The reference source is used as the alignment anchor. When indices
            are provided for the reference source, ROI-specific columns are
            created as `{feature}_roi{index}`. Every listed source is required
            to have the `time_column`, and indexed reference traces must share
            a common timebase.
        tol : float
            Merge tolerance in seconds for nearest-neighbor `merge_asof` joins.
        time_column : str
            Column name used for time alignment within each source.
        reference_source : str | None
            Optional source name to use as the alignment reference. Defaults to
            the first source in source_features.
        """
        source_features_list = []
        for entry in source_features:
            if len(entry) == 2:
                source, features = entry
                indices = None
            elif len(entry) == 3:
                source, features, indices = entry
            else:
                raise ValueError(
                    "source_features entries must be (source, features) or (source, features, indices)."
                )
            source_features_list.append((source, list(features), indices))
        if not source_features_list:
            raise ValueError("source_features must include at least one (source, features) pair.")
        if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 2:
            raise ValueError("Expected MultiIndex columns with (source, feature).")

        col_tuples = set(df.columns.tolist())
        for source, _, _ in source_features_list:
            if (source, time_column) not in col_tuples:
                raise ValueError(
                    f"Source {source!r} is missing required ('{source}', '{time_column}') column."
                )

        if reference_source is None:
            ref_idx = 0
        else:
            ref_idx = next(
                (i for i, (source, _, _) in enumerate(source_features_list) if source == reference_source),
                None,
            )
            if ref_idx is None:
                raise ValueError(
                    f"reference_source {reference_source!r} is not in source_features."
                )

        ref_source, ref_features, ref_indices = source_features_list[ref_idx]
        indexed_sources = [
            source for source, _, indices in source_features_list if indices is not None
        ]
        if len(indexed_sources) > 1:
            raise ValueError(
                "Only one source may specify indices when building long tables."
            )
        if indexed_sources and indexed_sources[0] != ref_source:
            raise ValueError(
                "The indexed source must be the reference source in source_features."
            )

        merge_sources = [
            entry for i, entry in enumerate(source_features_list) if i != ref_idx
        ]

        frames = []
        if ref_indices is None:
            ref_index_list = []
        elif isinstance(ref_indices, (list, tuple, np.ndarray)):
            ref_index_list = list(ref_indices)
        else:
            ref_index_list = [ref_indices]

        for idx, row in df.iterrows():
            if ref_indices is None:
                out = self._source_timeseries(
                    row,
                    ref_source,
                    ref_features,
                    time_column,
                    index=None,
                )
                if out is None:
                    continue
            else:
                roi_frames = []
                base_time = None
                for ref_index in ref_index_list:
                    roi_df = self._source_timeseries(
                        row,
                        ref_source,
                        ref_features,
                        time_column,
                        index=ref_index,
                    )
                    if roi_df is None:
                        continue

                    if base_time is None:
                        base_time = roi_df[time_column].to_numpy()
                    elif not np.array_equal(roi_df[time_column].to_numpy(), base_time):
                        raise ValueError(
                            f"Source {ref_source!r} ROI timebases differ; cannot align per-ROI columns."
                        )

                    rename = {
                        feature_name: f"{feature_name}_roi{ref_index}"
                        for feature_name in ref_features
                    }
                    roi_frames.append(roi_df.rename(columns=rename))

                if base_time is None:
                    continue

                out = pd.concat(
                    [roi_frames[0][[time_column]]]
                    + [frame.drop(columns=[time_column]) for frame in roi_frames],
                    axis=1,
                )

            out = out.sort_values(time_column)

            for source, features, _ in merge_sources:
                ts = self._source_timeseries(
                    row,
                    source,
                    features,
                    time_column,
                    index=None,
                )
                if ts is not None:
                    out = pd.merge_asof(
                        out,
                        ts.sort_values(time_column),
                        on=time_column,
                        direction="nearest",
                        tolerance=tol,
                    )
                else:
                    for feature_name in features:
                        out[feature_name] = np.nan

            subj, ses, task = idx
            out.insert(0, "Task", task)
            out.insert(0, "Session", ses)
            out.insert(0, "Subject", subj)

            frames.append(out)

        return pd.concat(frames, ignore_index=True)

    def run_feature_on_row(
		self,
		df: pd.DataFrame,
		feature: FeatureFn,
		subject: Optional[str] = None,
		session: Optional[str] = None,
		task: Optional[str] = None,
		debug: bool = False,
    ):
        """Compute one feature for a single row selection.

        Example:
                val = bench.run_feature_on_row(df, bench.data[0], subject="GS29", session="ses-01")
        """
        idx, row = get_row(df, subject, session, task)
        val = feature.run(row)
        if debug:
            print(f"{log_context(idx)} | {feature.name} = {val}")
        return val

    def filter_data(self, df: pd.DataFrame, drop_rows_list: Optional[tuple] = None) -> pd.DataFrame:
        """Filter rows using stored or provided drop rules.

        Example:
            df = bench.filter_data(df)
        """
        if drop_rows_list is None and self.filter_config is not None:
            drop_rows_list = self.filter_config.drop_rows
        return drop_rows(df, drop_rows_list or ())

    def save_table(self, df: pd.DataFrame, name: str = "table.csv", folder: str = "stats") -> Path:
        """Save a table to the configured output folders.

        Example:
            path = bench.save_table(table, name="summary.csv")
        """
        if self.output_paths is None:
            raise ValueError("Call setup() before save_table().")
        if not hasattr(self.output_paths, folder):
            raise ValueError(f"Unknown output folder: {folder!r}")
        out_dir = getattr(self.output_paths, folder)
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            df.to_csv(path)
        elif path.suffix == ".parquet":
            df.to_parquet(path)
        else:
            path = path.with_suffix(".csv")
            df.to_csv(path)
        self._track_output(path, "tables")
        return path

    def save_analysis_result_tables(
        self,
        result: AnalysisResult,
        prefix: Optional[str] = None,
        folder: str = "stats",
    ) -> Dict[str, Path]:
        """Persist DataFrame outputs from an AnalysisResult using consistent names.

        Saves:
        - `result.table` as `<prefix>_table.csv`
        - `result.data` if DataFrame as `<prefix>.csv`
        - each DataFrame in `result.data` when dict-like as `<prefix>_<key>.csv`
        """
        base = prefix or result.name
        saved: Dict[str, Path] = {}

        if isinstance(result.table, pd.DataFrame):
            saved["table"] = self.save_table(result.table, f"{base}_table.csv", folder=folder)

        if isinstance(result.data, pd.DataFrame):
            saved["data"] = self.save_table(result.data, f"{base}.csv", folder=folder)
        elif isinstance(result.data, dict):
            for key, value in result.data.items():
                if isinstance(value, pd.DataFrame):
                    safe_key = str(key).replace(" ", "_")
                    saved[safe_key] = self.save_table(value, f"{base}_{safe_key}.csv", folder=folder)

        return saved

    def save_provenance(self, name: str = "provenance.json") -> Path:
        """Write a lightweight provenance record for the current run.

        Example:
            path = bench.save_provenance()
        """
        if self.output_paths is None:
            raise ValueError("Call setup() before save_provenance().")
        config_dir = self.output_paths.config
        config_dir.mkdir(parents=True, exist_ok=True)

        used_feature_names = {
            name
            for entry in self._usage.get("features", [])
            for name in entry.get("names", [])
        }
        features = []
        for feat in self.data:
            if feat.name not in used_feature_names:
                continue
            try:
                params = asdict(feat)
            except Exception:
                params = {"name": feat.name, "label": feat.label}
            features.append({"name": feat.name, "class": feat.__class__.__name__, "params": params})

        used_analysis_names = {entry.get("name") for entry in self._usage.get("analyses", [])}
        analyses = [
            {"name": a.name, "class": a.__class__.__name__}
            for a in self._analyses.values()
            if a.name in used_analysis_names
        ]
        used_plotter_names = {entry.get("name") for entry in self._usage.get("plots", [])}
        plotters = [
            {"name": p.name, "class": p.__class__.__name__}
            for p in self._plotters.values()
            if p.name in used_plotter_names
        ]

        payload = {
            "io_config": None if self.io_config is None else asdict(self.io_config),
            "filter_config": None if self.filter_config is None else asdict(self.filter_config),
            "git_hash": self._get_git_hash(),
            "features": features,
            "analyses": analyses,
            "plotters": plotters,
            "usage": self._usage,
        }

        path = config_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        self._track_output(path, "other")
        return path

    def save_run_summary(
        self,
        name: str = "run.json",
        params: Optional[Mapping[str, Any]] = None,
        notes: Optional[str] = None,
        outputs: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Write a compact run summary for small-team workflows.

        Example:
            bench.save_run_summary(params={"max_freq": 20}, notes="pilot run")
        """
        if self.output_paths is None:
            raise ValueError("Call setup() before save_run_summary().")

        config_dir = self.output_paths.config
        config_dir.mkdir(parents=True, exist_ok=True)

        if params is None:
            params_payload: Dict[str, Any] = {
                "analyses": [
                    {"name": e.get("name"), "kwargs": e.get("kwargs", {})}
                    for e in self._usage.get("analyses", [])
                ],
                "plots": [
                    {"name": e.get("name"), "kwargs": e.get("kwargs", {})}
                    for e in self._usage.get("plots", [])
                ],
            }
        else:
            params_payload = dict(params)

        outputs_payload = dict(outputs) if outputs is not None else {
            "tables": list(self._saved_outputs.get("tables", [])),
            "figures": list(self._saved_outputs.get("figures", [])),
            "other": list(self._saved_outputs.get("other", [])),
        }

        payload = {
            "created_at": datetime.now().isoformat(),
            "git_hash": self._get_git_hash(),
            "params": params_payload,
            "notes": notes,
            "io_config": None if self.io_config is None else asdict(self.io_config),
            "filter_config": None if self.filter_config is None else asdict(self.filter_config),
            "usage": self._usage,
            "outputs": outputs_payload,
        }

        path = config_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        self._track_output(path, "other")
        return path

    def save_figure(
        self,
        fig,
        name: str,
        folder: str = "plots",
        dpi: int = 300,
        bbox_inches: str = "tight",
    ) -> Path:
        """Save a matplotlib figure to the output folders.

        Example:
            path = bench.save_figure(fig, name="overview.png")
        """
        if self.output_paths is None:
            raise ValueError("Call setup() before save_figure().")
        if not hasattr(self.output_paths, folder):
            raise ValueError(f"Unknown output folder: {folder!r}")
        out_dir = getattr(self.output_paths, folder)
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        self._track_output(path, "figures")
        return path

    def export_feature_report(
        self,
        df: pd.DataFrame,
        source: str,
        feature: str,
        name: str = "feature_report.csv",
        folder: str = "stats",
        id_sep: str = "_",
    ) -> Path:
        """Export a per-row feature report from a nested signal column.

        Example:
            path = bench.export_feature_report(df, source="meso", feature="meso_tiff")
        """
        if self.output_paths is None:
            raise ValueError("Call setup() before export_feature_report().")
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("Expected MultiIndex with Subject/Session/Task.")
        if not hasattr(self.output_paths, folder):
            raise ValueError(f"Unknown output folder: {folder!r}")

        cols = []
        series = []
        for idx, row in df.iterrows():
            subject, session, task = idx
            col_name = id_sep.join([str(subject), str(session), str(task)])
            x = row.get((source, feature))
            if x is None:
                continue
            arr = pd.Series(x).to_numpy().ravel()
            cols.append(col_name)
            series.append(arr)

        if not series:
            raise ValueError(f"No data found for ({source}, {feature}).")

        max_len = max(len(a) for a in series)
        data = {
            c: pd.Series(a).reindex(range(max_len)).to_numpy()
            for c, a in zip(cols, series)
        }

        out_df = pd.DataFrame(data)
        out_dir = getattr(self.output_paths, folder)
        out_path = out_dir / name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        self._track_output(out_path, "tables")
        return out_path

    @staticmethod
    def _make_output_paths(cfg: IOConfig) -> OutputPaths:
        run_dir = cfg.output_root / cfg.run_name / cfg.tag
        plots = run_dir / "plots"
        reports = run_dir / "reports"
        stats = run_dir / "stats"
        config = run_dir / "config"
        for p in (plots, reports, stats, config):
            p.mkdir(parents=True, exist_ok=True)
        return OutputPaths(run_dir=run_dir, plots=plots, reports=reports, stats=stats, config=config)

    @staticmethod
    def _get_git_hash() -> Optional[str]:
        try:
            repo_root = Path(__file__).resolve().parents[1]
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip() or None
        except Exception:
            return None

    @staticmethod
    def _load_df(path: Path) -> pd.DataFrame:
        path = Path(path)
        if path.suffix in {".pkl", ".pickle"}:
            return pd.read_pickle(path)
        if path.suffix in {".h5", ".hdf", ".hdf5"}:
            return pd.read_hdf(path, key="HFSA")
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported input format: {path.suffix}")
