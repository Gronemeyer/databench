from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type, Union, cast
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
from databench import provenance
from databench.registry import ANALYSIS_CLASSES, FEATURE_CLASSES, PLOTTER_CLASSES
from databench.utils import drop_rows, session_to_int
from databench.debug import get_row, log_context
from databench._utils._logger import get_logger


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
        features: Optional[Iterable[Any]] = None,
        analyses: Optional[Iterable[Any]] = None,
        plotters: Optional[Iterable[Any]] = None,
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
        self._provenance_notes: Optional[str] = None
        self._saved_outputs: Dict[str, List[str]] = {
            "tables": [],
            "figures": [],
            "other": [],
            "feature_plots": [],
        }
        self._logger = get_logger("databench")

        for feat in (features or FEATURE_CLASSES):
            self.register_feature(feat)
        for analysis in (analyses or ANALYSIS_CLASSES):
            self.register_analysis(analysis)
        for plotter in (plotters or PLOTTER_CLASSES):
            self.register_plotter(plotter)

    def register_feature(self, feat: Any) -> "Bench":
        """Register a feature class or instance.

        Example:
            bench.register_feature(MyFeature)
        """
        instance = feat() if isinstance(feat, type) else feat  # type: ignore[call-arg]
        self._features[instance.name] = instance
        return self

    def register_analysis(self, analysis: Any) -> "Bench":
        """Register an analysis class or instance.

        Example:
            bench.register_analysis(MyAnalysis)
        """
        instance = analysis() if isinstance(analysis, type) else analysis  # type: ignore[call-arg]
        self._analyses[instance.name] = instance
        return self

    def register_plotter(self, plotter: Any) -> "Bench":
        """Register a plotter class or instance.

        Example:
            bench.register_plotter(MyPlotter)
        """
        instance = plotter() if isinstance(plotter, type) else plotter  # type: ignore[call-arg]
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
        return self._features[name]

    @staticmethod
    def _serialize_component(component: Any) -> Dict[str, Any]:
        try:
            return asdict(component)
        except Exception:
            return {"class": component.__class__.__name__}

    def _resolve_analysis(self, analysis: Any) -> Analysis:
        return analysis() if isinstance(analysis, type) else analysis  # type: ignore[call-arg]

    def _resolve_plotter_or_analysis(self, plotter: Any) -> Union[Plotter, Analysis]:
        return plotter() if isinstance(plotter, type) else plotter  # type: ignore[call-arg]

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
        return dict(self._derived_column_meta[name])

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
        """No-op preflight kept for API compatibility."""
        return None

    def analyze(self, analysis: Union[Analysis, Type[Analysis]], *args, **kwargs) -> AnalysisResult:
        """Run a named analysis and return its result.

        Example:
            from databench.analysis import LongitudinalAnalysis
            result = bench.analyze(LongitudinalAnalysis(), table, y="speed_mean_cms")
        """
        analysis_obj = self._resolve_analysis(analysis)
        self._logger.info(f"Analyze: {analysis_obj.name}")
        self._usage["analyses"].append(
            {
                "name": analysis_obj.name,
                "class": analysis_obj.__class__.__name__,
                "class_module": analysis_obj.__class__.__module__,
                "doc": (analysis_obj.__class__.__doc__ or "").strip() or None,
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
        self._logger.debug(f"Plot: {plot_obj.name}")
        self._usage["plots"].append(
            {
                "name": plot_obj.name,
                "class": plot_obj.__class__.__name__,
                "class_module": plot_obj.__class__.__module__,
                "doc": (plot_obj.__class__.__doc__ or "").strip() or None,
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
        analysis = self._analyses[result.name]
        self._logger.info(f"Save result: {result.name}")
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
        scientist: Optional[str] = None,
        notes: Optional[str] = None,
        run_name: str = "databench",
        tag: Optional[str] = None,
    ) -> tuple[IOConfig, OutputPaths]:
        """Define input/output paths and create output folders.

        Example:
            io_cfg, paths = bench.setup("/path/to/input.pkl")
        """
        cfg = IOConfig(
            input_path=Path(input_path),
            output_root=output_root,
            scientist=scientist,
            run_name=run_name,
            tag=tag or "",
        )
        paths = self._make_output_paths(cfg)
        self.io_config = cfg
        self.output_paths = paths
        self._provenance_notes = notes
        self._saved_outputs = {
            "tables": [],
            "figures": [],
            "other": [],
            "feature_plots": [],
        }
        self._logger.info(f"Setup: input={cfg.input_path} output={paths.run_dir}")
        return cfg, paths

    def _track_output(self, path: Path, category: str) -> None:
        key = category if category in self._saved_outputs else "other"
        as_str = str(path)
        if as_str not in self._saved_outputs[key]:
            self._saved_outputs[key].append(as_str)

    def _log_saved_outputs(self, label: str, path: Path) -> None:
        self._logger.info(
            f"{label}: {path} (tables={len(self._saved_outputs.get('tables', []))}, "
            f"figures={len(self._saved_outputs.get('figures', []))}, "
            f"other={len(self._saved_outputs.get('other', []))})"
        )

    def load(self, input_path: Optional[Path] = None) -> pd.DataFrame:
        """Load the dataset from the configured input path.

        Example:
            df = bench.load()
        """
        if input_path is None:
            input_path = self.io_config.input_path  # type: ignore[union-attr]
        self._logger.info(f"Load dataset: {input_path}")
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
        self._logger.info(
            f"Build session table: features={len(use_features)} df_shape={df.shape} index_names={list(df.index.names)}"
        )
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
        x = row.get((source, feature))
        arr = np.asarray(x)
        if arr.ndim > 1 and index is not None:
            arr = arr[index]
        return arr

    @staticmethod
    def _source_timeseries(
        row: pd.Series,
        source: str,
        features: Iterable[str],
        time_column: str,
        index: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        t = Bench._extract_trace(row, source, time_column, index, "Source")
        t_arr = np.atleast_1d(t).astype(float, copy=False)
        data: Dict[str, Any] = {time_column: t_arr}
        for feature_name in features:
            x = Bench._extract_trace(row, source, feature_name, index, "Feature")
            data[feature_name] = np.atleast_1d(x)
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
        self._logger.info("Build long table")
        for entry in source_features:
            source = entry[0]
            features = entry[1]
            indices = entry[2] if len(entry) > 2 else None
            source_features_list.append((source, list(features), indices))

        ref_idx = 0
        if reference_source is not None:
            for i, (source, _, _) in enumerate(source_features_list):
                if source == reference_source:
                    ref_idx = i
                    break

        ref_source, ref_features, ref_indices = source_features_list[ref_idx]
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
                    ts = ts.dropna(subset=[time_column])
                    out = pd.merge_asof(
                        out,
                        ts.sort_values(time_column),
                        on=time_column,
                        direction="nearest",
                        tolerance=tol,  # type: ignore[arg-type]
                    )
                else:
                    for feature_name in features:
                        out[feature_name] = np.nan

            subj, ses, task = idx  # type: ignore[misc]
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
        self._logger.info(f"Run feature: {feature.name} | {log_context(idx)}")
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
        self._logger.info(f"Filter data: rows={len(drop_rows_list or ())}")
        return drop_rows(df, drop_rows_list or ())

    def save_table(self, df: pd.DataFrame, name: str = "table.csv", folder: str = "stats") -> Path:
        """Save a table to the configured output folders.

        Example:
            path = bench.save_table(table, name="summary.csv")
        """
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
        self._logger.info(f"Save analysis tables: {base}")
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
        config_dir = self.output_paths.config  # type: ignore[union-attr]
        config_dir.mkdir(parents=True, exist_ok=True)
        path, summary_path = provenance.save_provenance(self, output_dir=config_dir, name=name)
        self._track_output(path, "other")
        self._track_output(summary_path, "other")
        self._log_saved_outputs("Save provenance", path)
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
        config_dir = self.output_paths.config  # type: ignore[union-attr]
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
        self._log_saved_outputs("Save run summary", path)
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
        out_dir = getattr(self.output_paths, folder)
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        self._track_output(path, "figures")
        return path

    def save_feature_plot(
        self,
        fig,
        name: str,
        feature_name: str,
        plotter: Optional[Any] = None,
        folder: str = "plots",
        dpi: int = 300,
        bbox_inches: str = "tight",
    ) -> Path:
        """Save a figure and record its associated feature name explicitly."""
        path = self.save_figure(fig, name=name, folder=folder, dpi=dpi, bbox_inches=bbox_inches)
        record = {
            "feature_name": feature_name,
            "path": str(path),
        }
        if plotter is not None:
            record["plotter_class"] = plotter.__class__.__name__
            record["plotter_module"] = plotter.__class__.__module__
            record["plotter_doc"] = (plotter.__class__.__doc__ or "").strip() or None
            record["plotter_params"] = self._serialize_component(plotter)
        self._saved_outputs.setdefault("feature_plots", []).append(record)
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
        cols = []
        series = []
        for idx, row in df.iterrows():
            subject, session, task = idx  # type: ignore[misc]
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
            return cast(pd.DataFrame, pd.read_hdf(path, key="HFSA"))
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported input format: {path.suffix}")
