from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Type, Union
import json
from dataclasses import asdict
import subprocess

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

    def analyze(self, name: str, *args, **kwargs) -> AnalysisResult:
        """Run a named analysis and return its result.

        Example:
            result = bench.analyze("longitudinal_summary", table, y="speed_mean_cms")
        """
        try:
            analysis = self._analyses[name]
        except KeyError as exc:
            raise KeyError(f"Unknown analysis: {name!r}") from exc
        self._usage["analyses"].append({"name": name, "kwargs": kwargs})
        return analysis.run(*args, **kwargs)

    def plot(self, name: str, *args, **kwargs):
        """Run a named plotter or analysis plot.

        Example:
            fig, _ = bench.plot("feature", table, feature=bench.data[0])
        """
        if name in self._plotters:
            self._usage["plots"].append({"name": name, "kwargs": kwargs})
            return self._plotters[name].plot(*args, **kwargs)
        if name in self._analyses:
            self._usage["plots"].append({"name": name, "kwargs": kwargs})
            return self._analyses[name].plot(*args, **kwargs)
        raise KeyError(f"Unknown plotter or analysis: {name!r}")

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
        return cfg, paths

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
        return path

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
