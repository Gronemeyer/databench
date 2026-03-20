from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type, Union, cast
import json
from dataclasses import asdict
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from databench.analysis.base import Analysis, AnalysisResult
from databench.config import FilterConfig, IOConfig, OutputPaths, _resolve_dataset_alias_for_output
from databench.features import FeatureFn
from databench.plotting import Plotter
from databench import provenance
from databench.registry import (
    ANALYSIS_CLASSES, FEATURE_CLASSES, PLOTTER_CLASSES,
)
from databench.utils import drop_rows, session_to_int
from databench._utils._logger import get_logger


class Bench:
    """Chainable entry point for loading data, running analyses, and making plots.

    Most methods return ``self`` so calls can be chained::

        bench = Bench()
        (bench
            .setup(path, run_name="260218", tag="eta")
            .build_long(sources=[("mesomap", rois), ("treadmill", ["speed_mm"])])
            .label_conditions(ses_to_cond)
            .analyze(EtaAnalysis(...))
            .plot(EtaConditionPlotter(...), save="eta_onset.png")
            .save_tables(prefix="eta"))

    State is accessible via properties for non-chain usage::

        df   = bench.df       # raw loaded DataFrame
        long = bench.long     # long table from build_long
        res  = bench.result   # last AnalysisResult
    """

    def __init__(
        self,
        features: Optional[Iterable[Any]] = None,
        analyses: Optional[Iterable[Any]] = None,
        plotters: Optional[Iterable[Any]] = None,
    ) -> None:
        self._features: Dict[str, FeatureFn] = {}
        self._analyses: Dict[str, Analysis] = {}
        self._plotters: Dict[str, Plotter] = {}
        self.io_config: Optional[IOConfig] = None
        self.filter_config: Optional[FilterConfig] = None
        self.output_paths: Optional[OutputPaths] = None
        self._invoked: List[Dict[str, Any]] = []
        self._provenance_notes: Optional[str] = None
        self._saved_outputs: Dict[str, List[str]] = {
            "tables": [],
            "figures": [],
            "other": [],
            "feature_plots": [],
        }
        self._logger = get_logger("databench")

        # Chain state — populated by chainable methods
        self._df: Optional[pd.DataFrame] = None
        self._long: Optional[pd.DataFrame] = None
        self._session_table: Optional[pd.DataFrame] = None
        self._result: Optional[AnalysisResult] = None
        self._last_fig: Optional[Any] = None

        for feat in (features or FEATURE_CLASSES):
            self.register_feature(feat)
        for analysis in (analyses or ANALYSIS_CLASSES):
            self.register_analysis(analysis)
        for plotter in (plotters or PLOTTER_CLASSES):
            self.register_plotter(plotter)

    # -- RunContext entry point ------------------------------------------------

    def run(self, name: str, notes: Optional[str] = None) -> "provenance.RunContext":
        """Return a ``RunContext`` context manager for scoped execution.

        Usage::

            with bench.run("eta-analysis") as run:
                run.build_long(sources=[...])
                result = run.analyze(analysis, df=run.long)
                fig, ax = run.plot(plotter, result, save="plot.png")
                run.save_tables(result, prefix="eta")
            # provenance auto-saved, bench state restored

        On entry the bench chain state (df, long, session_table, result) is
        snapshotted.  On clean exit provenance is auto-saved and the
        snapshot is restored so successive ``run()`` blocks are isolated.
        """
        return provenance.RunContext(self, name, notes)

    # -- Chain-state properties ------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        """The raw loaded DataFrame (set by ``setup()``)."""
        if self._df is None:
            raise RuntimeError("No data loaded. Call .setup() first.")
        return self._df

    @property
    def long(self) -> pd.DataFrame:
        """The long table (set by ``build_long()``)."""
        if self._long is None:
            raise RuntimeError("No long table. Call .build_long() first.")
        return self._long

    @property
    def session_table(self) -> pd.DataFrame:
        """The session feature table (set by ``build_session_table()``)."""
        if self._session_table is None:
            raise RuntimeError("No session table. Call .build_session_table() first.")
        return self._session_table

    @property
    def result(self) -> AnalysisResult:
        """The most recent AnalysisResult (set by ``analyze()``)."""
        if self._result is None:
            raise RuntimeError("No result. Call .analyze() first.")
        return self._result

    def register_feature(self, feat: Any) -> "Bench":
        """Register a feature class or instance."""
        instance = feat() if isinstance(feat, type) else feat
        self._features[instance.name] = instance
        return self

    def register_analysis(self, analysis: Any) -> "Bench":
        """Register an analysis class or instance."""
        instance = analysis() if isinstance(analysis, type) else analysis
        self._analyses[instance.name] = instance
        return self

    def register_plotter(self, plotter: Any) -> "Bench":
        """Register a plotter class or instance."""
        instance = plotter() if isinstance(plotter, type) else plotter
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
        """Resolve to an Analysis instance and register it for provenance."""
        if isinstance(analysis, str):
            if analysis in self._analyses:
                return self._analyses[analysis]
            raise KeyError(f"No analysis registered with name={analysis!r}. Available: {self.list_analyses()}")
        instance = analysis() if isinstance(analysis, type) else analysis
        self._analyses[instance.name] = instance
        return instance

    def _resolve_plotter(self, plotter: Any) -> Union[Plotter, Analysis]:
        """Resolve to a Plotter instance and register it."""
        if isinstance(plotter, str):
            if plotter in self._plotters:
                return self._plotters[plotter]
            if plotter in self._analyses:
                return self._analyses[plotter]
            raise KeyError(f"No plotter registered with name={plotter!r}. Available: {self.list_plotters()}")
        instance = plotter() if isinstance(plotter, type) else plotter
        self._plotters[instance.name] = instance
        return instance

    def list_analyses(self) -> List[str]:
        """List registered analysis names."""
        return sorted(self._analyses.keys())

    def list_plotters(self) -> List[str]:
        """List registered plotter names."""
        return sorted(self._plotters.keys())

    def analyze(self, analysis: Union[str, Analysis, Type[Analysis]], df: Union[pd.DataFrame, pd.Series, None] = None) -> AnalysisResult:
        """Run an analysis and return its result.

        Also stores the result on ``self._result`` for later access.

        Parameters
        ----------
        analysis : str, Analysis, or Analysis subclass
            The analysis to run.
        df : DataFrame or Series
            Data to analyse.  Required — pass ``bench.long``, ``bench.df``,
            ``bench.session_table``, or a row explicitly.

        Returns
        -------
        AnalysisResult
        """
        analysis_obj = self._resolve_analysis(analysis)
        self._logger.info(f"Analyze: {analysis_obj.name}")
        self._invoked.append({"kind": "analysis", "name": analysis_obj.name})
        if df is None:
            if self._long is not None:
                df = self._long
            else:
                raise ValueError(
                    "No df supplied and no long table available. "
                    "Pass df= explicitly (bench.long, bench.df, etc.)."
                )
        if hasattr(analysis_obj, "validate") and isinstance(df, pd.DataFrame):
            analysis_obj.validate(df)
        self._result = analysis_obj.run(df)
        return self._result

    def plot(
        self,
        plotter: Union[str, Plotter, Type[Plotter], Analysis, Type[Analysis]],
        result: Optional[AnalysisResult] = None,
        save: Optional[str] = None,
    ):
        """Run a plotter and return ``(fig, axes)``.

        Parameters
        ----------
        plotter : str, Plotter, or Plotter subclass
            The plotter to use.
        result : AnalysisResult, optional
            Data to plot.  When *None*, uses ``self.result``.
        save : str, optional
            When provided the figure is saved and closed automatically.

        Returns
        -------
        ``(fig, axes)`` or whatever the plotter returns.
        """
        plot_obj = self._resolve_plotter(plotter)
        self._logger.debug(f"Plot: {plot_obj.name}")
        self._invoked.append({"kind": "plot", "name": plot_obj.name})
        if result is None:
            result = self._result
        out = plot_obj.plot(result)
        # Normalise return value
        fig = None
        if isinstance(out, tuple):
            fig = out[0]
        elif isinstance(out, dict):
            if save:
                for key, val in out.items():
                    sub_fig = val[0] if isinstance(val, tuple) else val
                    self.save_and_close(sub_fig, f"{save}_{key}.png")
                self._last_fig = None
                return out
        else:
            fig = out
        self._last_fig = fig
        if save and fig is not None:
            self.save_and_close(fig, save)
        return out

    def save(self, result: Optional[AnalysisResult] = None) -> list:
        """Save a result using the originating analysis."""
        if result is None:
            result = self.result
        analysis = self._analyses[result.name]
        self._logger.info(f"Save result: {result.name}")
        return analysis.save(result)

    def set_filters(self, drop_rows: tuple = ()) -> "Bench":
        """Set dataset filters for later use with ``filter_data()``. Returns self."""
        cfg = FilterConfig(drop_rows=drop_rows)
        self.filter_config = cfg
        return self

    def filter(self, drop_rows: Optional[tuple] = None, **kwargs) -> "Bench":
        """Apply filters to the loaded DataFrame in-place and return self.

        Keyword arguments are matched against index levels::

            bench.filter(Task="task-spont")

        Or use ``drop_rows`` for explicit multi-index tuple exclusion.
        """
        if drop_rows is not None:
            self.set_filters(drop_rows)
        df = self.df  # may raise if not loaded
        if self.filter_config is not None:
            df = self.filter_data(df)
        # Keyword-based index filtering
        for level, value in kwargs.items():
            if level in df.index.names:
                mask = df.index.get_level_values(level) == value
                df = df.loc[mask]
        self._df = df
        return self

    def setup(
        self,
        input_path: Path,
        output_root: Path = Path("outputs"),
        scientist: Optional[str] = None,
        analyst: Optional[str] = None,
        lab: Optional[str] = None,
        notes: Optional[str] = None,
        run_name: str = "databench",
        tag: Optional[str] = None,
    ) -> "Bench":
        """Define input/output paths, create output folders, and load the dataset.

        Returns self for chaining.
        """
        cfg = IOConfig(
            input_path=Path(input_path),
            output_root=output_root,
            scientist=scientist,
            analyst=analyst,
            lab=lab,
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
        self._df = self._load_df(cfg.input_path)
        self._logger.info(f"Loaded dataset: {cfg.input_path} shape={self._df.shape}")
        return self

    def track_output(self, path: Path, category: str) -> None:
        """Record a saved output path under the given category.

        Categories: ``tables``, ``figures``, ``other``, ``feature_plots``.
        """
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

    def build_session_table(
        self,
        df: Optional[pd.DataFrame] = None,
        features: Optional[Iterable[FeatureFn]] = None,
    ) -> "Bench":
        """Compute a wide feature table from a raw dataset.

        When ``df`` is omitted, uses ``self.df``. Stores result on
        ``self._session_table``. Returns self for chaining.
        """
        if df is None:
            df = self.df
        use_features = list(features) if features is not None else self.data
        self._logger.info(
            f"Build session table: features={len(use_features)} df_shape={df.shape} index_names={list(df.index.names)}"
        )
        for f in use_features:
            self._invoked.append({"kind": "feature", "name": f.name, "kwargs": {}})
        rows = []
        for _, row in df.iterrows():
            out = {}
            for feat in use_features:
                out[feat.name] = feat.run(row)
            rows.append(out)
        out = pd.DataFrame(rows, index=df.index)
        out["session_n"] = df.index.get_level_values("Session").map(session_to_int)
        self._session_table = out.sort_values(["Subject", "session_n"])
        return self

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
        df: Optional[pd.DataFrame] = None,
        source_features: Optional[Iterable[tuple[str, Iterable[str]]]] = None,
        sources: Optional[Iterable[tuple[str, Iterable[str]]]] = None,
        tol: float = 0.25,
        time_column: str = "time_elapsed_s",
        reference_source: Optional[str] = None,
    ) -> "Bench":
        """Build a long table by aligning multiple source timeseries. Returns self.

        Parameters
        ----------
        df : pd.DataFrame | None
            Input table. Defaults to ``self.df``.
        source_features / sources : iterable
            Ordered list of ``(source, features)`` or ``(source, features, indices)``
            pairs. ``sources`` is an alias for ``source_features``.
        """
        if df is None:
            df = self.df
        sf = source_features or sources
        if sf is None:
            raise ValueError("Provide source_features (or sources=) argument.")
        source_features_list = []
        self._logger.info("Build long table")
        for entry in sf:
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

        self._long = pd.concat(frames, ignore_index=True)
        return self

    def filter_data(self, df: pd.DataFrame, drop_rows_list: Optional[tuple] = None) -> pd.DataFrame:
        """Filter rows using stored or provided drop rules."""
        if drop_rows_list is None and self.filter_config is not None:
            drop_rows_list = self.filter_config.drop_rows
        self._logger.info(f"Filter data: rows={len(drop_rows_list or ())}")
        return drop_rows(df, drop_rows_list or ())

    # -- Convenience chainable helpers -----------------------------------------

    def label_conditions(self, mapping: Dict[str, str], column: str = "Condition") -> "Bench":
        """Map Session values to condition labels on the long table.

        ``mapping`` is ``{"ses-01": "baseline", "ses-02": "saline", ...}``.
        Returns self for chaining.
        """
        self.long[column] = self.long["Session"].map(mapping)
        return self

    def save_and_close(self, fig, name: str, folder: str = "plots", dpi: int = 300) -> Path:
        """Save a figure and close it. Returns the output path."""
        path = self.save_figure(fig, name=name, folder=folder, dpi=dpi)
        plt.close(fig)
        return path

    def save_tables(self, result: Optional[AnalysisResult] = None, prefix: Optional[str] = None, folder: str = "stats") -> "Bench":
        """Save tables from the current (or given) result. Returns self."""
        if result is None:
            result = self.result
        self.save_analysis_result_tables(result, prefix=prefix, folder=folder)
        return self

    def save_table(self, df: pd.DataFrame, name: str = "table.csv", folder: str = "stats") -> Path:
        """Save a table to the configured output folders."""
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
        self.track_output(path, "tables")
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
        """Write a provenance record for the current run."""
        config_dir = self.output_paths.config  # type: ignore[union-attr]
        config_dir.mkdir(parents=True, exist_ok=True)
        path, summary_path = provenance.save_provenance(self, output_dir=config_dir, name=name)
        self.track_output(path, "other")
        self.track_output(summary_path, "other")
        self._log_saved_outputs("Save provenance", path)
        return path

    def save_run_summary(
        self,
        name: str = "run.json",
        params: Optional[Mapping[str, Any]] = None,
        notes: Optional[str] = None,
        outputs: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Write a compact run summary."""
        config_dir = self.output_paths.config  # type: ignore[union-attr]
        config_dir.mkdir(parents=True, exist_ok=True)

        if params is None:
            params_payload: Dict[str, Any] = {
                "analyses": [
                    {"name": e["name"], "kwargs": e.get("kwargs", {})}
                    for e in self._invoked if e["kind"] == "analysis"
                ],
                "plots": [
                    {"name": e["name"], "kwargs": e.get("kwargs", {})}
                    for e in self._invoked if e["kind"] == "plot"
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
            "invoked": self._invoked,
            "outputs": outputs_payload,
        }

        path = config_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        self.track_output(path, "other")
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
        """Save a matplotlib figure to the output folders."""
        out_dir = getattr(self.output_paths, folder)
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        self.track_output(path, "figures")
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
        """Export a per-row feature report from a nested signal column."""
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
        self.track_output(out_path, "tables")
        return out_path

    @staticmethod
    def _make_output_paths(cfg: IOConfig) -> OutputPaths:
        # Structure: output_root / dataset_alias / script_name / YYMMDD / tag
        dataset_alias = _resolve_dataset_alias_for_output(cfg.input_path)
        script_name = cfg.script_name or "unknown"
        date_stamp = datetime.now().strftime("%y%m%d")
        tag = cfg.tag or "untagged"
        run_dir = cfg.output_root / dataset_alias / script_name / date_stamp / tag
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
