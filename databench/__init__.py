"""databench — reproducible analysis for multiindex neuroscience datasets.

The core is small: open a :class:`~databench.project.Project`, select
sessions, run your analysis, and save outputs through a
:class:`~databench.run.Run`, which records a single verifiable
``provenance.json`` for every run.

Analysis algorithms (locomotion, oscillation, eta, …) are plain library
functions under ``databench.analysis`` — import them directly where needed.
"""
from databench.provenance import _databench_version

__version__ = _databench_version()

from databench.utils.logger import setup_logging as _setup_logging

_setup_logging()

from databench.project import Project
from databench.config import resolve_dataset, dataset_params
from databench.plotting import set_theme

__all__ = [
    "Project",
    "resolve_dataset",
    "dataset_params",
    "set_theme",
    "__version__",
]
