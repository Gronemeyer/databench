"""Simple centralized logging for databench with ANSI colors and aligned columns.

Just import and use::

    from databench.utils.logger import get_logger
    _log = get_logger(__name__)
    _log.info("Hello world")

For analysis entry-points, decorate with ``@log_run``::

    from databench.utils.logger import log_run

    @log_run
    def run(self, df):
        ...

``log_run`` logs at INFO on entry/exit with smart arg summaries and
elapsed wall-clock time.  ``log_this_fr`` remains available for verbose
DEBUG-level tracing of every call (e.g. per-row feature extractors).
"""

import functools
import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger as _logger

_configured = False
_log_dir: Optional[Path] = None

_LOG_FMT = "{time:HH:mm:ss} | <level>{level:<8}</level> | [<cyan>{extra[name]}</cyan>] --> {message}"


def install_excepthook() -> None:
    """Log uncaught exceptions to the databench log file."""

    def _handle(exc_type, exc_value, exc_traceback):
        logger = get_logger("databench")
        logger.opt(exception=(exc_type, exc_value, exc_traceback)).error("Uncaught exception")
        if _log_dir is not None:
            log_file = _log_dir / "databench.log"
            print(f"Uncaught exception. See {log_file} for details.")

    sys.excepthook = _handle


def setup_logging(log_dir: Optional[str] = None, level: str = "INFO") -> None:
    global _configured, _log_dir
    if _configured:
        return

    if log_dir:
        _log_dir = Path(log_dir)
    else:
        project_root = Path(__file__).resolve().parent.parent
        _log_dir = project_root / "logs"

    _log_dir.mkdir(parents=True, exist_ok=True)

    _logger.remove()
    _logger.add(
        sys.stderr,
        level=level.upper(),
        format=_LOG_FMT,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    _logger.add(
        _log_dir / "databench.log",
        level="DEBUG",
        format=_LOG_FMT,
        rotation="00:00",
        retention=7,
        encoding="utf-8",
    )

    install_excepthook()

    _configured = True

    try:
        from databench._provenance import _databench_version
        get_logger("databench").info(f"databench {_databench_version()}")
    except Exception:
        pass


def log_this_fr(func):
    """Decorator that logs entry, exit (and exceptions) of the function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Entering {func.__qualname__} args={args!r}, kwargs={kwargs!r}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__qualname__} returned {result!r}")
            return result
        except Exception:
            logger.exception(f"Exception in {func.__qualname__}")
            raise

    return wrapper


# ── Helpers for log_run ────────────────────────────────────────────────────

def _summarize_arg(obj: object) -> str:
    """Return a compact, human-readable summary of *obj* for log messages."""
    import numpy as np
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return f"DataFrame({obj.shape[0]}x{obj.shape[1]})"
    if isinstance(obj, pd.Series):
        return f"Series(len={len(obj)})"
    if isinstance(obj, np.ndarray):
        return f"ndarray(shape={obj.shape})"
    # SessionGroup (has __len__ and 'label' is absent)
    cls = type(obj).__name__
    if cls == "SessionGroup":
        return f"SessionGroup(n={len(obj)})"
    if cls == "Session":
        label = getattr(obj, "label", None)
        return f"Session({label})" if label else "Session"
    if cls == "AlignedData":
        return "AlignedData"
    return cls


def _summarize_result(obj: object) -> str:
    """Return a compact summary of an analysis result for log messages."""
    import pandas as pd
    if isinstance(obj, pd.DataFrame):
        return f"DataFrame({obj.shape[0]}x{obj.shape[1]})"
    return type(obj).__name__


_LOG_RUN_ATTR = "__log_run_wrapped__"


def log_run(func):
    """INFO-level decorator for analysis entry points.

    Logs method name, summarized arguments, wall-clock elapsed time,
    and a compact summary of the return value.  On exception, logs at
    ERROR with full traceback.

    Safe to stack with ``@log_this_fr`` — this decorator sets a sentinel
    attribute (``__log_run_wrapped__``) so it won't double-wrap.
    """
    if getattr(func, _LOG_RUN_ATTR, False):
        return func  # already wrapped

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Resolve a readable name for the logger tag
        self = args[0] if args else None
        qualname = func.__qualname__
        tag = qualname

        logger = get_logger(tag)

        # Summarize positional args (skip self)
        parts = [_summarize_arg(a) for a in args[1:]]
        for k, v in kwargs.items():
            parts.append(f"{k}={_summarize_arg(v)}")
        arg_str = ", ".join(parts) if parts else ""

        logger.info(f"Starting | {arg_str}" if arg_str else "Starting")
        t0 = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            logger.info(f"Completed ({elapsed:.2f}s) | {_summarize_result(result)}")
            return result
        except Exception:
            elapsed = time.perf_counter() - t0
            logger.exception(f"Failed ({elapsed:.2f}s)")
            raise

    setattr(wrapper, _LOG_RUN_ATTR, True)
    return wrapper


def get_logger(name: str):
    if not _configured:
        setup_logging()
    return _logger.bind(name=name)
