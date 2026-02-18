"""Simple centralized logging for databench with ANSI colors and aligned columns.

Just import and use:
    from databench._utils._logger import get_logger
    logger = get_logger("MyClass")
    logger.info("Hello world")
"""

import functools
import sys
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


def get_logger(name: str):
    if not _configured:
        setup_logging()
    return _logger.bind(name=name)
