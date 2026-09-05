"""Colored console logging formatter.

Adds ANSI color coding to log output for visual clarity. Colors are applied
only when stdout is a TTY (interactive terminal); when output is redirected to
a file or pipe, colors are disabled automatically to keep logs clean.
"""

from __future__ import annotations

import logging
import sys

# ANSI escape codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_COLORS = {
    "DEBUG": "\033[36m",      # cyan
    "INFO": "\033[32m",       # green
    "WARNING": "\033[33m",    # yellow
    "ERROR": "\033[31m",      # red
    "CRITICAL": "\033[1;31m",  # bold red
}

# Color for the logger name (module) portion.
_NAME_COLOR = "\033[34m"  # blue
_TIME_COLOR = "\033[90m"  # bright black / gray


class ColoredFormatter(logging.Formatter):
    """Formatter that colorizes level names and module names."""

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, use_color: bool | None = None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        # Auto-detect TTY unless explicitly overridden.
        self.use_color = use_color if use_color is not None else sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_color:
            return super().format(record)

        level_color = _COLORS.get(record.levelname, "")
        # Colorize the level name and module name in-place.
        original_levelname = record.levelname
        original_name = record.name
        record.levelname = f"{level_color}{record.levelname}{_RESET}"
        record.name = f"{_NAME_COLOR}{record.name}{_RESET}"

        # Colorize the timestamp by wrapping the asctime portion.
        result = super().format(record)

        # Restore for any downstream handlers.
        record.levelname = original_levelname
        record.name = original_name

        return result


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a colored console handler."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter(
            fmt=f"{_TIME_COLOR}%(asctime)s{_RESET} [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
