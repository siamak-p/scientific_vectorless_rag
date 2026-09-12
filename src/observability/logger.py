"""Structured application logging.

Log records are emitted both to the console and to a rotating file inside the
application home directory. Every record carries a timestamp, level, module,
operation and — when relevant — structured error information.

The package is deliberately named ``observability`` rather than ``logging`` so
that it can never shadow the standard library module.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from typing import Any

from core.constants import APP_SLUG, LOG_DIR, ensure_directories

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_MAX_BYTES = 2 * 1024 * 1024
_BACKUP_COUNT = 3

_SENSITIVE_KEYS = ("api_key", "apikey", "token", "secret", "password", "authorization")

_configured = False


class _ConsoleFormatter(logging.Formatter):
    """Console output that keeps the diagnostic context of a problem.

    Only warnings and above carry their context: an operator watching the
    terminal needs to know *why* something failed without opening the JSON
    log, while routine progress lines stay on one readable line.
    """

    def __init__(self) -> None:
        super().__init__(_CONSOLE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        context = getattr(record, "context", None)
        if record.levelno < logging.WARNING or not isinstance(context, dict) or not context:
            return line

        details = " ".join(
            f"{key}={value}" for key, value in _redact(context).items() if value not in (None, "")
        )
        return f"{line} | {details}" if details else line


class _JsonFileFormatter(logging.Formatter):
    """Serialise log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "module": record.name,
            "operation": getattr(record, "operation", record.funcName),
            "message": record.getMessage(),
        }

        context = getattr(record, "context", None)
        if isinstance(context, dict) and context:
            payload["context"] = _redact(context)

        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            payload["error"] = {
                "type": exc_type.__name__ if exc_type else "Exception",
                "message": str(exc_value),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(payload, ensure_ascii=False, default=str)


def _redact(context: dict[str, Any]) -> dict[str, Any]:
    """Replace values of sensitive looking keys so secrets never reach the logs."""
    safe: dict[str, Any] = {}
    for key, value in context.items():
        if any(marker in key.lower() for marker in _SENSITIVE_KEYS):
            safe[key] = "***redacted***"
        elif isinstance(value, dict):
            safe[key] = _redact(value)
        else:
            safe[key] = value
    return safe


def configure_logging(level: int = logging.INFO) -> None:
    """Install the console and rotating file handlers exactly once."""
    global _configured
    if _configured:
        return

    ensure_directories()

    root = logging.getLogger(APP_SLUG)
    root.setLevel(level)
    root.propagate = False

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(_ConsoleFormatter())
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / f"{APP_SLUG}.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_JsonFileFormatter())
    root.addHandler(file_handler)

    # Third party libraries are noisy at INFO level.
    for noisy in ("httpx", "httpcore", "urllib3", "arxiv", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced application logger."""
    configure_logging()
    return logging.getLogger(f"{APP_SLUG}.{name}")


def log_event(
    logger: logging.Logger,
    operation: str,
    message: str,
    level: int = logging.INFO,
    exc_info: bool = False,
    **context: Any,
) -> None:
    """Emit a structured event with an explicit operation name and context."""
    logger.log(
        level, message, exc_info=exc_info, extra={"operation": operation, "context": context}
    )
