"""Structured JSON logging.

One line of JSON per event, with a correlation id so a request or a replay run
can be followed across the pipeline. Only the fields explicitly attached are
emitted: environment variables and configuration values are never logged, so a
stack trace cannot leak a path or a token into an aggregator.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from typing import Any

#: Correlation id for the current request or replay run.
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="-")

#: Attributes present on every LogRecord; anything else was attached by the
#: caller and is treated as structured context.
_STANDARD = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Renders records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Format one record.

        Args:
            record: The record to format.

        Returns:
            A JSON object on one line.
        """
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            # The type and message only; the full traceback goes to stderr in
            # development but is not embedded in the structured payload.
            exc_type = record.exc_info[0]
            payload["error_type"] = exc_type.__name__ if exc_type else "unknown"
            payload["error"] = str(record.exc_info[1])
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger.

    Args:
        level: Logging level name.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def new_correlation_id() -> str:
    """Generate and install a fresh correlation id."""
    value = uuid.uuid4().hex[:12]
    correlation_id.set(value)
    return value
