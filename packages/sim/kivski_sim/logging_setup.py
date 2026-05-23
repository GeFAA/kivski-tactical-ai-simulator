"""Centralized logging configuration for the simulator.

By default we render colourful, human-readable logs via ``rich``. With
``json=True`` we instead emit one JSON object per line which is easy to ingest
in CI logs, container stdout, or downstream tooling.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

__all__ = ["setup_logging"]


_LEVEL_MAP: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


class _JsonFormatter(logging.Formatter):
    """One-JSON-object-per-line formatter for structured environments."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Preserve any extra fields the caller attached.
        for k, v in record.__dict__.items():
            if k in payload or k.startswith("_"):
                continue
            if k in (
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "message",
                "module",
                "msecs",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "taskName",
            ):
                continue
            try:
                json.dumps(v, default=str)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)
        return json.dumps(payload, default=str)


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return _LEVEL_MAP.get(level.upper(), logging.INFO)


def setup_logging(level: str = "INFO", json: bool = False) -> None:
    """Configure the root logger.

    Repeated calls are idempotent: existing handlers are removed first so unit
    tests and CLI entry-points can safely re-initialise without duplicating
    log lines.
    """
    root = logging.getLogger()
    root.setLevel(_resolve_level(level))

    # Clear existing handlers to keep this call idempotent.
    for h in list(root.handlers):
        root.removeHandler(h)

    if json:
        handler: logging.Handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(_JsonFormatter())
    else:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
                markup=False,
            )
            handler.setFormatter(logging.Formatter(fmt="%(module)s:%(lineno)d | %(message)s"))
        except ImportError:  # pragma: no cover - rich is a hard dep, but be safe
            handler = logging.StreamHandler(stream=sys.stderr)
            handler.setFormatter(
                logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(module)s:%(lineno)d | %(message)s")
            )

    handler.setLevel(_resolve_level(level))
    root.addHandler(handler)
