"""Automatic run naming for telemetry output directories.

A "run" in Kivski corresponds to one invocation of the trainer (or one
evaluation sweep). The directory name has to be unique enough to never
collide between concurrent processes, sortable in chronological order,
and short enough to fit in a terminal window. The format used here is::

    {prefix}-YYYYMMDD-HHMMSS-{shortuid8}

For example: ``kivski-20260523-091122-3f9c1a04``.

The eight-hex-character suffix is taken from :func:`uuid.uuid4` and
provides ~3.4 * 10^9 distinct values per second, which is plenty to
de-dupe across a fleet of training workers.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

__all__ = [
    "generate_run_name",
    "latest_run_name",
    "list_runs",
]


def generate_run_name(prefix: str = "kivski") -> str:
    """Build a fresh run name with a timestamp and a short uid.

    Args:
        prefix: Leading token (e.g. ``"kivski"``, ``"eval"``,
            ``"sweep01"``). May not be empty.

    Returns:
        A string of the form ``"{prefix}-YYYYMMDD-HHMMSS-{shortuid8}"``.

    Raises:
        ValueError: If ``prefix`` is empty or contains whitespace.
    """
    if not prefix or any(ch.isspace() for ch in prefix):
        raise ValueError(f"prefix must be a non-empty whitespace-free string, got {prefix!r}")
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    short = uuid.uuid4().hex[:8]
    return f"{prefix}-{ts}-{short}"


def list_runs(log_dir: Path) -> list[str]:
    """Return all run-directory names in ``log_dir``, sorted oldest first.

    Sorting is done by mtime so that runs created in the same second are
    still ordered deterministically. Hidden directories (leading ``.``)
    are ignored.

    If ``log_dir`` does not exist, returns an empty list rather than
    raising -- this is the most useful behaviour for fresh checkouts.
    """
    root = Path(log_dir)
    if not root.exists():
        return []
    entries = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    entries.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return [p.name for p in entries]


def latest_run_name(log_dir: Path) -> str | None:
    """Return the most recently created run-directory name, or ``None``.

    Useful for resume-from-latest workflows in the trainer.
    """
    runs = list_runs(log_dir)
    return runs[-1] if runs else None
