"""Persistent training-time tracker.

Tracks two quantities the user actually wants to see in the viewer:

* **Total trained seconds** — cumulative across *every* training run on
  this machine. Persists to disk so PC-reboots, backend restarts, and
  watchdog-driven trainer restarts don't reset the counter.
* **Current session seconds** — wall-clock since the *currently running*
  trainer process started. Derived on-the-fly from the running
  :class:`TrainingJob`'s ``started_at`` field, so it auto-resets
  whenever a trainer is launched, killed, or crashes.

Persistence file: ``models/logs/training_clock.json`` with the
shape::

    {"total_trained_seconds": 19800.5, "last_update": 1779648000.123}

Concurrency model: the backend is single-process async, so we don't
need real cross-process locking — but we *do* use atomic
write-to-tmp + ``os.replace`` so a crash mid-write can't leave a
half-written JSON on disk that would crash the next boot.

The increment logic intentionally clamps the per-tick delta with a
small cap (``_MAX_TICK_DELTA_SECONDS``) so a system that slept for
hours doesn't suddenly inflate the counter by the entire sleep
duration — the cap is generous enough to absorb a watchdog restart
or a long ``tick`` interval but short enough to keep absurd
clock-skew jumps bounded.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

__all__ = [
    "TrainingClock",
    "get_clock",
    "set_clock_path",
]

_LOG = logging.getLogger("kivski_api.training_clock")

# Per-tick clamp: if the caller went silent for longer than this, attribute
# only this much to "training time" (anything more is almost certainly the
# host machine sleeping, a long pause between ticks, or a clock skew). The
# value is generous enough that a 15 s lifespan tick or a watchdog
# 10 s poll cycle is fully absorbed.
_MAX_TICK_DELTA_SECONDS: float = 60.0

# Default path -- callers (lifespan, tests) can override via set_clock_path.
_DEFAULT_PATH_RELATIVE = Path("models") / "logs" / "training_clock.json"


def _repo_root() -> Path:
    """Walk up from this file to the repo root (where pyproject.toml lives)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[3]


def _default_path() -> Path:
    """Resolve ``models/logs/training_clock.json`` relative to the repo."""
    return _repo_root() / _DEFAULT_PATH_RELATIVE


class TrainingClock:
    """File-backed counter of cumulative training time.

    Thread-safe via an internal ``threading.Lock`` so the lifespan tick
    task and any HTTP request handler can both call into ``tick`` /
    ``to_dict`` without racing on the in-memory state.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path: Path = Path(path) if path is not None else _default_path()
        self._lock = threading.Lock()
        self._total_seconds: float = 0.0
        # last_update tracks the wall-clock time of the previous tick
        # call. ``None`` means "no previous tick" -- the very first
        # tick after construction starts a fresh interval.
        self._last_update: float | None = None
        self._load()

    # ---------------------------------------------------------------- IO

    def _load(self) -> None:
        """Best-effort read of the on-disk state. Missing/broken file is fine."""
        try:
            if not self._path.is_file():
                return
            raw = self._path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                _LOG.warning("TrainingClock: %s has wrong shape, ignoring", self._path)
                return
            total = payload.get("total_trained_seconds", 0.0)
            last = payload.get("last_update")
            if isinstance(total, (int, float)) and total >= 0:
                self._total_seconds = float(total)
            if isinstance(last, (int, float)) and last > 0:
                self._last_update = float(last)
        except (OSError, ValueError):
            _LOG.exception("TrainingClock: failed to load %s, starting fresh", self._path)
            self._total_seconds = 0.0
            self._last_update = None

    def _persist_locked(self) -> None:
        """Write current state atomically. Caller must hold ``self._lock``."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "total_trained_seconds": float(self._total_seconds),
                "last_update": float(self._last_update) if self._last_update is not None else 0.0,
            }
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            # Atomic rename: replaces existing file in one syscall on POSIX
            # and Windows (Python uses MoveFileEx under the hood).
            os.replace(tmp, self._path)
        except OSError:
            _LOG.exception("TrainingClock: failed to persist to %s", self._path)

    # -------------------------------------------------------------- API

    def tick(self, now_unix: float, training_running: bool) -> None:
        """Advance the counter.

        If ``training_running`` is True and we have a previous
        ``last_update``, add the elapsed wall-clock to the total --
        clamped to ``_MAX_TICK_DELTA_SECONDS`` so a long sleep doesn't
        inflate the counter. Always updates ``last_update`` and
        persists the new state to disk.
        """
        with self._lock:
            prev = self._last_update
            if training_running and prev is not None:
                delta = max(0.0, float(now_unix) - float(prev))
                if delta > _MAX_TICK_DELTA_SECONDS:
                    delta = _MAX_TICK_DELTA_SECONDS
                self._total_seconds += delta
            self._last_update = float(now_unix)
            self._persist_locked()

    def to_dict(
        self,
        *,
        session_started_at: float | None = None,
        now_unix: float | None = None,
    ) -> dict[str, float]:
        """Snapshot for the API layer.

        ``session_started_at`` is the running trainer's ``started_at``
        (epoch seconds); when ``None`` the current session is considered
        idle and ``current_session_seconds`` is 0. ``now_unix`` is taken
        from the caller so HTTP responses see a coherent now (passed in
        once, used for both fields).
        """
        with self._lock:
            total = float(self._total_seconds)
        if session_started_at is not None and now_unix is not None:
            session = max(0.0, float(now_unix) - float(session_started_at))
        else:
            session = 0.0
        return {
            "total_seconds": total,
            "current_session_seconds": session,
        }

    # ----------------------------------------------------- introspection

    @property
    def path(self) -> Path:
        return self._path

    @property
    def total_seconds(self) -> float:
        with self._lock:
            return float(self._total_seconds)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


_CLOCK: TrainingClock | None = None


def set_clock_path(path: Path | str | None) -> None:
    """Reset the module-level clock to use a fresh path.

    Mostly useful for tests that want to point at a tmp_path; the FastAPI
    lifespan calls ``get_clock()`` with the default path on boot.
    """
    global _CLOCK
    _CLOCK = TrainingClock(path)


def get_clock() -> TrainingClock:
    """Return the process-wide :class:`TrainingClock` singleton."""
    global _CLOCK
    if _CLOCK is None:
        _CLOCK = TrainingClock()
    return _CLOCK
