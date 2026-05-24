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

    {
        "total_trained_seconds": 19800.5,
        "last_update": 1779648000.123,
        "runs": {
            "api-20260524-...": {
                "max_env_steps": 356352,
                "frame_skip": 4,
                "num_envs": 48,
                "tick_dt": 0.1,
                "scanned_at": 1779650000.0
            },
            ...
        }
    }

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

The aggregated *simulated* game-time accounting is the second
responsibility of this module: every parallel env in a vec-env runs
``frame_skip`` engine ticks per env_step, each tick advances the
simulated clock by ``tick_dt = 1/tick_rate_hz``. The clock scans all
per-run ``models/logs/<run>/metrics.jsonl`` files (and falls back to
``models/checkpoints/**/*.pt.json`` sidecars for runs that pre-date
the ``live/env_steps`` log key) and caches each run's max env_steps
+ frame_skip + tick_dt under ``runs`` so subsequent calls don't
re-tail every JSONL on disk.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
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

# Conservative defaults applied when a run's metrics.jsonl / checkpoint
# sidecars don't record the simulation cadence. Match the sim defaults
# in ``packages/sim/kivski_sim/config.py`` (tick_rate_hz=10, frame_skip=1).
# If frame_skip wasn't captured we *intentionally* use the trainer-side
# default rather than 1, because every training preset we ship overrides
# frame_skip > 1 (fast=4, turbo=6). Using 1 here would under-count by 4-6×.
_FALLBACK_FRAME_SKIP: float = 4.0
_FALLBACK_TICK_DT: float = 0.1  # 10 Hz

# Per-JSONL scan ceiling. Reading the entire file is fine — they're typically
# a few hundred KB — but we cap how much we scan in one call so a runaway
# log doesn't lock the event loop. Most runs finish well under this.
_MAX_JSONL_BYTES: int = 32 * 1024 * 1024  # 32 MiB


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


def _logs_root() -> Path:
    return _repo_root() / "models" / "logs"


def _checkpoints_root() -> Path:
    return _repo_root() / "models" / "checkpoints"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    # NaN/inf guard. NaN is the only value where ``x != x`` is True, so the
    # comparison filters it without importing math (keeps this module
    # zero-dep). ``inf`` and ``-inf`` would happily multiply through and
    # break the aggregator downstream, so they're guarded explicitly.
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return v


def _scan_jsonl_for_env_steps(path: Path) -> dict[str, float] | None:
    """Return ``{max_env_steps, frame_skip, tick_dt, num_envs}`` from a metrics.jsonl.

    Walks every record from newest-known relevant key first; we just
    sweep the file linearly because the JSONL is line-delimited and the
    max env_steps shows up monotonically — keeping a running max is
    cheap. Returns ``None`` when the file is missing, unreadable, or
    contained no env_steps record.

    Sniffs both ``live/env_steps`` (preferred — emitted by the trainer
    every update from May 2026 onwards) and ``train/env_steps``. Sim
    cadence keys (``*/frame_skip``, ``*/tick_dt``, ``*/num_envs``) are
    snapped to the last seen value.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    # Empty file = nothing to learn; very-large files would block the
    # event loop on a synchronous read so we bail and let the caller
    # fall back to checkpoint sidecars (which are tiny).
    if st.st_size == 0 or st.st_size > _MAX_JSONL_BYTES:
        return None
    max_env_steps = 0.0
    frame_skip: float | None = None
    tick_dt: float | None = None
    num_envs: float | None = None
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line[0] != "{":
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                # env_steps: prefer train/, fall back to live/.
                v = rec.get("train/env_steps")
                if v is None:
                    v = rec.get("live/env_steps")
                if isinstance(v, (int, float)):
                    fv = _safe_float(v, 0.0)
                    if fv > max_env_steps:
                        max_env_steps = fv
                # Sim cadence — last seen wins.
                fs = rec.get("train/frame_skip", rec.get("live/frame_skip"))
                if isinstance(fs, (int, float)):
                    frame_skip = _safe_float(fs, _FALLBACK_FRAME_SKIP)
                td = rec.get("train/tick_dt", rec.get("live/tick_dt"))
                if isinstance(td, (int, float)):
                    tick_dt = _safe_float(td, _FALLBACK_TICK_DT)
                ne = rec.get("train/num_envs", rec.get("live/num_envs"))
                if isinstance(ne, (int, float)):
                    num_envs = _safe_float(ne, 0.0)
    except OSError:
        return None

    if max_env_steps <= 0:
        return None
    return {
        "max_env_steps": float(max_env_steps),
        "frame_skip": float(frame_skip) if frame_skip is not None else _FALLBACK_FRAME_SKIP,
        "tick_dt": float(tick_dt) if tick_dt is not None else _FALLBACK_TICK_DT,
        "num_envs": float(num_envs) if num_envs is not None else 0.0,
    }


def _scan_checkpoint_sidecars_for_run(run_name: str) -> dict[str, float] | None:
    """Fallback: pull max env_steps from ``models/checkpoints/<run>/*.pt.json``.

    Used for historical runs that pre-date the ``train/env_steps`` log
    key — the trainer always wrote env_steps into the per-checkpoint
    sidecar even before metrics.jsonl carried it, so we can still
    aggregate retroactively.
    """
    ckpt_dir = _checkpoints_root() / run_name
    if not ckpt_dir.is_dir():
        return None
    max_env_steps = 0.0
    for sidecar in ckpt_dir.glob("*.pt.json"):
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        v = payload.get("env_steps")
        if isinstance(v, (int, float)):
            fv = _safe_float(v, 0.0)
            if fv > max_env_steps:
                max_env_steps = fv
    if max_env_steps <= 0:
        return None
    return {
        "max_env_steps": float(max_env_steps),
        "frame_skip": _FALLBACK_FRAME_SKIP,
        "tick_dt": _FALLBACK_TICK_DT,
        "num_envs": 0.0,
    }


def _scan_hparams_and_train_step_for_run(run_name: str) -> dict[str, float] | None:
    """Last-resort fallback: derive env_steps from ``hparams.json + max(train/step)``.

    For runs that pre-date the ``train/env_steps`` log key *and* had
    their checkpoints pruned, we can still reconstruct env_steps as::

        env_steps ≈ max(train/step) × rollout_steps × num_envs

    because every PPO update consumes exactly ``rollout_steps`` env
    steps from each of ``num_envs`` parallel envs. ``hparams.json``
    carries those constants and the JSONL carries the update-step
    counter, so the product is exact (modulo any in-progress rollout
    that didn't finish before the trainer was killed).

    Returns ``None`` if either file is missing or doesn't carry the
    relevant keys; in practice this hits for runs from a build that
    didn't yet log either env_steps or train/step.
    """
    run_dir = _logs_root() / run_name
    hparams_path = run_dir / "hparams.json"
    jsonl_path = run_dir / "metrics.jsonl"
    if not hparams_path.is_file() or not jsonl_path.is_file():
        return None

    try:
        hp = json.loads(hparams_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(hp, dict):
        return None
    num_envs = _safe_float(hp.get("num_envs"), 0.0)
    rollout_steps = _safe_float(hp.get("rollout_steps"), 0.0)
    if num_envs <= 0 or rollout_steps <= 0:
        return None
    tick_rate_hz = _safe_float(hp.get("tick_rate_hz"), 0.0)
    tick_dt = 1.0 / tick_rate_hz if tick_rate_hz > 0 else _FALLBACK_TICK_DT
    frame_skip = _safe_float(hp.get("frame_skip"), _FALLBACK_FRAME_SKIP)

    try:
        st = jsonl_path.stat()
    except OSError:
        return None
    if st.st_size == 0 or st.st_size > _MAX_JSONL_BYTES:
        return None

    max_train_step = 0.0
    try:
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line[0] != "{":
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                v = rec.get("train/step")
                if isinstance(v, (int, float)):
                    fv = _safe_float(v, 0.0)
                    if fv > max_train_step:
                        max_train_step = fv
    except OSError:
        return None

    if max_train_step <= 0:
        return None
    env_steps = max_train_step * rollout_steps * num_envs
    return {
        "max_env_steps": float(env_steps),
        "frame_skip": float(frame_skip),
        "tick_dt": float(tick_dt),
        "num_envs": float(num_envs),
    }


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
        # Per-run aggregation cache: {run_name: {max_env_steps, frame_skip, tick_dt, num_envs, scanned_at}}.
        self._runs: dict[str, dict[str, float]] = {}
        # Last time we refreshed the aggregated scan. Throttled to keep
        # `/api/training/clock` cheap even when polled every few seconds.
        self._last_scan_at: float = 0.0
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
            runs = payload.get("runs")
            if isinstance(runs, dict):
                cleaned: dict[str, dict[str, float]] = {}
                for name, info in runs.items():
                    if not isinstance(name, str) or not isinstance(info, dict):
                        continue
                    cleaned[name] = {
                        "max_env_steps": _safe_float(info.get("max_env_steps"), 0.0),
                        "frame_skip": _safe_float(
                            info.get("frame_skip"), _FALLBACK_FRAME_SKIP
                        ),
                        "tick_dt": _safe_float(info.get("tick_dt"), _FALLBACK_TICK_DT),
                        "num_envs": _safe_float(info.get("num_envs"), 0.0),
                        "scanned_at": _safe_float(info.get("scanned_at"), 0.0),
                    }
                self._runs = cleaned
        except (OSError, ValueError):
            _LOG.exception("TrainingClock: failed to load %s, starting fresh", self._path)
            self._total_seconds = 0.0
            self._last_update = None
            self._runs = {}

    def _persist_locked(self) -> None:
        """Write current state atomically. Caller must hold ``self._lock``."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "total_trained_seconds": float(self._total_seconds),
                "last_update": float(self._last_update) if self._last_update is not None else 0.0,
                "runs": self._runs,
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

    # ----------------------------- aggregated simulated-time accounting

    def compute_total_simulated_seconds(
        self,
        *,
        min_scan_interval_seconds: float = 10.0,
        current_run_name: str | None = None,
    ) -> dict[str, Any]:
        """Return aggregated *simulated* game-time across every run on disk.

        Walks ``models/logs/<run>/metrics.jsonl`` for env_steps, falling
        back to ``models/checkpoints/<run>/*.pt.json`` for runs where
        ``train/env_steps`` wasn't logged. Multiplies by frame_skip ×
        tick_dt → game-seconds, sums across runs → "agent game-time".

        The scan result is cached in ``self._runs`` and persisted to
        disk so subsequent calls (typically every 5 seconds from the
        frontend poll) only re-scan when:

          * the throttle elapsed (``min_scan_interval_seconds``), or
          * a new run dir appeared, or
          * ``current_run_name``'s JSONL grew since the last scan.

        Returns ``{total_simulated_seconds, current_session_simulated_seconds,
        runs_scanned, total_env_steps}`` ready for the API layer.
        """
        now = time.time()
        with self._lock:
            should_scan = (now - self._last_scan_at) >= float(min_scan_interval_seconds)
        if should_scan or current_run_name is not None:
            self._refresh_run_cache(current_run_name=current_run_name)

        with self._lock:
            total_sim_seconds = 0.0
            total_env_steps = 0.0
            for info in self._runs.values():
                env_steps = _safe_float(info.get("max_env_steps"), 0.0)
                frame_skip = _safe_float(info.get("frame_skip"), _FALLBACK_FRAME_SKIP)
                tick_dt = _safe_float(info.get("tick_dt"), _FALLBACK_TICK_DT)
                total_env_steps += env_steps
                total_sim_seconds += env_steps * frame_skip * tick_dt

            current_info: dict[str, float] | None = None
            if current_run_name and current_run_name in self._runs:
                current_info = dict(self._runs[current_run_name])

            runs_scanned = len(self._runs)

        if current_info is not None:
            current_env_steps = _safe_float(current_info.get("max_env_steps"), 0.0)
            current_frame_skip = _safe_float(
                current_info.get("frame_skip"), _FALLBACK_FRAME_SKIP
            )
            current_tick_dt = _safe_float(current_info.get("tick_dt"), _FALLBACK_TICK_DT)
            current_sim_seconds = current_env_steps * current_frame_skip * current_tick_dt
            current_num_envs = _safe_float(current_info.get("num_envs"), 0.0)
        else:
            current_env_steps = 0.0
            current_frame_skip = 0.0
            current_tick_dt = 0.0
            current_sim_seconds = 0.0
            current_num_envs = 0.0

        return {
            "total_simulated_seconds": float(total_sim_seconds),
            "total_env_steps": float(total_env_steps),
            "current_session_simulated_seconds": float(current_sim_seconds),
            "current_session_env_steps": float(current_env_steps),
            "current_session_num_envs": float(current_num_envs),
            "current_session_frame_skip": float(current_frame_skip),
            "current_session_tick_dt": float(current_tick_dt),
            "runs_scanned": int(runs_scanned),
        }

    def _refresh_run_cache(self, *, current_run_name: str | None = None) -> None:
        """Re-scan ``models/logs/*`` and ``models/checkpoints/*`` for env_steps.

        Caches each run's aggregate in ``self._runs`` and persists to
        disk. Safe to call from any thread.
        """
        logs_root = _logs_root()
        ckpt_root = _checkpoints_root()
        candidate_runs: set[str] = set()

        if logs_root.is_dir():
            for child in logs_root.iterdir():
                if child.is_dir():
                    candidate_runs.add(child.name)
        if ckpt_root.is_dir():
            for child in ckpt_root.iterdir():
                if child.is_dir():
                    candidate_runs.add(child.name)

        now = time.time()
        with self._lock:
            existing = dict(self._runs)

        updated: dict[str, dict[str, float]] = dict(existing)
        scanned_now = 0
        for run_name in candidate_runs:
            jsonl = logs_root / run_name / "metrics.jsonl"
            jsonl_mtime = 0.0
            if jsonl.is_file():
                try:
                    jsonl_mtime = jsonl.stat().st_mtime
                except OSError:
                    jsonl_mtime = 0.0
            cached = existing.get(run_name)
            # Skip re-scan when the JSONL hasn't been touched since the
            # last scan and we already have an entry — unless this is
            # the currently-running run, in which case we always re-tail
            # so the "current session" gauge tracks env_steps live.
            if (
                cached is not None
                and current_run_name != run_name
                and jsonl_mtime > 0
                and cached.get("scanned_at", 0.0) >= jsonl_mtime
                and cached.get("max_env_steps", 0.0) > 0
            ):
                continue
            info = _scan_jsonl_for_env_steps(jsonl) if jsonl.is_file() else None
            if info is None:
                # JSONL had no env_steps record; try checkpoint sidecars.
                info = _scan_checkpoint_sidecars_for_run(run_name)
            if info is None:
                # Final fallback: derive env_steps from update_step ×
                # rollout_steps × num_envs using hparams.json. This
                # covers historical runs whose metrics.jsonl pre-dated
                # the env_steps key *and* whose per-episode checkpoints
                # got pruned (only ``best.pt`` survives long retention).
                info = _scan_hparams_and_train_step_for_run(run_name)
            if info is None:
                # Nothing to learn — but keep any previous cache entry
                # so a transient empty scan doesn't wipe history.
                continue
            info = dict(info)
            info["scanned_at"] = now
            updated[run_name] = info
            scanned_now += 1

        with self._lock:
            self._runs = updated
            self._last_scan_at = now
            self._persist_locked()
        if scanned_now > 0:
            _LOG.debug(
                "TrainingClock: refreshed %d run aggregates (total cached=%d)",
                scanned_now,
                len(updated),
            )

    # ----------------------------------------------------- introspection

    @property
    def path(self) -> Path:
        return self._path

    @property
    def total_seconds(self) -> float:
        with self._lock:
            return float(self._total_seconds)

    @property
    def runs(self) -> dict[str, dict[str, float]]:
        """Snapshot of the per-run cache. Returned shallow-copied."""
        with self._lock:
            return {k: dict(v) for k, v in self._runs.items()}


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
