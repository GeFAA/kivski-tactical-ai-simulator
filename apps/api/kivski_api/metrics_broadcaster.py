"""Live training-metrics broadcaster.

The trainer writes line-delimited JSON records to
``models/logs/<run>/metrics.jsonl`` once per optimizer update (via
:class:`kivski_agents.telemetry.JSONLSink`). This module owns a single
``asyncio.Task`` that, every :data:`MetricsBroadcaster.POLL_SEC` seconds,
tails every running :class:`kivski_api.session.TrainingJob`'s JSONL file
and pushes the parsed records out to every live match WebSocket as

* ``{"type": "metrics_sample", "data": {...}}`` -- one per record, shaped
  to match the frontend's :ts:type:`MetricsSample` interface.
* ``{"type": "training_status", "data": {...}}`` -- one per record, mirrors
  :ts:type:`TrainingStatus` and is what flips the sparklines in
  :ts:type:`TrainingPanel`.

V1 fan-out strategy is simple: every metrics frame goes to every match's
subscriber set. There's no per-match filter because a training run is a
project-wide artefact, not bound to a specific match preview. If the
broadcaster cannot find a JSONL file yet (trainer hasn't created it),
that job is skipped silently — the next poll will pick it up. Corrupted
lines (malformed JSON) are logged and skipped without bringing down the
broadcaster loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kivski_api.session import SessionRegistry, TrainingJob

__all__ = [
    "MetricsBroadcaster",
    "broadcast_training_status_to_all",
    "set_active_broadcaster",
]

_LOG = logging.getLogger("kivski_api.metrics_broadcaster")


# ---------------------------------------------------------------------------
# Module-level handle so non-async callers (e.g. /api/training/start) can
# trigger a single broadcast without owning the broadcaster instance.
# ---------------------------------------------------------------------------


_ACTIVE: MetricsBroadcaster | None = None


def set_active_broadcaster(broadcaster: MetricsBroadcaster | None) -> None:
    """Register the running broadcaster so module-level helpers can use it."""
    global _ACTIVE
    _ACTIVE = broadcaster


async def broadcast_training_status_to_all(data: dict[str, Any]) -> None:
    """Send a single ``training_status`` frame to every match subscriber.

    Safe to call when the broadcaster hasn't been started yet -- in that
    case the call is a no-op (used by ``/api/training/start`` to flip the
    UI pill immediately).
    """
    if _ACTIVE is None:
        return
    await _ACTIVE._broadcast({"type": "training_status", "data": dict(data)})


# ---------------------------------------------------------------------------
# Broadcaster
# ---------------------------------------------------------------------------


class MetricsBroadcaster:
    """Polls live training jobs' JSONL log and fans new records out via WS.

    Lifecycle: instantiate with a :class:`SessionRegistry`, then
    ``await start()`` from the FastAPI ``lifespan``. The broadcaster runs
    a single background task and shuts down cleanly on ``await stop()``.

    Per-job state tracks the last byte offset already read so subsequent
    polls only stream new lines. When a job goes away (process exit +
    removed from the registry) its offset entry is pruned. Per-job state
    also keeps the "latest known" value for every metric key so the
    consolidated frame the broadcaster emits always carries the most
    recent reading even if the just-arrived record happens to only
    contain ``train/*`` keys.
    """

    POLL_SEC: float = 1.0

    def __init__(self, registry: SessionRegistry, poll_sec: float | None = None) -> None:
        self.registry = registry
        self.poll_sec: float = float(poll_sec) if poll_sec is not None else self.POLL_SEC
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        # job_id -> last byte offset successfully read from metrics.jsonl
        self._offsets: dict[str, int] = {}
        # job_id -> latest value seen for every metric key. Lets us emit a
        # consolidated frame even when only some keys are present in the
        # latest record (the trainer logs train/ + episode/ + live/ in
        # separate log_dict calls, each becoming its own JSONL line).
        self._latest: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch the background polling task. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._loop(), name="metrics-broadcaster")
        set_active_broadcaster(self)
        _LOG.info("MetricsBroadcaster started (poll=%.2fs)", self.poll_sec)

    async def stop(self) -> None:
        """Signal the background task to exit and await it."""
        self._stop_event.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            self._task.cancel()
            with contextlib.suppress(BaseException):
                await self._task
        self._task = None
        set_active_broadcaster(None)
        _LOG.info("MetricsBroadcaster stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                # Snapshot the registry so a concurrent register/remove
                # doesn't mutate the dict mid-iteration.
                jobs = list(self.registry.training.items())
                for job_id, job in jobs:
                    try:
                        await self._poll_job(job)
                    except Exception:  # noqa: BLE001 - keep the loop alive
                        _LOG.exception("poll_job failed for %s", job_id)
                # Drop offsets / latest state for jobs that have been removed.
                live_ids = {jid for jid, _ in jobs}
                for stale in set(self._offsets) - live_ids:
                    self._offsets.pop(stale, None)
                    self._latest.pop(stale, None)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_sec)
        except asyncio.CancelledError:
            raise
        finally:
            _LOG.info("MetricsBroadcaster loop exiting")

    # ------------------------------------------------------------------
    # Per-job polling
    # ------------------------------------------------------------------

    async def _poll_job(self, job: TrainingJob) -> None:
        path = self._resolve_jsonl_path(job)
        if path is None or not path.is_file():
            return
        offset = self._offsets.get(job.job_id, 0)
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size <= offset:
            # No new bytes (or file rotated/shrunk; recover by re-seeking 0).
            if size < offset:
                self._offsets[job.job_id] = 0
            return
        try:
            with path.open("rb") as fh:
                fh.seek(offset)
                chunk = fh.read(size - offset)
        except OSError:
            return

        # Split into lines but only consume complete ones (last line might
        # be a partial write -- keep its bytes in the offset for next poll).
        text = chunk.decode("utf-8", errors="replace")
        new_offset = offset
        for raw_line in text.splitlines(keepends=True):
            if not raw_line.endswith(("\n", "\r")):
                # Partial trailing line; bail without advancing.
                break
            new_offset += len(raw_line.encode("utf-8", errors="replace"))
            line = raw_line.rstrip("\r\n").strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                _LOG.warning("skip corrupted jsonl line in %s: %s", path, exc)
                continue
            if not isinstance(record, dict):
                continue
            await self._handle_record(job, record)
        self._offsets[job.job_id] = new_offset

    def _resolve_jsonl_path(self, job: TrainingJob) -> Path | None:
        """Return the JSONL path for ``job``, falling back to a glob if missing."""
        if job.metrics_jsonl_path is not None:
            return Path(job.metrics_jsonl_path)
        # Older jobs (or jobs spawned outside the API) may not have the path
        # pinned. Best-effort: probe ``models/logs/<run_name>/metrics.jsonl``
        # if a run_name is known.
        if job.run_name:
            candidate = Path("models/logs") / job.run_name / "metrics.jsonl"
            if candidate.is_file():
                return candidate
        return None

    # ------------------------------------------------------------------
    # Record handling
    # ------------------------------------------------------------------

    async def _handle_record(self, job: TrainingJob, record: dict[str, Any]) -> None:
        """Merge ``record`` into the per-job latest state and broadcast frames."""
        latest = self._latest.setdefault(job.job_id, {})
        latest.update(record)

        episodes_target = int(job.episodes or 0)

        # MetricsSample (frontend shape, camelCase).
        metrics_sample: dict[str, Any] = {
            "episode": int(_pick_int(latest, "live/episode", "train/episode", "episode/episode", default=0)),
        }
        winrate_random = _pick_float(latest, "live/winrate_vs_random", "eval/random/yellow_winrate")
        if winrate_random is not None:
            metrics_sample["winrateVsRandom"] = winrate_random
        winrate_scripted = _pick_float(
            latest,
            "live/winrate_vs_scripted",
            "eval/scripted_rush/yellow_winrate",
            "eval/scripted_hold/yellow_winrate",
        )
        if winrate_scripted is not None:
            metrics_sample["winrateVsScripted"] = winrate_scripted
        pol = _pick_float(latest, "live/policy_loss", "train/policy_loss")
        if pol is not None:
            metrics_sample["policyLoss"] = pol
        val = _pick_float(latest, "live/value_loss", "train/value_loss")
        if val is not None:
            metrics_sample["valueLoss"] = val
        ent = _pick_float(latest, "live/entropy", "train/entropy")
        if ent is not None:
            metrics_sample["entropy"] = ent

        await self._broadcast({"type": "metrics_sample", "data": metrics_sample})

        # TrainingStatus (frontend shape, camelCase).
        training_status: dict[str, Any] = {
            "running": bool(job.is_running()),
            "episode": metrics_sample["episode"],
            "totalEpisodes": episodes_target,
        }
        if pol is not None:
            training_status["policyLoss"] = pol
        if val is not None:
            training_status["valueLoss"] = val
        if ent is not None:
            training_status["entropy"] = ent
        await self._broadcast({"type": "training_status", "data": training_status})

    # ------------------------------------------------------------------
    # WS fan-out
    # ------------------------------------------------------------------

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        """Send ``payload`` to every subscriber across every live match."""
        # Snapshot the session list so removals during iteration are safe.
        for session in list(self.registry.sessions.values()):
            if not session.subscribers:
                continue
            dead = []
            # list() the subscribers so a discard() inside the loop is safe.
            for ws in list(session.subscribers):
                try:
                    await ws.send_json(payload)
                except Exception:  # noqa: BLE001 - any failure → prune
                    dead.append(ws)
            for ws in dead:
                session.subscribers.discard(ws)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_float(state: dict[str, Any], *keys: str) -> float | None:
    """Return the first key present in ``state`` coerced to ``float``."""
    for key in keys:
        if key in state:
            try:
                return float(state[key])
            except (TypeError, ValueError):
                continue
    return None


def _pick_int(state: dict[str, Any], *keys: str, default: int = 0) -> int:
    """Return the first key present in ``state`` coerced to ``int``."""
    for key in keys:
        if key in state:
            try:
                return int(float(state[key]))
            except (TypeError, ValueError):
                continue
    return int(default)
