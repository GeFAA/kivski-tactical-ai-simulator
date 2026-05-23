"""In-process state shared by the FastAPI routes.

We deliberately keep the storage simple (module-level singleton, plain dicts)
because the V1 server is intended to host a handful of concurrent match
viewers, not a multi-tenant cluster. Adding a real persistence layer later is
straightforward because every consumer goes through :class:`SessionRegistry`.

A :class:`MatchSession` owns:

* the deterministic :class:`Engine`
* the loaded :class:`MapData`
* the policy adapters for each side (yellow / blue)
* an asyncio task driving the tick loop
* a set of subscribed :class:`fastapi.WebSocket` connections

The tick loop emits one ``{"type": "snapshot", "data": ...}`` JSON message per
tick to every live subscriber at ``cfg.server.tick_broadcast_hz * speed`` Hz.
Disconnected sockets are pruned silently.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kivski_sim.config import KivskiConfig, load_config
from kivski_sim.engine import Engine
from kivski_sim.map_loader import MapData, load_map
from kivski_sim.replay import ReplayActionFrame, ReplayEventFrame, ReplayHeader, ReplayWriter
from kivski_sim.utils import now_unix

from kivski_api.policies import (
    PolicyAdapter,
    RandomPolicy,
    latest_checkpoint_path,
    load_policy,
)

if TYPE_CHECKING:  # avoid the heavy fastapi import at module import time
    from fastapi import WebSocket

__all__ = [
    "MatchSession",
    "SessionRegistry",
    "TrainingJob",
    "TrainingWatchdog",
    "REGISTRY",
]

# How long the watchdog sleeps between poll cycles. Long enough that it's
# basically free; short enough that a crashed trainer doesn't go more
# than ~10s without being noticed.
_WATCHDOG_INTERVAL_SECONDS: float = 10.0
# Cap on auto-restart attempts per job. A persistently-crashing trainer
# (bad checkpoint, broken config) is a real user problem -- silently
# spinning up doomed processes forever is worse than surfacing the failure.
_WATCHDOG_MAX_RESTARTS: int = 3

_LOG = logging.getLogger("kivski_api.session")


class _EventOnlyReplayWriter:
    """A :class:`ReplayWriter`-shaped facade that drops action frames.

    Live matches run at 10-20 Hz for many minutes, so persisting every
    action frame would balloon disk usage to 50k+ frames per match.
    V1 intentionally keeps only event frames (round_start, plant,
    defuse, kill, round_end) — they're small, semantically dense, and
    enough for post-hoc match analysis. The full ReplayReader can still
    parse the resulting files because the format treats absent action
    frames as legal.
    """

    def __init__(self, inner: ReplayWriter) -> None:
        self._inner: ReplayWriter | None = inner

    def write_actions(self, _frame: ReplayActionFrame) -> None:  # noqa: D401
        # Intentional no-op: see class docstring.
        return None

    def write_event(self, frame: ReplayEventFrame) -> None:
        if self._inner is not None:
            self._inner.write_event(frame)

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
            self._inner = None


# ---------------------------------------------------------------------------
# Training job descriptor
# ---------------------------------------------------------------------------


@dataclass
class TrainingJob:
    """Bookkeeping for a child training process spawned by the API.

    The :class:`subprocess.Popen` handle is owned exclusively by the API
    process; if the API restarts the orphaned job continues to run but is no
    longer manageable through the HTTP interface.
    """

    job_id: str
    config_path: str
    log_path: Path
    started_at: float
    pid: int | None = None
    process: subprocess.Popen[bytes] | None = None
    episodes: int | None = None
    resume_from: str | None = None
    exit_code: int | None = None
    # Run identifier + on-disk log directory that the trainer writes its
    # metrics.jsonl into. Filled in by the /api/training/start endpoint
    # when it computes the run name so the broadcaster can tail it without
    # having to rediscover the path.
    run_name: str | None = None
    metrics_jsonl_path: Path | None = None
    # Watchdog state. ``stop_requested`` flips to True when the user
    # explicitly calls /api/training/stop so the watchdog won't auto-
    # restart what was intentionally stopped. ``restart_count`` caps the
    # number of crashes we'll silently recover from per job.
    stop_requested: bool = False
    restart_count: int = 0

    def is_running(self) -> bool:
        if self.process is None:
            return False
        rc = self.process.poll()
        if rc is None:
            return True
        # Cache the exit code so the next status call still sees it.
        self.exit_code = int(rc)
        return False

    def tail_log(self, n_lines: int = 50) -> list[str]:
        """Return the last ``n_lines`` of the log file (best-effort, never raises)."""
        if not self.log_path.is_file():
            return []
        try:
            # The log can grow without bound -- read from the end to stay cheap.
            with self.log_path.open("rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                # Heuristic: 256 B/line * n is plenty for our trainer logs.
                chunk = min(size, max(8192, n_lines * 256))
                fh.seek(max(0, size - chunk))
                raw = fh.read().decode("utf-8", errors="replace")
        except OSError:
            return []
        lines = raw.splitlines()
        return lines[-n_lines:]


# ---------------------------------------------------------------------------
# Match session
# ---------------------------------------------------------------------------


@dataclass
class MatchSession:
    """One running (or paused) match streamed to one or more WebSocket clients."""

    id: str
    engine: Engine
    map_data: MapData
    cfg: KivskiConfig
    paused: bool = False
    speed: float = 1.0
    subscribers: set[WebSocket] = field(default_factory=set)
    policy_yellow: PolicyAdapter = field(default_factory=RandomPolicy)
    policy_blue: PolicyAdapter = field(default_factory=RandomPolicy)
    # Human-readable identifiers for the API + UI. Filled in by
    # :meth:`SessionRegistry.create_match` based on the resolved adapter so
    # the live header can render e.g. "Yellow: checkpoint:main_ep_12000".
    policy_yellow_name: str | None = None
    policy_blue_name: str | None = None
    selected_agent: int | None = None
    created_at: float = field(default_factory=now_unix)
    last_tick_at: float = 0.0
    _task: asyncio.Task[None] | None = None
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    _broadcast_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Auto-save: every live match streams its event frames (round_start,
    # plant, defuse, kill, round_end) into a replay file under
    # ``models/replays/match-<id>-<timestamp>.replay``. The writer is
    # created lazily on the first start() call so unit tests that
    # construct a MatchSession without ever running it don't litter disk.
    _replay_writer: _EventOnlyReplayWriter | None = field(default=None, repr=False)
    _replay_path: Path | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Kick off the background tick task if not already running.

        Side-effect: creates the auto-save :class:`ReplayWriter` if one
        isn't attached yet, so events captured by the very first engine
        tick land in the on-disk replay.
        """
        if self._task is not None and not self._task.done():
            return
        self._ensure_replay_writer()
        self._stop_event = asyncio.Event()
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self.run_loop(), name=f"match-loop-{self.id}")

    async def stop(self) -> None:
        """Signal the loop to stop and wait for it to exit cleanly."""
        self._stop_event.set()
        if self._task is None:
            self._close_replay_writer()
            return
        try:
            await asyncio.wait_for(self._task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        self._task = None
        # The loop's finally{} normally handles this, but stop() may be
        # invoked before the loop ever ran (e.g. registry shutdown right
        # after create_match).
        self._close_replay_writer()

    def reset(self) -> None:
        """Reset the underlying engine. Loop continues running.

        Auto-save policy: closes the current replay file and opens a fresh
        one so each reset produces its own ``.replay`` artefact rather
        than concatenating across resets (which would confuse the format,
        because the engine emits a fresh ``round_start`` event for tick 0).
        """
        # Close + reopen so each reset starts a clean .replay file.
        self._close_replay_writer()
        self.engine.reset()
        self.policy_yellow.reset()
        self.policy_blue.reset()
        self._ensure_replay_writer()

    def _replay_dir(self) -> Path:
        """Resolve ``models/replays`` relative to the repo root."""
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "models" / "replays"
            # Accept either an existing dir or one we can create as
            # ``models/replays`` next to an existing ``models``.
            if candidate.parent.is_dir():
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
        fallback = here.parents[3] / "models" / "replays"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _ensure_replay_writer(self) -> None:
        """Open the per-match ``.replay`` file and wire it into the engine.

        Idempotent: a second call while a writer is already attached is a
        no-op so reset()+start() doesn't double-open. Failure is logged
        and swallowed -- replays are nice-to-have, not load-bearing.
        """
        if self._replay_writer is not None:
            return
        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = self._replay_dir() / f"match-{self.id}-{ts}.replay"
            header = ReplayHeader(
                seed=int(getattr(self.engine.state, "seed", 0) or 0),
                config_hash="",
                map_name=str(self.map_data.name),
                team_size=int(self.cfg.simulation.team_size),
                created_at=float(time.time()),
                kivski_version="0.1.0",
            )
            raw_writer = ReplayWriter(path, header)
            # Wrap in an event-only facade so we don't write 10+ frames/sec of
            # action payloads. The engine treats both writer types identically.
            self._replay_writer = _EventOnlyReplayWriter(raw_writer)
            self._replay_path = path
            # The facade is a structural ReplayWriter (duck-typed); the
            # engine only calls write_actions / write_event / close.
            self.engine.set_replay_writer(self._replay_writer)  # type: ignore[arg-type]
            # The engine's initial reset() fires before we attach so the
            # very first ``round_start`` would be lost. Write a synthetic
            # one here so the replay always has at least one frame describing
            # round 0; subsequent events (plant/defuse/kill/round_end and
            # later round_starts) flow through the engine as normal.
            with contextlib.suppress(Exception):
                self._replay_writer.write_event(
                    ReplayEventFrame(
                        tick=int(getattr(self.engine.state, "tick", 0) or 0),
                        kind="round_start",
                        data={
                            "round_id": int(getattr(self.engine.state, "round_id", 0) or 0),
                            "seed": int(getattr(self.engine.state, "seed", 0) or 0),
                            "source": "session_open",
                        },
                    )
                )
            _LOG.info("Match %s auto-saving replay (event-only) to %s", self.id, path)
        except Exception:
            _LOG.exception("Match %s failed to open replay writer (auto-save disabled)", self.id)
            self._replay_writer = None
            self._replay_path = None

    def _close_replay_writer(self) -> None:
        """Detach + close the replay writer if one is attached."""
        if self._replay_writer is None:
            return
        with contextlib.suppress(Exception):
            self.engine.set_replay_writer(None)
        try:
            self._replay_writer.close()
        except Exception:
            _LOG.exception("Match %s failed to close replay writer", self.id)
        self._replay_writer = None

    def set_speed(self, multiplier: float) -> None:
        # Clamp to a sensible range so a typo can't peg the CPU at 1000x.
        m = float(multiplier)
        if m < 0.1:
            m = 0.1
        if m > 16.0:
            m = 16.0
        self.speed = m

    def set_selected_agent(self, agent_id: int | None) -> None:
        if agent_id is None:
            self.selected_agent = None
            return
        # Validate against the current roster.
        ids = {int(a.agent_id) for a in self.engine.state.agents}
        self.selected_agent = int(agent_id) if int(agent_id) in ids else None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _effective_dt(self) -> float:
        """Wall-clock seconds to wait between broadcast frames."""
        hz = max(1, int(self.cfg.server.tick_broadcast_hz))
        return 1.0 / (float(hz) * max(0.1, float(self.speed)))

    def _build_observations(self) -> dict[int, dict[str, Any]]:
        """Build a (lightweight) observation dict keyed by agent id.

        Real observation tensors come from the env wrapper in Task 4. For the
        V1 server we only need *something* keyed by id so the policy adapter
        knows which agents are present.
        """
        return {int(a.agent_id): {"alive": bool(a.alive)} for a in self.engine.state.agents}

    async def run_loop(self) -> None:
        """Drive the engine and broadcast snapshots until stopped or done."""
        _LOG.info("Match %s loop starting (tick_broadcast_hz=%s)", self.id, self.cfg.server.tick_broadcast_hz)
        try:
            # Emit an initial snapshot so newly-connected clients render at
            # least one frame even when paused.
            await self._broadcast_snapshot()

            while not self._stop_event.is_set():
                dt = self._effective_dt()

                if self.paused:
                    await asyncio.sleep(min(dt, 0.05))
                    continue

                # Gather actions from both policies. We split the obs dict by
                # team so each side gets only its own agents.
                obs_all = self._build_observations()
                yellow_obs: dict[int, dict[str, Any]] = {}
                blue_obs: dict[int, dict[str, Any]] = {}
                for a in self.engine.state.agents:
                    if int(a.agent_id) not in obs_all:
                        continue
                    if int(a.team) == 0:  # Team.YELLOW == 0
                        yellow_obs[int(a.agent_id)] = obs_all[int(a.agent_id)]
                    else:
                        blue_obs[int(a.agent_id)] = obs_all[int(a.agent_id)]

                try:
                    actions_y = self.policy_yellow.act(yellow_obs)
                    actions_b = self.policy_blue.act(blue_obs)
                except Exception:
                    _LOG.exception("Policy raised in match %s -- pausing", self.id)
                    self.paused = True
                    continue
                actions = {**actions_y, **actions_b}

                # Step the engine.
                try:
                    _snap, _rewards, done = self.engine.step(actions)
                except Exception:
                    _LOG.exception("Engine step failed in match %s -- pausing", self.id)
                    self.paused = True
                    continue
                self.last_tick_at = now_unix()

                await self._broadcast_snapshot()

                if done:
                    await self._broadcast_event({"type": "match_done", "match_id": self.id})
                    _LOG.info("Match %s finished, loop exiting", self.id)
                    break

                await asyncio.sleep(dt)
        except asyncio.CancelledError:
            _LOG.info("Match %s loop cancelled", self.id)
            raise
        finally:
            self._close_replay_writer()
            _LOG.info("Match %s loop stopped", self.id)

    # ------------------------------------------------------------------
    # Broadcast helpers
    # ------------------------------------------------------------------

    async def _broadcast_snapshot(self) -> None:
        snap = self.engine.snapshot()
        payload = {
            "type": "snapshot",
            "match_id": self.id,
            "paused": self.paused,
            "speed": self.speed,
            "selected_agent": self.selected_agent,
            "data": snap.to_json_dict(),
        }
        await self._broadcast_event(payload)

    async def _broadcast_event(self, payload: dict[str, Any]) -> None:
        """Send ``payload`` to every subscriber, pruning dead sockets."""
        if not self.subscribers:
            return
        async with self._broadcast_lock:
            dead: list[WebSocket] = []
            for ws in list(self.subscribers):
                try:
                    await ws.send_json(payload)
                except Exception:
                    # Catch-all: any send failure means the socket is gone.
                    dead.append(ws)
            for ws in dead:
                self.subscribers.discard(ws)

    def map_info_payload(self) -> dict[str, Any]:
        """Lightweight initial frame describing the map for new subscribers."""
        m = self.map_data
        return {
            "type": "map_info",
            "match_id": self.id,
            "data": {
                "name": m.name,
                "version": int(m.version),
                "width": int(m.width),
                "height": int(m.height),
                "tile_size": float(m.tile_size),
                "bombsites": {
                    name: {
                        "center": [float(s.center[0]), float(s.center[1])],
                        "polygon": [[float(p[0]), float(p[1])] for p in s.polygon],
                    }
                    for name, s in m.bombsites.items()
                },
                "walls": [
                    {
                        "polygon": [[float(p[0]), float(p[1])] for p in ob.polygon],
                        "blocks_movement": bool(ob.blocks_movement),
                        "blocks_sight": bool(ob.blocks_sight),
                        "low": bool(ob.low),
                    }
                    for ob in m.walls
                ],
                "cover": [
                    {
                        "polygon": [[float(p[0]), float(p[1])] for p in ob.polygon],
                        "blocks_movement": bool(ob.blocks_movement),
                        "blocks_sight": bool(ob.blocks_sight),
                        "low": bool(ob.low),
                    }
                    for ob in m.cover
                ],
                "named_areas": [
                    {
                        "name": area.name,
                        "polygon": [[float(p[0]), float(p[1])] for p in area.polygon],
                    }
                    for area in m.named_areas
                ],
            },
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SessionRegistry:
    """Module-level singleton holding all live matches and training jobs."""

    def __init__(self) -> None:
        self.sessions: dict[str, MatchSession] = {}
        self.training: dict[str, TrainingJob] = {}
        self.loaded_checkpoint: str | None = None

    # ----- Matches -----------------------------------------------------

    def create_match(
        self,
        *,
        map_name: str = "dustline",
        seed: int | None = None,
        config_path: str | None = None,
        policy_yellow: str | None = None,
        policy_blue: str | None = None,
    ) -> MatchSession:
        cfg = load_config(config_path) if config_path else load_config()
        map_data = load_map(map_name)
        engine = Engine(config=cfg, map_data=map_data, seed=seed)
        engine.reset()
        match_id = uuid.uuid4().hex[:12]

        # Auto-checkpoint behaviour: when neither side is explicitly named we
        # default *both* to the latest checkpoint (or fall back to random when
        # no checkpoint exists). Without this the live viewer would always
        # show two RandomPolicy adapters even after days of training -- the
        # exact bug v0.3.0 set out to fix.
        auto_default = policy_yellow is None and policy_blue is None
        if auto_default:
            ckpt = latest_checkpoint_path()
            if ckpt is not None:
                default_spec: str | None = "latest"
            else:
                default_spec = "random"
                _LOG.warning("create_match: no checkpoint available, defaulting both sides to random")
            yellow_spec = default_spec
            blue_spec = default_spec
        else:
            yellow_spec = policy_yellow
            blue_spec = policy_blue

        adapter_y = load_policy(yellow_spec)
        adapter_b = load_policy(blue_spec)

        session = MatchSession(
            id=match_id,
            engine=engine,
            map_data=map_data,
            cfg=cfg,
            policy_yellow=adapter_y,
            policy_blue=adapter_b,
            policy_yellow_name=str(getattr(adapter_y, "name", yellow_spec or "random")),
            policy_blue_name=str(getattr(adapter_b, "name", blue_spec or "random")),
        )
        self.sessions[match_id] = session
        _LOG.info(
            "Created match %s (map=%s, seed=%s, yellow=%s, blue=%s)",
            match_id,
            map_name,
            seed,
            session.policy_yellow_name,
            session.policy_blue_name,
        )
        return session

    def get_match(self, match_id: str) -> MatchSession | None:
        return self.sessions.get(match_id)

    async def remove_match(self, match_id: str) -> None:
        session = self.sessions.pop(match_id, None)
        if session is None:
            return
        await session.stop()
        # Close any remaining sockets (best-effort).
        for ws in list(session.subscribers):
            with contextlib.suppress(Exception):
                await ws.close()
        session.subscribers.clear()
        _LOG.info("Removed match %s", match_id)

    async def shutdown(self) -> None:
        """Stop every running session -- called from the FastAPI lifespan."""
        for match_id in list(self.sessions.keys()):
            await self.remove_match(match_id)
        for job in list(self.training.values()):
            if job.process is not None and job.is_running():
                with contextlib.suppress(Exception):
                    job.process.terminate()

    # ----- Training ----------------------------------------------------

    def register_training(self, job: TrainingJob) -> None:
        self.training[job.job_id] = job

    def latest_training(self) -> TrainingJob | None:
        if not self.training:
            return None
        return max(self.training.values(), key=lambda j: j.started_at)


REGISTRY = SessionRegistry()


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


class TrainingWatchdog:
    """Background task that auto-restarts a crashed training subprocess.

    Lifecycle: started from the FastAPI lifespan, polls every
    :data:`_WATCHDOG_INTERVAL_SECONDS`. For each registered
    :class:`TrainingJob` it checks ``process.poll()`` -- if the process
    exited with a non-zero code *and* the user didn't explicitly stop
    the job *and* we haven't exhausted the restart budget, we spawn a
    fresh ``python -m scripts.train train`` re-using the same config
    and ``--resume``-ing from the newest checkpoint on disk.

    Implementation note: spawning the trainer requires the route-layer
    helpers (``_find_resumable_checkpoint``, ``_log_dir``, ...) which
    would create an import cycle if imported at module load. We import
    them lazily inside ``_restart_job`` instead.
    """

    def __init__(self, registry: SessionRegistry) -> None:
        self._registry = registry
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run(), name="training-watchdog")
        _LOG.info(
            "TrainingWatchdog started (interval=%.1fs, max_restarts=%d)",
            _WATCHDOG_INTERVAL_SECONDS,
            _WATCHDOG_MAX_RESTARTS,
        )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        self._task = None

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._check_once()
            except Exception:
                _LOG.exception("TrainingWatchdog poll cycle raised")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=_WATCHDOG_INTERVAL_SECONDS)
            except TimeoutError:
                continue

    def _check_once(self) -> None:
        for job in list(self._registry.training.values()):
            self._maybe_restart(job)

    def _maybe_restart(self, job: TrainingJob) -> None:
        if job.process is None:
            return
        if job.is_running():
            return
        if job.stop_requested:
            return
        if job.exit_code is None:
            return  # process never started; nothing to recover
        if job.exit_code == 0:
            return  # clean exit, training finished normally
        if job.restart_count >= _WATCHDOG_MAX_RESTARTS:
            _LOG.warning(
                "TrainingWatchdog: job %s exhausted restart budget (exit=%s)",
                job.job_id,
                job.exit_code,
            )
            return
        self._restart_job(job)

    def _restart_job(self, job: TrainingJob) -> None:
        # Lazy import to avoid the routes layer importing session and vice
        # versa at module-load time.
        from kivski_api.routes.training import (
            _find_resumable_checkpoint,
            _log_dir,
            _repo_root,
        )

        resume_target = _find_resumable_checkpoint()
        resume_str = str(resume_target) if resume_target is not None else job.resume_from
        new_job_id = uuid.uuid4().hex[:12]
        log_path = _log_dir() / f"train-{new_job_id}.log"

        cmd: list[str] = [
            sys.executable,
            "-m",
            "scripts.train",
            "train",
            "--config",
            job.config_path or "configs/default.yaml",
        ]
        if job.episodes is not None:
            cmd.extend(["--episodes", str(int(job.episodes))])
        if resume_str:
            cmd.extend(["--resume", str(resume_str)])
        run_name = f"watchdog-{time.strftime('%Y%m%d-%H%M%S')}-{new_job_id}"
        metrics_jsonl_path = _repo_root() / "models" / "logs" / run_name / "metrics.jsonl"
        cmd.extend(["--run-name", run_name, "--telemetry", "all"])

        _LOG.warning(
            "TrainingWatchdog: restarting crashed job %s (exit=%s, attempt=%d/%d) -> %s",
            job.job_id,
            job.exit_code,
            job.restart_count + 1,
            _WATCHDOG_MAX_RESTARTS,
            " ".join(cmd),
        )
        try:
            log_fh = log_path.open("wb")
        except OSError:
            _LOG.exception("TrainingWatchdog: cannot open log file %s", log_path)
            return
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(_repo_root()),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                close_fds=(os.name != "nt"),
            )
        except OSError:
            log_fh.close()
            _LOG.exception("TrainingWatchdog: failed to spawn restart for job %s", job.job_id)
            return

        new_job = TrainingJob(
            job_id=new_job_id,
            config_path=job.config_path,
            log_path=log_path,
            started_at=now_unix(),
            pid=proc.pid,
            process=proc,
            episodes=job.episodes,
            resume_from=resume_str,
            run_name=run_name,
            metrics_jsonl_path=metrics_jsonl_path,
            restart_count=job.restart_count + 1,
        )
        self._registry.register_training(new_job)
