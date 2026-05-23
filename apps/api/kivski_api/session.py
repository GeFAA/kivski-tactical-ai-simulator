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
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kivski_sim.config import KivskiConfig, load_config
from kivski_sim.engine import Engine
from kivski_sim.map_loader import MapData, load_map
from kivski_sim.utils import now_unix

from kivski_api.policies import (
    PolicyAdapter,
    RandomPolicy,
    latest_checkpoint_path,
    load_latest_checkpoint_policy,
    load_policy,
)

if TYPE_CHECKING:  # avoid the heavy fastapi import at module import time
    from fastapi import WebSocket

__all__ = [
    "MatchSession",
    "SessionRegistry",
    "TrainingJob",
    "REGISTRY",
]

_LOG = logging.getLogger("kivski_api.session")


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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Kick off the background tick task if not already running."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self.run_loop(), name=f"match-loop-{self.id}")

    async def stop(self) -> None:
        """Signal the loop to stop and wait for it to exit cleanly."""
        self._stop_event.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        self._task = None

    def reset(self) -> None:
        """Reset the underlying engine. Loop continues running."""
        self.engine.reset()
        self.policy_yellow.reset()
        self.policy_blue.reset()

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
                _LOG.warning(
                    "create_match: no checkpoint available, defaulting both sides to random"
                )
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
