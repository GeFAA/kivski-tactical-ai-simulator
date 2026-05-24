"""Deterministic 5v5 bomb-defuse engine.

This is the heart of Kivski: a headless, seeded, fully-observable simulation
that progresses one tick at a time given a per-agent ``ActionBundle`` map.
Given identical ``(seed, config, action_stream)`` the engine produces a
bit-exact identical trajectory -- a property exercised by the unit tests
and relied upon by the replay system.

High-level loop per ``step``:

1. Decrement ``phase_ticks_remaining`` and accumulate per-tick events.
2. Dispatch to ``_step_buy`` / ``_step_live`` / ``_step_post_plant`` based
   on the current ``Phase``.
3. Check round-end conditions and (if needed) call ``_end_round``.
4. Snapshot the resulting state and stream it to the replay writer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kivski_sim.combat import (
    angle_from_to,
    compute_damage,
    compute_hit_probability,
    sample_reaction_time,
    shots_per_tick,
)
from kivski_sim.config import KivskiConfig
from kivski_sim.economy import apply_buy_choice, kill_reward, round_end_payouts
from kivski_sim.map_loader import MapData
from kivski_sim.replay import ReplayActionFrame, ReplayEventFrame, ReplayWriter
from kivski_sim.rng import RngHub
from kivski_sim.state import AgentState, BombState, MatchState, TeamState
from kivski_sim.types import (
    WEAPONS,
    ActionBundle,
    AgentId,
    BombPhase,
    BuyChoice,
    CombatEvent,
    CommAction,
    MatchOutcome,
    Message,
    MicroAction,
    Phase,
    RoundOutcome,
    RoundSummary,
    Side,
    SoundEvent,
    Team,
    WeaponClass,
)
from kivski_sim.visibility import DEFAULT_FOV_RADIANS, compute_fov, compute_los

__all__ = ["Engine", "EngineConfig", "Snapshot"]


# Movement constants -- tuned so a default-walking agent crosses ~4.5 tiles
# per second at 10 Hz. SPRINT 1.4x, CROUCH 0.5x, FALL_BACK 0.7x match the
# spec in the engine task.
_BASE_SPEED_TILES_PER_TICK_AT_10HZ: float = 0.45

_SPEED_MULT: dict[MicroAction, float] = {
    MicroAction.DEFAULT: 1.0,
    MicroAction.CROUCH_HOLD: 0.5,
    MicroAction.PEEK: 0.45,
    MicroAction.SPRINT: 1.4,
    MicroAction.FALL_BACK: 0.7,
    MicroAction.INTERACT: 0.0,
}

# Sound intensities per posture for footstep emission.
_FOOTSTEP_INTENSITY: dict[MicroAction, float] = {
    MicroAction.DEFAULT: 0.5,
    MicroAction.CROUCH_HOLD: 0.10,
    MicroAction.PEEK: 0.15,
    MicroAction.SPRINT: 1.0,
    MicroAction.FALL_BACK: 0.35,
    MicroAction.INTERACT: 0.0,
}

_BOMB_PICKUP_DISTANCE: float = 0.8
_INTERACT_RADIUS: float = 0.5  # max distance from carrier for plant/defuse stop checks
_AGENT_RADIUS: float = 0.30


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """Higher-level config used by the engine, derived from ``KivskiConfig``.

    Wraps the typed pydantic config and exposes the most-frequently-used
    knobs at the top level for cheap attribute access inside the hot loop.
    """

    cfg: KivskiConfig
    map_data: MapData
    max_rounds: int = field(init=False)
    tick_rate_hz: int = field(init=False)
    dt: float = field(init=False)

    def __post_init__(self) -> None:
        self.max_rounds = int(self.cfg.simulation.max_rounds)
        self.tick_rate_hz = int(self.cfg.simulation.tick_rate_hz)
        self.dt = 1.0 / float(self.tick_rate_hz)

    @classmethod
    def from_config(
        cls,
        cfg: KivskiConfig,
        map_data: MapData,
        max_rounds: int | None = None,
    ) -> EngineConfig:
        ec = cls(cfg=cfg, map_data=map_data)
        if max_rounds is not None:
            ec.max_rounds = int(max_rounds)
        return ec


@dataclass
class Snapshot:
    """Immutable per-tick snapshot for viewers / training / replay.

    The shape of every list is fixed for the duration of a match so that
    downstream consumers (PixiJS viewer, observation builder) can rely on
    stable indices.
    """

    tick: int
    round_id: int
    phase: Phase
    bomb_phase: BombPhase
    yellow_score: int
    blue_score: int
    seconds_left: float
    plant_progress: float
    defuse_progress: float
    agents: list[dict[str, Any]]
    bomb: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    sounds: list[dict[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe nested dict (enums become ints, np arrays become lists)."""
        return {
            "tick": int(self.tick),
            "round_id": int(self.round_id),
            "phase": int(self.phase),
            "bomb_phase": int(self.bomb_phase),
            "yellow_score": int(self.yellow_score),
            "blue_score": int(self.blue_score),
            "seconds_left": float(self.seconds_left),
            "plant_progress": float(self.plant_progress),
            "defuse_progress": float(self.defuse_progress),
            "agents": list(self.agents),
            "bomb": dict(self.bomb),
            "events": list(self.events),
            "messages": list(self.messages),
            "sounds": list(self.sounds),
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Engine:
    """Headless deterministic engine.

    Typical usage::

        engine = Engine(cfg=cfg, map_data=map_data, seed=42)
        snap = engine.reset()
        while not engine.is_done():
            actions = policy.act(snap)
            snap, rewards, done = engine.step(actions)
    """

    # ------------------------------------------------------------------
    # Construction & lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: KivskiConfig,
        map_data: MapData,
        seed: int | None = None,
    ) -> None:
        if seed is None:
            seed = int(config.seed)
        self._engine_cfg = EngineConfig.from_config(config, map_data)
        self._cfg = config
        self._map = map_data
        self._rng = RngHub(int(seed))
        self._state: MatchState = self._build_initial_state(int(seed))
        # Per-tick scratch buffers for events / messages / sounds.
        self._tick_events: list[CombatEvent] = []
        self._tick_messages: list[Message] = []
        self._tick_sounds: list[SoundEvent] = []
        self._replay_writer: ReplayWriter | None = None
        # Cache derived numbers used many times per tick.
        self._team_size: int = int(config.simulation.team_size)

    # ------------------------------------------------------------------

    @property
    def state(self) -> MatchState:
        """Direct (read-mostly) access to the match state, for observers."""
        return self._state

    @property
    def map(self) -> MapData:
        return self._map

    @property
    def config(self) -> KivskiConfig:
        return self._cfg

    # ------------------------------------------------------------------
    # Public step API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> Snapshot:
        """Reset the engine to the start of a fresh match.

        If ``seed`` is provided it replaces the seed; otherwise the existing
        seed is reused. After a reset the RNG hub is fresh and the first
        snapshot reflects round 0 in the BUY phase.
        """
        if seed is not None:
            self._rng = RngHub(int(seed))
            self._state = self._build_initial_state(int(seed))
        else:
            self._rng = RngHub(self._rng.seed)
            self._state = self._build_initial_state(self._rng.seed)
        self._tick_events.clear()
        self._tick_messages.clear()
        self._tick_sounds.clear()
        snap = self.snapshot()
        if self._replay_writer is not None:
            self._replay_writer.write_event(
                ReplayEventFrame(
                    tick=0,
                    kind="round_start",
                    data={"round_id": 0, "seed": int(self._rng.seed)},
                )
            )
        return snap

    def step(
        self,
        actions: dict[AgentId, ActionBundle] | dict[int, ActionBundle],
        *,
        light_snapshot: bool = False,
    ) -> tuple[Snapshot, dict[AgentId, float], bool]:
        """Advance the simulation by one tick.

        Returns ``(snapshot, rewards, done)`` where:

        * ``snapshot`` is the post-step :class:`Snapshot` -- or a *light*
          snapshot when ``light_snapshot=True`` (training mode). The light
          snapshot omits the agent/bomb/event list serialisation that the
          PixiJS viewer needs, which is a measurable chunk of CPU at 32+
          parallel envs. The training loop uses
          :attr:`Engine.state` directly so it does not need the heavy
          serialisation.
        * ``rewards`` is a dict ``agent_id -> float`` (round-win/loss + simple
          shaping for damage). This is the *dense* reward used by training.
        * ``done`` is True when the match has ended (``MatchOutcome != NONE``).
        """
        if self._state.match_outcome != MatchOutcome.NONE:
            # Match already done -- return current snapshot with no rewards.
            return (
                self._light_snapshot() if light_snapshot else self.snapshot(),
                self._zero_reward_dict(),
                True,
            )

        # 1) Normalise + record actions.
        normalised = self._normalise_actions(actions)
        if self._replay_writer is not None:
            self._replay_writer.write_actions(self._build_action_frame(normalised))

        # 2) Clear per-tick scratch buffers (events/messages/sounds).
        self._tick_events.clear()
        self._tick_messages.clear()
        self._tick_sounds.clear()

        # 3) Dispatch by phase.
        self._state.tick += 1
        rewards = self._zero_reward_dict()
        round_just_ended = False
        if self._state.phase == Phase.BUY:
            self._step_buy(normalised)
            if self._state.phase_ticks_remaining > 0:
                self._state.phase_ticks_remaining -= 1
            if self._state.phase_ticks_remaining <= 0:
                self._transition_to_live()
        elif self._state.phase == Phase.LIVE:
            self._step_live(normalised, rewards)
            self._state.phase_ticks_remaining = max(0, self._state.phase_ticks_remaining - 1)
            outcome = self._check_round_end_live()
            if outcome != RoundOutcome.NONE:
                self._end_round(outcome, rewards)
                round_just_ended = True
        elif self._state.phase == Phase.POST_PLANT:
            self._step_post_plant(normalised, rewards)
            self._state.phase_ticks_remaining = max(0, self._state.phase_ticks_remaining - 1)
            outcome = self._check_round_end_post_plant()
            if outcome != RoundOutcome.NONE:
                self._end_round(outcome, rewards)
                round_just_ended = True

        # 4) Stash sounds/messages into state for any observer/agent that wants
        # them as part of the next observation.
        self._state.sounds = list(self._tick_sounds)
        self._state.messages = list(self._tick_messages)

        snap = self._light_snapshot() if light_snapshot else self.snapshot()
        done = self._state.match_outcome != MatchOutcome.NONE
        if round_just_ended and self._replay_writer is not None and not done:
            self._replay_writer.write_event(
                ReplayEventFrame(
                    tick=int(self._state.tick),
                    kind="round_start",
                    data={"round_id": int(self._state.round_id)},
                )
            )
        return snap, rewards, done

    # ------------------------------------------------------------------
    # Lightweight snapshot used by training callers
    # ------------------------------------------------------------------

    def _light_snapshot(self) -> Snapshot:
        """Return a :class:`Snapshot` skeleton with only the cheap fields.

        Training never reads the agent/event/message/sound lists out of the
        return value (it accesses :attr:`Engine.state` directly), so packing
        them is pure waste at training time. The fields the viewer needs
        (``tick``, ``round_id``, ``phase``, scores, bomb phase) are still
        populated so loggers / progress prints keep working.
        """
        s = self._state
        return Snapshot(
            tick=int(s.tick),
            round_id=int(s.round_id),
            phase=s.phase,
            bomb_phase=s.bomb.phase,
            yellow_score=int(s.teams[Team.YELLOW].score) if Team.YELLOW in s.teams else 0,
            blue_score=int(s.teams[Team.BLUE].score) if Team.BLUE in s.teams else 0,
            seconds_left=float(s.phase_ticks_remaining) * self._engine_cfg.dt,
            plant_progress=float(s.bomb.plant_progress),
            defuse_progress=float(s.bomb.defuse_progress),
            agents=[],
            bomb={},
            events=[],
            messages=[],
            sounds=[],
        )

    def snapshot(self) -> Snapshot:
        """Build a fresh :class:`Snapshot` from the current state."""
        s = self._state
        agents_payload: list[dict[str, Any]] = []
        for a in s.agents:
            agents_payload.append(
                {
                    "id": int(a.agent_id),
                    "team": int(a.team),
                    "side": int(a.side),
                    "alive": bool(a.alive),
                    "hp": float(a.hp),
                    "armor": float(a.armor),
                    "pos": [float(a.pos[0]), float(a.pos[1])],
                    "facing": float(a.facing),
                    "weapon": int(a.weapon),
                    "money": int(a.money),
                    "has_bomb": bool(a.has_bomb),
                    "has_defuse_kit": bool(a.has_defuse_kit),
                    "kills_round": int(a.kills_round),
                    "deaths_round": int(a.deaths_round),
                }
            )

        bomb_payload: dict[str, Any] = {
            "phase": int(s.bomb.phase),
            "carrier": int(s.bomb.carrier),
            "pos": [float(s.bomb.pos[0]), float(s.bomb.pos[1])],
            "plant_progress": float(s.bomb.plant_progress),
            "defuse_progress": float(s.bomb.defuse_progress),
            "defuser": int(s.bomb.defuser),
            "time_since_plant": float(s.bomb.time_since_plant),
            "site": s.bomb.site if s.bomb.site is not None else "",
        }

        events_payload = [self._combat_event_to_dict(e) for e in self._tick_events]
        messages_payload = [self._message_to_dict(m) for m in self._tick_messages]
        sounds_payload = [self._sound_to_dict(snd) for snd in self._tick_sounds]

        return Snapshot(
            tick=int(s.tick),
            round_id=int(s.round_id),
            phase=s.phase,
            bomb_phase=s.bomb.phase,
            yellow_score=int(s.teams[Team.YELLOW].score) if Team.YELLOW in s.teams else 0,
            blue_score=int(s.teams[Team.BLUE].score) if Team.BLUE in s.teams else 0,
            seconds_left=float(s.phase_ticks_remaining) * self._engine_cfg.dt,
            plant_progress=float(s.bomb.plant_progress),
            defuse_progress=float(s.bomb.defuse_progress),
            agents=agents_payload,
            bomb=bomb_payload,
            events=events_payload,
            messages=messages_payload,
            sounds=sounds_payload,
        )

    def is_done(self) -> bool:
        return self._state.match_outcome != MatchOutcome.NONE

    def set_replay_writer(self, writer: ReplayWriter | None) -> None:
        """Attach a replay writer to capture action and event frames."""
        self._replay_writer = writer

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def _build_initial_state(self, seed: int) -> MatchState:
        state = MatchState(seed=int(seed))
        state.teams = {
            Team.YELLOW: TeamState(team=Team.YELLOW, side=Side.ATTACKER),
            Team.BLUE: TeamState(team=Team.BLUE, side=Side.DEFENDER),
        }
        state.agents = self._spawn_agents()
        state.bomb = BombState()
        self._assign_bomb_random(state)
        # Start in BUY phase.
        state.phase = Phase.BUY
        state.phase_ticks_remaining = int(
            round(self._cfg.simulation.buy_time_seconds * self._engine_cfg.tick_rate_hz)
        )
        state.round_id = 0
        state.tick = 0
        return state

    def _spawn_agents(self) -> list[AgentState]:
        """Create both teams' agents and place them at their map spawns."""
        agents: list[AgentState] = []
        size = int(self._cfg.simulation.team_size)
        starting = int(self._cfg.simulation.starting_money)
        agent_id = 0
        for team, side in ((Team.YELLOW, Side.ATTACKER), (Team.BLUE, Side.DEFENDER)):
            for idx in range(size):
                spawn = self._map.nearest_spawn(side, idx)
                pos = np.array(spawn, dtype=np.float32)
                # Defenders face roughly back toward their spawn->mid axis.
                facing = math.pi if side == Side.DEFENDER else 0.0
                a = AgentState(
                    agent_id=agent_id,
                    team=team,
                    side=side,
                    pos=pos,
                    vel=np.zeros(2, dtype=np.float32),
                    facing=float(facing),
                    alive=True,
                    hp=100.0,
                    armor=0.0,
                    money=starting,
                    weapon=WeaponClass.SIDEARM,
                    secondary=WeaponClass.KNIFE,
                )
                agents.append(a)
                agent_id += 1
        return agents

    def _assign_bomb_random(self, state: MatchState) -> None:
        """Give the bomb to a deterministically-chosen attacker."""
        attackers = [a for a in state.agents if a.side == Side.ATTACKER]
        if not attackers:
            return
        # Reset all carrier flags.
        for a in state.agents:
            a.has_bomb = False
        rng = self._rng.channel("spawn")
        idx = int(rng.integers(0, len(attackers)))
        carrier = attackers[idx]
        carrier.has_bomb = True
        state.bomb.phase = BombPhase.CARRIED
        state.bomb.carrier = int(carrier.agent_id)
        state.bomb.pos = np.array(carrier.pos, dtype=np.float32)
        state.bomb.plant_progress = 0.0
        state.bomb.defuse_progress = 0.0
        state.bomb.defuser = -1
        state.bomb.time_since_plant = 0.0
        state.bomb.site = None

    # ------------------------------------------------------------------
    # Phase: BUY
    # ------------------------------------------------------------------

    def _step_buy(self, actions: dict[int, ActionBundle]) -> None:
        """During BUY agents may purchase a weapon and/or armor.

        Movement is disabled (agents are pinned to spawn). All other action
        fields are ignored. Buy choices are validated by :func:`apply_buy_choice`
        so an unaffordable selection silently fails.
        """
        for agent in self._state.agents:
            if not agent.alive:
                continue
            ab = actions.get(int(agent.agent_id))
            if ab is None:
                continue
            if ab.buy != BuyChoice.NONE:
                apply_buy_choice(agent, ab.buy, self._cfg.economy)

    def _transition_to_live(self) -> None:
        self._state.phase = Phase.LIVE
        self._state.phase_ticks_remaining = int(
            round(self._cfg.simulation.round_time_seconds * self._engine_cfg.tick_rate_hz)
        )

    # ------------------------------------------------------------------
    # Phase: LIVE
    # ------------------------------------------------------------------

    def _step_live(
        self,
        actions: dict[int, ActionBundle],
        rewards: dict[AgentId, float],
    ) -> None:
        """One LIVE tick: movement, combat, bomb interaction, comms."""
        # Snapshot pre-step positions so we know who actually moved (used
        # for the footstep sound emission).
        prev_positions: dict[int, np.ndarray] = {
            int(a.agent_id): a.pos.copy() for a in self._state.agents if a.alive
        }
        # 1) Movement -- updates positions, facing, and bomb position for the
        # carrier.
        for agent in self._state.agents:
            if not agent.alive:
                continue
            ab = actions.get(int(agent.agent_id))
            self._apply_movement(agent, ab)
            # Carry the bomb with the carrier.
            if agent.has_bomb:
                self._state.bomb.pos = np.array(agent.pos, dtype=np.float32)

        # 2) Generate footstep sounds for any agent that moved.
        for agent in self._state.agents:
            if not agent.alive:
                continue
            prev = prev_positions.get(int(agent.agent_id))
            if prev is None:
                continue
            moved = float(np.linalg.norm(agent.pos - prev))
            ab = actions.get(int(agent.agent_id))
            micro = ab.micro if ab is not None else MicroAction.DEFAULT
            if moved > 0.01 and _FOOTSTEP_INTENSITY[micro] > 0.0:
                self._emit_sound(agent, kind="step", intensity=_FOOTSTEP_INTENSITY[micro])

        # 3) Comms -- broadcast to teammates.
        for agent in self._state.agents:
            if not agent.alive:
                continue
            ab = actions.get(int(agent.agent_id))
            if ab is None or ab.comm == CommAction.NONE:
                continue
            self._emit_message(agent, ab)

        # 4) Combat (one pass per attacker -> all visible enemies).
        for agent in self._state.agents:
            if not agent.alive:
                continue
            ab = actions.get(int(agent.agent_id))
            self._resolve_agent_combat(agent, ab, rewards)

        # 5) Decrement reaction cooldowns.
        for agent in self._state.agents:
            if agent.reaction_cooldown > 0:
                agent.reaction_cooldown -= 1

        # 6) Bomb interaction (planting only happens in LIVE; defuse happens
        # in POST_PLANT).
        self._handle_bomb_interaction_live(actions)

    # ----- Movement ----------------------------------------------------

    def _apply_movement(
        self,
        agent: AgentState,
        action: ActionBundle | None,
    ) -> None:
        """Update ``agent.pos`` and ``agent.facing`` based on the action.

        Movement is continuous (v0.4): ``action.move_vec`` is a 2D vector
        in ``[-1, 1]^2``. The magnitude is the speed factor (0 = HOLD,
        1 = full speed) and the direction is the heading. To prevent
        diagonal "free speed" we clamp the magnitude onto the unit circle.
        """
        if action is None:
            return
        if action.micro == MicroAction.INTERACT:
            # Planting/defusing -- no movement. Facing stays the same.
            return
        mv = np.asarray(action.move_vec, dtype=np.float32).reshape(-1)
        if mv.shape[0] < 2:
            # Degenerate / missing vector -- treat as HOLD.
            self._maybe_face_aim(agent, action)
            return
        mv = mv[:2]
        # Clamp NaN/inf to zero so a broken policy can't poison the engine.
        if not np.all(np.isfinite(mv)):
            mv = np.zeros(2, dtype=np.float32)
        # Element-wise clamp to [-1, 1] (defensive: the policy should
        # already bound the output, but Gaussian sampling can overshoot).
        mv = np.clip(mv, -1.0, 1.0)
        mag = float(np.linalg.norm(mv))
        if mag <= 0.08:
            # Tiny magnitude -> HOLD. The 0.08 deadband prevents an
            # untrained Gaussian policy (mean=0, std=1) from producing
            # near-zero magnitudes that translate to crawling speed and
            # episodes that never end. Threshold is small enough that a
            # trained policy can still issue near-zero moves on purpose.
            self._maybe_face_aim(agent, action)
            return
        # *** v0.4.1: normalise to UNIT magnitude. Continuous policy chooses
        # the *direction* (any angle), but speed is always full (modulated
        # by MicroAction below). Without this, the untrained Gaussian
        # produces magnitudes ~0.3-0.6 → agents creep → matches timeout
        # → 0 episodes complete → no PPO updates. Trained policy can still
        # express "move slowly" via MicroAction.CROUCH_HOLD.
        mv = mv / mag
        mag = 1.0

        speed = _BASE_SPEED_TILES_PER_TICK_AT_10HZ * _SPEED_MULT.get(action.micro, 1.0)
        # Scale tick-rate -- our base is 10 Hz.
        speed *= 10.0 * self._engine_cfg.dt

        delta_x = float(mv[0]) * speed
        delta_y = float(mv[1]) * speed
        new_pos = np.array(
            [float(agent.pos[0]) + delta_x, float(agent.pos[1]) + delta_y],
            dtype=np.float32,
        )

        # Collision: if the destination is blocked, try axis-aligned slides.
        if self._map.is_blocked(new_pos):
            slide_x = np.array(
                [float(agent.pos[0]) + delta_x, float(agent.pos[1])],
                dtype=np.float32,
            )
            slide_y = np.array(
                [float(agent.pos[0]), float(agent.pos[1]) + delta_y],
                dtype=np.float32,
            )
            if not self._map.is_blocked(slide_x):
                new_pos = slide_x
            elif not self._map.is_blocked(slide_y):
                new_pos = slide_y
            else:
                # Completely stuck this tick -- no movement.
                return

        # Avoid stacking agents at exactly identical coords (cheap repulsion).
        for other in self._state.agents:
            if not other.alive or other.agent_id == agent.agent_id:
                continue
            dist = float(np.linalg.norm(other.pos - new_pos))
            if dist < 2.0 * _AGENT_RADIUS:
                # Sliding nudge -- step half-way back to current pos.
                new_pos = (new_pos.astype(np.float32) + agent.pos.astype(np.float32)) * 0.5

        agent.pos = new_pos.astype(np.float32)
        # Face in the direction of travel if no aim target is set.
        if action.aim_target < 0:
            agent.facing = float(math.atan2(delta_y, delta_x))
        else:
            self._maybe_face_aim(agent, action)

    def _maybe_face_aim(self, agent: AgentState, action: ActionBundle) -> None:
        if action.aim_target < 0:
            return
        target = self._find_agent(action.aim_target)
        if target is None or not target.alive:
            return
        agent.facing = angle_from_to(agent.pos, target.pos)

    # ----- Combat ------------------------------------------------------

    def _resolve_agent_combat(
        self,
        agent: AgentState,
        action: ActionBundle | None,
        rewards: dict[AgentId, float],
    ) -> None:
        """Sample shots from ``agent`` to any visible enemies this tick."""
        if agent.reaction_cooldown > 0:
            return
        if action is not None and action.micro == MicroAction.INTERACT:
            return
        weapon = WEAPONS[agent.weapon]
        max_range = weapon.max_range

        enemies = [
            (int(other.agent_id), other.pos)
            for other in self._state.agents
            if other.alive and other.side != agent.side
        ]
        if not enemies:
            return

        visible = compute_fov(
            self._map,
            agent.pos.astype(np.float64),
            float(agent.facing),
            DEFAULT_FOV_RADIANS,
            float(max_range),
            enemies,
        )
        if not visible:
            return

        # Prefer the agent's aim target if it's visible; otherwise pick the
        # nearest visible enemy.
        chosen_id: int | None = None
        if action is not None and action.aim_target >= 0 and action.aim_target in visible:
            chosen_id = int(action.aim_target)
        else:
            best_dist = float("inf")
            for eid in visible:
                target = self._find_agent(eid)
                if target is None:
                    continue
                d = float(np.linalg.norm(target.pos - agent.pos))
                if d < best_dist:
                    best_dist = d
                    chosen_id = eid
        if chosen_id is None:
            return
        target = self._find_agent(chosen_id)
        if target is None or not target.alive:
            return

        # Compute hit probability and resolve damage (possibly multi-shot for
        # high-fire-rate weapons).
        n_shots = max(1, int(round(shots_per_tick(weapon, self._engine_cfg.dt))))
        rng = self._rng.channel("combat")
        for _ in range(n_shots):
            if not target.alive:
                break
            visible_now, dist, through_cover = compute_los(
                self._map,
                agent.pos.astype(np.float64),
                target.pos.astype(np.float64),
                float(weapon.max_range),
            )
            if not visible_now:
                break
            target_micro = MicroAction.DEFAULT  # we don't see target's action here -- approximate
            p_hit = compute_hit_probability(
                weapon,
                float(dist),
                action.micro if action is not None else MicroAction.DEFAULT,
                target_micro,
                self._cfg.combat,
                through_cover=through_cover,
            )
            roll = float(rng.random())
            if roll > p_hit:
                continue
            hp_dmg, armor_dmg = compute_damage(
                weapon,
                float(dist),
                float(target.armor),
                bool(through_cover),
                float(self._cfg.combat.cover_damage_multiplier),
            )
            target.armor = max(0.0, float(target.armor) - float(armor_dmg))
            target.hp = float(target.hp) - float(hp_dmg)
            agent.damage_done_round += float(hp_dmg + armor_dmg)
            target.damage_taken_round += float(hp_dmg + armor_dmg)
            killed = target.hp <= 0.0
            if killed:
                target.hp = 0.0
                target.alive = False
                target.deaths_round += 1
                target.deaths_match += 1
                agent.kills_round += 1
                agent.kills_match += 1
                reward = kill_reward(agent.weapon, self._cfg.economy)
                agent.money = min(agent.money + int(reward), 16000)
                # Drop the bomb if the dead agent was the carrier.
                if target.has_bomb:
                    target.has_bomb = False
                    self._state.bomb.phase = BombPhase.DROPPED
                    self._state.bomb.carrier = -1
                    self._state.bomb.pos = np.array(target.pos, dtype=np.float32)
                # Reward shaping for training: small per-kill bonus to the attacker.
                rewards[AgentId(int(agent.agent_id))] = rewards.get(AgentId(int(agent.agent_id)), 0.0) + 1.0
                rewards[AgentId(int(target.agent_id))] = rewards.get(AgentId(int(target.agent_id)), 0.0) - 1.0
            self._tick_events.append(
                CombatEvent(
                    tick=int(self._state.tick),
                    attacker=AgentId(int(agent.agent_id)),
                    victim=AgentId(int(target.agent_id)),
                    weapon=agent.weapon,
                    damage=float(hp_dmg + armor_dmg),
                    killed=bool(killed),
                    distance=float(dist),
                    through_cover=bool(through_cover),
                )
            )
            self._emit_sound(agent, kind="shot", intensity=1.2)
            if self._replay_writer is not None:
                self._replay_writer.write_event(
                    ReplayEventFrame(
                        tick=int(self._state.tick),
                        kind="kill" if killed else "hit",
                        data={
                            "attacker": int(agent.agent_id),
                            "victim": int(target.agent_id),
                            "weapon": int(agent.weapon),
                            "damage": float(hp_dmg + armor_dmg),
                            "killed": bool(killed),
                        },
                    )
                )
        # Apply reaction cooldown after the shot pass.
        agent.last_shot_tick = int(self._state.tick)
        agent.reaction_cooldown = int(sample_reaction_time(rng, self._cfg.combat))

    # ----- Bomb (LIVE) -------------------------------------------------

    def _handle_bomb_interaction_live(self, actions: dict[int, ActionBundle]) -> None:
        """Planting + bomb pickup logic that runs during the LIVE phase."""
        bomb = self._state.bomb
        # If bomb is dropped, any agent who walks within pickup range *and* is on
        # the attacking side can pick it back up. Defenders cannot move the bomb.
        if bomb.phase == BombPhase.DROPPED:
            for agent in self._state.agents:
                if not agent.alive or agent.side != Side.ATTACKER:
                    continue
                if float(np.linalg.norm(agent.pos - bomb.pos)) <= _BOMB_PICKUP_DISTANCE:
                    agent.has_bomb = True
                    bomb.phase = BombPhase.CARRIED
                    bomb.carrier = int(agent.agent_id)
                    self._emit_sound(agent, kind="bomb_pickup", intensity=0.4)
                    break

        # Planting: only the carrier may plant, and only inside a bombsite, and only
        # while pressing INTERACT.
        if bomb.phase in (BombPhase.CARRIED, BombPhase.PLANTING) and bomb.carrier >= 0:
            carrier = self._find_agent(bomb.carrier)
            if carrier is None or not carrier.alive:
                return
            ab = actions.get(int(carrier.agent_id))
            site = self._map.is_in_bombsite(carrier.pos.astype(np.float64))
            if ab is not None and ab.micro == MicroAction.INTERACT and site is not None:
                bomb.phase = BombPhase.PLANTING
                bomb.plant_progress += self._engine_cfg.dt / max(
                    1e-3, float(self._cfg.simulation.plant_time_seconds)
                )
                bomb.site = site
                self._emit_sound(carrier, kind="plant", intensity=0.3)
                if bomb.plant_progress >= 1.0:
                    bomb.plant_progress = 1.0
                    bomb.phase = BombPhase.PLANTED
                    carrier.has_bomb = False
                    self._transition_to_post_plant()
            else:
                # Plant interrupted -- reset progress but stay carried.
                if bomb.plant_progress > 0.0 and bomb.phase == BombPhase.PLANTING:
                    bomb.plant_progress = 0.0
                    bomb.phase = BombPhase.CARRIED

    def _transition_to_post_plant(self) -> None:
        self._state.phase = Phase.POST_PLANT
        self._state.phase_ticks_remaining = int(
            round(self._cfg.simulation.bomb_timer_seconds * self._engine_cfg.tick_rate_hz)
        )
        self._state.bomb.time_since_plant = 0.0
        if self._replay_writer is not None:
            self._replay_writer.write_event(
                ReplayEventFrame(
                    tick=int(self._state.tick),
                    kind="plant",
                    data={
                        "site": self._state.bomb.site or "",
                        "carrier": int(self._state.bomb.carrier),
                    },
                )
            )

    # ------------------------------------------------------------------
    # Phase: POST_PLANT
    # ------------------------------------------------------------------

    def _step_post_plant(
        self,
        actions: dict[int, ActionBundle],
        rewards: dict[AgentId, float],
    ) -> None:
        """During post-plant, attackers can still fight; defenders may defuse."""
        # Movement + combat are the same as LIVE (re-use the LIVE handlers).
        prev_positions = {int(a.agent_id): a.pos.copy() for a in self._state.agents if a.alive}
        for agent in self._state.agents:
            if not agent.alive:
                continue
            ab = actions.get(int(agent.agent_id))
            self._apply_movement(agent, ab)
        for agent in self._state.agents:
            if not agent.alive:
                continue
            prev = prev_positions.get(int(agent.agent_id))
            if prev is None:
                continue
            moved = float(np.linalg.norm(agent.pos - prev))
            ab = actions.get(int(agent.agent_id))
            micro = ab.micro if ab is not None else MicroAction.DEFAULT
            if moved > 0.01 and _FOOTSTEP_INTENSITY[micro] > 0.0:
                self._emit_sound(agent, kind="step", intensity=_FOOTSTEP_INTENSITY[micro])
        for agent in self._state.agents:
            if not agent.alive:
                continue
            ab = actions.get(int(agent.agent_id))
            if ab is None or ab.comm == CommAction.NONE:
                continue
            self._emit_message(agent, ab)
        for agent in self._state.agents:
            if not agent.alive:
                continue
            ab = actions.get(int(agent.agent_id))
            self._resolve_agent_combat(agent, ab, rewards)
        for agent in self._state.agents:
            if agent.reaction_cooldown > 0:
                agent.reaction_cooldown -= 1

        # Bomb ticking + defuse logic.
        bomb = self._state.bomb
        bomb.time_since_plant += self._engine_cfg.dt

        # Defuse: any alive defender standing on the bomb pressing INTERACT.
        defuser_agent: AgentState | None = None
        for agent in self._state.agents:
            if not agent.alive or agent.side != Side.DEFENDER:
                continue
            ab = actions.get(int(agent.agent_id))
            if ab is None or ab.micro != MicroAction.INTERACT:
                continue
            if float(np.linalg.norm(agent.pos - bomb.pos)) <= _INTERACT_RADIUS + 0.5:
                defuser_agent = agent
                break

        if defuser_agent is not None:
            bomb.phase = BombPhase.DEFUSING
            bomb.defuser = int(defuser_agent.agent_id)
            defuse_time = (
                float(self._cfg.simulation.defuse_time_with_kit_seconds)
                if defuser_agent.has_defuse_kit
                else float(self._cfg.simulation.defuse_time_seconds)
            )
            bomb.defuse_progress += self._engine_cfg.dt / max(1e-3, defuse_time)
            self._emit_sound(defuser_agent, kind="defuse", intensity=0.25)
            if bomb.defuse_progress >= 1.0:
                bomb.defuse_progress = 1.0
                bomb.phase = BombPhase.DEFUSED
        else:
            if bomb.phase == BombPhase.DEFUSING:
                # Defuse interrupted -- reset progress and clear defuser.
                bomb.phase = BombPhase.PLANTED
                bomb.defuse_progress = 0.0
                bomb.defuser = -1

    # ------------------------------------------------------------------
    # Round end checks
    # ------------------------------------------------------------------

    def _check_round_end_live(self) -> RoundOutcome:
        alive_attackers = [a for a in self._state.agents if a.alive and a.side == Side.ATTACKER]
        alive_defenders = [a for a in self._state.agents if a.alive and a.side == Side.DEFENDER]
        if not alive_defenders:
            return RoundOutcome.ATTACKERS_ELIM
        if not alive_attackers:
            return RoundOutcome.DEFENDERS_ELIM
        if self._state.phase_ticks_remaining <= 0:
            return RoundOutcome.TIMEOUT
        return RoundOutcome.NONE

    def _check_round_end_post_plant(self) -> RoundOutcome:
        bomb = self._state.bomb
        if bomb.phase == BombPhase.DEFUSED:
            return RoundOutcome.BOMB_DEFUSED
        if self._state.phase_ticks_remaining <= 0:
            bomb.phase = BombPhase.DETONATED
            return RoundOutcome.BOMB_DETONATED
        return RoundOutcome.NONE

    # ------------------------------------------------------------------
    # End-of-round bookkeeping
    # ------------------------------------------------------------------

    def _end_round(self, outcome: RoundOutcome, rewards: dict[AgentId, float]) -> None:
        """Apply scores, payouts, and start the next round (or end the match)."""
        winning_side = self._winning_side_for_outcome(outcome)
        # Update scores.
        side_to_team = {ts.side: ts.team for ts in self._state.teams.values()}
        win_team = side_to_team.get(winning_side)
        if win_team is not None:
            self._state.teams[win_team].score += 1

        # Reward shaping for training: +1 to every alive agent on the winning
        # team, -1 to losers.
        for a in self._state.agents:
            base = 1.0 if a.side == winning_side else -1.0
            rewards[AgentId(int(a.agent_id))] = rewards.get(AgentId(int(a.agent_id)), 0.0) + base

        # Pay out economy + update loss streaks.
        round_end_payouts(self._state, outcome, winning_side, self._cfg.economy)

        # Record summary.
        summary = RoundSummary(
            round_id=int(self._state.round_id),
            outcome=outcome,
            winning_side=winning_side,
            yellow_score=int(self._state.teams[Team.YELLOW].score),
            blue_score=int(self._state.teams[Team.BLUE].score),
            bomb_planted=self._state.bomb.site is not None,
            bomb_planted_site=self._state.bomb.site,
            duration_ticks=int(self._state.tick),
            survivors_yellow=sum(1 for a in self._state.agents if a.alive and a.team == Team.YELLOW),
            survivors_blue=sum(1 for a in self._state.agents if a.alive and a.team == Team.BLUE),
        )
        self._state.round_summaries.append(summary)
        self._state.last_round_outcome = outcome

        if self._replay_writer is not None:
            self._replay_writer.write_event(
                ReplayEventFrame(
                    tick=int(self._state.tick),
                    kind="round_end",
                    data={
                        "round_id": int(self._state.round_id),
                        "outcome": int(outcome),
                        "winning_side": int(winning_side),
                        "yellow_score": int(self._state.teams[Team.YELLOW].score),
                        "blue_score": int(self._state.teams[Team.BLUE].score),
                    },
                )
            )

        # Check match end before advancing the round id.
        if self._check_match_end():
            return

        # Otherwise prepare the next round.
        self._state.round_id += 1
        # Side switch happens at the configured round boundary.
        if self._state.round_id == int(self._cfg.simulation.side_switch_round):
            self._swap_sides()
        self._reset_for_new_round()

    def _winning_side_for_outcome(self, outcome: RoundOutcome) -> Side:
        if outcome == RoundOutcome.ATTACKERS_ELIM:
            return Side.ATTACKER
        if outcome == RoundOutcome.BOMB_DETONATED:
            return Side.ATTACKER
        if outcome == RoundOutcome.DEFENDERS_ELIM:
            # Subtle: if the bomb is planted, defenders only win when the bomb is
            # defused (handled separately). Killing every attacker while the bomb
            # ticks is *not* an immediate win.
            if self._state.phase == Phase.POST_PLANT and self._state.bomb.phase != BombPhase.DEFUSED:
                return Side.ATTACKER  # bomb will still detonate => attackers win
            return Side.DEFENDER
        if outcome == RoundOutcome.BOMB_DEFUSED:
            return Side.DEFENDER
        if outcome == RoundOutcome.TIMEOUT:
            return Side.DEFENDER
        return Side.DEFENDER

    def _check_match_end(self) -> bool:
        max_rounds = int(self._engine_cfg.max_rounds)
        needed = max_rounds // 2 + 1
        y = int(self._state.teams[Team.YELLOW].score)
        b = int(self._state.teams[Team.BLUE].score)
        if y >= needed:
            self._state.match_outcome = MatchOutcome.YELLOW_WIN
            self._state.phase = Phase.MATCH_OVER
            return True
        if b >= needed:
            self._state.match_outcome = MatchOutcome.BLUE_WIN
            self._state.phase = Phase.MATCH_OVER
            return True
        if self._state.round_id + 1 >= max_rounds:
            if y > b:
                self._state.match_outcome = MatchOutcome.YELLOW_WIN
            elif b > y:
                self._state.match_outcome = MatchOutcome.BLUE_WIN
            else:
                self._state.match_outcome = MatchOutcome.DRAW
            self._state.phase = Phase.MATCH_OVER
            return True
        return False

    def _swap_sides(self) -> None:
        for ts in self._state.teams.values():
            ts.side = Side.DEFENDER if ts.side == Side.ATTACKER else Side.ATTACKER
        for a in self._state.agents:
            a.side = Side.DEFENDER if a.side == Side.ATTACKER else Side.ATTACKER

    def _reset_for_new_round(self) -> None:
        # Reset all agents to spawns, fresh HP/armor (armor lost unless bought
        # again next buy phase), keep money & weapon.
        int(self._cfg.simulation.team_size)
        for team, side in (
            (Team.YELLOW, self._state.teams[Team.YELLOW].side),
            (Team.BLUE, self._state.teams[Team.BLUE].side),
        ):
            for idx, a in enumerate(self._state.agents_on_team(team)):
                spawn = self._map.nearest_spawn(side, idx)
                a.pos = np.array(spawn, dtype=np.float32)
                a.vel = np.zeros(2, dtype=np.float32)
                a.facing = math.pi if side == Side.DEFENDER else 0.0
                a.alive = True
                a.hp = 100.0
                a.armor = 0.0
                a.side = side
                a.has_bomb = False
                a.kills_round = 0
                a.deaths_round = 0
                a.damage_done_round = 0.0
                a.damage_taken_round = 0.0
                a.reaction_cooldown = 0
                # V1 simplification: weapon persists, but defuse kit doesn't.
                a.has_defuse_kit = False
        self._assign_bomb_random(self._state)
        self._state.phase = Phase.BUY
        self._state.phase_ticks_remaining = int(
            round(self._cfg.simulation.buy_time_seconds * self._engine_cfg.tick_rate_hz)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise_actions(
        self,
        actions: dict[AgentId, ActionBundle] | dict[int, ActionBundle] | None,
    ) -> dict[int, ActionBundle]:
        """Build a clean ``int -> ActionBundle`` dict with HOLD for missing ids."""
        result: dict[int, ActionBundle] = {}
        if actions:
            for k, v in actions.items():
                result[int(k)] = v
        for a in self._state.agents:
            if int(a.agent_id) not in result:
                result[int(a.agent_id)] = ActionBundle()
        return result

    def _zero_reward_dict(self) -> dict[AgentId, float]:
        return {AgentId(int(a.agent_id)): 0.0 for a in self._state.agents}

    def _find_agent(self, agent_id: int) -> AgentState | None:
        if 0 <= agent_id < len(self._state.agents):
            cand = self._state.agents[agent_id]
            if int(cand.agent_id) == agent_id:
                return cand
        for a in self._state.agents:
            if int(a.agent_id) == agent_id:
                return a
        return None

    def _emit_sound(self, agent: AgentState, *, kind: str, intensity: float) -> None:
        # Radius scales with intensity -- louder sounds reach further.
        radius = 4.0 + 12.0 * float(intensity)
        ev = SoundEvent(
            tick=int(self._state.tick),
            pos=(float(agent.pos[0]), float(agent.pos[1])),
            radius=float(radius),
            intensity=float(intensity),
            source_team=agent.team,
            kind=str(kind),
        )
        self._tick_sounds.append(ev)

    def _emit_message(self, agent: AgentState, action: ActionBundle) -> None:
        teammates = tuple(
            AgentId(int(other.agent_id))
            for other in self._state.agents
            if other.team == agent.team and other.agent_id != agent.agent_id and other.alive
        )
        msg = Message(
            tick=int(self._state.tick),
            sender=AgentId(int(agent.agent_id)),
            receivers=teammates,
            action=action.comm,
            payload=action.comm_payload,
            pos=(float(agent.pos[0]), float(agent.pos[1])),
        )
        self._tick_messages.append(msg)

    # ----- Serialization helpers --------------------------------------

    def _combat_event_to_dict(self, ev: CombatEvent) -> dict[str, Any]:
        return {
            "tick": int(ev.tick),
            "attacker": int(ev.attacker),
            "victim": int(ev.victim),
            "weapon": int(ev.weapon),
            "damage": float(ev.damage),
            "killed": bool(ev.killed),
            "distance": float(ev.distance),
            "through_cover": bool(ev.through_cover),
        }

    def _message_to_dict(self, msg: Message) -> dict[str, Any]:
        return {
            "tick": int(msg.tick),
            "sender": int(msg.sender),
            "receivers": [int(r) for r in msg.receivers],
            "action": int(msg.action),
            "pos": list(msg.pos) if msg.pos is not None else None,
        }

    def _sound_to_dict(self, snd: SoundEvent) -> dict[str, Any]:
        return {
            "tick": int(snd.tick),
            "pos": [float(snd.pos[0]), float(snd.pos[1])],
            "radius": float(snd.radius),
            "intensity": float(snd.intensity),
            "source_team": int(snd.source_team),
            "kind": str(snd.kind),
        }

    def _build_action_frame(self, actions: dict[int, ActionBundle]) -> ReplayActionFrame:
        rows: list[dict[str, Any]] = []
        for aid, ab in sorted(actions.items()):
            mv = np.asarray(ab.move_vec, dtype=np.float32).reshape(-1)
            if mv.shape[0] < 2:
                mv = np.zeros(2, dtype=np.float32)
            rows.append(
                {
                    "agent_id": int(aid),
                    # v0.4: continuous move stored as two floats for replays.
                    "move_x": float(mv[0]),
                    "move_y": float(mv[1]),
                    "micro": int(ab.micro),
                    "aim_target": int(ab.aim_target),
                    "comm": int(ab.comm),
                    "buy": int(ab.buy),
                }
            )
        return ReplayActionFrame(tick=int(self._state.tick), actions=rows)
