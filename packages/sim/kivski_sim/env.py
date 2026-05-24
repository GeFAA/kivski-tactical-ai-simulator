"""PettingZoo Parallel API wrapper around the Kivski deterministic engine.

This module bridges the deterministic :class:`kivski_sim.engine.Engine` and the
MARL trainers. It provides:

* :class:`KivskiParallelEnv` -- a PettingZoo ``ParallelEnv`` whose ``step`` and
  ``reset`` accept / return dicts keyed by agent name (``"agent_0"`` ...).
* A flat float32 *egocentric* observation per agent with partial observability:
  agents only see what their FoV / sound / message channel reveals. The
  observation layout is documented exhaustively in :mod:`kivski_sim.obs_decoder`
  and the per-section sizes are derived from the live :class:`KivskiConfig` so
  changing ``observation.teammate_slots`` (etc.) does not require touching this
  file.
* A ``MultiDiscrete`` action space with ``[move, micro, comm, buy, aim_target]``.
* A :meth:`KivskiParallelEnv.step_with_comms` helper that lets a TarMAC-style
  policy hand in continuous communication payload vectors alongside the
  discrete action heads.
* A dense reward function that gates on :attr:`RewardShapingConfig.enabled` and
  the per-env shaping factor (decay schedule lives in the trainer).

Determinism: ``env.reset(seed=int)`` is fully deterministic given identical
config and action streams (the engine itself is bit-exact; the wrapper adds
only deterministic memory updates).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from kivski_sim.config import KivskiConfig
from kivski_sim.engine import Engine, Snapshot
from kivski_sim.map_loader import MapData, load_map
from kivski_sim.types import (
    WEAPONS,
    ActionBundle,
    BombPhase,
    BuyChoice,
    CommAction,
    MatchOutcome,
    MicroAction,
    Phase,
    Side,
    Team,
    WeaponClass,
)
from kivski_sim.visibility import DEFAULT_FOV_RADIANS, compute_fov, sound_audible

__all__ = ["KivskiParallelEnv", "agent_name", "agent_index"]


# Number of distinct weapon classes (for one-hot encoding in the obs vector).
_NUM_WEAPONS: int = len(WeaponClass)

# v0.4: continuous move replaces the 9-way ``MoveIntent`` enum. The 4 remaining
# discrete heads share the same semantics as before.
_CONTINUOUS_MOVE_DIM: int = 2
_NUM_MICRO_ACTIONS: int = len(MicroAction)  # 6
_NUM_COMM_ACTIONS: int = len(CommAction)  # 9
_NUM_BUY_OPTIONS: int = len(BuyChoice)  # 8

# Sound-event kind ids used in the observation packing.
_SOUND_KINDS: dict[str, int] = {
    "step": 0,
    "shot": 1,
    "plant": 2,
    "defuse": 3,
    "bomb_pickup": 4,
}
_NUM_SOUND_KINDS: int = 5

# Phase one-hot uses these four "macro" phases (warmup is folded into buy).
_PHASE_ONEHOT_ORDER: tuple[Phase, ...] = (Phase.BUY, Phase.LIVE, Phase.POST_PLANT, Phase.ROUND_OVER)
_NUM_PHASES_OBS: int = len(_PHASE_ONEHOT_ORDER)  # 4

# Trade window for the "useful trade" shaping signal.
_TRADE_WINDOW_SECONDS: float = 3.0


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def agent_name(agent_id: int) -> str:
    """Return the canonical PettingZoo agent name for an integer engine id."""
    return f"agent_{int(agent_id)}"


def agent_index(name: str) -> int:
    """Parse the integer engine id out of a PettingZoo agent name."""
    return int(name.split("_", 1)[1])


# ---------------------------------------------------------------------------
# Per-agent memory used to build partially-observable observations.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _LastKnownEnemy:
    """One entry in an agent's "last known enemy" memory."""

    enemy_id: int
    last_pos: np.ndarray  # shape (2,) float32
    last_tick: int
    last_weapon: WeaponClass
    was_alive: bool
    last_distance: float


@dataclass(slots=True)
class _HeardSound:
    """A sound the agent perceived this tick (or recently)."""

    tick: int
    pos: tuple[float, float]
    intensity: float
    kind: str


@dataclass(slots=True)
class _ReceivedMessage:
    """A teammate comm message received this tick."""

    tick: int
    sender_id: int
    sender_team_idx: int  # 0..team_size-1 index within the agent's team
    action: CommAction
    pos: tuple[float, float] | None
    payload: np.ndarray | None


@dataclass(slots=True)
class _AgentMemory:
    """Per-agent partial-observability memory the wrapper maintains."""

    # enemy_id -> last_known entry. Aged each tick by reading current engine tick.
    last_known: dict[int, _LastKnownEnemy] = field(default_factory=dict)
    # Bounded buffers updated each step; older entries fall off the back.
    sounds: list[_HeardSound] = field(default_factory=list)
    messages: list[_ReceivedMessage] = field(default_factory=list)
    # Bookkeeping for trade / pointless-death shaping.
    last_death_tick: int = -10_000
    last_kill_tick: int = -10_000
    dealt_damage_round: float = 0.0
    received_damage_round: float = 0.0


# ---------------------------------------------------------------------------
# Observation layout (computed from config; documented in obs_decoder.py)
# ---------------------------------------------------------------------------


# Per-slot widths (kept here, mirrored in obs_decoder for reproducibility).
_SELF_BLOCK_WIDTH: int = 7 + _NUM_WEAPONS  # 14 when _NUM_WEAPONS=7
_SELF_POS_WIDTH: int = 3
_TEAMMATE_SLOT_WIDTH: int = 8
_ENEMY_SLOT_WIDTH: int = 6
_SOUND_SLOT_WIDTH: int = 5
_MESSAGE_SLOT_WIDTH: int = 7
_MAP_CTX_WIDTH: int = 6 + _NUM_PHASES_OBS  # 10
_TEAM_CTX_WIDTH: int = 6


def _section_widths(cfg: KivskiConfig) -> dict[str, int]:
    """Return the byte-width of each observation section for the given config."""
    obs_cfg = cfg.agent.observation
    return {
        "self": _SELF_BLOCK_WIDTH,
        "self_pos": _SELF_POS_WIDTH,
        "teammates": _TEAMMATE_SLOT_WIDTH * int(obs_cfg.teammate_slots),
        "enemies": _ENEMY_SLOT_WIDTH * int(obs_cfg.last_known_enemies),
        "sounds": _SOUND_SLOT_WIDTH * int(obs_cfg.sound_event_slots),
        "messages": _MESSAGE_SLOT_WIDTH * int(obs_cfg.received_message_slots),
        "map_ctx": _MAP_CTX_WIDTH,
        "team_ctx": _TEAM_CTX_WIDTH,
    }


def _observation_dim(cfg: KivskiConfig) -> int:
    """Total length of the flat observation vector for one agent."""
    return sum(_section_widths(cfg).values())


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class KivskiParallelEnv(ParallelEnv):
    """PettingZoo Parallel wrapper around :class:`Engine`.

    Lifecycle:

    1. ``reset(seed=...)``  -> ``(observations, infos)`` keyed by agent name.
    2. Repeated ``step(actions)`` -> ``(obs, rewards, terminations,
       truncations, infos)`` where ``actions`` is a dict of either an int
       (encoded action index) or a length-5 numpy array
       ``[move, micro, comm, buy, aim_target]``.
    3. When the *match* finishes ``terminations[a]`` is True for every agent
       and the env should be re-``reset``.

    Notes on partial observability: the wrapper maintains a per-agent
    "last-known enemy" map and bounded buffers of recent sounds / comm
    messages. These are deterministic functions of the engine snapshot and
    the agent's FoV, so the entire pipeline stays reproducible.
    """

    metadata: dict[str, Any] = {"name": "kivski_v0", "is_parallelizable": True}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: KivskiConfig,
        map_name: str = "dustline",
        seed: int | None = None,
        *,
        map_data: MapData | None = None,
        frame_skip: int | None = None,
    ) -> None:
        super().__init__()
        self._cfg: KivskiConfig = config
        self._map_name: str = str(map_name)
        self._map: MapData = map_data if map_data is not None else load_map(self._map_name)
        # Honor either an explicit env seed or the config seed.
        self._seed: int = int(seed) if seed is not None else int(config.seed)
        self._engine: Engine = Engine(config=config, map_data=self._map, seed=self._seed)
        # Episode counter for shaping decay (the trainer can override directly).
        self._episodes_done: int = 0
        # The wrapper-controlled "shaping factor" multiplies every dense reward.
        # 1.0 = full shaping (subject to config flag); 0.0 = pure outcome rewards.
        self._shaping_factor: float = 1.0 if bool(config.reward_shaping.enabled) else 0.0

        # Frame-skip (action repeat). Constructor arg wins over cfg so tests
        # can override without rebuilding the config. Clamped to >=1 so
        # `range(self._frame_skip)` always runs at least once.
        if frame_skip is None:
            frame_skip = int(getattr(config.simulation, "frame_skip", 1) or 1)
        self._frame_skip: int = max(1, int(frame_skip))

        # Reward-curriculum gate. When the trainer flips a stage we filter
        # the dense reward components in :meth:`_compute_rewards` to only
        # those buckets present in `_active_reward_features` (or all when
        # the gate is None = curriculum disabled).
        self._active_reward_features: set[str] | None = None
        self._curriculum_stage_name: str = "default"

        # Number of agents per team and total -- locked for the whole episode.
        self._team_size: int = int(config.simulation.team_size)
        n = 2 * self._team_size
        self._possible_agents: list[str] = [agent_name(i) for i in range(n)]
        # PettingZoo's "agents" list shrinks only when an agent leaves the env
        # permanently. For our use-case we keep it constant across the episode
        # and rely on the ``terminations`` dict to express "this agent is dead".
        self._agents: list[str] = list(self._possible_agents)

        # Cache observation length so we can sanity-check shapes downstream.
        self._obs_dim: int = _observation_dim(config)
        self._section_widths: dict[str, int] = _section_widths(config)

        # Build the gym spaces (one per agent, all identical -- agents are
        # symmetric, only the *contents* differ).
        self._observation_space: spaces.Box = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        # aim_target slots: 0 = no target, 1..N = pointer into the *other*
        # agents (excludes self). Total = 2 * team_size (other agents) + 1.
        aim_targets = 2 * self._team_size  # excludes self
        self._aim_target_dim: int = aim_targets + 1
        # v0.4 mixed action space:
        #   - ``move``     -> Box(-1, 1, shape=(2,)) continuous heading + speed
        #   - ``discrete`` -> MultiDiscrete([micro, comm, buy, aim_target])
        self._discrete_action_dims: np.ndarray = np.array(
            [
                _NUM_MICRO_ACTIONS,
                _NUM_COMM_ACTIONS,
                _NUM_BUY_OPTIONS,
                self._aim_target_dim,
            ],
            dtype=np.int64,
        )
        self._action_space: spaces.Dict = spaces.Dict(
            {
                "move": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(_CONTINUOUS_MOVE_DIM,),
                    dtype=np.float32,
                ),
                "discrete": spaces.MultiDiscrete(self._discrete_action_dims),
            }
        )

        # Per-agent partial-observability memory.
        self._memory: dict[str, _AgentMemory] = {name: _AgentMemory() for name in self._possible_agents}
        # Used by the reward shaping bookkeeping.
        self._round_id_at_last_step: int = 0
        # Memory of pre-step (alive, hp, money, has_bomb, side) for delta-based rewards.
        self._prev_agent_snapshot: list[dict[str, Any]] = []
        self._prev_score: dict[Team, int] = {Team.YELLOW: 0, Team.BLUE: 0}
        # Communication scratch buffer (cleared each step) -- exposed via infos.
        self._latest_comm_messages: dict[str, dict[int, np.ndarray]] = {
            name: {} for name in self._possible_agents
        }
        # Buffer for the next reset call.
        self._needs_reset: bool = True

    # ------------------------------------------------------------------
    # PettingZoo API surface
    # ------------------------------------------------------------------

    @property
    def possible_agents(self) -> list[str]:
        return list(self._possible_agents)

    @property
    def agents(self) -> list[str]:
        return list(self._agents)

    @agents.setter
    def agents(self, value: Iterable[str]) -> None:
        # PettingZoo writes here from external utilities (e.g. wrappers).
        self._agents = list(value)

    @property
    def num_agents(self) -> int:
        return len(self._agents)

    @property
    def max_num_agents(self) -> int:
        return len(self._possible_agents)

    @property
    def engine(self) -> Engine:
        """Direct access to the underlying engine (useful for tests/viewers)."""
        return self._engine

    @property
    def map(self) -> MapData:
        return self._map

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def section_widths(self) -> dict[str, int]:
        """Width (number of floats) per observation section -- read-only copy."""
        return dict(self._section_widths)

    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:  # type: ignore[override]
        del agent  # all agents share an identical space
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Space:  # type: ignore[override]
        del agent
        return self._action_space

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Reset the engine, clear memory, and return the initial observations."""
        del options
        if seed is not None:
            self._seed = int(seed)
        # Build a fresh engine so RNG channels and replay state are pristine.
        self._engine = Engine(config=self._cfg, map_data=self._map, seed=self._seed)
        self._engine.reset(seed=self._seed)
        self._memory = {name: _AgentMemory() for name in self._possible_agents}
        self._latest_comm_messages = {name: {} for name in self._possible_agents}
        self._agents = list(self._possible_agents)
        self._round_id_at_last_step = 0
        self._prev_score = {Team.YELLOW: 0, Team.BLUE: 0}
        self._prev_agent_snapshot = self._snapshot_agent_minisnapshot()
        self._needs_reset = False

        observations = self._build_all_observations()
        infos = self._build_all_infos()
        return observations, infos

    # ------------------------------------------------------------------

    def step(
        self,
        actions: dict[str, int | np.ndarray | list[int]],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Advance the env one tick. See class docstring for shapes."""
        return self.step_with_comms(actions, comm_payloads=None)

    def step_with_comms(
        self,
        actions: dict[str, int | np.ndarray | list[int]],
        comm_payloads: dict[str, np.ndarray] | None = None,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Same as :meth:`step` but accepts a per-agent comm payload vector.

        The payloads are forwarded into the engine via
        :attr:`ActionBundle.comm_payload` and surfaced to receivers through
        ``info["comm_messages"]`` so policies can run TarMAC-style attention
        over them.

        When :attr:`frame_skip` > 1, the same engine action is replayed
        for ``frame_skip`` inner ticks and rewards are summed before they
        are returned to the caller. Observations / terminations come from
        the *last* inner tick. Memory updates run after each inner tick
        so partial-observability buffers stay coherent.
        """
        if self._needs_reset:
            raise RuntimeError("KivskiParallelEnv.step called before reset()")

        # 1) Translate the dict[str, action] into an engine-level
        # dict[int, ActionBundle], merging in any optional comm payloads.
        engine_actions = self._decode_actions(actions, comm_payloads)

        rewards: dict[str, float] = dict.fromkeys(self._possible_agents, 0.0)
        observations: dict[str, np.ndarray] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}
        any_terminal = False

        for _ in range(self._frame_skip):
            # 2) Remember pre-step state so we can compute reward deltas.
            pre_snapshot = self._snapshot_agent_minisnapshot()
            pre_match_outcome = self._engine.state.match_outcome
            pre_round_id = int(self._engine.state.round_id)

            # 3) Drive the engine forward. ``light_snapshot=True`` skips the
            # JSON-serialisable agent/event/message lists in the engine's
            # return value -- the env wrapper reads engine.state directly so
            # those fields are pure overhead during training.
            snap, engine_rewards, _engine_done = self._engine.step(engine_actions, light_snapshot=True)

            # 4) Update the per-agent memory using the post-step snapshot.
            self._update_memory(snap)

            # 5) Build observations / rewards / terminations / truncations / infos.
            observations = self._build_all_observations()
            inner_rewards = self._compute_rewards(
                engine_rewards=engine_rewards,
                pre_snapshot=pre_snapshot,
                pre_round_id=pre_round_id,
                pre_match_outcome=pre_match_outcome,
            )
            for name, r in inner_rewards.items():
                rewards[name] = float(rewards[name]) + float(r)
            terminations = self._compute_terminations()
            truncations = self._compute_truncations()
            infos = self._build_all_infos()

            # Cache for next inner step.
            self._prev_agent_snapshot = self._snapshot_agent_minisnapshot()
            self._round_id_at_last_step = int(self._engine.state.round_id)

            if all(terminations.values()) or all(truncations.values()):
                any_terminal = True
                break

        if any_terminal:
            self._needs_reset = True
            self._episodes_done += 1
        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------

    def render(self) -> Snapshot:
        """Return the underlying engine snapshot for external renderers."""
        return self._engine.snapshot()

    def close(self) -> None:
        """Nothing to clean up -- engine is pure Python objects."""
        self._needs_reset = True

    # ------------------------------------------------------------------
    # Shaping controls (used by the trainer to decay shaping over time)
    # ------------------------------------------------------------------

    def set_shaping_factor(self, factor: float) -> None:
        """Override the dense-reward multiplier (``0.0`` = outcome-only)."""
        self._shaping_factor = float(max(0.0, min(1.0, factor)))

    def get_shaping_factor(self) -> float:
        return float(self._shaping_factor)

    # ------------------------------------------------------------------
    # Reward curriculum controls
    # ------------------------------------------------------------------

    def set_curriculum_stage(
        self, stage_name: str, features: list[str] | tuple[str, ...] | set[str] | None
    ) -> None:
        """Activate a reward-curriculum stage by name.

        Args:
            stage_name: Free-form label for logging (no behavioural effect).
            features: Bucket names allowed to contribute to dense rewards
                this stage. ``["all"]`` or ``None`` enables every bucket
                (= disables the curriculum gate). Recognised buckets:
                ``kill``, ``survive``, ``damage_dealt``, ``damage_received``,
                ``bomb_pickup``, ``bomb_plant``, ``bomb_defuse``,
                ``useful_trade``, ``pointless_death``, ``map_control``.
        """
        self._curriculum_stage_name = str(stage_name)
        if features is None:
            self._active_reward_features = None
            return
        feats = {str(f).strip().lower() for f in features}
        if not feats or "all" in feats:
            self._active_reward_features = None
            return
        self._active_reward_features = feats

    def get_curriculum_stage(self) -> tuple[str, set[str] | None]:
        """Return ``(stage_name, active_features_or_None)`` (None == all)."""
        return (
            self._curriculum_stage_name,
            set(self._active_reward_features) if self._active_reward_features is not None else None,
        )

    def _feature_enabled(self, feature: str) -> bool:
        """Return True iff ``feature`` is in the current curriculum stage."""
        if self._active_reward_features is None:
            return True
        return feature in self._active_reward_features

    # ------------------------------------------------------------------
    # Frame-skip introspection
    # ------------------------------------------------------------------

    @property
    def frame_skip(self) -> int:
        """Number of engine ticks consumed per env.step()."""
        return int(self._frame_skip)

    def set_frame_skip(self, frame_skip: int) -> None:
        """Override frame-skip at runtime. Clamped to >=1."""
        self._frame_skip = max(1, int(frame_skip))

    def shaping_factor_for_episode(self) -> float:
        """Compute the schedule-implied shaping factor for the current episode.

        The trainer is free to ignore this and call :meth:`set_shaping_factor`
        explicitly. We keep the logic here too so single-process scripts work
        without an explicit schedule callback.
        """
        if not bool(self._cfg.reward_shaping.enabled):
            return 0.0
        decay_after = int(self._cfg.reward_shaping.decay_after_episodes)
        if decay_after <= 0:
            return 1.0
        return float(max(0.0, 1.0 - self._episodes_done / decay_after))

    # ------------------------------------------------------------------
    # Internal: action decoding
    # ------------------------------------------------------------------

    def _decode_actions(
        self,
        actions: dict[str, Any],
        comm_payloads: dict[str, np.ndarray] | None,
    ) -> dict[int, ActionBundle]:
        """Convert PettingZoo actions into an engine ``dict[int, ActionBundle]``.

        Accepts the new dict format ``{"move": [mx, my], "discrete": [micro, comm,
        buy, aim]}`` as well as a v0.3-style flat ``[move_x, move_y, micro, comm,
        buy, aim]`` numpy array / list for ad-hoc test usage.
        """
        out: dict[int, ActionBundle] = {}
        team_size = self._team_size
        for name in self._possible_agents:
            aid = agent_index(name)
            raw = actions.get(name)
            if raw is None:
                # Missing action -- the engine treats this as HOLD.
                out[aid] = ActionBundle()
                continue
            mv, micro_i, comm_i, buy_i, aim_i = _coerce_mixed_action(raw)
            micro = MicroAction(int(np.clip(micro_i, 0, _NUM_MICRO_ACTIONS - 1)))
            comm = CommAction(int(np.clip(comm_i, 0, _NUM_COMM_ACTIONS - 1)))
            buy = BuyChoice(int(np.clip(buy_i, 0, _NUM_BUY_OPTIONS - 1)))

            aim_target = self._decode_aim_target(aid, int(aim_i), team_size)
            payload = None
            if comm_payloads is not None and name in comm_payloads:
                payload = np.asarray(comm_payloads[name], dtype=np.float32)

            out[aid] = ActionBundle(
                move_vec=mv,
                micro=micro,
                aim_target=aim_target,
                comm=comm,
                comm_payload=payload,
                buy=buy,
            )
        return out

    def _decode_aim_target(self, self_aid: int, aim_idx: int, team_size: int) -> int:
        """Translate a slot index in [0, 2*team_size] -> a concrete agent id.

        Slot 0 means "no specific target". Slots 1..2*team_size index into the
        ordered list of *other* agents (skip ``self_aid``).
        """
        if aim_idx <= 0:
            return -1
        total = 2 * team_size
        slot = int(aim_idx) - 1
        if slot < 0 or slot >= total:
            return -1
        # Build an ordered list of "other agents" (skip self) deterministically.
        others = [i for i in range(total) if i != self_aid]
        if slot >= len(others):
            return -1
        return int(others[slot])

    # ------------------------------------------------------------------
    # Internal: observation construction
    # ------------------------------------------------------------------

    def _build_all_observations(self) -> dict[str, np.ndarray]:
        # We previously called ``self._engine.snapshot()`` here and threaded
        # it through ``_build_observation``. The observation builder, however,
        # only ever reads from ``self._engine.state`` directly -- the snapshot
        # was unused. Skipping the snapshot here saves an O(agents + events)
        # serialisation per step, which is a meaningful chunk of training-time
        # throughput at 32 envs.
        obs: dict[str, np.ndarray] = {}
        for name in self._possible_agents:
            obs[name] = self._build_observation(name, None)
        return obs

    def _build_observation(self, agent: str, snap: Snapshot | None) -> np.ndarray:
        """Pack the per-agent egocentric observation into a flat float32 vector."""
        del snap  # reserved for callers that pass an explicit snapshot
        aid = agent_index(agent)
        self_state = self._engine.state.agents[aid]
        mem = self._memory[agent]
        vec = np.zeros(self._obs_dim, dtype=np.float32)
        cursor = 0

        # ----- Self block -----------------------------------------------
        # [hp/100, armor/100, money/8000, weapon_onehot(7), has_bomb, alive]
        vec[cursor] = float(self_state.hp) / 100.0
        vec[cursor + 1] = float(self_state.armor) / 100.0
        vec[cursor + 2] = float(self_state.money) / 8000.0
        # Weapon one-hot
        weapon_idx = int(self_state.weapon)
        if 0 <= weapon_idx < _NUM_WEAPONS:
            vec[cursor + 3 + weapon_idx] = 1.0
        vec[cursor + 3 + _NUM_WEAPONS] = 1.0 if self_state.has_bomb else 0.0
        vec[cursor + 4 + _NUM_WEAPONS] = 1.0 if self_state.alive else 0.0
        cursor += _SELF_BLOCK_WIDTH

        # ----- Self position --------------------------------------------
        w, h = float(self._map.width), float(self._map.height)
        vec[cursor] = float(self_state.pos[0]) / max(w, 1.0)
        vec[cursor + 1] = float(self_state.pos[1]) / max(h, 1.0)
        vec[cursor + 2] = float(self_state.facing) / (2.0 * math.pi)
        cursor += _SELF_POS_WIDTH

        # ----- Teammates -------------------------------------------------
        team_slots = int(self._cfg.agent.observation.teammate_slots)
        teammates = [
            other
            for other in self._engine.state.agents
            if other.team == self_state.team and other.agent_id != self_state.agent_id
        ]
        # Sort by id for determinism, take first ``team_slots``.
        teammates.sort(key=lambda a: int(a.agent_id))
        for slot in range(team_slots):
            base = cursor + slot * _TEAMMATE_SLOT_WIDTH
            if slot >= len(teammates):
                continue  # padded zero
            tm = teammates[slot]
            dx = float(tm.pos[0] - self_state.pos[0])
            dy = float(tm.pos[1] - self_state.pos[1])
            dist = math.sqrt(dx * dx + dy * dy)
            vec[base + 0] = 1.0 if tm.alive else 0.0
            vec[base + 1] = float(tm.hp) / 100.0
            vec[base + 2] = dx / max(w, 1.0)
            vec[base + 3] = dy / max(h, 1.0)
            vec[base + 4] = float(min(dist / max(w, 1.0), 1.0))
            vec[base + 5] = 1.0 if tm.has_bomb else 0.0
            vec[base + 6] = float(int(tm.weapon)) / float(max(_NUM_WEAPONS - 1, 1))
            vec[base + 7] = float(tm.money) / 8000.0
        cursor += team_slots * _TEAMMATE_SLOT_WIDTH

        # ----- Last-known enemies ---------------------------------------
        enemy_slots = int(self._cfg.agent.observation.last_known_enemies)
        # Sort last-known entries by recency, freshest first.
        entries = sorted(mem.last_known.values(), key=lambda e: -int(e.last_tick))
        max_age_ticks = float(self._cfg.simulation.max_ticks_per_round)
        for slot in range(enemy_slots):
            base = cursor + slot * _ENEMY_SLOT_WIDTH
            if slot >= len(entries):
                continue
            entry = entries[slot]
            age = max(0, int(self._engine.state.tick) - int(entry.last_tick))
            dx = float(entry.last_pos[0] - self_state.pos[0])
            dy = float(entry.last_pos[1] - self_state.pos[1])
            vec[base + 0] = float(age) / max(max_age_ticks, 1.0)
            vec[base + 1] = dx / max(w, 1.0)
            vec[base + 2] = dy / max(h, 1.0)
            vec[base + 3] = float(int(entry.last_weapon)) / float(max(_NUM_WEAPONS - 1, 1))
            vec[base + 4] = 1.0 if entry.was_alive else 0.0
            vec[base + 5] = float(min(entry.last_distance / max(w, 1.0), 1.0))
        cursor += enemy_slots * _ENEMY_SLOT_WIDTH

        # ----- Sound events ---------------------------------------------
        sound_slots = int(self._cfg.agent.observation.sound_event_slots)
        # Most recent first.
        recent_sounds = sorted(mem.sounds, key=lambda s: -int(s.tick))[:sound_slots]
        for slot in range(sound_slots):
            base = cursor + slot * _SOUND_SLOT_WIDTH
            if slot >= len(recent_sounds):
                continue
            snd = recent_sounds[slot]
            age = max(0, int(self._engine.state.tick) - int(snd.tick))
            dx = float(snd.pos[0]) - float(self_state.pos[0])
            dy = float(snd.pos[1]) - float(self_state.pos[1])
            kind_id = _SOUND_KINDS.get(snd.kind, 0)
            vec[base + 0] = float(age) / 30.0
            vec[base + 1] = dx / max(w, 1.0)
            vec[base + 2] = dy / max(h, 1.0)
            vec[base + 3] = float(snd.intensity)
            vec[base + 4] = float(kind_id) / float(max(_NUM_SOUND_KINDS - 1, 1))
        cursor += sound_slots * _SOUND_SLOT_WIDTH

        # ----- Received messages ----------------------------------------
        msg_slots = int(self._cfg.agent.observation.received_message_slots)
        recent_msgs = sorted(mem.messages, key=lambda m: -int(m.tick))[:msg_slots]
        for slot in range(msg_slots):
            base = cursor + slot * _MESSAGE_SLOT_WIDTH
            if slot >= len(recent_msgs):
                continue
            msg = recent_msgs[slot]
            age = max(0, int(self._engine.state.tick) - int(msg.tick))
            dx = 0.0
            dy = 0.0
            if msg.pos is not None:
                dx = float(msg.pos[0]) - float(self_state.pos[0])
                dy = float(msg.pos[1]) - float(self_state.pos[1])
            payload_norm = 0.0
            if msg.payload is not None and msg.payload.size > 0:
                payload_norm = float(np.linalg.norm(msg.payload))
            vec[base + 0] = float(age) / 30.0
            vec[base + 1] = float(msg.sender_team_idx) / float(max(self._team_size - 1, 1))
            vec[base + 2] = float(int(msg.action)) / float(max(_NUM_COMM_ACTIONS - 1, 1))
            vec[base + 3] = dx / max(w, 1.0)
            vec[base + 4] = dy / max(h, 1.0)
            vec[base + 5] = 1.0 if msg.payload is not None else 0.0
            vec[base + 6] = payload_norm
        cursor += msg_slots * _MESSAGE_SLOT_WIDTH

        # ----- Map context ----------------------------------------------
        # [bombsite_a_dx, bombsite_a_dy, bombsite_b_dx, bombsite_b_dy,
        #  time_in_round_norm, phase_onehot4]
        site_a = self._map.bombsites.get("A")
        site_b = self._map.bombsites.get("B")
        if site_a is not None:
            vec[cursor + 0] = (float(site_a.center[0]) - float(self_state.pos[0])) / max(w, 1.0)
            vec[cursor + 1] = (float(site_a.center[1]) - float(self_state.pos[1])) / max(h, 1.0)
        if site_b is not None:
            vec[cursor + 2] = (float(site_b.center[0]) - float(self_state.pos[0])) / max(w, 1.0)
            vec[cursor + 3] = (float(site_b.center[1]) - float(self_state.pos[1])) / max(h, 1.0)
        # Time-in-round normalized: lower = end of phase.
        max_phase_ticks = max(1, int(self._cfg.simulation.max_ticks_per_round))
        vec[cursor + 4] = float(self._engine.state.phase_ticks_remaining) / float(max_phase_ticks)
        # Reserved slot (kept zero -- room for a future map id one-hot).
        vec[cursor + 5] = 0.0
        # Phase one-hot.
        phase = self._engine.state.phase
        for i, p in enumerate(_PHASE_ONEHOT_ORDER):
            vec[cursor + 6 + i] = 1.0 if phase == p else 0.0
        cursor += _MAP_CTX_WIDTH

        # ----- Team context ---------------------------------------------
        # [teammates_alive/team_size, enemies_alive_known/team_size,
        #  bomb_phase/7, my_team_score/max_rounds,
        #  enemy_team_score/max_rounds, consecutive_losses/5]
        teammates_alive = sum(
            1
            for other in self._engine.state.agents
            if other.team == self_state.team and other.agent_id != self_state.agent_id and other.alive
        )
        enemies_alive_known = sum(1 for entry in mem.last_known.values() if entry.was_alive)
        bomb_phase_val = int(self._engine.state.bomb.phase) / float(max(len(BombPhase) - 1, 1))
        max_rounds = max(1, int(self._cfg.simulation.max_rounds))
        my_team = self_state.team
        enemy_team = Team.BLUE if my_team == Team.YELLOW else Team.YELLOW
        my_score = int(self._engine.state.teams[my_team].score)
        enemy_score = int(self._engine.state.teams[enemy_team].score)
        consec = int(self._engine.state.teams[my_team].consecutive_losses)
        vec[cursor + 0] = float(teammates_alive) / float(max(self._team_size, 1))
        vec[cursor + 1] = float(enemies_alive_known) / float(max(self._team_size, 1))
        vec[cursor + 2] = bomb_phase_val
        vec[cursor + 3] = float(my_score) / float(max_rounds)
        vec[cursor + 4] = float(enemy_score) / float(max_rounds)
        vec[cursor + 5] = float(consec) / 5.0
        cursor += _TEAM_CTX_WIDTH

        # Sanity guard: we should have filled the whole vector exactly.
        assert cursor == self._obs_dim, (cursor, self._obs_dim)
        return vec

    # ------------------------------------------------------------------
    # Internal: memory updates
    # ------------------------------------------------------------------

    def _update_memory(self, snap: Snapshot) -> None:
        """Refresh each agent's last-known / sound / message buffers."""
        del snap  # we read directly from engine state for determinism
        state = self._engine.state
        tick = int(state.tick)
        # Reset comm scratch buffer.
        self._latest_comm_messages = {name: {} for name in self._possible_agents}

        # 1) Update last-known enemies using FoV.
        for self_state in state.agents:
            name = agent_name(int(self_state.agent_id))
            mem = self._memory[name]
            if not self_state.alive:
                # Dead agents stop accumulating new memories.
                self._age_sounds_and_messages(mem, tick)
                continue
            enemies = [
                (int(other.agent_id), other.pos.astype(np.float64))
                for other in state.agents
                if other.side != self_state.side and other.alive
            ]
            max_range = float(WEAPONS[self_state.weapon].max_range)
            # FoV detection range is bounded by weapon max_range, capped by map.
            # We still want vision when holding a knife, so floor it.
            vis_range = max(max_range, 12.0)
            visible_ids = compute_fov(
                self._map,
                self_state.pos.astype(np.float64),
                float(self_state.facing),
                DEFAULT_FOV_RADIANS,
                vis_range,
                enemies,
            )
            for eid in visible_ids:
                target = state.agents[eid]
                dist = float(np.linalg.norm(target.pos - self_state.pos))
                mem.last_known[int(eid)] = _LastKnownEnemy(
                    enemy_id=int(eid),
                    last_pos=np.array(target.pos, dtype=np.float32),
                    last_tick=tick,
                    last_weapon=target.weapon,
                    was_alive=bool(target.alive),
                    last_distance=dist,
                )

        # 2) Sound events: every agent perceives sounds via sound_audible.
        for self_state in state.agents:
            name = agent_name(int(self_state.agent_id))
            mem = self._memory[name]
            if not self_state.alive:
                continue
            for snd in state.sounds:
                if snd.source_team == self_state.team:
                    continue  # don't surface own-team sounds to the agent
                heard, _strength, approx_pos = sound_audible(
                    self._map,
                    self_state.pos.astype(np.float64),
                    np.array(snd.pos, dtype=np.float64),
                    float(snd.intensity),
                    float(snd.radius),
                )
                if heard:
                    mem.sounds.append(
                        _HeardSound(
                            tick=tick,
                            pos=approx_pos,
                            intensity=float(snd.intensity),
                            kind=str(snd.kind),
                        )
                    )

        # 3) Comm messages: broadcast to teammates (engine already filters).
        for msg in state.messages:
            sender_aid = int(msg.sender)
            sender_state = state.agents[sender_aid]
            # Compute the sender's intra-team index for the receiver-side feature.
            sender_idx = sum(
                1 for other in state.agents if other.team == sender_state.team and other.agent_id < sender_aid
            )
            payload = msg.payload
            for rid in msg.receivers:
                receiver_name = agent_name(int(rid))
                self._memory[receiver_name].messages.append(
                    _ReceivedMessage(
                        tick=tick,
                        sender_id=sender_aid,
                        sender_team_idx=sender_idx,
                        action=msg.action,
                        pos=msg.pos,
                        payload=payload,
                    )
                )
                if payload is not None:
                    self._latest_comm_messages[receiver_name][sender_aid] = payload.astype(
                        np.float32, copy=False
                    )

        # 4) Age + bound the sound / message buffers.
        for mem in self._memory.values():
            self._age_sounds_and_messages(mem, tick)

    def _age_sounds_and_messages(self, mem: _AgentMemory, tick: int) -> None:
        """Drop entries older than 30 ticks and cap buffer length."""
        max_age = 30
        # Sounds.
        mem.sounds = [s for s in mem.sounds if tick - s.tick <= max_age]
        if len(mem.sounds) > 64:
            mem.sounds = mem.sounds[-64:]
        # Messages.
        mem.messages = [m for m in mem.messages if tick - m.tick <= max_age]
        if len(mem.messages) > 64:
            mem.messages = mem.messages[-64:]

    # ------------------------------------------------------------------
    # Internal: info dicts
    # ------------------------------------------------------------------

    def _build_all_infos(self) -> dict[str, dict[str, Any]]:
        infos: dict[str, dict[str, Any]] = {}
        for name in self._possible_agents:
            comm_msgs = dict(self._latest_comm_messages[name])
            mask = np.zeros(2 * self._team_size, dtype=np.float32)
            for sender_id in comm_msgs:
                # Mask is indexed by global agent id for simplicity (callers
                # can re-key it if they prefer team-local indices).
                if 0 <= sender_id < mask.shape[0]:
                    mask[sender_id] = 1.0
            self_state = self._engine.state.agents[agent_index(name)]
            infos[name] = {
                "comm_messages": comm_msgs,
                "comm_attention_mask": mask,
                "alive": bool(self_state.alive),
                "team": int(self_state.team),
                "side": int(self_state.side),
                "round_id": int(self._engine.state.round_id),
                "phase": int(self._engine.state.phase),
                "tick": int(self._engine.state.tick),
            }
        return infos

    # ------------------------------------------------------------------
    # Internal: termination / truncation
    # ------------------------------------------------------------------

    def _compute_terminations(self) -> dict[str, bool]:
        """Per-agent termination flag.

        PettingZoo Parallel semantics expect ``terminations[a]`` to flip True
        once an agent is "done forever" for this episode. Our episode == the
        full match: even agents that die mid-round respawn at the next round
        start, so we only terminate every agent simultaneously when the match
        is over. Per-round death is surfaced through ``info["alive"]`` so
        trainers that care can mask gradients on that signal.
        """
        match_done = self._engine.state.match_outcome != MatchOutcome.NONE
        return {name: bool(match_done) for name in self._possible_agents}

    def _compute_truncations(self) -> dict[str, bool]:
        # We do not artificially truncate -- the engine drives match length.
        return dict.fromkeys(self._possible_agents, False)

    # ------------------------------------------------------------------
    # Internal: reward computation
    # ------------------------------------------------------------------

    def _snapshot_agent_minisnapshot(self) -> list[dict[str, Any]]:
        """Snapshot the subset of agent state needed for reward deltas."""
        out: list[dict[str, Any]] = []
        for a in self._engine.state.agents:
            out.append(
                {
                    "id": int(a.agent_id),
                    "alive": bool(a.alive),
                    "hp": float(a.hp),
                    "money": int(a.money),
                    "has_bomb": bool(a.has_bomb),
                    "team": int(a.team),
                    "side": int(a.side),
                    "damage_done_round": float(a.damage_done_round),
                    "damage_taken_round": float(a.damage_taken_round),
                }
            )
        return out

    def _compute_rewards(
        self,
        engine_rewards: dict[int, float],
        pre_snapshot: list[dict[str, Any]],
        pre_round_id: int,
        pre_match_outcome: MatchOutcome,
    ) -> dict[str, float]:
        """Combine engine rewards with dense shaping into a per-agent dict.

        Strategy:

        * The engine already emits +1/-1 for kills and round-win/-loss bonuses,
          plus zero for everything else. We forward those unconditionally as
          "outcome rewards" so the agents always feel the outcome signal.
        * On top we layer (subject to shaping factor) the dense rewards listed
          in :class:`RewardShapingConfig`.
        """
        del pre_match_outcome  # currently unused -- reserved for future signals
        rs = self._cfg.reward_shaping
        factor = float(self._shaping_factor) * (1.0 if rs.enabled else 0.0)
        dt = 1.0 / float(self._cfg.simulation.tick_rate_hz)

        rewards: dict[str, float] = dict.fromkeys(self._possible_agents, 0.0)
        # Forward outcome rewards from engine, scaled.
        for aid, r in engine_rewards.items():
            rewards[agent_name(int(aid))] = float(r)

        if factor <= 0.0:
            return rewards

        # Build a quick lookup from agent_id -> pre/post snapshots.
        pre_by_id: dict[int, dict[str, Any]] = {int(p["id"]): p for p in pre_snapshot}

        # Track per-tick deltas.
        round_changed = int(self._engine.state.round_id) != int(pre_round_id)
        team_kills_this_tick: dict[Team, list[int]] = {Team.YELLOW: [], Team.BLUE: []}
        team_deaths_this_tick: dict[Team, list[int]] = {Team.YELLOW: [], Team.BLUE: []}

        for a in self._engine.state.agents:
            aid = int(a.agent_id)
            pre = pre_by_id.get(aid)
            if pre is None:
                continue
            name = agent_name(aid)
            shaping = 0.0

            # Damage dealt this tick.
            dmg_dealt = float(a.damage_done_round) - float(pre["damage_done_round"])
            if round_changed:
                # round_summaries reset damage_done_round, so we use the post-step
                # value directly as the delta-since-round-start.
                dmg_dealt = float(a.damage_done_round)
            if dmg_dealt > 0.0 and self._feature_enabled("damage_dealt"):
                shaping += float(rs.damage_dealt_per_hp) * dmg_dealt

            # Damage received this tick.
            dmg_recv = float(a.damage_taken_round) - float(pre["damage_taken_round"])
            if round_changed:
                dmg_recv = float(a.damage_taken_round)
            if dmg_recv > 0.0 and self._feature_enabled("damage_received"):
                shaping += float(rs.damage_received_per_hp) * dmg_recv

            # Survival reward: small per-tick bonus while alive.
            if a.alive and self._feature_enabled("survive"):
                shaping += float(rs.survival_per_second) * dt

            # Bomb pickup: had no bomb before, has it now (and we are attacker).
            if (
                (not bool(pre["has_bomb"]))
                and a.has_bomb
                and a.side == Side.ATTACKER
                and self._feature_enabled("bomb_pickup")
            ):
                shaping += float(rs.bomb_pickup)

            # Death bookkeeping.
            if bool(pre["alive"]) and not a.alive:
                team_deaths_this_tick[a.team].append(aid)
                mem = self._memory[name]
                mem.last_death_tick = int(self._engine.state.tick)
                # Pointless death: died without dealing any damage in the round.
                if float(pre["damage_done_round"]) <= 0.0 and self._feature_enabled("pointless_death"):
                    shaping += float(rs.pointless_death)
            # Kill bookkeeping (alive agents that increased kill count).
            if a.alive:
                pass

            rewards[name] = float(rewards[name]) + float(shaping) * factor

        # Useful-trade bonus: a kill by team X within TRADE_WINDOW seconds of
        # a teammate's death.
        trade_window_ticks = int(round(_TRADE_WINDOW_SECONDS * float(self._cfg.simulation.tick_rate_hz)))
        cur_tick = int(self._engine.state.tick)
        for a in self._engine.state.agents:
            name = agent_name(int(a.agent_id))
            mem = self._memory[name]
            pre = pre_by_id.get(int(a.agent_id))
            if pre is None or not a.alive:
                continue
            # Heuristic for "did I just kill someone this tick?":
            #   engine emits +1 per kill in the same dict it uses for outcome
            #   rewards, but it also emits +1 for round-win to *every* alive
            #   teammate -- so we additionally require this agent to have
            #   dealt positive damage this tick.
            this_step_engine_reward = float(engine_rewards.get(int(a.agent_id), 0.0))
            dmg_dealt_this_step = float(a.damage_done_round) - float(pre["damage_done_round"])
            if round_changed:
                dmg_dealt_this_step = float(a.damage_done_round)
            killed_this_tick = this_step_engine_reward > 0.0 and dmg_dealt_this_step > 0.0
            if killed_this_tick:
                mem.last_kill_tick = cur_tick
                team_kills_this_tick[a.team].append(int(a.agent_id))
                # Look for a same-team death within the trade window.
                recent_team_death = False
                for teammate in self._engine.state.agents:
                    if teammate.team != a.team or teammate.agent_id == a.agent_id:
                        continue
                    tmem = self._memory[agent_name(int(teammate.agent_id))]
                    if cur_tick - int(tmem.last_death_tick) <= trade_window_ticks:
                        recent_team_death = True
                        break
                if recent_team_death and self._feature_enabled("useful_trade"):
                    rewards[name] = float(rewards[name]) + float(rs.useful_trade) * factor

        # Plant / defuse bonuses: detect transitions to PLANTED / DEFUSED.
        bomb_phase = self._engine.state.bomb.phase
        if (
            bomb_phase == BombPhase.PLANTED
            and self._round_id_at_last_step == int(self._engine.state.round_id)
            and self._feature_enabled("bomb_plant")
        ):
            # All attackers get a slice (rewarded once per phase change is hard to
            # detect cheaply; we let the engine's outcome reward handle the bulk
            # of the signal and use this as a smaller dense bonus per tick the
            # bomb stays planted).
            for a in self._engine.state.agents:
                if a.side == Side.ATTACKER and a.alive:
                    rewards[agent_name(int(a.agent_id))] += float(rs.successful_plant) * factor * 0.05
        elif bomb_phase == BombPhase.DEFUSED and self._feature_enabled("bomb_defuse"):
            for a in self._engine.state.agents:
                if a.side == Side.DEFENDER and a.alive:
                    rewards[agent_name(int(a.agent_id))] += float(rs.successful_defuse) * factor * 0.5

        return rewards


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_move() -> np.ndarray:
    return np.zeros(2, dtype=np.float32)


def _coerce_mixed_action(
    raw: Any,
) -> tuple[np.ndarray, int, int, int, int]:
    """Coerce a wide range of input formats into a mixed action tuple.

    Returns ``(move_vec[2 float32], micro:int, comm:int, buy:int, aim:int)``.
    Accepted encodings:

    * Dict with keys ``"move"`` and ``"discrete"`` (the v0.4 canonical format).
    * Dict with explicit named keys ``move``/``move_vec``/``move_x``/``move_y``,
      ``micro``, ``comm``, ``buy``, ``aim_target``.
    * Flat numpy array / list / tuple of length >= 6 laid out as
      ``[move_x, move_y, micro, comm, buy, aim]`` (legacy fixture friendly).
    * Length-5 array ``[move_index, micro, comm, buy, aim]`` is interpreted as
      v0.3-style discrete movement via the :data:`MOVE_VECTORS` table.
    * A single int treated as a legacy :class:`MoveIntent` (test convenience).
    """
    if isinstance(raw, dict):
        # Canonical v0.4 dict.
        if "move" in raw and "discrete" in raw:
            mv = np.asarray(raw["move"], dtype=np.float32).reshape(-1)
            mv = _zero_move() if mv.shape[0] < 2 else mv[:2].copy()
            disc = np.asarray(raw["discrete"], dtype=np.int64).reshape(-1)
            if disc.shape[0] < 4:
                padded = np.zeros(4, dtype=np.int64)
                padded[: disc.shape[0]] = disc
                disc = padded
            return mv, int(disc[0]), int(disc[1]), int(disc[2]), int(disc[3])
        # Field-named dict.
        if "move_vec" in raw or "move_x" in raw or "move_y" in raw:
            if "move_vec" in raw:
                mv = np.asarray(raw["move_vec"], dtype=np.float32).reshape(-1)
                mv = _zero_move() if mv.shape[0] < 2 else mv[:2].copy()
            else:
                mv = np.array(
                    [float(raw.get("move_x", 0.0)), float(raw.get("move_y", 0.0))],
                    dtype=np.float32,
                )
            return (
                mv,
                int(raw.get("micro", 0)),
                int(raw.get("comm", 0)),
                int(raw.get("buy", 0)),
                int(raw.get("aim_target", 0)),
            )
        if "move" in raw:
            from kivski_sim.types import MOVE_VECTORS, MoveIntent

            try:
                intent = MoveIntent(int(raw["move"]))
            except (ValueError, TypeError):
                intent = MoveIntent.HOLD
            dx, dy = MOVE_VECTORS[intent]
            return (
                np.array([dx, dy], dtype=np.float32),
                int(raw.get("micro", 0)),
                int(raw.get("comm", 0)),
                int(raw.get("buy", 0)),
                int(raw.get("aim_target", 0)),
            )
        # Empty dict -> HOLD.
        return _zero_move(), 0, 0, 0, 0
    if isinstance(raw, (int, np.integer)):
        from kivski_sim.types import MOVE_VECTORS, MoveIntent

        try:
            intent = MoveIntent(int(raw))
        except (ValueError, TypeError):
            intent = MoveIntent.HOLD
        dx, dy = MOVE_VECTORS[intent]
        return np.array([dx, dy], dtype=np.float32), 0, 0, 0, 0
    if isinstance(raw, (np.ndarray, list, tuple)):
        arr = np.asarray(raw).reshape(-1)
        if arr.shape[0] >= 6:
            mv = arr[:2].astype(np.float32, copy=False)
            return (
                np.array([float(mv[0]), float(mv[1])], dtype=np.float32),
                int(arr[2]),
                int(arr[3]),
                int(arr[4]),
                int(arr[5]),
            )
        # Length-5 legacy ([move_idx, micro, comm, buy, aim]).
        if arr.shape[0] == 5:
            from kivski_sim.types import MOVE_VECTORS, MoveIntent

            try:
                intent = MoveIntent(int(arr[0]))
            except (ValueError, TypeError):
                intent = MoveIntent.HOLD
            dx, dy = MOVE_VECTORS[intent]
            return (
                np.array([dx, dy], dtype=np.float32),
                int(arr[1]),
                int(arr[2]),
                int(arr[3]),
                int(arr[4]),
            )
        if arr.shape[0] == 2:
            mv = arr.astype(np.float32, copy=False)
            return np.array([float(mv[0]), float(mv[1])], dtype=np.float32), 0, 0, 0, 0
        # Pad-short -> HOLD.
        return _zero_move(), 0, 0, 0, 0
    raise TypeError(f"Unsupported action type: {type(raw)!r}")


# Legacy alias retained for any third-party caller that imported it; the
# coercion is now the mixed-action variant above.
_coerce_action_array = _coerce_mixed_action
