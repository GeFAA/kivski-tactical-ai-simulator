"""Synchronous vectorised wrapper around N :class:`KivskiParallelEnv` envs.

The wrapper batches observations / rewards across the N envs into NumPy
arrays of shape ``[num_envs, ...]``. There is no subprocess parallelism in
V1: the simulator is already numpy-heavy and most contention comes from
the engine's pure-Python control flow, so adding ``multiprocessing`` would
add real overhead without much throughput win on the CPU-bound critical
path. The wrapper is intentionally tiny and predictable.

Auto-reset semantics: when every agent in env ``i`` reports
``terminations[a] == True`` (i.e. the match has ended), we call
``envs[i].reset(seed=...)`` and store the freshly reset observations as
the new starting obs for that env. The pre-reset terminal info is
surfaced via ``infos[i]["episode_done"] = True`` along with an aggregated
:class:`EpisodeStats` record under ``infos[i]["episode_stats"]`` so the
rollout collector can record end-of-episode metrics without re-walking
the engine state.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from kivski_sim.config import KivskiConfig
from kivski_sim.engine import Snapshot
from kivski_sim.env import KivskiParallelEnv, agent_index, agent_name
from kivski_sim.map_loader import MapData, load_map
from kivski_sim.types import MatchOutcome, RoundOutcome, Team
from kivski_sim.utils import now_unix

from kivski_agents.metrics import EpisodeStats

__all__ = ["VecEnvWrapper", "VecEnvStep"]


# ---------------------------------------------------------------------------
# Per-env episode accumulator
# ---------------------------------------------------------------------------


@dataclass
class _EpisodeAccumulator:
    """Running totals for one in-flight episode (cleared on reset)."""

    episode: int = 0
    total_rewards_yellow: float = 0.0
    total_rewards_blue: float = 0.0
    last_alive: dict[int, bool] | None = None

    def reset(self, episode: int, alive: dict[int, bool]) -> None:
        self.episode = int(episode)
        self.total_rewards_yellow = 0.0
        self.total_rewards_blue = 0.0
        self.last_alive = dict(alive)

    def add_step_rewards(self, env: KivskiParallelEnv, rewards: dict[str, float]) -> None:
        for name, reward in rewards.items():
            try:
                aid = agent_index(name)
            except (ValueError, IndexError):
                continue
            if aid < 0 or aid >= len(env.engine.state.agents):
                continue
            team = env.engine.state.agents[aid].team
            if team == Team.YELLOW:
                self.total_rewards_yellow += float(reward)
            else:
                self.total_rewards_blue += float(reward)


@dataclass
class VecEnvStep:
    """Return payload of :meth:`VecEnvWrapper.step` / :meth:`reset`.

    ``observations`` / ``rewards`` / ``terminations`` / ``truncations`` are
    dicts keyed by agent name; each value is a length-``num_envs`` numpy
    array (one row per env). ``infos`` is a list of per-env info dicts.
    """

    observations: dict[str, np.ndarray]
    rewards: dict[str, np.ndarray]
    terminations: dict[str, np.ndarray]
    truncations: dict[str, np.ndarray]
    infos: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Vectorised wrapper
# ---------------------------------------------------------------------------


class VecEnvWrapper:
    """Synchronous vector wrapper around ``num_envs`` :class:`KivskiParallelEnv`.

    Args:
        num_envs: How many parallel envs to drive each step.
        cfg: Configuration shared by every env. Side-switch should already
            be disabled (set to a large value) by the trainer so the
            "training side" assignment stays stable across the match.
        map_name: Map name to load (``"dustline"`` by default).
        base_seed: Each env gets ``base_seed + env_idx`` so the rollouts
            are independent but reproducible.

    Notes on auto-reset: when a match finishes for env ``i`` we re-seed
    with a freshly derived value so the next match doesn't replay the
    same scenario; the derivation is deterministic given the original
    ``base_seed`` and the per-env episode counter.
    """

    def __init__(
        self,
        num_envs: int,
        cfg: KivskiConfig,
        map_name: str,
        base_seed: int,
        *,
        map_data: MapData | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.num_envs: int = int(num_envs)
        self.cfg: KivskiConfig = cfg
        self.map_name: str = str(map_name)
        # Share a single MapData across envs: ``MapData`` is read-only.
        self.map_data: MapData = map_data if map_data is not None else load_map(self.map_name)
        self._base_seed: int = int(base_seed)
        # Per-env counter used to derive deterministic reseed values.
        self._episode_counter: list[int] = [0 for _ in range(self.num_envs)]
        self.envs: list[KivskiParallelEnv] = [
            KivskiParallelEnv(
                config=cfg,
                map_name=self.map_name,
                seed=self._base_seed + i,
                map_data=self.map_data,
            )
            for i in range(self.num_envs)
        ]
        self.agent_names: list[str] = list(self.envs[0].possible_agents)
        self.obs_dim: int = int(self.envs[0].observation_dim)
        self.team_size: int = int(cfg.simulation.team_size)
        # Read action heads from the env's space so changes in the env stay in sync.
        nvec = np.asarray(self.envs[0].action_space(self.agent_names[0]).nvec, dtype=np.int64)
        self.n_heads: int = int(nvec.shape[0])
        self.action_dims: np.ndarray = nvec
        # Per-env accumulators for episode_stats.
        self._acc: list[_EpisodeAccumulator] = [_EpisodeAccumulator(episode=0) for _ in range(self.num_envs)]
        # Initial observation buffer for the very first reset.
        self._current_obs: dict[str, np.ndarray] | None = None
        self._current_infos: list[dict[str, Any]] | None = None
        # Pre-allocate scratch arrays used by step().
        self._rewards_buf: dict[str, np.ndarray] = {
            name: np.zeros(self.num_envs, dtype=np.float32) for name in self.agent_names
        }
        self._term_buf: dict[str, np.ndarray] = {
            name: np.zeros(self.num_envs, dtype=np.bool_) for name in self.agent_names
        }
        self._trunc_buf: dict[str, np.ndarray] = {
            name: np.zeros(self.num_envs, dtype=np.bool_) for name in self.agent_names
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> VecEnvStep:
        """Reset every env in the batch."""
        obs_batch: dict[str, np.ndarray] = {
            name: np.zeros((self.num_envs, self.obs_dim), dtype=np.float32) for name in self.agent_names
        }
        infos: list[dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            seed = self._derive_seed(i, self._episode_counter[i])
            obs, info = env.reset(seed=seed)
            for name, vec in obs.items():
                obs_batch[name][i] = np.asarray(vec, dtype=np.float32)
            self._acc[i].reset(
                episode=int(self._episode_counter[i]),
                alive=self._alive_map(env),
            )
            infos.append({"per_agent": info, "episode_done": False})
        # Rewards / terminations zeroed on reset.
        rewards = {name: np.zeros(self.num_envs, dtype=np.float32) for name in self.agent_names}
        terms = {name: np.zeros(self.num_envs, dtype=np.bool_) for name in self.agent_names}
        truncs = {name: np.zeros(self.num_envs, dtype=np.bool_) for name in self.agent_names}
        self._current_obs = obs_batch
        self._current_infos = infos
        return VecEnvStep(obs_batch, rewards, terms, truncs, infos)

    # ------------------------------------------------------------------

    def step(
        self,
        actions: dict[str, np.ndarray],
        comm_payloads: dict[str, np.ndarray] | None = None,
    ) -> VecEnvStep:
        """Advance every env by one tick.

        Args:
            actions: ``{agent_name: int64[num_envs, n_heads]}``.
            comm_payloads: Optional ``{agent_name: float32[num_envs, value_dim]}``.

        Returns:
            A :class:`VecEnvStep` with the batched results. When a single env
            finishes its match, it is auto-reset; the returned ``observations``
            for that env contain the *fresh* starting obs (and the
            corresponding ``infos[i]["episode_done"] == True`` along with an
            ``EpisodeStats`` record under ``"episode_stats"``).
        """
        if self._current_obs is None:
            raise RuntimeError("VecEnvWrapper.step called before reset()")

        # Allocate output batches.
        obs_batch: dict[str, np.ndarray] = {
            name: np.zeros((self.num_envs, self.obs_dim), dtype=np.float32) for name in self.agent_names
        }
        infos: list[dict[str, Any]] = []

        for name in self.agent_names:
            self._rewards_buf[name].fill(0.0)
            self._term_buf[name].fill(False)
            self._trunc_buf[name].fill(False)

        for i, env in enumerate(self.envs):
            # Pull this env's slice from the batched dicts.
            env_actions: dict[str, np.ndarray] = {}
            env_payloads: dict[str, np.ndarray] | None = {} if comm_payloads is not None else None
            for name in self.agent_names:
                if name in actions:
                    env_actions[name] = np.asarray(actions[name][i], dtype=np.int64)
                else:
                    env_actions[name] = np.zeros(self.n_heads, dtype=np.int64)
                if comm_payloads is not None and name in comm_payloads:
                    env_payloads[name] = np.asarray(  # type: ignore[index]
                        comm_payloads[name][i], dtype=np.float32
                    )

            # Step the env (use the comm-aware variant if payloads were given).
            if comm_payloads is not None and env_payloads:
                obs, rewards, terms, truncs, env_info = env.step_with_comms(
                    env_actions, comm_payloads=env_payloads
                )
            else:
                obs, rewards, terms, truncs, env_info = env.step(env_actions)

            # Tally episode-level reward totals before the (possible) reset.
            self._acc[i].add_step_rewards(env, rewards)

            for name in self.agent_names:
                self._rewards_buf[name][i] = float(rewards.get(name, 0.0))
                self._term_buf[name][i] = bool(terms.get(name, False))
                self._trunc_buf[name][i] = bool(truncs.get(name, False))

            episode_done = all(self._term_buf[name][i] for name in self.agent_names) or all(
                self._trunc_buf[name][i] for name in self.agent_names
            )

            if episode_done:
                # Capture the terminal stats from the *current* engine state,
                # then auto-reset the env.
                stats = self._build_episode_stats(env, i)
                self._episode_counter[i] += 1
                reset_seed = self._derive_seed(i, self._episode_counter[i])
                new_obs, new_info = env.reset(seed=reset_seed)
                self._acc[i].reset(
                    episode=int(self._episode_counter[i]),
                    alive=self._alive_map(env),
                )
                for name, vec in new_obs.items():
                    obs_batch[name][i] = np.asarray(vec, dtype=np.float32)
                infos.append(
                    {
                        "per_agent": new_info,
                        "episode_done": True,
                        "episode_stats": stats,
                    }
                )
            else:
                for name, vec in obs.items():
                    obs_batch[name][i] = np.asarray(vec, dtype=np.float32)
                infos.append({"per_agent": env_info, "episode_done": False})

        self._current_obs = obs_batch
        self._current_infos = infos
        return VecEnvStep(
            observations=obs_batch,
            rewards={name: self._rewards_buf[name].copy() for name in self.agent_names},
            terminations={name: self._term_buf[name].copy() for name in self.agent_names},
            truncations={name: self._trunc_buf[name].copy() for name in self.agent_names},
            infos=infos,
        )

    # ------------------------------------------------------------------

    def render(self, env_idx: int = 0) -> Snapshot:
        """Return the engine snapshot for ``env_idx`` (mostly for debugging)."""
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        return self.envs[env_idx].render()

    def current_step(self) -> VecEnvStep | None:
        """Return the last step's batched payload (or ``None`` if never stepped).

        Useful when a fresh consumer (e.g. a new :class:`RolloutCollector`) wants
        to pick up where the previous one left off without re-resetting the envs.
        Rewards / terminations are zeroed because they reflect the *last*
        transition, not the in-flight state.
        """
        if self._current_obs is None or self._current_infos is None:
            return None
        rewards = {name: np.zeros(self.num_envs, dtype=np.float32) for name in self.agent_names}
        terms = {name: np.zeros(self.num_envs, dtype=np.bool_) for name in self.agent_names}
        truncs = {name: np.zeros(self.num_envs, dtype=np.bool_) for name in self.agent_names}
        return VecEnvStep(
            observations=self._current_obs,
            rewards=rewards,
            terminations=terms,
            truncations=truncs,
            infos=self._current_infos,
        )

    def close(self) -> None:
        """Release every wrapped env."""
        for env in self.envs:
            # The engine has no I/O to clean up; we just want to be
            # tolerant of partial state during shutdown.
            with contextlib.suppress(Exception):
                env.close()

    # ------------------------------------------------------------------
    # Reward curriculum (broadcasts to every wrapped env)
    # ------------------------------------------------------------------

    def set_curriculum_stage(self, stage_name: str, features: list[str] | None) -> None:
        """Forward a reward-curriculum stage flip to every wrapped env."""
        for env in self.envs:
            with contextlib.suppress(Exception):
                env.set_curriculum_stage(stage_name, features)

    # ------------------------------------------------------------------
    # Side / team helpers
    # ------------------------------------------------------------------

    def agent_team_indices(self, env_idx: int = 0) -> dict[str, int]:
        """Return ``{agent_name: int(Team)}`` for the env at ``env_idx``.

        Useful for the rollout collector to split actions by training-side
        vs opponent-side. Teams are stable across the whole episode (we
        disable side-switching during training).
        """
        env = self.envs[env_idx]
        return {agent_name(int(a.agent_id)): int(a.team) for a in env.engine.state.agents}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _derive_seed(self, env_idx: int, episode: int) -> int:
        """Deterministically derive a new seed for ``env_idx`` at ``episode``."""
        # Combine the base seed with the env and episode index in a way
        # that is stable across runs and avoids collisions on the first
        # few episodes (where ``episode * num_envs`` could repeat).
        return (
            int(
                (int(self._base_seed) & 0x7FFF_FFFF)
                ^ ((int(env_idx) + 1) * 0x9E37_79B9)
                ^ ((int(episode) + 1) * 0x85EB_CA77)
            )
            & 0x7FFF_FFFF
        )

    @staticmethod
    def _alive_map(env: KivskiParallelEnv) -> dict[int, bool]:
        return {int(a.agent_id): bool(a.alive) for a in env.engine.state.agents}

    def _build_episode_stats(self, env: KivskiParallelEnv, env_idx: int) -> EpisodeStats:
        """Aggregate a terminal :class:`EpisodeStats` payload for env ``env_idx``."""
        state = env.engine.state
        outcome = state.match_outcome
        if outcome == MatchOutcome.YELLOW_WIN:
            winner = "yellow"
        elif outcome == MatchOutcome.BLUE_WIN:
            winner = "blue"
        else:
            winner = "draw"
        yellow_score = int(state.teams[Team.YELLOW].score)
        blue_score = int(state.teams[Team.BLUE].score)
        summaries = list(state.round_summaries)
        total_rounds = len(summaries)
        avg_duration = (
            float(sum(int(s.duration_ticks) for s in summaries)) / float(max(total_rounds, 1))
            if total_rounds > 0
            else 0.0
        )
        total_kills = int(sum(s.survivors_yellow + s.survivors_blue for s in summaries))
        # "Deaths" approximation: every match starts with ``team_size`` alive on
        # each side and ends with ``survivors_*`` alive in the last summary.
        total_deaths = 0
        for s in summaries:
            total_deaths += int(2 * self.team_size - s.survivors_yellow - s.survivors_blue)
        bombs_planted = int(sum(1 for s in summaries if bool(s.bomb_planted)))
        bombs_defused = int(
            sum(1 for s in summaries if RoundOutcome(int(s.outcome)) == RoundOutcome.BOMB_DEFUSED)
        )
        bombs_detonated = int(
            sum(1 for s in summaries if RoundOutcome(int(s.outcome)) == RoundOutcome.BOMB_DETONATED)
        )
        acc = self._acc[env_idx]
        return EpisodeStats(
            episode=int(acc.episode),
            match_done=True,
            yellow_score=yellow_score,
            blue_score=blue_score,
            winner=winner,
            total_rounds=int(total_rounds),
            avg_round_duration_ticks=float(avg_duration),
            total_kills=int(total_kills),
            total_deaths=int(total_deaths),
            bombs_planted=int(bombs_planted),
            bombs_defused=int(bombs_defused),
            bombs_detonated=int(bombs_detonated),
            total_rewards_yellow=float(acc.total_rewards_yellow),
            total_rewards_blue=float(acc.total_rewards_blue),
            timestamp=float(now_unix()),
        )
