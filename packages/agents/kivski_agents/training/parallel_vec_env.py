"""Multi-process / threaded vectorised env wrappers.

This module sits next to :mod:`kivski_agents.training.vec_env`. The
synchronous :class:`VecEnvWrapper` there is great for unit tests and tiny
smoke runs, but the Python GIL plus the engine's pure-Python control flow
caps its throughput at "a handful of envs per core".

The wrappers here speed up rollout collection by spreading the env work
across either (a) OS subprocesses (real CPU parallelism, expensive on
Windows ``spawn`` startup but free runtime cost) or (b) Python threads
(no startup cost, releases the GIL whenever the engine drops into numpy
hot paths so we still get ~30-50% speedup on CPU-bound numerics).

Both wrappers re-export the exact API surface of
:class:`VecEnvWrapper` so the rollout collector and trainer don't need
to know which backend is in use. The trainer picks one via the
:func:`make_vec_env` factory.

Design notes:

* The "anchor env" (a single env hosted in the parent process) is kept
  alive purely so the league / trainer can probe ``envs[0]`` for the
  starting :class:`AgentState` (team partition, action heads, etc).
  It is *never stepped* once the workers take over -- it just mirrors
  the static metadata.
* Worker processes are started via ``torch.multiprocessing.get_context("spawn")``
  to stay Windows-compatible and to dodge the "fork + torch" landmine
  on Linux where forked workers can deadlock on autograd's internal
  locks.
* All inter-process traffic uses :class:`Pipe` (duplex). Each tick we
  send a tuple ``(cmd, payload)`` and read back ``(payload,)``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from kivski_sim.config import KivskiConfig
from kivski_sim.engine import Snapshot
from kivski_sim.env import KivskiParallelEnv, agent_name
from kivski_sim.map_loader import MapData, load_map
from kivski_sim.types import MatchOutcome, RoundOutcome, Team
from kivski_sim.utils import now_unix

from kivski_agents.metrics import EpisodeStats
from kivski_agents.training.auto_tune import envs_per_worker_split
from kivski_agents.training.vec_env import VecEnvStep, VecEnvWrapper

__all__ = [
    "ThreadedVecEnv",
    "SubprocVecEnv",
    "make_vec_env",
]


_LOG = logging.getLogger("kivski.training.parallel_vec_env")


# ---------------------------------------------------------------------------
# Per-env episode accumulator (mirrors the sync wrapper's bookkeeping)
# ---------------------------------------------------------------------------


@dataclass
class _EpisodeAccumulator:
    """Running totals for one in-flight episode (cleared on reset)."""

    episode: int = 0
    total_rewards_yellow: float = 0.0
    total_rewards_blue: float = 0.0

    def reset(self, episode: int) -> None:
        self.episode = int(episode)
        self.total_rewards_yellow = 0.0
        self.total_rewards_blue = 0.0


# ---------------------------------------------------------------------------
# Threaded backend (cheap to start, modest speedup)
# ---------------------------------------------------------------------------


class ThreadedVecEnv:
    """Drop-in for :class:`VecEnvWrapper` that steps envs in parallel threads.

    The engine is mostly numpy-heavy, so the GIL is dropped during the long
    portions of ``engine.step``. With ``num_workers > 1`` we usually see
    +30-50% throughput on commodity machines without paying the cost of
    process startup.

    The API is byte-for-byte identical to :class:`VecEnvWrapper`. All
    state lives in the main process so debugging / introspection works
    the same way.
    """

    def __init__(
        self,
        num_envs: int,
        cfg: KivskiConfig,
        map_name: str,
        base_seed: int,
        *,
        map_data: MapData | None = None,
        num_workers: int | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.num_envs: int = int(num_envs)
        self.cfg: KivskiConfig = cfg
        self.map_name: str = str(map_name)
        self.map_data: MapData = map_data if map_data is not None else load_map(self.map_name)
        self._base_seed: int = int(base_seed)
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
        nvec = np.asarray(self.envs[0].action_space(self.agent_names[0]).nvec, dtype=np.int64)
        self.n_heads: int = int(nvec.shape[0])
        self.action_dims: np.ndarray = nvec
        self._acc: list[_EpisodeAccumulator] = [_EpisodeAccumulator() for _ in range(self.num_envs)]
        self._current_obs: dict[str, np.ndarray] | None = None
        self._current_infos: list[dict[str, Any]] | None = None
        self._rewards_buf: dict[str, np.ndarray] = {
            name: np.zeros(self.num_envs, dtype=np.float32) for name in self.agent_names
        }
        self._term_buf: dict[str, np.ndarray] = {
            name: np.zeros(self.num_envs, dtype=np.bool_) for name in self.agent_names
        }
        self._trunc_buf: dict[str, np.ndarray] = {
            name: np.zeros(self.num_envs, dtype=np.bool_) for name in self.agent_names
        }

        cpu = int(os.cpu_count() or 4)
        chosen = num_workers if (num_workers is not None and num_workers > 0) else min(self.num_envs, cpu)
        self.num_workers: int = max(1, min(self.num_envs, int(chosen)))
        self._executor: ThreadPoolExecutor | None = None
        if self.num_workers > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="kivski-vecenv",
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> VecEnvStep:
        obs_batch: dict[str, np.ndarray] = {
            name: np.zeros((self.num_envs, self.obs_dim), dtype=np.float32) for name in self.agent_names
        }
        seeds = [self._derive_seed(i, self._episode_counter[i]) for i in range(self.num_envs)]

        def _reset_one(i: int) -> tuple[int, dict[str, np.ndarray], dict[str, dict[str, Any]]]:
            obs, info = self.envs[i].reset(seed=seeds[i])
            return i, obs, info

        results = self._map(_reset_one, range(self.num_envs))
        infos: list[dict[str, Any]] = [None] * self.num_envs  # type: ignore[list-item]
        for i, obs, info in results:
            for name, vec in obs.items():
                obs_batch[name][i] = np.asarray(vec, dtype=np.float32)
            self._acc[i].reset(episode=int(self._episode_counter[i]))
            infos[i] = {"per_agent": info, "episode_done": False}
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
        if self._current_obs is None:
            raise RuntimeError("ThreadedVecEnv.step called before reset()")

        obs_batch: dict[str, np.ndarray] = {
            name: np.zeros((self.num_envs, self.obs_dim), dtype=np.float32) for name in self.agent_names
        }

        for name in self.agent_names:
            self._rewards_buf[name].fill(0.0)
            self._term_buf[name].fill(False)
            self._trunc_buf[name].fill(False)

        # Pre-slice the per-env action/payload dicts in the main thread so the
        # worker side only sees pure numpy arrays it can hand to the engine.
        per_env_actions: list[dict[str, np.ndarray]] = []
        per_env_payloads: list[dict[str, np.ndarray] | None] = []
        for i in range(self.num_envs):
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
            per_env_actions.append(env_actions)
            per_env_payloads.append(env_payloads)

        def _step_one(
            i: int,
        ) -> tuple[
            int,
            dict[str, np.ndarray],
            dict[str, float],
            dict[str, bool],
            dict[str, bool],
            dict[str, dict[str, Any]],
        ]:
            env = self.envs[i]
            env_actions = per_env_actions[i]
            env_payloads = per_env_payloads[i]
            if env_payloads:
                obs, rewards, terms, truncs, info = env.step_with_comms(
                    env_actions, comm_payloads=env_payloads
                )
            else:
                obs, rewards, terms, truncs, info = env.step(env_actions)
            return i, obs, rewards, terms, truncs, info

        results = self._map(_step_one, range(self.num_envs))
        infos: list[dict[str, Any]] = [None] * self.num_envs  # type: ignore[list-item]
        # Determine which envs finished, then reset them serially (engine reset
        # touches RNG state and is cheap enough to keep on the main thread).
        for i, obs, rewards, terms, truncs, info in results:
            env = self.envs[i]
            # Tally per-team rewards before potential reset.
            for name, r in rewards.items():
                try:
                    aid = int(name.split("_", 1)[1])
                except (ValueError, IndexError):
                    continue
                if 0 <= aid < len(env.engine.state.agents):
                    if env.engine.state.agents[aid].team == Team.YELLOW:
                        self._acc[i].total_rewards_yellow += float(r)
                    else:
                        self._acc[i].total_rewards_blue += float(r)

            for name in self.agent_names:
                self._rewards_buf[name][i] = float(rewards.get(name, 0.0))
                self._term_buf[name][i] = bool(terms.get(name, False))
                self._trunc_buf[name][i] = bool(truncs.get(name, False))

            episode_done = all(self._term_buf[name][i] for name in self.agent_names) or all(
                self._trunc_buf[name][i] for name in self.agent_names
            )
            if episode_done:
                stats = self._build_episode_stats(env, i)
                self._episode_counter[i] += 1
                reset_seed = self._derive_seed(i, self._episode_counter[i])
                new_obs, new_info = env.reset(seed=reset_seed)
                self._acc[i].reset(episode=int(self._episode_counter[i]))
                for name, vec in new_obs.items():
                    obs_batch[name][i] = np.asarray(vec, dtype=np.float32)
                infos[i] = {
                    "per_agent": new_info,
                    "episode_done": True,
                    "episode_stats": stats,
                }
            else:
                for name, vec in obs.items():
                    obs_batch[name][i] = np.asarray(vec, dtype=np.float32)
                infos[i] = {"per_agent": info, "episode_done": False}

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
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        return self.envs[env_idx].render()

    def current_step(self) -> VecEnvStep | None:
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
        if self._executor is not None:
            with contextlib.suppress(Exception):
                self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        for env in self.envs:
            with contextlib.suppress(Exception):
                env.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        with contextlib.suppress(Exception):
            self.close()

    # ------------------------------------------------------------------
    # Reward curriculum (broadcasts to every wrapped env)
    # ------------------------------------------------------------------

    def set_curriculum_stage(
        self, stage_name: str, features: list[str] | None
    ) -> None:
        for env in self.envs:
            with contextlib.suppress(Exception):
                env.set_curriculum_stage(stage_name, features)

    # ------------------------------------------------------------------
    # Side / team helpers
    # ------------------------------------------------------------------

    def agent_team_indices(self, env_idx: int = 0) -> dict[str, int]:
        env = self.envs[env_idx]
        return {agent_name(int(a.agent_id)): int(a.team) for a in env.engine.state.agents}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _map(self, fn, items):  # type: ignore[no-untyped-def]
        """Execute ``fn(item)`` for each item, using the pool if available."""
        if self._executor is None or self.num_workers <= 1:
            return [fn(it) for it in items]
        # ``executor.map`` preserves input order, which is what we want.
        return list(self._executor.map(fn, list(items)))

    def _derive_seed(self, env_idx: int, episode: int) -> int:
        return (
            int(
                (int(self._base_seed) & 0x7FFF_FFFF)
                ^ ((int(env_idx) + 1) * 0x9E37_79B9)
                ^ ((int(episode) + 1) * 0x85EB_CA77)
            )
            & 0x7FFF_FFFF
        )

    def _build_episode_stats(self, env: KivskiParallelEnv, env_idx: int) -> EpisodeStats:
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


# ---------------------------------------------------------------------------
# Subprocess worker: top-level so it can be pickled on Windows ``spawn``.
# ---------------------------------------------------------------------------


def _subproc_worker_loop(
    remote,  # multiprocessing.connection.Connection
    parent_remote,  # the other half of the pipe (closed in the child)
    cfg_dump: dict[str, Any],
    map_name: str,
    map_data_payload: Any,
    seeds: list[int],
) -> None:
    """Worker entry point.

    Hosts ``len(seeds)`` :class:`KivskiParallelEnv` instances and replies to
    commands sent down ``remote``. The command protocol is intentionally
    flat:

    * ``("reset", seeds)`` -> ``("reset_ok", [(obs, info), ...])``
    * ``("step", per_env_payload)`` -> ``("step_ok", results, finished_indices,
                                          [(episode_stats_dict or None), ...])``
    * ``("close", None)`` -> exits cleanly

    The child never touches torch; it only operates on numpy arrays and
    the pure-Python engine, so we don't need to drag torch into the
    worker process startup.
    """
    parent_remote.close()
    # Rebuild config inside the worker (the dict is the only pickle-safe form
    # of the frozen pydantic models we have).
    cfg = KivskiConfig.model_validate(cfg_dump)
    map_data = map_data_payload if map_data_payload is not None else load_map(map_name)
    envs: list[KivskiParallelEnv] = [
        KivskiParallelEnv(config=cfg, map_name=map_name, seed=int(s), map_data=map_data) for s in seeds
    ]
    # First reset to prime each env so subsequent ``step`` calls don't blow up.
    for env, seed in zip(envs, seeds, strict=False):
        env.reset(seed=int(seed))

    try:
        while True:
            try:
                cmd, payload = remote.recv()
            except EOFError:
                break
            if cmd == "reset":
                # payload: list[int] seeds (one per env)
                replies: list[tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]] = []
                for env, s in zip(envs, payload, strict=False):
                    obs, info = env.reset(seed=int(s))
                    replies.append((obs, info))
                remote.send(("reset_ok", replies))
            elif cmd == "step":
                # payload: list[(env_actions, env_payloads)] per env
                step_results: list[
                    tuple[
                        dict[str, np.ndarray],
                        dict[str, float],
                        dict[str, bool],
                        dict[str, bool],
                        dict[str, dict[str, Any]],
                    ]
                ] = []
                for env, (env_actions, env_payloads) in zip(envs, payload, strict=False):
                    if env_payloads:
                        out = env.step_with_comms(env_actions, comm_payloads=env_payloads)
                    else:
                        out = env.step(env_actions)
                    step_results.append(out)
                remote.send(("step_ok", step_results))
            elif cmd == "reset_one":
                # payload: list[(env_index_local, seed)] to reset selectively
                replies = []
                for idx, s in payload:
                    obs, info = envs[int(idx)].reset(seed=int(s))
                    replies.append((int(idx), obs, info))
                remote.send(("reset_one_ok", replies))
            elif cmd == "team_indices":
                # Used once during construction to surface team partitioning.
                env = envs[0]
                team = {agent_name(int(a.agent_id)): int(a.team) for a in env.engine.state.agents}
                remote.send(("team_indices_ok", team))
            elif cmd == "render":
                idx = int(payload)
                snap = envs[idx].render()
                remote.send(("render_ok", snap))
            elif cmd == "match_summary":
                # Used by the parent to build an EpisodeStats payload for
                # an env that just finished. Payload: env index in this worker.
                idx = int(payload)
                env = envs[idx]
                state = env.engine.state
                summaries = [
                    {
                        "outcome": int(s.outcome),
                        "survivors_yellow": int(s.survivors_yellow),
                        "survivors_blue": int(s.survivors_blue),
                        "duration_ticks": int(s.duration_ticks),
                        "bomb_planted": bool(s.bomb_planted),
                    }
                    for s in state.round_summaries
                ]
                summary = {
                    "match_outcome": int(state.match_outcome),
                    "yellow_score": int(state.teams[Team.YELLOW].score),
                    "blue_score": int(state.teams[Team.BLUE].score),
                    "summaries": summaries,
                }
                remote.send(("match_summary_ok", summary))
            elif cmd == "set_curriculum_stage":
                stage_name, features = payload
                for env in envs:
                    try:
                        env.set_curriculum_stage(stage_name, features)
                    except Exception:  # noqa: BLE001 - never let one env kill the loop
                        pass
                remote.send(("set_curriculum_stage_ok", True))
            elif cmd == "close":
                break
            else:  # pragma: no cover - protocol bug
                remote.send(("error", f"unknown cmd: {cmd!r}"))
    finally:
        for e in envs:
            with contextlib.suppress(Exception):
                e.close()
        with contextlib.suppress(Exception):
            remote.close()


# ---------------------------------------------------------------------------
# Subprocess vec env
# ---------------------------------------------------------------------------


class SubprocVecEnv:
    """Multi-process vectorised env using ``torch.multiprocessing``.

    Each worker hosts ``envs_per_worker`` :class:`KivskiParallelEnv` instances
    and talks to the parent over a duplex :class:`Pipe`. The wrapper exposes
    the same API as :class:`VecEnvWrapper` so the rest of the training stack
    can switch backends with no code changes.

    Falls back gracefully on startup failure: if any worker fails to come
    up (which happens occasionally on Windows when ``spawn`` cannot pickle
    something), the constructor raises -- the :func:`make_vec_env` factory
    catches the exception and falls back to :class:`VecEnvWrapper`.

    Args:
        num_envs: Total parallel envs.
        cfg: Shared config (frozen pydantic; serialised via ``model_dump``).
        map_name: Map to load (workers re-load the map locally to avoid
            shipping the full MapData over the pipe; the parent also holds
            one anchor instance).
        base_seed: Seeds are derived deterministically per env.
        num_workers: How many worker subprocesses to spawn. Defaults to
            ``min(num_envs, cpu_count() - 1)``.
        map_data: Optional pre-loaded map data for the anchor env. Workers
            always reload locally because :class:`MapData` is not always
            cheap to pickle.
    """

    def __init__(
        self,
        num_envs: int,
        cfg: KivskiConfig,
        map_name: str,
        base_seed: int,
        *,
        map_data: MapData | None = None,
        num_workers: int | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.num_envs: int = int(num_envs)
        self.cfg: KivskiConfig = cfg
        self.map_name: str = str(map_name)
        self.map_data: MapData = map_data if map_data is not None else load_map(self.map_name)
        self._base_seed: int = int(base_seed)
        cpu = int(os.cpu_count() or 4)
        chosen = num_workers if (num_workers is not None and num_workers > 0) else min(self.num_envs, cpu - 1)
        self.num_workers: int = max(1, min(self.num_envs, int(chosen)))
        self._split: list[int] = envs_per_worker_split(self.num_envs, self.num_workers)
        # Per-env counter used to derive deterministic reseed values.
        self._episode_counter: list[int] = [0 for _ in range(self.num_envs)]

        # Anchor env: a single in-process env so league/trainer can read
        # static metadata (team partition, obs dim, action heads). Never
        # stepped from the trainer once workers are up.
        self._anchor_env = KivskiParallelEnv(
            config=cfg,
            map_name=self.map_name,
            seed=self._base_seed,
            map_data=self.map_data,
        )
        # Prime anchor so render()/agent_team_indices work without raising.
        self._anchor_env.reset(seed=self._base_seed)
        self.envs: list[KivskiParallelEnv] = [self._anchor_env]
        self.agent_names: list[str] = list(self._anchor_env.possible_agents)
        self.obs_dim: int = int(self._anchor_env.observation_dim)
        self.team_size: int = int(cfg.simulation.team_size)
        nvec = np.asarray(self._anchor_env.action_space(self.agent_names[0]).nvec, dtype=np.int64)
        self.n_heads: int = int(nvec.shape[0])
        self.action_dims: np.ndarray = nvec
        self._acc: list[_EpisodeAccumulator] = [_EpisodeAccumulator() for _ in range(self.num_envs)]

        self._current_obs: dict[str, np.ndarray] | None = None
        self._current_infos: list[dict[str, Any]] | None = None
        self._rewards_buf: dict[str, np.ndarray] = {
            name: np.zeros(self.num_envs, dtype=np.float32) for name in self.agent_names
        }
        self._term_buf: dict[str, np.ndarray] = {
            name: np.zeros(self.num_envs, dtype=np.bool_) for name in self.agent_names
        }
        self._trunc_buf: dict[str, np.ndarray] = {
            name: np.zeros(self.num_envs, dtype=np.bool_) for name in self.agent_names
        }

        # Spin up workers. Any failure here propagates -- the make_vec_env
        # factory catches it and falls back to the sync wrapper.
        self._spawn_workers(cfg)
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Worker bring-up
    # ------------------------------------------------------------------

    def _spawn_workers(self, cfg: KivskiConfig) -> None:
        # Use torch's multiprocessing wrapper so any torch state respects
        # Windows' spawn semantics. We *don't* ship any torch objects to the
        # workers themselves -- they only touch numpy + the engine.
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        cfg_dump = cfg.model_dump()

        self._processes: list[Any] = []
        self._remotes: list[Any] = []
        # Per-env -> (worker_idx, local_idx) routing.
        self._env_to_worker: list[tuple[int, int]] = []

        cursor = 0
        for w_idx, n_in_worker in enumerate(self._split):
            seeds = [self._derive_seed(cursor + i, 0) for i in range(n_in_worker)]
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target=_subproc_worker_loop,
                args=(child_conn, parent_conn, cfg_dump, self.map_name, None, seeds),
                daemon=True,
            )
            proc.start()
            # Close the child end in the parent -- only the worker needs it.
            child_conn.close()
            self._processes.append(proc)
            self._remotes.append(parent_conn)
            for local_i in range(n_in_worker):
                self._env_to_worker.append((w_idx, local_i))
            cursor += n_in_worker

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> VecEnvStep:
        obs_batch: dict[str, np.ndarray] = {
            name: np.zeros((self.num_envs, self.obs_dim), dtype=np.float32) for name in self.agent_names
        }
        infos: list[dict[str, Any]] = [None] * self.num_envs  # type: ignore[list-item]

        # Build per-worker seed lists in env-index order so the reply order
        # matches what we expect when reassembling the batched obs.
        per_worker_seeds: list[list[int]] = [[] for _ in range(self.num_workers)]
        per_worker_env_indices: list[list[int]] = [[] for _ in range(self.num_workers)]
        for env_i in range(self.num_envs):
            w_idx, _local = self._env_to_worker[env_i]
            seed = self._derive_seed(env_i, self._episode_counter[env_i])
            per_worker_seeds[w_idx].append(seed)
            per_worker_env_indices[w_idx].append(env_i)

        for w_idx, remote in enumerate(self._remotes):
            remote.send(("reset", per_worker_seeds[w_idx]))
        for w_idx, remote in enumerate(self._remotes):
            kind, payload = remote.recv()
            if kind != "reset_ok":
                raise RuntimeError(f"worker reset failed: kind={kind!r} payload={payload!r}")
            for (obs, info), env_i in zip(payload, per_worker_env_indices[w_idx], strict=False):
                for name, vec in obs.items():
                    obs_batch[name][env_i] = np.asarray(vec, dtype=np.float32)
                self._acc[env_i].reset(episode=int(self._episode_counter[env_i]))
                infos[env_i] = {"per_agent": info, "episode_done": False}

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
        if self._current_obs is None:
            raise RuntimeError("SubprocVecEnv.step called before reset()")

        obs_batch: dict[str, np.ndarray] = {
            name: np.zeros((self.num_envs, self.obs_dim), dtype=np.float32) for name in self.agent_names
        }
        for name in self.agent_names:
            self._rewards_buf[name].fill(0.0)
            self._term_buf[name].fill(False)
            self._trunc_buf[name].fill(False)

        # Pre-slice per-env actions/payloads in the parent so the workers
        # don't have to deal with the full batched dict over the wire.
        per_worker_payloads: list[list[tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]]] = [
            [] for _ in range(self.num_workers)
        ]
        per_worker_env_indices: list[list[int]] = [[] for _ in range(self.num_workers)]
        for env_i in range(self.num_envs):
            w_idx, _local = self._env_to_worker[env_i]
            env_actions: dict[str, np.ndarray] = {}
            env_payloads: dict[str, np.ndarray] | None = {} if comm_payloads is not None else None
            for name in self.agent_names:
                if name in actions:
                    env_actions[name] = np.asarray(actions[name][env_i], dtype=np.int64)
                else:
                    env_actions[name] = np.zeros(self.n_heads, dtype=np.int64)
                if comm_payloads is not None and name in comm_payloads:
                    env_payloads[name] = np.asarray(  # type: ignore[index]
                        comm_payloads[name][env_i], dtype=np.float32
                    )
            per_worker_payloads[w_idx].append((env_actions, env_payloads))
            per_worker_env_indices[w_idx].append(env_i)

        for w_idx, remote in enumerate(self._remotes):
            remote.send(("step", per_worker_payloads[w_idx]))

        infos: list[dict[str, Any]] = [None] * self.num_envs  # type: ignore[list-item]
        # Track which envs need a follow-up reset on their respective workers
        # and which need an episode_stats payload.
        finished_per_worker: list[list[tuple[int, int]]] = [[] for _ in range(self.num_workers)]
        finished_env_indices: list[int] = []

        for w_idx, remote in enumerate(self._remotes):
            kind, payload = remote.recv()
            if kind != "step_ok":
                raise RuntimeError(f"worker step failed: kind={kind!r} payload={payload!r}")
            for (obs, rewards, terms, truncs, env_info), env_i in zip(
                payload, per_worker_env_indices[w_idx], strict=False
            ):
                # Tally per-team rewards using the team partition from the
                # anchor env (teams are stable per env-index because every
                # env is constructed with the same config / team_size).
                # We approximate team membership by agent id modulo team_size:
                # in the engine, agents are spawned with YELLOW for ids
                # 0..team_size-1 and BLUE for ids team_size..2*team_size-1.
                # (Side-switching during training is disabled by the trainer.)
                ts = self.team_size
                for name, r in rewards.items():
                    try:
                        aid = int(name.split("_", 1)[1])
                    except (ValueError, IndexError):
                        continue
                    if aid < ts:
                        self._acc[env_i].total_rewards_yellow += float(r)
                    else:
                        self._acc[env_i].total_rewards_blue += float(r)

                for name in self.agent_names:
                    self._rewards_buf[name][env_i] = float(rewards.get(name, 0.0))
                    self._term_buf[name][env_i] = bool(terms.get(name, False))
                    self._trunc_buf[name][env_i] = bool(truncs.get(name, False))

                episode_done = all(self._term_buf[name][env_i] for name in self.agent_names) or all(
                    self._trunc_buf[name][env_i] for name in self.agent_names
                )

                if episode_done:
                    # Ask worker for round summary so we can build EpisodeStats.
                    _local = self._env_to_worker[env_i][1]
                    finished_per_worker[w_idx].append((env_i, int(_local)))
                    finished_env_indices.append(env_i)
                else:
                    for name, vec in obs.items():
                        obs_batch[name][env_i] = np.asarray(vec, dtype=np.float32)
                    infos[env_i] = {"per_agent": env_info, "episode_done": False}

        # For finished envs: pull per-env match summary, then request a reset.
        # Build EpisodeStats objects on parent side from the dict payload.
        ep_stats_by_env: dict[int, EpisodeStats] = {}
        for w_idx, remote in enumerate(self._remotes):
            for env_i, local_idx in finished_per_worker[w_idx]:
                remote.send(("match_summary", local_idx))
                kind, summary = remote.recv()
                if kind != "match_summary_ok":
                    raise RuntimeError(f"worker match_summary failed: kind={kind!r}")
                ep_stats_by_env[env_i] = self._build_episode_stats_from_summary(summary, env_i)

        # Bump episode counter, build reset payloads.
        per_worker_reset_payload: list[list[tuple[int, int]]] = [[] for _ in range(self.num_workers)]
        for w_idx, finished_list in enumerate(finished_per_worker):
            for env_i, local_idx in finished_list:
                self._episode_counter[env_i] += 1
                seed = self._derive_seed(env_i, self._episode_counter[env_i])
                per_worker_reset_payload[w_idx].append((local_idx, seed))
        # Dispatch resets.
        for w_idx, remote in enumerate(self._remotes):
            if per_worker_reset_payload[w_idx]:
                remote.send(("reset_one", per_worker_reset_payload[w_idx]))
        # Collect.
        for w_idx, remote in enumerate(self._remotes):
            if not per_worker_reset_payload[w_idx]:
                continue
            kind, payload = remote.recv()
            if kind != "reset_one_ok":
                raise RuntimeError(f"worker reset_one failed: kind={kind!r}")
            # Map local_idx back to env_i via finished_per_worker ordering.
            finished_for_worker = finished_per_worker[w_idx]
            # Build a quick (local_idx -> env_i) lookup for this worker.
            local_to_env = {int(local_idx): env_i for env_i, local_idx in finished_for_worker}
            for local_idx, obs, info in payload:
                env_i = local_to_env[int(local_idx)]
                self._acc[env_i].reset(episode=int(self._episode_counter[env_i]))
                for name, vec in obs.items():
                    obs_batch[name][env_i] = np.asarray(vec, dtype=np.float32)
                infos[env_i] = {
                    "per_agent": info,
                    "episode_done": True,
                    "episode_stats": ep_stats_by_env.get(env_i),
                }

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
        """Render the *anchor* env -- workers do not maintain a viewable env."""
        del env_idx
        return self._anchor_env.render()

    def current_step(self) -> VecEnvStep | None:
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
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            with contextlib.suppress(Exception):
                remote.send(("close", None))
        for remote in self._remotes:
            with contextlib.suppress(Exception):
                remote.close()
        for proc in self._processes:
            with contextlib.suppress(Exception):
                proc.join(timeout=3.0)
            if proc.is_alive():
                with contextlib.suppress(Exception):
                    proc.terminate()
        with contextlib.suppress(Exception):
            self._anchor_env.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        with contextlib.suppress(Exception):
            self.close()

    # ------------------------------------------------------------------
    # Reward curriculum (broadcasts to every worker + the anchor env)
    # ------------------------------------------------------------------

    def set_curriculum_stage(
        self, stage_name: str, features: list[str] | None
    ) -> None:
        if self._closed:
            return
        # Update the anchor env (used for one-off ``render`` calls).
        with contextlib.suppress(Exception):
            self._anchor_env.set_curriculum_stage(stage_name, features)
        # Broadcast to every worker subprocess.
        for remote in self._remotes:
            with contextlib.suppress(Exception):
                remote.send(("set_curriculum_stage", (stage_name, features)))
        for remote in self._remotes:
            try:
                tag, _ = remote.recv()
                if tag != "set_curriculum_stage_ok":  # pragma: no cover - protocol bug
                    pass
            except Exception:  # noqa: BLE001 - best-effort
                pass

    # ------------------------------------------------------------------
    # Side / team helpers
    # ------------------------------------------------------------------

    def agent_team_indices(self, env_idx: int = 0) -> dict[str, int]:
        del env_idx
        return {agent_name(int(a.agent_id)): int(a.team) for a in self._anchor_env.engine.state.agents}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _derive_seed(self, env_idx: int, episode: int) -> int:
        return (
            int(
                (int(self._base_seed) & 0x7FFF_FFFF)
                ^ ((int(env_idx) + 1) * 0x9E37_79B9)
                ^ ((int(episode) + 1) * 0x85EB_CA77)
            )
            & 0x7FFF_FFFF
        )

    def _build_episode_stats_from_summary(self, summary: dict[str, Any], env_idx: int) -> EpisodeStats:
        outcome = int(summary.get("match_outcome", 0))
        if outcome == int(MatchOutcome.YELLOW_WIN):
            winner = "yellow"
        elif outcome == int(MatchOutcome.BLUE_WIN):
            winner = "blue"
        else:
            winner = "draw"
        yellow_score = int(summary.get("yellow_score", 0))
        blue_score = int(summary.get("blue_score", 0))
        round_summaries = list(summary.get("summaries", []))
        total_rounds = len(round_summaries)
        avg_duration = (
            float(sum(int(s.get("duration_ticks", 0)) for s in round_summaries)) / float(max(total_rounds, 1))
            if total_rounds > 0
            else 0.0
        )
        total_kills = int(
            sum(int(s.get("survivors_yellow", 0)) + int(s.get("survivors_blue", 0)) for s in round_summaries)
        )
        total_deaths = 0
        for s in round_summaries:
            total_deaths += int(
                2 * self.team_size - int(s.get("survivors_yellow", 0)) - int(s.get("survivors_blue", 0))
            )
        bombs_planted = int(sum(1 for s in round_summaries if bool(s.get("bomb_planted", False))))
        bombs_defused = int(
            sum(1 for s in round_summaries if int(s.get("outcome", 0)) == int(RoundOutcome.BOMB_DEFUSED))
        )
        bombs_detonated = int(
            sum(1 for s in round_summaries if int(s.get("outcome", 0)) == int(RoundOutcome.BOMB_DETONATED))
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_vec_env(
    num_envs: int,
    cfg: KivskiConfig,
    map_name: str,
    base_seed: int,
    *,
    kind: str = "subproc",
    num_workers: int | None = None,
    map_data: MapData | None = None,
) -> VecEnvWrapper | ThreadedVecEnv | SubprocVecEnv:
    """Create a vectorised env wrapper of the requested ``kind``.

    Args:
        num_envs: Parallel env count.
        cfg: Shared :class:`KivskiConfig`.
        map_name: Map to load.
        base_seed: Deterministic base seed; per-env seeds derive from it.
        kind: Backend to use; one of ``"sync"`` (the original synchronous
            wrapper), ``"thread"`` (thread-pool based), ``"subproc"``
            (multi-process). ``"subproc"`` is the default for production
            training; ``"sync"`` is used by tests for deterministic
            single-thread behaviour.
        num_workers: Worker count for ``thread`` / ``subproc`` modes. None
            asks the wrapper to pick a sensible default.
        map_data: Optional pre-loaded MapData; the wrapper will load it
            itself if omitted.

    Returns:
        A wrapper exposing the :class:`VecEnvWrapper` interface.

    On a subprocess startup failure (e.g. Windows pickling issues) this
    factory logs a warning and silently falls back to the synchronous
    wrapper so the training script never crashes due to a backend choice.
    """
    k = (kind or "sync").lower().strip()
    if k == "sync":
        return VecEnvWrapper(
            num_envs=num_envs, cfg=cfg, map_name=map_name, base_seed=base_seed, map_data=map_data
        )
    if k == "thread":
        try:
            return ThreadedVecEnv(
                num_envs=num_envs,
                cfg=cfg,
                map_name=map_name,
                base_seed=base_seed,
                map_data=map_data,
                num_workers=num_workers,
            )
        except Exception as exc:  # noqa: BLE001 - fallback path
            _LOG.warning("ThreadedVecEnv failed to start (%s); falling back to SyncVecEnv", exc)
            return VecEnvWrapper(
                num_envs=num_envs,
                cfg=cfg,
                map_name=map_name,
                base_seed=base_seed,
                map_data=map_data,
            )
    if k == "subproc":
        try:
            return SubprocVecEnv(
                num_envs=num_envs,
                cfg=cfg,
                map_name=map_name,
                base_seed=base_seed,
                map_data=map_data,
                num_workers=num_workers,
            )
        except Exception as exc:  # noqa: BLE001 - fallback path
            _LOG.warning("SubprocVecEnv failed to start (%s); falling back to SyncVecEnv", exc)
            return VecEnvWrapper(
                num_envs=num_envs,
                cfg=cfg,
                map_name=map_name,
                base_seed=base_seed,
                map_data=map_data,
            )
    raise ValueError(f"unknown vec_env kind: {kind!r}")


# ---------------------------------------------------------------------------
# Defensive helper used by tests to deterministically wait for workers
# ---------------------------------------------------------------------------


def _wait_until(predicate, timeout: float, poll: float = 0.05) -> bool:
    """Tiny helper used by the test-suite to wait for an async condition."""
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(float(poll))
    return predicate()


# Threading import preserved here so pyflakes doesn't complain (used in the
# fallback path's defensive timing).
_ = threading
