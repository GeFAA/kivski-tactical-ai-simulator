"""Roll out T steps across N envs, populating a :class:`RolloutBuffer`.

The collector wires together the training :class:`PolicyRunner`, an
:class:`OpponentSampler`, and the :class:`VecEnvWrapper`. The training
policy plays the YELLOW side and the opponent plays the BLUE side; this
mapping is stable across the whole episode because the trainer disables
side-switching while training (the engine's side-switch logic is still
exercised in eval).

For each tick:

1. Slice the per-env observation dict into a YELLOW dict and a BLUE dict
   keyed by agent name.
2. Run the training :class:`PolicyRunner` over the YELLOW agents to get
   actions + comm payloads + log-probs + the centralised value.
3. Run the :class:`OpponentSampler` over the BLUE agents to get their
   actions / payloads (no log-probs needed; we don't train the opponent).
4. Merge into a single action / payload dict and call ``vec_env.step``.
5. Store the transition into the :class:`RolloutBuffer`.
6. On per-env episode-done flags, capture :class:`EpisodeStats` and reset
   the training policy's hidden state for the affected agents.

The collected per-step shapes line up with what :class:`RolloutBuffer`
expects (``[N_envs, n_agents, ...]``); see the buffer docstring for the
exact contract.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from kivski_sim.config import KivskiConfig
from kivski_sim.env import agent_index

from kivski_agents.buffer import RolloutBuffer
from kivski_agents.metrics import CommUsageStats, EpisodeStats
from kivski_agents.policy_runner import PolicyRunner
from kivski_agents.training.league import OpponentSampler
from kivski_agents.training.vec_env import VecEnvWrapper

__all__ = ["RolloutCollector", "CollectionResult"]


# ---------------------------------------------------------------------------
# Return payload
# ---------------------------------------------------------------------------


@dataclass
class CollectionResult:
    """Bundle returned from :meth:`RolloutCollector.collect`."""

    buffer: RolloutBuffer
    episode_stats: list[EpisodeStats]
    comm_usage: CommUsageStats
    last_value: torch.Tensor  # [N_envs] bootstrap value for GAE
    total_env_steps: int
    fps: float


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class RolloutCollector:
    """Drives ``T`` steps across ``N`` envs, writing into a :class:`RolloutBuffer`."""

    def __init__(
        self,
        vec_env: VecEnvWrapper,
        training_runner: PolicyRunner,
        opponent_sampler: OpponentSampler,
        buffer: RolloutBuffer,
        cfg: KivskiConfig,
        device: torch.device | str = "cpu",
        initial_hidden: torch.Tensor | None = None,
    ) -> None:
        self.vec_env: VecEnvWrapper = vec_env
        self.runner: PolicyRunner = training_runner
        self.opponent: OpponentSampler = opponent_sampler
        self.buffer: RolloutBuffer = buffer
        self.cfg: KivskiConfig = cfg
        self.device: torch.device = torch.device(device)
        self.team_size: int = int(cfg.simulation.team_size)
        self.obs_dim: int = int(vec_env.obs_dim)
        self.n_heads: int = int(vec_env.n_heads)

        # Static role partition. The trainer pins side-switching off, so this
        # mapping is stable across the whole match.
        agent_team = vec_env.agent_team_indices(env_idx=0)
        from kivski_sim.types import Team  # local to avoid cycles

        self.yellow_names: list[str] = sorted(
            (name for name, tid in agent_team.items() if tid == int(Team.YELLOW)),
            key=agent_index,
        )
        self.blue_names: list[str] = sorted(
            (name for name, tid in agent_team.items() if tid == int(Team.BLUE)),
            key=agent_index,
        )
        if len(self.yellow_names) != self.team_size or len(self.blue_names) != self.team_size:
            raise RuntimeError(
                f"Team partition mismatch: yellow={self.yellow_names!r}, "
                f"blue={self.blue_names!r}, expected team_size={self.team_size}"
            )

        # Initialise side-internal state if it hasn't been set up yet. We do
        # NOT reset an env that already has in-flight episodes: each rollout
        # picks up where the previous one stopped, and per-env auto-resets in
        # the vec-env handle terminal transitions cleanly.
        if not getattr(self.runner, "_agent_names", None):
            self.runner.reset(self.yellow_names)
        self.opponent.reset(self.blue_names)

        # If the vec env has never produced a starting observation, bootstrap
        # it now. Otherwise reuse whatever the previous collector left behind.
        cur = self.vec_env.current_step()
        if cur is None:
            self._last_step = self.vec_env.reset()
        else:
            self._last_step = cur

        # Per-comm aggregation across the rollout.
        self._comm_counter: Counter[int] = Counter()
        self._payload_norm_sum: float = 0.0
        self._payload_norm_count: int = 0

        # Per-collector batched hidden state: [L, num_envs * team_size, H].
        # If ``initial_hidden`` is provided (typical: the trainer hands over the
        # previous collector's final hidden so consecutive rollouts share
        # recurrent context), we adopt it; otherwise we seed with zeros.
        L = int(self.runner.model.gru_layers)
        H = int(self.runner.model.hidden_size)
        ne = int(vec_env.num_envs)
        ts = int(self.team_size)
        if initial_hidden is not None and tuple(initial_hidden.shape) == (L, ne * ts, H):
            self._batched_hidden: torch.Tensor = initial_hidden.detach().to(self.device).clone()
        else:
            self._batched_hidden = torch.zeros(L, ne * ts, H, device=self.device)

    # ------------------------------------------------------------------

    @property
    def batched_hidden(self) -> torch.Tensor:
        """Read-only access to the current ``[L, ne*team_size, H]`` hidden state."""
        return self._batched_hidden

    # ------------------------------------------------------------------

    def collect(self, T: int) -> CollectionResult:
        """Collect ``T`` transitions per env into :attr:`buffer`."""
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")

        self.buffer.clear()
        self._comm_counter = Counter()
        self._payload_norm_sum = 0.0
        self._payload_norm_count = 0
        episode_stats_out: list[EpisodeStats] = []
        ne = self.vec_env.num_envs
        cv_dim = int(self.buffer.comm_value_dim)

        # Per-env per-agent reset mask. Cleared between calls.
        self._needs_hidden_reset: list[set[int]] = [set() for _ in range(ne)]

        # Wall-clock timer for FPS reporting.
        import time

        t_start = time.perf_counter()

        # Pre-allocate per-tick scratch tensors that are written into the buffer.
        for t in range(T):
            obs_batch = self._last_step.observations  # dict[name, [N_envs, obs_dim]]

            # ---- 1) Build hidden-state masks for the runner ------------
            # Mask shape used by PolicyRunner is per-agent (the runner stores
            # [num_layers, n_agents_yellow, H]). We pass mask=0 to force a hidden
            # reset for an agent whose env auto-reset on the previous step.
            yellow_mask = self._compute_runner_masks_for_yellow(ne)

            # ---- 2) Stack YELLOW observations across envs --------------
            # Result shape: [N_envs * team_size, obs_dim]
            yellow_obs_np = np.zeros((ne * self.team_size, self.obs_dim), dtype=np.float32)
            for env_i in range(ne):
                for agent_j, name in enumerate(self.yellow_names):
                    yellow_obs_np[env_i * self.team_size + agent_j] = obs_batch[name][env_i]
            yellow_obs = torch.from_numpy(yellow_obs_np).to(self.device)

            # Joint observation per env = concat of YELLOW obs (the centralised
            # critic only sees its own team -- this is the standard MAPPO
            # convention for symmetric two-team games).
            joint_obs_np = yellow_obs_np.reshape(ne, self.team_size * self.obs_dim)
            joint_obs = torch.from_numpy(joint_obs_np).to(self.device)

            # ---- 3) Build the received-comm tensor for YELLOW ----------
            # We pull comm payloads from the env infos written on the previous
            # step. We average payloads received from teammates (matching the
            # PolicyRunner.act averaging) before the forward pass.
            received_comm_np = self._gather_received_comm(
                obs_owner_names=self.yellow_names,
                payload_dim=cv_dim,
            )
            received_comm = torch.from_numpy(received_comm_np.reshape(ne * self.team_size, cv_dim)).to(
                self.device
            )

            # ---- 4) Forward training policy ----------------------------
            # The pre-step hidden state is the collector-owned batched tensor
            # of shape ``[L, ne * team_size, H]`` -- we snapshot it before the
            # forward pass so the buffer can replay the GRU during PPO updates.
            pre_hidden = self._batched_hidden.detach().clone()
            (
                yellow_move_np,
                yellow_disc_np,
                yellow_payloads_np,
                log_probs_np,
                value_np,
                new_hidden,
                entropy_np,
            ) = self._forward_training_policy(
                yellow_obs=yellow_obs,
                received_comm=received_comm,
                joint_obs=joint_obs,
                mask=yellow_mask,
                pre_hidden=pre_hidden,
            )

            # The forward call uses ``pre_hidden`` as input and returns
            # ``new_hidden`` of the same shape; this becomes the next tick's
            # pre-step hidden.
            self._batched_hidden = new_hidden  # [L, ne * team_size, H]

            # ---- 5) Forward opponent (BLUE) ----------------------------
            blue_obs_dict = {name: obs_batch[name] for name in self.blue_names}
            blue_move_np, blue_disc_np, blue_payloads_np = self._forward_opponent(blue_obs_dict)

            # ---- 6) Merge actions / payloads ---------------------------
            # The env expects ``{agent: {"move": [N_envs, D], "discrete":
            # [N_envs, n_heads]}}`` per agent (decoded one tick at a time
            # inside ``VecEnvWrapper.step``).
            merged_actions: dict[str, dict[str, np.ndarray]] = {}
            merged_payloads: dict[str, np.ndarray] = {}
            for j, name in enumerate(self.yellow_names):
                merged_actions[name] = {
                    "move": yellow_move_np[:, j, :],  # [N_envs, move_dim]
                    "discrete": yellow_disc_np[:, j, :],  # [N_envs, n_heads]
                }
                merged_payloads[name] = yellow_payloads_np[:, j, :]
            for j, name in enumerate(self.blue_names):
                merged_actions[name] = {
                    "move": blue_move_np[:, j, :],
                    "discrete": blue_disc_np[:, j, :],
                }
                merged_payloads[name] = blue_payloads_np[:, j, :]

            # ---- 7) Step the vec env -----------------------------------
            step_out = self.vec_env.step(merged_actions, comm_payloads=merged_payloads)

            # ---- 8) Build buffer per-step tensors ----------------------
            # The buffer's "n_agents" dimension is the *training-side* team only.
            # We only place the training-side per-agent tensors in the buffer;
            # rewards from the opponent side are not used for PPO updates.
            yellow_rewards_np = np.zeros((ne, self.team_size), dtype=np.float32)
            for j, name in enumerate(self.yellow_names):
                yellow_rewards_np[:, j] = step_out.rewards[name]

            # Pack the mixed action storage for the buffer.
            actions_payload = {
                "move": torch.from_numpy(yellow_move_np).to(self.device),
                "discrete": torch.from_numpy(yellow_disc_np).to(self.device),
            }

            # Per-agent alive mask (1 = alive, 0 = dead or done).
            # The mask is per-tick, taken *after* the env step but before any
            # auto-reset's fresh state is observed. We use info["per_agent"][name]["alive"].
            alive_mask_np = np.zeros((ne, self.team_size), dtype=np.float32)
            for env_i in range(ne):
                # In auto-reset envs the per_agent info corresponds to the new
                # fresh episode (all alive). We treat that as alive=1.0 so the
                # next step has a valid mask.
                per_agent = step_out.infos[env_i].get("per_agent", {})
                for j, name in enumerate(self.yellow_names):
                    info_for_agent = per_agent.get(name, {})
                    alive_mask_np[env_i, j] = 1.0 if bool(info_for_agent.get("alive", True)) else 0.0

            # Comm masks per agent (we don't have explicit attention masks at this
            # layer; the model uses an internal default). Provide a 1-mask of
            # shape [N_envs, team_size, team_size] (broadcastable to whatever
            # the buffer needs as n_teammates).
            n_teammates = int(self.buffer.n_teammates)
            comm_masks_np = np.ones((ne, self.team_size, n_teammates), dtype=np.float32)

            # Hidden state in buffer shape: [L, N_envs, n_agents, H].
            # pre_hidden is [L, N_envs * team_size, H] -- our chosen flatten
            # order is (env_i * team_size + agent_j), so reshaping back to
            # [L, N_envs, team_size, H] reverses that without further work.
            buffer_hidden = pre_hidden.reshape(
                self.buffer.gru_layers, ne, self.team_size, self.buffer.hidden_size
            )

            # ---- 9) Write into the buffer ------------------------------
            self.buffer.add(
                step=t,
                observations=yellow_obs.view(ne, self.team_size, self.obs_dim),
                joint_obs=joint_obs,
                actions=actions_payload,
                log_probs=torch.from_numpy(log_probs_np).to(self.device),
                value=torch.from_numpy(value_np).to(self.device),
                rewards=torch.from_numpy(yellow_rewards_np).to(self.device),
                masks=torch.from_numpy(alive_mask_np).to(self.device),
                hidden_states=buffer_hidden,
                received_comms=torch.from_numpy(received_comm_np.reshape(ne, self.team_size, cv_dim)).to(
                    self.device
                ),
                comm_masks=torch.from_numpy(comm_masks_np).to(self.device),
            )

            # ---- 10) Tally comm usage ----------------------------------
            # v0.4 discrete head layout: [micro=0, comm=1, buy=2, aim=3].
            yellow_comm_actions = yellow_disc_np[:, :, 1].reshape(-1)
            self._comm_counter.update(int(x) for x in yellow_comm_actions.tolist())
            # Mean payload norm (across YELLOW, this tick).
            payload_norms = np.linalg.norm(yellow_payloads_np.reshape(-1, cv_dim), axis=-1)
            self._payload_norm_sum += float(payload_norms.sum())
            self._payload_norm_count += int(payload_norms.shape[0])

            # ---- 11) Capture episode stats + queue hidden resets ------
            for env_i, info in enumerate(step_out.infos):
                if info.get("episode_done"):
                    stats = info.get("episode_stats")
                    if stats is not None:
                        episode_stats_out.append(stats)
                    # Queue hidden reset for every YELLOW agent in this env.
                    for j in range(self.team_size):
                        self._needs_hidden_reset[env_i].add(j)
                    # Also reset the opponent's internal state for this env.
                    with contextlib.suppress(Exception):
                        self.opponent.reset(self.blue_names)

            self._last_step = step_out

        # ---- After the loop: build the bootstrap value for GAE ------------
        last_obs_batch = self._last_step.observations
        last_yellow_obs_np = np.zeros((ne * self.team_size, self.obs_dim), dtype=np.float32)
        for env_i in range(ne):
            for agent_j, name in enumerate(self.yellow_names):
                last_yellow_obs_np[env_i * self.team_size + agent_j] = last_obs_batch[name][env_i]
        last_joint_obs_np = last_yellow_obs_np.reshape(ne, self.team_size * self.obs_dim)
        last_joint_obs = torch.from_numpy(last_joint_obs_np).to(self.device)
        with torch.no_grad():
            last_value = self.runner.model.value_head(last_joint_obs).view(ne).detach()

        # Aggregate comm-usage stats.
        total_messages = int(sum(self._comm_counter.values()))
        if total_messages > 0:
            probs = np.array([c / total_messages for c in self._comm_counter.values()], dtype=np.float64)
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        else:
            entropy = 0.0
        mean_payload_norm = self._payload_norm_sum / float(max(self._payload_norm_count, 1))
        comm_usage = CommUsageStats(
            counts={int(k): int(v) for k, v in self._comm_counter.items()},
            entropy=float(entropy),
            mean_payload_norm=float(mean_payload_norm),
        )

        # Wall-clock FPS.
        elapsed = max(time.perf_counter() - t_start, 1e-9)
        total_env_steps = T * ne
        fps = float(total_env_steps) / float(elapsed)

        return CollectionResult(
            buffer=self.buffer,
            episode_stats=episode_stats_out,
            comm_usage=comm_usage,
            last_value=last_value,
            total_env_steps=int(total_env_steps),
            fps=float(fps),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_runner_masks_for_yellow(self, num_envs: int) -> torch.Tensor:
        """Return a ``[num_envs * team_size]`` 0/1 mask (0 = reset hidden)."""
        mask = np.ones(num_envs * self.team_size, dtype=np.float32)
        for env_i in range(num_envs):
            if self._needs_hidden_reset[env_i]:
                for j in self._needs_hidden_reset[env_i]:
                    mask[env_i * self.team_size + j] = 0.0
                self._needs_hidden_reset[env_i].clear()
        return torch.from_numpy(mask).to(self.device)

    def _forward_training_policy(
        self,
        yellow_obs: torch.Tensor,  # [N_envs * team_size, obs_dim]
        received_comm: torch.Tensor,  # [N_envs * team_size, cv_dim]
        joint_obs: torch.Tensor,  # [N_envs, joint_obs_dim]
        mask: torch.Tensor,  # [N_envs * team_size]
        pre_hidden: torch.Tensor,  # [L, N_envs * team_size, H]
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        torch.Tensor,
        np.ndarray,
    ]:
        """Run the training model over all YELLOW agents across all envs.

        Returns numpy arrays sized:
            * move_actions: ``[N_envs, team_size, continuous_move_dim]`` float32
            * discrete_actions: ``[N_envs, team_size, n_heads]`` int64
            * payloads: ``[N_envs, team_size, cv_dim]``
            * log_probs: ``[N_envs, team_size]``
            * value: ``[N_envs]``
            * new_hidden: ``[L, N_envs * team_size, H]`` torch tensor
            * entropy: ``[N_envs, team_size]``
        """
        ne = self.vec_env.num_envs
        ts = self.team_size
        cv_dim = int(self.buffer.comm_value_dim)
        move_dim = int(self.buffer.continuous_move_dim)

        with torch.no_grad():
            out = self.runner.model.act(
                obs=yellow_obs,
                hidden_state=pre_hidden,
                received_comm=received_comm,
                joint_obs=None,  # value computed separately on per-env joint obs
                masks=mask,
                deterministic=False,
            )
            value = self.runner.model.value_head(joint_obs).view(ne).detach()

        move_actions = (
            out["move_actions"].view(ne, ts, move_dim).cpu().numpy().astype(np.float32)
        )
        disc_actions = (
            out["discrete_actions"].view(ne, ts, self.n_heads).cpu().numpy().astype(np.int64)
        )
        log_probs = out["log_probs"].view(ne, ts).cpu().numpy().astype(np.float32)
        entropy = out["entropy"].view(ne, ts).cpu().numpy().astype(np.float32)
        payloads = out["comm_payload"].view(ne, ts, cv_dim).cpu().numpy().astype(np.float32)
        new_hidden = out["new_hidden"]
        value_np = value.cpu().numpy().astype(np.float32)
        return move_actions, disc_actions, payloads, log_probs, value_np, new_hidden, entropy

    def _forward_opponent(
        self,
        blue_obs_per_env: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the opponent sampler env-by-env and return stacked arrays.

        Returns:
            * move_actions: ``[N_envs, team_size, move_dim]`` float32
            * discrete_actions: ``[N_envs, team_size, n_heads]`` int64
            * payloads: ``[N_envs, team_size, cv_dim]``
        """
        ne = self.vec_env.num_envs
        ts = self.team_size
        cv_dim = int(self.buffer.comm_value_dim)
        move_dim = int(self.buffer.continuous_move_dim)
        move_out = np.zeros((ne, ts, move_dim), dtype=np.float32)
        disc_out = np.zeros((ne, ts, self.n_heads), dtype=np.int64)
        payloads_out = np.zeros((ne, ts, cv_dim), dtype=np.float32)

        for env_i in range(ne):
            obs_for_env: dict[str, np.ndarray] = {}
            for name in self.blue_names:
                obs_for_env[name] = blue_obs_per_env[name][env_i]
            actions_dict, payloads_dict = self.opponent.act(obs_for_env)
            for j, name in enumerate(self.blue_names):
                if name in actions_dict:
                    move_arr, disc_arr = _split_baseline_action(
                        actions_dict[name], move_dim=move_dim, n_heads=self.n_heads
                    )
                    move_out[env_i, j] = move_arr
                    disc_out[env_i, j] = disc_arr
                if payloads_dict and name in payloads_dict:
                    pl = np.asarray(payloads_dict[name], dtype=np.float32).reshape(-1)
                    if pl.shape[0] >= cv_dim:
                        payloads_out[env_i, j] = pl[:cv_dim]
                    else:
                        payloads_out[env_i, j, : pl.shape[0]] = pl
        return move_out, disc_out, payloads_out

    def _gather_received_comm(
        self,
        obs_owner_names: list[str],
        payload_dim: int,
    ) -> np.ndarray:
        """Build the average-received comm tensor for the given side.

        Returns:
            ``[N_envs, team_size, payload_dim]`` float32.
        """
        ne = self.vec_env.num_envs
        ts = len(obs_owner_names)
        out = np.zeros((ne, ts, payload_dim), dtype=np.float32)
        # We pull per-agent comm_messages from the previous step's per-env info.
        if self._last_step is None:
            return out
        for env_i, info in enumerate(self._last_step.infos):
            per_agent = info.get("per_agent", {})
            for j, name in enumerate(obs_owner_names):
                msgs = per_agent.get(name, {}).get("comm_messages", {})
                if not msgs:
                    continue
                accum = np.zeros(payload_dim, dtype=np.float32)
                count = 0
                for payload in msgs.values():
                    arr = np.asarray(payload, dtype=np.float32).reshape(-1)
                    if arr.shape[0] >= payload_dim:
                        accum += arr[:payload_dim]
                    else:
                        tmp = np.zeros(payload_dim, dtype=np.float32)
                        tmp[: arr.shape[0]] = arr
                        accum += tmp
                    count += 1
                if count > 0:
                    out[env_i, j] = accum / float(count)
        return out


# ---------------------------------------------------------------------------
# Tiny standalone helper (kept here so the trainer can stay lean)
# ---------------------------------------------------------------------------


def _split_baseline_action(
    action: Any,
    *,
    move_dim: int,
    n_heads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a baseline action into (move_vec, discrete_heads).

    Accepts the v0.4 dict ``{"move": [D], "discrete": [n_heads]}`` as well as
    a legacy flat int64 vector ``[move_idx, micro, comm, buy, aim]`` (which
    gets mapped via :data:`kivski_sim.types.MOVE_VECTORS`). Any missing
    heads are zero-padded.
    """
    move = np.zeros(move_dim, dtype=np.float32)
    disc = np.zeros(n_heads, dtype=np.int64)
    if isinstance(action, dict):
        if "move" in action:
            mv = np.asarray(action["move"], dtype=np.float32).reshape(-1)
            move[: min(move_dim, mv.shape[0])] = mv[: min(move_dim, mv.shape[0])]
        if "discrete" in action:
            d = np.asarray(action["discrete"], dtype=np.int64).reshape(-1)
            disc[: min(n_heads, d.shape[0])] = d[: min(n_heads, d.shape[0])]
        return move, disc
    arr = np.asarray(action).reshape(-1)
    # Legacy flat encoding: assume first entry is the legacy MoveIntent.
    if arr.shape[0] == 5:
        from kivski_sim.types import MOVE_VECTORS, MoveIntent

        try:
            intent = MoveIntent(int(arr[0]))
        except (ValueError, TypeError):
            intent = MoveIntent.HOLD
        dx, dy = MOVE_VECTORS[intent]
        move[0] = float(dx)
        if move_dim >= 2:
            move[1] = float(dy)
        tail = np.asarray(arr[1:], dtype=np.int64)
        disc[: min(n_heads, tail.shape[0])] = tail[: min(n_heads, tail.shape[0])]
        return move, disc
    # Length-4 = pure discrete heads (already in v0.4 format).
    if arr.shape[0] == n_heads:
        disc[:] = np.asarray(arr, dtype=np.int64)
        return move, disc
    return move, disc


def required_buffer_capacity(num_envs: int, rollout_steps: int, team_size: int) -> int:
    """Return the total number of (T, N, agent) transitions stored per rollout."""
    if num_envs <= 0 or rollout_steps <= 0 or team_size <= 0:
        return 0
    return int(num_envs) * int(rollout_steps) * int(team_size)
