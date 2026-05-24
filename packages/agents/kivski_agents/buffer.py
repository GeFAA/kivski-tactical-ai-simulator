"""Multi-agent rollout buffer for recurrent MAPPO.

The buffer is laid out as a set of pre-allocated tensors keyed by
``[T, N_envs, n_agents, ...]`` (or ``[T, N_envs, ...]`` for environment-
global quantities like the centralised value). All tensors are allocated
once at construction time and re-used across rollouts -- the trainer just
calls :meth:`clear` between cycles.

GAE-Lambda advantages are computed against the *centralised* value (one
per env per timestep) and then broadcast to all agents. Per-agent rewards
are subtracted from a common baseline so that each agent gets its own
credit assignment signal, MAPPO-style.

The buffer also yields tensor-typed :class:`RolloutBatch` mini-batches for
the PPO update; trajectories are flattened across ``T * N_envs * n_agents``
so each minibatch row is one (timestep, env, agent) transition. The
hidden state slice for that row is the *pre-step* hidden state, which the
caller feeds back into :meth:`KivskiActorCritic.evaluate` to recompute
log-probs.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["RolloutBatch", "RolloutBuffer"]


# ---------------------------------------------------------------------------
# Data class for one minibatch
# ---------------------------------------------------------------------------


@dataclass
class RolloutBatch:
    """One PPO minibatch (v0.4 mixed action space).

    All tensors share a leading batch dimension ``[BS]`` where
    ``BS = minibatch_size`` (or less for the trailing partial batch).

    The ``actions`` attribute is a dict ``{"move": [BS, move_dim] float,
    "discrete": [BS, n_heads] int64}`` -- ready to hand to
    :meth:`KivskiActorCritic.evaluate`.
    """

    observations: Tensor  # [BS, obs_dim]
    joint_observations: Tensor  # [BS, joint_obs_dim]
    actions: dict[str, Tensor]  # {"move": [BS, D], "discrete": [BS, n_heads]}
    old_log_probs: Tensor  # [BS]
    old_values: Tensor  # [BS]
    returns: Tensor  # [BS]
    advantages: Tensor  # [BS]
    masks: Tensor  # [BS]
    hidden_states: Tensor  # [num_layers, BS, hidden_size]
    received_comms: Tensor  # [BS, comm_value_dim]


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """Collects per-step transitions across ``N_envs`` parallel environments.

    The expected per-step shapes are:

    * ``observations``  ``[N_envs, n_agents, obs_dim]``
    * ``joint_obs``     ``[N_envs, joint_obs_dim]``
    * ``actions``       ``[N_envs, n_agents, n_heads]`` (int64)
    * ``log_probs``     ``[N_envs, n_agents]``
    * ``value``         ``[N_envs]`` (centralised, one per env)
    * ``rewards``       ``[N_envs, n_agents]``
    * ``masks``         ``[N_envs, n_agents]`` (1 = alive AND match not done)
    * ``hidden_states`` ``[num_layers, N_envs, n_agents, hidden_size]`` *or*
      ``[num_layers, N_envs * n_agents, hidden_size]`` -- both accepted.
    * ``received_comms`` ``[N_envs, n_agents, comm_value_dim]``
    * ``comm_masks``    ``[N_envs, n_agents, n_teammates]``

    After ``T`` steps have been added, :meth:`compute_advantages` populates
    ``returns`` / ``advantages`` and :meth:`minibatch_iter` yields shuffled
    :class:`RolloutBatch` objects for the PPO epochs.
    """

    def __init__(
        self,
        T: int,
        N_envs: int,
        n_agents: int,
        obs_dim: int,
        joint_obs_dim: int,
        n_heads: int,
        hidden_size: int,
        comm_value_dim: int,
        device: torch.device | str = "cpu",
        n_teammates: int | None = None,
        gru_layers: int = 1,
        continuous_move_dim: int = 2,
    ) -> None:
        if T <= 0 or N_envs <= 0 or n_agents <= 0:
            raise ValueError("T, N_envs, n_agents must be positive")
        if obs_dim <= 0 or joint_obs_dim <= 0 or n_heads <= 0:
            raise ValueError("obs_dim, joint_obs_dim, n_heads must be positive")
        if hidden_size <= 0 or comm_value_dim <= 0:
            raise ValueError("hidden_size, comm_value_dim must be positive")
        if gru_layers <= 0:
            raise ValueError("gru_layers must be positive")
        if continuous_move_dim <= 0:
            raise ValueError("continuous_move_dim must be positive")

        self.T: int = int(T)
        self.N_envs: int = int(N_envs)
        self.n_agents: int = int(n_agents)
        self.obs_dim: int = int(obs_dim)
        self.joint_obs_dim: int = int(joint_obs_dim)
        self.n_heads: int = int(n_heads)
        self.continuous_move_dim: int = int(continuous_move_dim)
        self.hidden_size: int = int(hidden_size)
        self.comm_value_dim: int = int(comm_value_dim)
        self.gru_layers: int = int(gru_layers)
        self.n_teammates: int = int(n_teammates) if n_teammates is not None else int(n_agents)
        self.device: torch.device = torch.device(device)

        d = self.device

        # Per-agent tensors -- [T, N_envs, n_agents, ...]
        self.observations = torch.zeros(self.T, self.N_envs, self.n_agents, self.obs_dim, device=d)
        # v0.4: split action storage -- continuous move + discrete heads.
        self.actions_move = torch.zeros(
            self.T, self.N_envs, self.n_agents, self.continuous_move_dim, device=d
        )
        self.actions_discrete = torch.zeros(
            self.T, self.N_envs, self.n_agents, self.n_heads, dtype=torch.int64, device=d
        )
        self.log_probs = torch.zeros(self.T, self.N_envs, self.n_agents, device=d)
        self.rewards = torch.zeros(self.T, self.N_envs, self.n_agents, device=d)
        self.masks = torch.zeros(self.T, self.N_envs, self.n_agents, device=d)
        self.received_comms = torch.zeros(self.T, self.N_envs, self.n_agents, self.comm_value_dim, device=d)
        self.comm_masks = torch.zeros(self.T, self.N_envs, self.n_agents, self.n_teammates, device=d)
        # Hidden state is "pre-step" -- one slot per (T, N_envs, n_agents).
        self.hidden_states = torch.zeros(
            self.T, self.gru_layers, self.N_envs, self.n_agents, self.hidden_size, device=d
        )

        # Env-global tensors -- [T, N_envs, ...]
        self.joint_observations = torch.zeros(self.T, self.N_envs, self.joint_obs_dim, device=d)
        self.values = torch.zeros(self.T, self.N_envs, device=d)

        # Computed in compute_advantages.
        self.returns = torch.zeros(self.T, self.N_envs, device=d)
        # Per-agent advantages: returns - value (broadcast across agents).
        self.advantages = torch.zeros(self.T, self.N_envs, self.n_agents, device=d)

        self._step: int = 0

    # --------------------------------------------------------------
    # Population
    # --------------------------------------------------------------

    @property
    def step(self) -> int:
        return self._step

    @property
    def full(self) -> bool:
        return self._step >= self.T

    @property
    def actions(self) -> Tensor:
        """Legacy alias: the discrete-action tensor.

        Older test code references ``buffer.actions`` directly to inspect
        sampled discrete actions. The continuous move tensor is exposed
        separately as :attr:`actions_move` / :attr:`actions_discrete`.
        """
        return self.actions_discrete

    def add(
        self,
        step: int,
        observations: Tensor,
        joint_obs: Tensor,
        actions: Tensor | dict[str, Tensor],
        log_probs: Tensor,
        value: Tensor,
        rewards: Tensor,
        masks: Tensor,
        hidden_states: Tensor,
        received_comms: Tensor,
        comm_masks: Tensor,
    ) -> None:
        """Insert one timestep of transitions at index ``step``.

        ``actions`` may be either:
          * a dict ``{"move": [N, A, D] float, "discrete": [N, A, n_heads]
            int64}`` (canonical v0.4 format), or
          * a single int64 tensor ``[N, A, n_heads]`` (legacy: the move
            tensor is zeroed, useful for tests that only care about discrete
            shapes).

        Tensors are converted to the buffer's device/dtype as needed.
        Shapes are validated to catch the most common batching mistakes early.
        """
        if step < 0 or step >= self.T:
            raise IndexError(f"step out of range [0, {self.T}): got {step}")

        ne, na = self.N_envs, self.n_agents

        def _check(t: Tensor, name: str, expected: tuple[int, ...]) -> Tensor:
            t = t.to(self.device)
            if tuple(t.shape) != expected:
                raise ValueError(f"{name} has shape {tuple(t.shape)}, expected {expected}")
            return t

        self.observations[step] = _check(
            observations.to(torch.float32), "observations", (ne, na, self.obs_dim)
        )
        self.joint_observations[step] = _check(
            joint_obs.to(torch.float32), "joint_obs", (ne, self.joint_obs_dim)
        )
        if isinstance(actions, dict):
            move = actions["move"].to(torch.float32)
            disc = actions["discrete"].to(torch.int64)
            self.actions_move[step] = _check(
                move, "actions['move']", (ne, na, self.continuous_move_dim)
            )
            self.actions_discrete[step] = _check(
                disc, "actions['discrete']", (ne, na, self.n_heads)
            )
        else:
            self.actions_discrete[step] = _check(
                actions.to(torch.int64), "actions", (ne, na, self.n_heads)
            )
            # Move defaults to zeros (HOLD) when only discrete actions are stored.
            self.actions_move[step] = torch.zeros(
                ne, na, self.continuous_move_dim, device=self.device
            )
        self.log_probs[step] = _check(log_probs.to(torch.float32), "log_probs", (ne, na))
        # Value may arrive as [N_envs] or [N_envs, 1]; squeeze.
        v = value.to(self.device, torch.float32).view(ne)
        self.values[step] = v
        self.rewards[step] = _check(rewards.to(torch.float32), "rewards", (ne, na))
        self.masks[step] = _check(masks.to(torch.float32), "masks", (ne, na))
        self.received_comms[step] = _check(
            received_comms.to(torch.float32),
            "received_comms",
            (ne, na, self.comm_value_dim),
        )
        self.comm_masks[step] = _check(
            comm_masks.to(torch.float32),
            "comm_masks",
            (ne, na, self.n_teammates),
        )

        # Hidden state: accept several shape conventions.
        hs = hidden_states.to(self.device, torch.float32)
        if hs.dim() == 4 and tuple(hs.shape) == (self.gru_layers, ne, na, self.hidden_size):
            self.hidden_states[step] = hs
        elif hs.dim() == 3 and tuple(hs.shape) == (self.gru_layers, ne * na, self.hidden_size):
            self.hidden_states[step] = hs.view(self.gru_layers, ne, na, self.hidden_size)
        else:
            raise ValueError(
                f"hidden_states has shape {tuple(hs.shape)}; expected "
                f"({self.gru_layers}, {ne}, {na}, {self.hidden_size}) "
                f"or ({self.gru_layers}, {ne * na}, {self.hidden_size})"
            )

        self._step = max(self._step, step + 1)

    # --------------------------------------------------------------
    # Advantage estimation
    # --------------------------------------------------------------

    def compute_advantages(
        self,
        last_value: Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Run GAE-Lambda over the centralised value to fill ``returns`` /
        ``advantages``.

        ``last_value`` is the bootstrap value at ``T`` (shape ``[N_envs]``
        or ``[N_envs, 1]``). Per-agent advantages are computed as
        ``returns - value`` broadcast across agents and *modulated* by the
        per-agent alive mask: dead agents see zero advantage that step, so
        their gradients vanish.
        """
        if self._step == 0:
            return
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if not (0.0 <= gae_lambda <= 1.0):
            raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")

        t_max = self._step
        lv = last_value.to(self.device, torch.float32).view(self.N_envs)
        # Convert per-agent masks into an env-level "is_active" mask: a
        # rollout step is active if *any* agent in that env is alive
        # (otherwise the centralised value is meaningless). We still keep
        # the per-agent mask around for the actor-loss weighting.
        env_active = (self.masks[:t_max].sum(dim=-1) > 0).to(self.values.dtype)
        # Average per-agent reward serves as the centralised reward; the
        # per-agent residual reward feeds into the advantage broadcast.
        avg_reward = self.rewards[:t_max].mean(dim=-1)  # [t_max, N_envs]

        advantages = torch.zeros(t_max, self.N_envs, device=self.device)
        gae = torch.zeros(self.N_envs, device=self.device)
        next_value = lv
        next_active = torch.ones(self.N_envs, device=self.device)
        for step in reversed(range(t_max)):
            active = env_active[step]
            delta = avg_reward[step] + gamma * next_value * next_active - self.values[step]
            gae = delta + gamma * gae_lambda * next_active * gae
            advantages[step] = gae
            next_value = self.values[step]
            next_active = active

        returns = advantages + self.values[:t_max]
        self.returns[:t_max] = returns
        # Broadcast advantage to per-agent and mask out dead agents.
        adv_per_agent = advantages.unsqueeze(-1).expand(-1, -1, self.n_agents)
        self.advantages[:t_max] = adv_per_agent * self.masks[:t_max]

    # --------------------------------------------------------------
    # Minibatch iteration
    # --------------------------------------------------------------

    def minibatch_iter(
        self,
        minibatch_size: int,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> Iterator[RolloutBatch]:
        """Yield :class:`RolloutBatch` slices flattened across (T, N_envs, n_agents).

        Args:
            minibatch_size: Number of transitions per minibatch. The final
                batch may be smaller if the total count is not divisible.
            shuffle: Whether to shuffle the flattened indices.
            generator: Optional ``torch.Generator`` for deterministic
                shuffling.

        Yields:
            :class:`RolloutBatch` objects with all tensors flattened to a
            shared leading batch dim.
        """
        if minibatch_size <= 0:
            raise ValueError(f"minibatch_size must be positive, got {minibatch_size}")
        t_max = self._step
        if t_max == 0:
            return

        ne, na = self.N_envs, self.n_agents
        flat_count = t_max * ne * na

        # Flatten per-agent tensors.
        obs_flat = self.observations[:t_max].reshape(flat_count, self.obs_dim)
        actions_move_flat = self.actions_move[:t_max].reshape(flat_count, self.continuous_move_dim)
        actions_disc_flat = self.actions_discrete[:t_max].reshape(flat_count, self.n_heads)
        log_probs_flat = self.log_probs[:t_max].reshape(flat_count)
        rewards_flat = self.rewards[:t_max].reshape(flat_count)
        masks_flat = self.masks[:t_max].reshape(flat_count)
        comms_flat = self.received_comms[:t_max].reshape(flat_count, self.comm_value_dim)
        adv_flat = self.advantages[:t_max].reshape(flat_count)
        # Hidden states: [t_max, L, N_envs, n_agents, H] -> [L, t_max*N_envs*n_agents, H]
        hs_flat = (
            self.hidden_states[:t_max]
            .permute(1, 0, 2, 3, 4)
            .reshape(self.gru_layers, flat_count, self.hidden_size)
        )

        # Env-global tensors need broadcasting along the agent axis.
        joint_obs_per_agent = (
            self.joint_observations[:t_max]
            .unsqueeze(2)
            .expand(t_max, ne, na, self.joint_obs_dim)
            .reshape(flat_count, self.joint_obs_dim)
        )
        values_per_agent = self.values[:t_max].unsqueeze(-1).expand(t_max, ne, na).reshape(flat_count)
        returns_per_agent = self.returns[:t_max].unsqueeze(-1).expand(t_max, ne, na).reshape(flat_count)
        del rewards_flat  # not needed in the batch -- kept for symmetry above

        # Indices.
        if shuffle:
            indices = torch.randperm(flat_count, device=self.device, generator=generator)
        else:
            indices = torch.arange(flat_count, device=self.device)

        for start in range(0, flat_count, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield RolloutBatch(
                observations=obs_flat.index_select(0, idx),
                joint_observations=joint_obs_per_agent.index_select(0, idx),
                actions={
                    "move": actions_move_flat.index_select(0, idx),
                    "discrete": actions_disc_flat.index_select(0, idx),
                },
                old_log_probs=log_probs_flat.index_select(0, idx),
                old_values=values_per_agent.index_select(0, idx),
                returns=returns_per_agent.index_select(0, idx),
                advantages=adv_flat.index_select(0, idx),
                masks=masks_flat.index_select(0, idx),
                hidden_states=hs_flat.index_select(1, idx),
                received_comms=comms_flat.index_select(0, idx),
            )

    # --------------------------------------------------------------
    # House-keeping
    # --------------------------------------------------------------

    def clear(self) -> None:
        """Reset the write cursor; tensor storage is reused next rollout."""
        self._step = 0

    def __len__(self) -> int:
        return self._step
