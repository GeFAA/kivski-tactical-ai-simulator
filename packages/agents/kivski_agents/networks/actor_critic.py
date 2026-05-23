"""Recurrent MAPPO actor-critic with TarMAC communication.

The top-level :class:`KivskiActorCritic` is what the trainer and the
inference runner instantiate. It wires together:

* a feed-forward :class:`ObservationEncoder` that lifts the flat env
  observation into a hidden-size embedding;
* a :class:`RecurrentCore` GRU over ``(obs_encoding, received_comm)`` that
  carries history across ticks;
* the TarMAC :class:`~kivski_agents.networks.comm.CommEncoder`,
  :class:`~kivski_agents.networks.comm.CommAttention`, and
  :class:`~kivski_agents.networks.comm.CommGate` from :mod:`.comm`;
* an autoregressive :class:`ActorHeads` module that samples five discrete
  heads in order: ``move -> micro -> comm_action -> buy -> aim_target``;
* a centralised :class:`ValueHead` for CTDE-style training.

The conventions used throughout:

* Batches are flat: ``[B, ...]``. The recurrent core can also operate on
  ``[T, B, ...]`` tensors for BPTT, but the top-level public methods
  ``act`` / ``evaluate`` only operate per timestep.
* Hidden states are shaped ``[num_layers, B, hidden_size]`` (PyTorch GRU
  convention). :meth:`initial_hidden_state` returns zeros of that shape.
* Centralised critic input is the *joint* observation -- concatenated
  per-agent observation vectors -- so the trainer is in charge of building
  that vector before each call.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from kivski_agents.networks.comm import CommAttention, CommEncoder, CommGate

__all__ = [
    "ActorHeads",
    "KivskiActorCritic",
    "ObservationEncoder",
    "RecurrentCore",
    "ValueHead",
]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ObservationEncoder(nn.Module):
    """Two-layer MLP that maps the flat observation vector to ``hidden_size``."""

    def __init__(self, obs_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.obs_dim: int = int(obs_dim)
        self.hidden_size: int = int(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )

    def forward(self, obs: Tensor) -> Tensor:
        """Encode ``obs`` of shape ``[B, obs_dim]`` to ``[B, hidden_size]``."""
        if obs.dim() != 2:
            raise ValueError(f"ObservationEncoder expects [B, obs_dim], got {tuple(obs.shape)}")
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"ObservationEncoder obs_dim mismatch: expected {self.obs_dim}, got {obs.shape[-1]}"
            )
        return self.net(obs)


class RecurrentCore(nn.Module):
    """GRU over ``concat(obs_encoding, received_comm)``.

    The module supports two call shapes:

    * Per-step:  input ``[B, input_dim]``, hidden ``[L, B, H]``
                  -> output ``[B, H]``, new hidden ``[L, B, H]``.
    * Sequence:  input ``[T, B, input_dim]``, hidden ``[L, B, H]``
                  -> output ``[T, B, H]``, new hidden ``[L, B, H]``.

    The ``masks`` argument (per-step or per-T*B) optionally resets the
    hidden state on episode boundaries: a 0 in ``masks`` zeroes the prior
    hidden state for that batch entry before the GRU step. This is the
    convention used in the rollout buffer / PPO update.
    """

    def __init__(self, input_dim: int, hidden_size: int = 256, num_layers: int = 1) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.input_dim: int = int(input_dim)
        self.hidden_size: int = int(hidden_size)
        self.num_layers: int = int(num_layers)
        # batch_first=False (timestep first) matches the BPTT convention used
        # by the PPO loop, even though our public top-level API uses [B, ...].
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=False,
        )

    def initial_hidden(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        """Return a zero hidden state ``[num_layers, batch_size, hidden_size]``."""
        return torch.zeros(self.num_layers, int(batch_size), self.hidden_size, device=device)

    def forward(
        self,
        inputs: Tensor,
        hidden: Tensor,
        masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run the GRU.

        Args:
            inputs: ``[B, input_dim]`` for per-step or ``[T, B, input_dim]``
                for sequence mode.
            hidden: ``[num_layers, B, hidden_size]`` prior hidden state.
            masks: Optional 0/1 mask. Per-step ``[B]`` / ``[B, 1]`` or
                sequence ``[T, B]`` / ``[T, B, 1]``. ``0`` resets hidden
                state for that entry **before** the GRU step.

        Returns:
            ``(output, new_hidden)`` -- output has the same leading dims as
            ``inputs`` and trailing ``hidden_size``; new hidden is the
            standard ``[num_layers, B, hidden_size]``.
        """
        if hidden.dim() != 3 or hidden.shape[0] != self.num_layers:
            raise ValueError(
                f"hidden must be [num_layers={self.num_layers}, B, H], got {tuple(hidden.shape)}"
            )

        if inputs.dim() == 2:
            # Per-step mode -- run a single GRU step.
            if masks is not None:
                m = masks.view(1, inputs.shape[0], 1).to(hidden.dtype)
                hidden = hidden * m
            out, new_hidden = self.gru(inputs.unsqueeze(0), hidden)
            return out.squeeze(0), new_hidden

        if inputs.dim() == 3:
            t, b, _ = inputs.shape
            if masks is None:
                out, new_hidden = self.gru(inputs, hidden)
                return out, new_hidden
            # When masks are provided we iterate timesteps so the hidden
            # state can be zeroed at episode boundaries.
            m = masks.view(t, b, 1).to(hidden.dtype)
            outputs = torch.empty(t, b, self.hidden_size, device=inputs.device, dtype=inputs.dtype)
            cur_hidden = hidden
            for step in range(t):
                cur_hidden = cur_hidden * m[step].view(1, b, 1)
                step_out, cur_hidden = self.gru(inputs[step].unsqueeze(0), cur_hidden)
                outputs[step] = step_out.squeeze(0)
            return outputs, cur_hidden

        raise ValueError(f"inputs must be 2D or 3D, got shape {tuple(inputs.shape)}")


# ---------------------------------------------------------------------------
# Actor / critic heads
# ---------------------------------------------------------------------------


class ActorHeads(nn.Module):
    """Autoregressive multi-head categorical actor.

    There are five action heads, sampled in a fixed order so each head can
    condition on the previously sampled (embedded) actions:

    1. ``move``      -- 9 categories (compass + hold).
    2. ``micro``     -- 6 categories (posture / interact).
    3. ``comm``      -- 9 categories (discrete comm token id).
    4. ``buy``       -- 8 categories (purchase choice).
    5. ``aim_target`` -- ``2 * team_size + 1`` categories (0 = none, 1.. =
       pointer into the other agents).

    Two modes:

    * :meth:`sample` -- autoregressive sampling for rollouts.
    * :meth:`evaluate` -- given actions, run all heads in parallel to
      recover per-head log probabilities, summed entropies, and the joint
      log-probability (sum across heads).
    """

    def __init__(
        self,
        hidden_size: int,
        action_dims: Sequence[int],
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        action_dims_list = [int(a) for a in action_dims]
        if not action_dims_list:
            raise ValueError("action_dims must not be empty")
        if any(a <= 0 for a in action_dims_list):
            raise ValueError(f"action_dims must all be positive, got {action_dims_list}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        self.hidden_size: int = int(hidden_size)
        self.action_dims: list[int] = list(action_dims_list)
        self.num_heads: int = len(self.action_dims)
        self.embedding_dim: int = int(embedding_dim)

        # One Linear per head -- input width grows with the number of
        # previously sampled actions (each embedded into ``embedding_dim``).
        heads: list[nn.Linear] = []
        embeddings: list[nn.Embedding] = []
        for i, n_cat in enumerate(self.action_dims):
            in_dim = self.hidden_size + i * self.embedding_dim
            heads.append(nn.Linear(in_dim, int(n_cat)))
            embeddings.append(nn.Embedding(int(n_cat), self.embedding_dim))
        self.heads = nn.ModuleList(heads)
        self.embeddings = nn.ModuleList(embeddings)

    # --------------------------------------------------------------
    # Sampling
    # --------------------------------------------------------------

    def sample(
        self,
        hidden: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Autoregressively sample one action per head.

        Args:
            hidden: ``[B, hidden_size]`` actor hidden state.
            deterministic: If True, take the argmax per head instead of
                sampling. Useful for eval / deployment.

        Returns:
            ``(actions, log_probs, entropy)``:
                * ``actions``: ``[B, num_heads]`` int64 sampled categories.
                * ``log_probs``: ``[B]`` -- summed log-prob across heads.
                * ``entropy``: ``[B]`` -- summed per-head categorical
                  entropy. (For the actually-sampled trajectory, this is
                  the *policy* entropy used in the PPO loss; we keep it
                  detached from the sampled action itself.)
        """
        if hidden.dim() != 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden must be [B, {self.hidden_size}], got {tuple(hidden.shape)}")
        b = hidden.shape[0]
        device = hidden.device

        actions = torch.empty(b, self.num_heads, dtype=torch.int64, device=device)
        log_prob_sum = torch.zeros(b, device=device)
        entropy_sum = torch.zeros(b, device=device)
        prev_embeddings: list[Tensor] = []
        for i, (head, emb) in enumerate(zip(self.heads, self.embeddings, strict=False)):
            head_in = torch.cat([hidden, *prev_embeddings], dim=-1) if prev_embeddings else hidden
            logits = head(head_in)
            dist = torch.distributions.Categorical(logits=logits)
            act = logits.argmax(dim=-1) if deterministic else dist.sample()
            actions[:, i] = act
            log_prob_sum = log_prob_sum + dist.log_prob(act)
            entropy_sum = entropy_sum + dist.entropy()
            prev_embeddings.append(emb(act))
        return actions, log_prob_sum, entropy_sum

    # --------------------------------------------------------------
    # Evaluation given pre-recorded actions
    # --------------------------------------------------------------

    def evaluate(
        self,
        hidden: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute joint log-prob + summed entropy for the given actions.

        Args:
            hidden: ``[B, hidden_size]`` actor hidden state.
            actions: ``[B, num_heads]`` int64 previously sampled actions.

        Returns:
            ``(log_probs, entropy)`` each shape ``[B]``.
        """
        if hidden.dim() != 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden must be [B, {self.hidden_size}], got {tuple(hidden.shape)}")
        if actions.dim() != 2 or actions.shape[1] != self.num_heads:
            raise ValueError(f"actions must be [B, num_heads={self.num_heads}], got {tuple(actions.shape)}")
        b = hidden.shape[0]
        log_prob_sum = torch.zeros(b, device=hidden.device)
        entropy_sum = torch.zeros(b, device=hidden.device)
        prev_embeddings: list[Tensor] = []
        actions_long = actions.detach().to(torch.int64)
        for i, (head, emb) in enumerate(zip(self.heads, self.embeddings, strict=False)):
            head_in = torch.cat([hidden, *prev_embeddings], dim=-1) if prev_embeddings else hidden
            logits = head(head_in)
            dist = torch.distributions.Categorical(logits=logits)
            # Use out-of-place ``clamp`` so we don't mutate a tensor that may
            # be reused across PPO epochs (would break autograd).
            act = actions_long[:, i].clamp(0, self.action_dims[i] - 1)
            log_prob_sum = log_prob_sum + dist.log_prob(act)
            entropy_sum = entropy_sum + dist.entropy()
            prev_embeddings.append(emb(act))
        return log_prob_sum, entropy_sum


class ValueHead(nn.Module):
    """Centralised critic MLP that consumes the joint observation."""

    def __init__(self, joint_obs_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        if joint_obs_dim <= 0:
            raise ValueError(f"joint_obs_dim must be positive, got {joint_obs_dim}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.joint_obs_dim: int = int(joint_obs_dim)
        self.hidden_size: int = int(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(self.joint_obs_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, joint_obs: Tensor) -> Tensor:
        """Return ``[B, 1]`` value estimates for ``joint_obs``."""
        if joint_obs.dim() != 2:
            raise ValueError(f"ValueHead expects [B, joint_obs_dim], got {tuple(joint_obs.shape)}")
        if joint_obs.shape[-1] != self.joint_obs_dim:
            raise ValueError(
                f"ValueHead joint_obs_dim mismatch: expected {self.joint_obs_dim}, got {joint_obs.shape[-1]}"
            )
        return self.net(joint_obs)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class KivskiActorCritic(nn.Module):
    """Recurrent MAPPO actor-critic with TarMAC comms.

    The model exposes two public entry points:

    * :meth:`act` -- per-tick inference. Takes the current per-agent obs,
      the prior hidden state, and the received-comm tensor; returns
      sampled actions, joint log-probs, the centralised value, the new
      hidden state, and the outgoing comm payload (signature, value, gate
      open mask).
    * :meth:`evaluate` -- training-time re-evaluation. Takes the same
      inputs **plus** the recorded actions and returns new log-probs and
      summed entropy alongside the value. This is what the PPO update
      calls in its minibatch loop.

    Constructor args mirror :class:`kivski_sim.config.MLConfig` fields so
    the factory can pass values straight through.
    """

    def __init__(
        self,
        obs_dim: int,
        joint_obs_dim: int,
        action_dims: Sequence[int],
        hidden_size: int = 256,
        comm_signature_dim: int = 32,
        comm_value_dim: int = 32,
        comm_attention_heads: int = 4,
        gumbel_temp: float = 1.0,
        gru_layers: int = 1,
        actor_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.obs_dim: int = int(obs_dim)
        self.joint_obs_dim: int = int(joint_obs_dim)
        self.action_dims: list[int] = [int(a) for a in action_dims]
        self.hidden_size: int = int(hidden_size)
        self.comm_signature_dim: int = int(comm_signature_dim)
        self.comm_value_dim: int = int(comm_value_dim)
        self.comm_attention_heads: int = int(comm_attention_heads)
        self.gumbel_temp: float = float(gumbel_temp)
        self.gru_layers: int = int(gru_layers)
        self.actor_embedding_dim: int = int(actor_embedding_dim)

        # ---- sub-modules ----
        self.observation_encoder = ObservationEncoder(self.obs_dim, self.hidden_size)
        # The GRU input is (obs_encoding | aggregated received comm).
        self.recurrent_core = RecurrentCore(
            input_dim=self.hidden_size + self.comm_value_dim,
            hidden_size=self.hidden_size,
            num_layers=self.gru_layers,
        )
        # Comm: encoder consumes the actor hidden (post-GRU), gate looks at
        # the same hidden, attention aggregates received teammate values.
        self.comm_encoder = CommEncoder(
            input_dim=self.hidden_size,
            signature_dim=self.comm_signature_dim,
            value_dim=self.comm_value_dim,
        )
        self.comm_attention = CommAttention(
            signature_dim=self.comm_signature_dim,
            value_dim=self.comm_value_dim,
            num_heads=self.comm_attention_heads,
        )
        self.comm_gate = CommGate(input_dim=self.hidden_size)
        # Actor + critic heads.
        self.actor_heads = ActorHeads(
            hidden_size=self.hidden_size,
            action_dims=self.action_dims,
            embedding_dim=self.actor_embedding_dim,
        )
        self.value_head = ValueHead(self.joint_obs_dim, self.hidden_size)

    # --------------------------------------------------------------
    # Lifecycle helpers
    # --------------------------------------------------------------

    def initial_hidden_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> Tensor:
        """Return a zero hidden state ``[num_layers, batch_size, hidden_size]``."""
        return self.recurrent_core.initial_hidden(batch_size, device=device)

    @property
    def device(self) -> torch.device:
        """Device the model's parameters live on (best-effort)."""
        try:
            return next(self.parameters()).device
        except StopIteration:  # pragma: no cover - no params
            return torch.device("cpu")

    # --------------------------------------------------------------
    # Internal forward chain (shared by act / evaluate)
    # --------------------------------------------------------------

    def _forward_core(
        self,
        obs: Tensor,
        hidden_state: Tensor,
        received_comm: Tensor,
        masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode observation + received comm, run the GRU one step.

        Args:
            obs: ``[B, obs_dim]``.
            hidden_state: ``[num_layers, B, hidden_size]``.
            received_comm: ``[B, comm_value_dim]`` -- the aggregated comm
                vector for this agent this tick.
            masks: Optional ``[B]`` reset mask (0 = reset hidden).

        Returns:
            ``(actor_hidden, new_hidden_state)``.
        """
        obs_enc = self.observation_encoder(obs)
        gru_in = torch.cat([obs_enc, received_comm], dim=-1)
        actor_hidden, new_hidden = self.recurrent_core(gru_in, hidden_state, masks=masks)
        return actor_hidden, new_hidden

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    def act(
        self,
        obs: Tensor,
        hidden_state: Tensor,
        received_comm: Tensor,
        joint_obs: Tensor | None = None,
        masks: Tensor | None = None,
        deterministic: bool = False,
    ) -> dict[str, Tensor]:
        """One inference step. Returns a dict of named tensors.

        Args:
            obs: ``[B, obs_dim]`` per-agent observation.
            hidden_state: ``[num_layers, B, hidden_size]`` prior GRU hidden.
            received_comm: ``[B, comm_value_dim]`` aggregated incoming comm.
            joint_obs: optional ``[B, joint_obs_dim]`` for the centralised
                value head; if ``None`` the value tensor is omitted from
                the returned dict (useful for non-training inference).
            masks: optional ``[B]`` hidden-state reset mask.
            deterministic: argmax sampling if True.

        Returns:
            Dict with keys:

            * ``actions``      ``[B, num_heads]`` int64
            * ``log_probs``    ``[B]`` summed joint log prob
            * ``entropy``      ``[B]`` summed entropy across heads
            * ``new_hidden``   ``[num_layers, B, hidden_size]``
            * ``comm_signature`` ``[B, comm_signature_dim]``
            * ``comm_value``   ``[B, comm_value_dim]`` (raw value, pre-gate)
            * ``comm_gate``    ``[B, 1]`` open mask in ``[0, 1]``
            * ``comm_gate_logits`` ``[B, 1]``
            * ``comm_payload`` ``[B, comm_value_dim]`` -- gated value to broadcast
            * ``value`` (optional) ``[B, 1]`` if ``joint_obs`` is provided
        """
        actor_hidden, new_hidden = self._forward_core(obs, hidden_state, received_comm, masks=masks)
        actions, log_probs, entropy = self.actor_heads.sample(actor_hidden, deterministic=deterministic)
        sig, val = self.comm_encoder(actor_hidden)
        gate_logits, gate_open = self.comm_gate(actor_hidden, temperature=self.gumbel_temp)
        payload = val * gate_open
        out: dict[str, Tensor] = {
            "actions": actions,
            "log_probs": log_probs,
            "entropy": entropy,
            "new_hidden": new_hidden,
            "comm_signature": sig,
            "comm_value": val,
            "comm_gate": gate_open,
            "comm_gate_logits": gate_logits,
            "comm_payload": payload,
        }
        if joint_obs is not None:
            out["value"] = self.value_head(joint_obs)
        return out

    def evaluate(
        self,
        obs: Tensor,
        hidden_state: Tensor,
        received_comm: Tensor,
        prev_actions: Tensor,
        joint_obs: Tensor,
        masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Re-evaluate logged actions for the PPO update.

        Returns a dict with ``log_probs``, ``entropy``, ``value``, and the
        new hidden state for completeness.
        """
        actor_hidden, new_hidden = self._forward_core(obs, hidden_state, received_comm, masks=masks)
        log_probs, entropy = self.actor_heads.evaluate(actor_hidden, prev_actions)
        value = self.value_head(joint_obs)
        return {
            "log_probs": log_probs,
            "entropy": entropy,
            "value": value,
            "new_hidden": new_hidden,
        }
