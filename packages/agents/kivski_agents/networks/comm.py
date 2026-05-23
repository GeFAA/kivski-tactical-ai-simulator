"""TarMAC-style learned communication for multi-agent policies.

This module implements the three primitives the Kivski actor-critic needs to
let teammates exchange continuous messages:

* :class:`CommEncoder` -- per-agent head that maps a "thought vector"
  (typically the GRU hidden state of the agent) into a *(signature, value)*
  pair. The signature is the attention key the *receiver* reads against,
  while the value carries the actual payload that gets aggregated.
* :class:`CommAttention` -- multi-head attention over the bank of teammate
  signatures / values. Receivers query with their own signature, attention
  weights are softmaxed across teammates with an explicit mask so that dead
  teammates or teammates that did not broadcast this tick are ignored.
* :class:`CommGate` -- a Gumbel-Sigmoid binary gate that lets the sender
  decide whether to broadcast at all. Hard at evaluation time (0/1), soft
  during training so gradients can flow.

The dimensionality is deliberately small. The trainer treats the value
vector as the comm payload that flows back into the env's
:meth:`KivskiParallelEnv.step_with_comms`, so the value width must match the
env's expectation. The default 32 is a good starting point but can be
overridden via :class:`kivski_sim.config.MLConfig.comm_embedding_dim`.

All modules in this file are deterministic given a seeded ``torch`` RNG and
do not perform any per-instance state mutation. They are safe to put behind
``torch.no_grad()`` for inference and behind ``torch.compile`` for
performance experiments.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["CommAttention", "CommEncoder", "CommGate"]


# ---------------------------------------------------------------------------
# Sender side: encoder + gate
# ---------------------------------------------------------------------------


class CommEncoder(nn.Module):
    """Map a per-agent thought vector to ``(signature, value)`` tensors.

    The encoder is two small Linear heads on top of the input vector. The
    signature is used as the attention key by *receivers*, the value is the
    actual content. Both are L2-normalised on the signature side to keep
    attention logits in a sane range; the value is left raw so the network
    can encode magnitude information.

    Args:
        input_dim: Dimensionality of the thought vector (typically the GRU
            hidden size of the actor).
        signature_dim: Size of the attention key vector. Must be divisible
            by ``num_heads`` in :class:`CommAttention` if that module is
            wired up downstream.
        value_dim: Size of the message payload vector that receivers
            aggregate. The env's :meth:`step_with_comms` uses this width
            verbatim, so changing it requires the env / runner to agree.
    """

    def __init__(
        self,
        input_dim: int,
        signature_dim: int = 32,
        value_dim: int = 32,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if signature_dim <= 0:
            raise ValueError(f"signature_dim must be positive, got {signature_dim}")
        if value_dim <= 0:
            raise ValueError(f"value_dim must be positive, got {value_dim}")

        self.input_dim: int = int(input_dim)
        self.signature_dim: int = int(signature_dim)
        self.value_dim: int = int(value_dim)

        # Small two-layer trunk gives the heads a shared, slightly non-linear
        # representation without inflating parameter count.
        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
        )
        self.signature_head = nn.Linear(self.input_dim, self.signature_dim)
        self.value_head = nn.Linear(self.input_dim, self.value_dim)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Encode ``hidden`` into ``(signature, value)`` of shape ``[B, D]``.

        Args:
            hidden: Input tensor of shape ``[B, input_dim]``.

        Returns:
            A two-tuple ``(signature, value)``:
                * ``signature``: L2-normalised ``[B, signature_dim]`` tensor.
                * ``value``:    raw   ``[B, value_dim]`` tensor.
        """
        if hidden.dim() != 2:
            raise ValueError(f"CommEncoder expects [B, input_dim], got shape {tuple(hidden.shape)}")
        if hidden.shape[-1] != self.input_dim:
            raise ValueError(
                f"CommEncoder input dim mismatch: expected {self.input_dim}, got {hidden.shape[-1]}"
            )

        h = self.trunk(hidden)
        sig = self.signature_head(h)
        val = self.value_head(h)
        # L2-normalise the signature so attention dot-products live in a
        # bounded range and softmax behaves well across training.
        sig = F.normalize(sig, p=2.0, dim=-1, eps=1e-6)
        return sig, val


class CommGate(nn.Module):
    """Differentiable binary broadcast gate.

    The gate produces a single logit per agent. At training time it returns
    a soft Gumbel-Sigmoid sample so gradients can flow through the (0..1)
    multiplicative mask applied to outgoing messages. At evaluation time
    it returns a hard 0/1 decision by thresholding the sigmoid at 0.5.

    Args:
        input_dim: Dimensionality of the input thought vector.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self.input_dim: int = int(input_dim)
        self.gate = nn.Linear(self.input_dim, 1)

    def forward(
        self,
        hidden: Tensor,
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """Compute gate logits and the (possibly soft) open mask.

        Args:
            hidden: Input tensor of shape ``[B, input_dim]``.
            temperature: Gumbel-Sigmoid temperature (smaller -> harder).

        Returns:
            ``(gate_logits, gate_open)``:
                * ``gate_logits``: raw linear-layer output, shape ``[B, 1]``.
                * ``gate_open``: in train mode a Gumbel-Sigmoid sample in
                  ``(0, 1)``; in eval mode a hard ``{0, 1}`` indicator.
        """
        if hidden.dim() != 2:
            raise ValueError(f"CommGate expects [B, input_dim], got shape {tuple(hidden.shape)}")
        gate_logits = self.gate(hidden)
        if self.training:
            # Differentiable Bernoulli via Gumbel-Sigmoid.
            #   y = sigmoid((g_logit + (g1 - g2)) / tau)
            # where g1, g2 ~ Gumbel(0, 1). This is a numerically stable form
            # that does not require log/exp of the input.
            tau = max(float(temperature), 1e-3)
            u1 = torch.rand_like(gate_logits).clamp_(1e-6, 1.0 - 1e-6)
            u2 = torch.rand_like(gate_logits).clamp_(1e-6, 1.0 - 1e-6)
            g1 = -torch.log(-torch.log(u1))
            g2 = -torch.log(-torch.log(u2))
            gate_open = torch.sigmoid((gate_logits + (g1 - g2)) / tau)
        else:
            gate_open = (torch.sigmoid(gate_logits) > 0.5).to(gate_logits.dtype)
        return gate_logits, gate_open


# ---------------------------------------------------------------------------
# Receiver side: multi-head attention
# ---------------------------------------------------------------------------


class CommAttention(nn.Module):
    """Multi-head attention over teammate messages.

    Receivers form a query from their own signature, dot-product it against
    every teammate's signature, mask out dead / silent teammates, softmax
    across the remaining ones, and aggregate the corresponding values.

    The number of heads partitions the *signature* and *value* dimensions
    in parallel -- both must be divisible by ``num_heads``. The aggregated
    output has the same width as the input value vector and is returned in
    full (no extra projection layer here -- the caller can add one).

    Args:
        signature_dim: Width of the per-head signature (key) vector.
        value_dim: Width of the per-head value vector.
        num_heads: Number of parallel attention heads.
    """

    def __init__(
        self,
        signature_dim: int,
        value_dim: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if signature_dim <= 0:
            raise ValueError(f"signature_dim must be positive, got {signature_dim}")
        if value_dim <= 0:
            raise ValueError(f"value_dim must be positive, got {value_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if signature_dim % num_heads != 0:
            raise ValueError(f"signature_dim ({signature_dim}) must be divisible by num_heads ({num_heads})")
        if value_dim % num_heads != 0:
            raise ValueError(f"value_dim ({value_dim}) must be divisible by num_heads ({num_heads})")

        self.signature_dim: int = int(signature_dim)
        self.value_dim: int = int(value_dim)
        self.num_heads: int = int(num_heads)
        self.head_sig_dim: int = self.signature_dim // self.num_heads
        self.head_val_dim: int = self.value_dim // self.num_heads
        self._scale: float = 1.0 / math.sqrt(max(self.head_sig_dim, 1))

        # Output mixing layer so heads can be linearly recombined.
        self.out_proj = nn.Linear(self.value_dim, self.value_dim)

    def forward(
        self,
        query: Tensor,
        sigs: Tensor,
        vals: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run multi-head attention.

        Args:
            query: Shape ``[B, signature_dim]`` -- this agent's read query.
            sigs:  Shape ``[B, N, signature_dim]`` -- teammate signatures.
            vals:  Shape ``[B, N, value_dim]``     -- teammate values.
            mask:  Shape ``[B, N]`` with 1 = teammate is alive AND sent a
                message this tick, 0 = ignore.

        Returns:
            ``(aggregated, weights)``:
                * ``aggregated``: ``[B, value_dim]`` softmax-weighted sum of
                  ``vals``. Rows where the entire mask is zero are returned
                  as the exact zero vector (no spurious uniform attention).
                * ``weights``: ``[B, N]`` mean attention weights across heads
                  -- useful for visualisation. For rows with an all-zero
                  mask, the weight tensor is all zeros.
        """
        if query.dim() != 2 or query.shape[-1] != self.signature_dim:
            raise ValueError(f"query must be [B, {self.signature_dim}], got {tuple(query.shape)}")
        if sigs.dim() != 3 or sigs.shape[-1] != self.signature_dim:
            raise ValueError(f"sigs must be [B, N, {self.signature_dim}], got {tuple(sigs.shape)}")
        if vals.dim() != 3 or vals.shape[-1] != self.value_dim:
            raise ValueError(f"vals must be [B, N, {self.value_dim}], got {tuple(vals.shape)}")
        if mask.dim() != 2:
            raise ValueError(f"mask must be [B, N], got {tuple(mask.shape)}")
        if sigs.shape[0] != query.shape[0] or sigs.shape[1] != mask.shape[1]:
            raise ValueError(
                "Batch / teammate counts disagree across query/sigs/vals/mask: "
                f"query={tuple(query.shape)}, sigs={tuple(sigs.shape)}, mask={tuple(mask.shape)}"
            )

        b, n, _ = sigs.shape
        h, d_s, d_v = self.num_heads, self.head_sig_dim, self.head_val_dim

        # Reshape to per-head layouts:
        #   q: [B, H, 1, d_s]
        #   k: [B, H, N, d_s]
        #   v: [B, H, N, d_v]
        q = query.view(b, h, 1, d_s)
        k = sigs.view(b, n, h, d_s).permute(0, 2, 1, 3).contiguous()
        v = vals.view(b, n, h, d_v).permute(0, 2, 1, 3).contiguous()

        # Attention logits per head: [B, H, 1, N] -> squeeze -> [B, H, N]
        logits = torch.matmul(q, k.transpose(-2, -1)).squeeze(-2) * self._scale  # [B, H, N]

        # Mask: 1 = valid, 0 = ignore. We force masked logits to a very
        # negative number so they vanish under softmax.
        mask_bool = mask.bool()
        # Broadcast across heads: [B, 1, N] -> [B, H, N]
        mask_b = mask_bool.unsqueeze(1).expand(b, h, n)
        very_neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask_b, very_neg)

        # Detect rows where the mask is all-zero. Those rows would otherwise
        # produce NaN under softmax. We softmax with a safe replacement and
        # zero them back out afterwards.
        any_valid = mask_bool.any(dim=1)  # [B]
        safe_logits = torch.where(
            any_valid.view(b, 1, 1).expand(b, h, n),
            logits,
            torch.zeros_like(logits),
        )
        weights = F.softmax(safe_logits, dim=-1)  # [B, H, N]
        # Zero out rows with no valid teammates so the aggregation is the
        # exact zero vector instead of an averaged-over-nothing artefact.
        weights = weights * any_valid.view(b, 1, 1).to(weights.dtype)

        # Weighted sum of values per head: [B, H, N] x [B, H, N, d_v] -> [B, H, d_v]
        agg_per_head = torch.einsum("bhn,bhnd->bhd", weights, v)
        aggregated = agg_per_head.reshape(b, h * d_v)
        aggregated = self.out_proj(aggregated)

        # Average attention weights across heads for downstream visualisation.
        weights_mean = weights.mean(dim=1)  # [B, N]
        return aggregated, weights_mean
