"""Behavior-cloning loss for KivskiActorCritic.

Continuous move: Gaussian NLL of the demo action under the policy's (mean, sigma).
Discrete heads: per-head cross-entropy on the demo categorical action.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def bc_loss(
    net_out: dict[str, object],
    demo_move: torch.Tensor,
    demo_discrete: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Compute total BC loss.

    Args:
        net_out: Must contain:
            "move_mean"   -> [B, 2]
            "move_log_std" -> [B, 2]
            "discrete_logits" -> list of 4 tensors [B, K_i]
        demo_move:     [B, 2] float, in [-1, 1]
        demo_discrete: [B, 4] int64

    Returns:
        (total_loss, breakdown) -- breakdown carries scalar floats.
    """
    move_mean = net_out["move_mean"]
    move_log_std = net_out["move_log_std"]
    discrete_logits = net_out["discrete_logits"]

    # Gaussian NLL per dim: 0.5 * ((x - mu)^2 / sigma^2 + log(2*pi*sigma^2))
    var = torch.exp(2.0 * move_log_std)
    sq_err = (demo_move - move_mean) ** 2
    move_nll = 0.5 * (sq_err / var + 2.0 * move_log_std + math.log(2.0 * math.pi))
    move_nll = move_nll.sum(dim=-1).mean()

    ce_per_head: list[float] = []
    discrete_ce_total = torch.zeros((), device=demo_move.device)
    for head_idx, logits in enumerate(discrete_logits):
        ce = F.cross_entropy(logits, demo_discrete[:, head_idx])
        discrete_ce_total = discrete_ce_total + ce
        ce_per_head.append(float(ce.detach().cpu().item()))

    total = move_nll + discrete_ce_total
    breakdown = {
        "move_nll": float(move_nll.detach().cpu().item()),
        "discrete_ce_per_head": ce_per_head,
    }
    return total, breakdown
