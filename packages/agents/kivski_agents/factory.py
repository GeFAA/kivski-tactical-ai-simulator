"""Constructor helpers that turn :class:`KivskiConfig` into model + trainer.

This is the single place that the rest of the codebase calls when it wants
"the standard recurrent-MAPPO actor-critic for this config". Keeping the
construction logic isolated here means we can tune defaults (hidden size,
comm width, gru depth) in :class:`kivski_sim.config.MLConfig` without
chasing through every call site.

Both helpers accept an explicit ``device`` so the trainer can pin to CUDA
when available while tests stay on CPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from kivski_agents.mappo import MAPPOTrainer
from kivski_agents.networks.actor_critic import KivskiActorCritic
from kivski_sim.config import KivskiConfig, MLConfig


__all__ = [
    "build_model",
    "build_trainer",
    "default_action_dims",
    "infer_joint_obs_dim",
]


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def default_action_dims(team_size: int) -> list[int]:
    """Return the standard MAPPO MultiDiscrete action dims for a given team size.

    Mirrors :class:`kivski_sim.env.KivskiParallelEnv` exactly:
    ``[move=9, micro=6, comm=9, buy=8, aim_target=2*team_size+1]``.
    """
    if team_size <= 0:
        raise ValueError(f"team_size must be positive, got {team_size}")
    return [9, 6, 9, 8, 2 * int(team_size) + 1]


def infer_joint_obs_dim(obs_dim: int, team_size: int, global_features: int = 0) -> int:
    """Joint-observation width for the centralised critic.

    The default is ``team_size * obs_dim`` (concatenated teammate views).
    Extra global features (e.g. one-hot of bomb state) can be added via
    ``global_features`` if the trainer chooses to inject them.
    """
    if obs_dim <= 0 or team_size <= 0:
        raise ValueError("obs_dim and team_size must be positive")
    return int(team_size) * int(obs_dim) + int(global_features)


def build_model(
    cfg: KivskiConfig,
    obs_dim: int,
    joint_obs_dim: int,
    action_dims: Sequence[int],
    device: torch.device | str = "cpu",
) -> KivskiActorCritic:
    """Construct :class:`KivskiActorCritic` from :class:`KivskiConfig`.

    Args:
        cfg: Full config; the relevant subsection is :attr:`KivskiConfig.ml`.
        obs_dim: Per-agent observation length (typically from
            :func:`kivski_sim.obs_decoder.get_observation_dim`).
        joint_obs_dim: Centralised critic input width
            (see :func:`infer_joint_obs_dim`).
        action_dims: Per-head category counts; usually
            :func:`default_action_dims` of the team size.
        device: ``"cpu"`` / ``"cuda"`` / etc.
    """
    ml: MLConfig = cfg.ml
    # comm_embedding_dim from config is split evenly between signature
    # (key) and value width by default. We round the signature down to the
    # nearest multiple of the head count so :class:`CommAttention` accepts
    # it. The value width is computed symmetrically.
    heads = max(1, int(ml.comm_attention_heads))
    total = max(2 * heads, int(ml.comm_embedding_dim))
    half = total // 2
    # Round each side up to a multiple of ``heads``.
    sig_dim = max(heads, ((half + heads - 1) // heads) * heads)
    val_dim = sig_dim  # symmetric default; trainer can override later

    model = KivskiActorCritic(
        obs_dim=int(obs_dim),
        joint_obs_dim=int(joint_obs_dim),
        action_dims=list(action_dims),
        hidden_size=int(ml.hidden_size),
        comm_signature_dim=int(sig_dim),
        comm_value_dim=int(val_dim),
        comm_attention_heads=heads,
        gumbel_temp=float(ml.gumbel_temperature),
        gru_layers=int(ml.gru_layers),
        actor_embedding_dim=32,
    )
    model.to(torch.device(device))
    return model


def build_trainer(
    model: KivskiActorCritic,
    cfg: KivskiConfig,
    device: torch.device | str = "cpu",
) -> MAPPOTrainer:
    """Wrap an existing model with a :class:`MAPPOTrainer` using ``cfg.ml``."""
    return MAPPOTrainer(model=model, cfg=cfg.ml, device=device)
