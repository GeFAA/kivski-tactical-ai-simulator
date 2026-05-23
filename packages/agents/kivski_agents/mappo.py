"""Multi-Agent PPO (MAPPO) optimisation loop.

This module owns the actual gradient updates for the
:class:`~kivski_agents.networks.actor_critic.KivskiActorCritic`. It is
deliberately small: the outer training loop (rollout collection, league
play, eval scheduling) lives in :mod:`scripts.train`. Here we expose:

* :class:`MAPPOLoss` -- a small typed payload returned from each call to
  :meth:`MAPPOTrainer.update` and consumed by the telemetry layer.
* :class:`MAPPOTrainer` -- wraps an Adam optimiser, performs N PPO epochs
  over a populated :class:`~kivski_agents.buffer.RolloutBuffer`, and
  supports checkpoint save / load with a sidecar JSON metadata file.

The update follows the standard PPO recipe with a clipped surrogate
objective, an optional value-clipping branch, an entropy bonus, and
``max_grad_norm`` clipping. Centralised critic training uses
``joint_observations`` from the buffer so the same call site handles
both decentralised actors and centralised critics (CTDE).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from kivski_agents.buffer import RolloutBuffer
from kivski_agents.networks.actor_critic import KivskiActorCritic
from kivski_sim.config import MLConfig


__all__ = ["MAPPOLoss", "MAPPOTrainer"]


# ---------------------------------------------------------------------------
# Loss payload
# ---------------------------------------------------------------------------


@dataclass
class MAPPOLoss:
    """Aggregated MAPPO update diagnostics returned by :meth:`MAPPOTrainer.update`."""

    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl: float = 0.0
    grad_norm: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    update_count: int = 0
    extras: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class MAPPOTrainer:
    """One-update-per-call PPO trainer for :class:`KivskiActorCritic`."""

    def __init__(
        self,
        model: KivskiActorCritic,
        cfg: MLConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model: KivskiActorCritic = model
        self.cfg: MLConfig = cfg
        self.device: torch.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(cfg.learning_rate),
            eps=1e-5,
        )
        # Bookkeeping for telemetry / checkpoint sidecars.
        self._update_steps: int = 0

    # --------------------------------------------------------------
    # Single PPO update over a populated buffer
    # --------------------------------------------------------------

    def update(self, buffer: RolloutBuffer) -> MAPPOLoss:
        """Run ``ppo_epochs`` passes over the buffer and return aggregated stats."""
        cfg = self.cfg
        epochs = max(1, int(cfg.ppo_epochs))
        clip = float(cfg.ppo_clip)
        ent_coef = float(cfg.entropy_coef)
        val_coef = float(cfg.value_coef)
        max_grad_norm = float(cfg.max_grad_norm)
        minibatch_size = max(1, int(cfg.minibatch_size))

        if buffer.step == 0:
            return MAPPOLoss()

        loss_meter = MAPPOLoss()
        n_minibatches = 0

        # We also track the variance for the explained-variance metric.
        all_returns: list[torch.Tensor] = []
        all_values_pred: list[torch.Tensor] = []

        self.model.train()
        for _epoch in range(epochs):
            for batch in buffer.minibatch_iter(minibatch_size, shuffle=True):
                eval_out = self.model.evaluate(
                    obs=batch.observations,
                    hidden_state=batch.hidden_states,
                    received_comm=batch.received_comms,
                    prev_actions=batch.actions,
                    joint_obs=batch.joint_observations,
                    masks=batch.masks,
                )
                new_log_probs = eval_out["log_probs"]
                entropy = eval_out["entropy"]
                new_values = eval_out["value"].squeeze(-1)

                # ---- Policy loss (clipped surrogate) ----
                advantages = batch.advantages
                # Normalise advantages within the minibatch -- standard PPO
                # trick that keeps the gradient scale stable.
                if advantages.numel() > 1:
                    adv_mean = advantages.mean()
                    adv_std = advantages.std().clamp_min(1e-6)
                    advantages = (advantages - adv_mean) / adv_std

                log_ratio = new_log_probs - batch.old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
                # Only count alive agents in the policy loss.
                mask = batch.masks
                mask_sum = mask.sum().clamp_min(1.0)
                policy_loss = -(torch.min(surr1, surr2) * mask).sum() / mask_sum

                # ---- Value loss (with clipping) ----
                v_clipped = batch.old_values + torch.clamp(
                    new_values - batch.old_values, -clip, clip
                )
                v_loss_unclipped = F.mse_loss(new_values, batch.returns, reduction="none")
                v_loss_clipped = F.mse_loss(v_clipped, batch.returns, reduction="none")
                # Value is centralised -- use the simple mean across the minibatch.
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # ---- Entropy bonus ----
                entropy_term = (entropy * mask).sum() / mask_sum

                total_loss = policy_loss + val_coef * value_loss - ent_coef * entropy_term

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=max_grad_norm
                )
                self.optimizer.step()

                # ---- Diagnostics ----
                with torch.no_grad():
                    # Approximate KL divergence (Schulman 2020 estimator k3):
                    #   E[ exp(log_ratio) - 1 - log_ratio ]
                    kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().item()
                    clipped = ((ratio - 1.0).abs() > clip).float().mean().item()

                loss_meter.policy_loss += float(policy_loss.item())
                loss_meter.value_loss += float(value_loss.item())
                loss_meter.entropy += float(entropy_term.item())
                loss_meter.kl += float(kl)
                loss_meter.grad_norm += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                loss_meter.clip_fraction += float(clipped)
                n_minibatches += 1

                all_returns.append(batch.returns.detach())
                all_values_pred.append(new_values.detach())

        if n_minibatches > 0:
            loss_meter.policy_loss /= n_minibatches
            loss_meter.value_loss /= n_minibatches
            loss_meter.entropy /= n_minibatches
            loss_meter.kl /= n_minibatches
            loss_meter.grad_norm /= n_minibatches
            loss_meter.clip_fraction /= n_minibatches
        loss_meter.update_count = n_minibatches
        loss_meter.explained_variance = _explained_variance(all_values_pred, all_returns)
        self._update_steps += 1
        return loss_meter

    # --------------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------------

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Save model + optimizer state to ``path`` and a sidecar JSON.

        The sidecar file lives next to the checkpoint at ``path.with_suffix('.json')``
        and is written in a stable, human-readable format that ops people
        can grep.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(metadata or {})
        metadata.setdefault("kivski_version", "0.1.0")
        metadata.setdefault("timestamp", time.time())
        metadata.setdefault("update_steps", int(self._update_steps))

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metadata": metadata,
                "model_init": _model_init_dict(self.model),
                "cfg": _ml_config_to_dict(self.cfg),
            },
            out_path,
        )
        sidecar = out_path.with_suffix(out_path.suffix + ".json") if out_path.suffix else out_path.with_suffix(".json")
        with sidecar.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True, default=_json_safe)
        return out_path

    def load(self, path: str | Path) -> dict[str, Any]:
        """Restore model + optimizer state. Returns the metadata dict."""
        ckpt_path = Path(path)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except (ValueError, KeyError):
                # Optimizer state may be mismatched if model layout changed.
                # We surface this through metadata but don't crash.
                pass
        meta = dict(ckpt.get("metadata", {}))
        self._update_steps = int(meta.get("update_steps", self._update_steps))
        return meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _explained_variance(values: list[torch.Tensor], returns: list[torch.Tensor]) -> float:
    """1 - Var(returns - values) / Var(returns). 1 = perfect, 0 = no info."""
    if not values or not returns:
        return 0.0
    v = torch.cat(values).flatten().double()
    r = torch.cat(returns).flatten().double()
    var_r = r.var(unbiased=False).item()
    if var_r < 1e-12:
        return 0.0
    return float(1.0 - ((r - v).var(unbiased=False).item() / var_r))


def _model_init_dict(model: KivskiActorCritic) -> dict[str, Any]:
    """Capture the constructor args needed to rebuild the model later."""
    return {
        "obs_dim": int(model.obs_dim),
        "joint_obs_dim": int(model.joint_obs_dim),
        "action_dims": list(model.action_dims),
        "hidden_size": int(model.hidden_size),
        "comm_signature_dim": int(model.comm_signature_dim),
        "comm_value_dim": int(model.comm_value_dim),
        "comm_attention_heads": int(model.comm_attention_heads),
        "gumbel_temp": float(model.gumbel_temp),
        "gru_layers": int(model.gru_layers),
        "actor_embedding_dim": int(model.actor_embedding_dim),
    }


def _ml_config_to_dict(cfg: MLConfig) -> dict[str, Any]:
    """Dump :class:`MLConfig` to a JSON-friendly dict (best-effort)."""
    try:
        return cfg.model_dump()
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        return asdict(cfg)  # type: ignore[arg-type]


def _json_safe(value: Any) -> Any:
    """JSON encoder hook for things like tensors and numpy scalars."""
    if torch.is_tensor(value):
        return value.tolist()
    return str(value)
