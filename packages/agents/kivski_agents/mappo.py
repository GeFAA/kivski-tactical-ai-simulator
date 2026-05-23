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

import contextlib
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from kivski_sim.config import MLConfig

from kivski_agents.buffer import RolloutBuffer
from kivski_agents.networks.actor_critic import KivskiActorCritic
from kivski_agents.persistence.checkpoint_compat import (
    CheckpointIncompatibleError,
    build_compat_metadata,
    check_compat,
    write_sidecar_metadata,
)

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

        # ---- Mixed precision (autocast bf16) ----
        # bfloat16 is numerically stable on Ampere+ (compute 8.0+) without
        # needing a GradScaler. We enable it automatically when running on
        # a CUDA device that supports bf16. Falls back transparently to
        # fp32 on CPU or on older GPUs.
        self.amp_enabled: bool = False
        self.amp_dtype: torch.dtype = torch.float32
        if self.device.type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    self.amp_enabled = True
                    self.amp_dtype = torch.bfloat16
            except Exception:
                # Defensive: any probe failure means we stay in fp32.
                self.amp_enabled = False

        # ---- Mixed precision (autocast bf16) ----
        # bfloat16 is numerically stable on Ampere+ (compute 8.0+) without
        # needing a GradScaler. We enable it automatically when running on
        # a CUDA device that supports bf16. Falls back transparently to
        # fp32 on CPU or on older GPUs.
        self.amp_enabled: bool = False
        self.amp_dtype: torch.dtype = torch.float32
        if self.device.type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    self.amp_enabled = True
                    self.amp_dtype = torch.bfloat16
            except Exception:
                # Defensive: any probe failure means we stay in fp32.
                self.amp_enabled = False

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

        # Autocast context for mixed-precision forward+loss on CUDA. bf16
        # has the same exponent range as fp32 so we can skip the GradScaler
        # entirely (which fp16 would need to avoid underflow).
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)
            if self.amp_enabled
            else nullcontext()
        )

        self.model.train()
        for _epoch in range(epochs):
            for batch in buffer.minibatch_iter(minibatch_size, shuffle=True):
                with amp_ctx:
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
                    v_clipped = batch.old_values + torch.clamp(new_values - batch.old_values, -clip, clip)
                    v_loss_unclipped = F.mse_loss(new_values, batch.returns, reduction="none")
                    v_loss_clipped = F.mse_loss(v_clipped, batch.returns, reduction="none")
                    # Value is centralised -- use the simple mean across the minibatch.
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    # ---- Entropy bonus ----
                    entropy_term = (entropy * mask).sum() / mask_sum

                    total_loss = policy_loss + val_coef * value_loss - ent_coef * entropy_term

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
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

    def save(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
        env_shape: dict[str, Any] | None = None,
    ) -> Path:
        """Save model + optimizer state to ``path`` and a sidecar JSON.

        The blob written to ``path`` always includes a compat metadata
        section (model arch + env shape) so :meth:`load` -- or any other
        consumer like :class:`PolicyBundle.from_checkpoint` -- can refuse
        to load it into a differently-shaped model instead of crashing
        the trainer with a ``size mismatch`` ``RuntimeError`` (which the
        watchdog used to interpret as a transient failure and respawn
        forever).

        The sidecar file lives next to the checkpoint at
        ``path.with_suffix('.json')`` and is written in a stable,
        human-readable format that ops people can grep.

        Args:
            path: Destination checkpoint path. Parent dir is created.
            metadata: Caller-supplied extras (episode, run_name, score,
                ...). Merged into the saved blob's ``metadata`` field.
            env_shape: Optional ``{obs_dim, n_heads, team_size}`` blob
                from the live vec_env. When ``None`` only ``team_size``
                (inferred from ``action_dims[4]``) is recorded.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Build the strict compat block first so callers can't silently
        # overwrite ``model_arch`` / ``env_shape`` from their own metadata.
        compat = build_compat_metadata(
            model_arch=_model_init_dict(self.model),
            env_shape=env_shape or _env_shape_from_model(self.model),
            extra=None,
        )
        # User-supplied metadata is layered *under* the compat block, so
        # arbitrary keys (episode, run_name, opponent, score, ...) are
        # preserved without ever overriding ``model_arch`` / ``env_shape``.
        meta_out: dict[str, Any] = dict(metadata or {})
        meta_out.setdefault("kivski_version", compat["kivski_version"])
        meta_out.setdefault("timestamp", compat["timestamp"])
        meta_out.setdefault("update_steps", int(self._update_steps))
        # Strict block overrides any stale arch/env the caller may have set.
        meta_out["schema_version"] = compat["schema_version"]
        meta_out["model_arch"] = compat["model_arch"]
        meta_out["env_shape"] = compat["env_shape"]

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metadata": meta_out,
                "model_init": _model_init_dict(self.model),
                "cfg": _ml_config_to_dict(self.cfg),
            },
            out_path,
        )
        # Sidecar JSON (torch-free read path).
        sidecar_payload: dict[str, Any] = dict(meta_out)
        # tensors / numpy scalars shouldn't appear here but be defensive.
        for k, v in list(sidecar_payload.items()):
            if torch.is_tensor(v):
                sidecar_payload[k] = v.tolist()
        write_sidecar_metadata(out_path, sidecar_payload)
        return out_path

    def load(
        self,
        path: str | Path,
        env_shape: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Restore model + optimizer state. Returns the metadata dict.

        Validates the saved compat metadata against the currently-built
        model and ``env_shape``. On any mismatch raises
        :class:`CheckpointIncompatibleError` -- the API watchdog catches
        this specifically and refuses to auto-restart with the same
        ``--resume`` target, preventing the infinite-respawn cascade we
        hit when a stale ``best.pt`` had a wrong ``hidden_size``.

        For backwards-compatibility: checkpoints without metadata are
        loaded best-effort, but any ``RuntimeError`` from
        ``load_state_dict`` (typically a "size mismatch") is converted
        into :class:`CheckpointIncompatibleError` so the watchdog short-
        circuit still fires.
        """
        ckpt_path = Path(path)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict):
            # Lone state_dict. Try the load directly; failure -> incompat.
            try:
                self.model.load_state_dict(ckpt)
            except RuntimeError as exc:
                raise CheckpointIncompatibleError(
                    f"Checkpoint {ckpt_path.name} (bare state_dict) cannot be "
                    f"loaded into the current model: {exc}"
                ) from exc
            return {}

        saved_meta = dict(ckpt.get("metadata", {}) or {})
        expected = {
            "model_arch": _model_init_dict(self.model),
            "env_shape": env_shape or _env_shape_from_model(self.model),
        }
        if saved_meta and "model_arch" in saved_meta:
            # Hard fail with a clear message before torch even tries the
            # state-dict copy.
            check_compat(saved_meta, expected, source=ckpt_path.name)
        else:
            # No metadata -> warn but proceed. The actual load below will
            # surface the size mismatch if there is one.
            pass

        try:
            self.model.load_state_dict(ckpt["model"])
        except RuntimeError as exc:
            # Translate the cryptic "size mismatch for X.weight ..." error
            # into something the watchdog can recognise.
            raise CheckpointIncompatibleError(
                f"Checkpoint {ckpt_path.name} does not match current model "
                f"arch (state_dict load failed): {exc}"
            ) from exc

        if "optimizer" in ckpt:
            # Optimizer state may be mismatched if model layout changed.
            # We surface this through metadata but don't crash.
            with contextlib.suppress(ValueError, KeyError):
                self.optimizer.load_state_dict(ckpt["optimizer"])
        self._update_steps = int(saved_meta.get("update_steps", self._update_steps))
        return saved_meta


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


def _env_shape_from_model(model: KivskiActorCritic) -> dict[str, Any]:
    """Best-effort env shape inferred from a model when no live env is around.

    ``team_size`` is reconstructed from ``action_dims[4] = 2 * team_size + 1``
    (see :func:`kivski_agents.factory.default_action_dims`). ``n_heads`` is
    the number of discrete action heads (== ``len(action_dims)``).
    ``obs_dim`` is read straight from the model.
    """
    action_dims = list(model.action_dims) if model.action_dims else []
    n_heads = len(action_dims)
    team_size: int | None = None
    if n_heads >= 5:
        last = int(action_dims[4])
        if last >= 3 and (last - 1) % 2 == 0:
            team_size = (last - 1) // 2
    return {
        "obs_dim": int(model.obs_dim),
        "n_heads": int(n_heads),
        "team_size": int(team_size) if team_size is not None else None,
    }


def _ml_config_to_dict(cfg: MLConfig) -> dict[str, Any]:
    """Dump :class:`MLConfig` to a JSON-friendly dict (best-effort)."""
    try:
        return cfg.model_dump()
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        return asdict(cfg)  # type: ignore[arg-type]


def _json_safe(value: Any) -> Any:
    """JSON encoder hook for things like tensors and numpy scalars.

    Kept for backwards-compat: callers used to pass this to
    ``json.dump(..., default=_json_safe)``. The current save path
    delegates sidecar writing to ``write_sidecar_metadata`` which uses
    ``default=str``, so this helper is now an opt-in convenience.
    """
    if torch.is_tensor(value):
        return value.tolist()
    return str(value)
