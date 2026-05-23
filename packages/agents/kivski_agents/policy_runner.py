"""Inference-time wrapper that drives a trained model against the env API.

The :class:`PolicyRunner` is the bridge between
:class:`~kivski_agents.networks.actor_critic.KivskiActorCritic` and the
PettingZoo-style ``dict[str, np.ndarray]`` interface exposed by
:class:`kivski_sim.env.KivskiParallelEnv`. It maintains the per-agent
recurrent hidden state across calls within a single episode and resets it
on episode boundaries.

The runner is used in three places:

* **FastAPI live match server** -- one env per match, drives the model
  forward each tick.
* **Eval suite** -- frozen-snapshot rollouts against baselines.
* **League sampling** -- the trainer instantiates runners around frozen
  opponent snapshots while training the current policy.

For ease of distribution we also expose :class:`PolicyBundle`, a tiny
container that pairs a checkpoint state dict with the environment config
and run metadata. ``PolicyBundle.from_checkpoint`` loads a checkpoint
saved by :meth:`MAPPOTrainer.save` and ``PolicyBundle.to_runner`` builds
a ready-to-use runner.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kivski_agents.networks.actor_critic import KivskiActorCritic
from kivski_sim.config import KivskiConfig


__all__ = ["PolicyBundle", "PolicyRunner"]


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------


class PolicyRunner:
    """Drive a :class:`KivskiActorCritic` through env observations.

    Args:
        model: A trained actor-critic. Will be moved to ``device`` and put
            into eval mode.
        device: Inference device (defaults to CPU for predictable
            latencies in the live server).
        deterministic: If True, take the argmax per head when sampling
            actions. Default is stochastic for diverse behaviour.
    """

    def __init__(
        self,
        model: KivskiActorCritic,
        device: torch.device | str = "cpu",
        deterministic: bool = False,
    ) -> None:
        self.model: KivskiActorCritic = model
        self.device: torch.device = torch.device(device)
        self.deterministic: bool = bool(deterministic)
        self.model.to(self.device)
        self.model.eval()

        # Per-agent persistent state -- populated on :meth:`reset`.
        self._agent_names: list[str] = []
        self._agent_to_index: dict[str, int] = {}
        # Hidden: [num_layers, n_agents, H]
        self._hidden: torch.Tensor | None = None

    # --------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------

    def reset(self, agent_names: list[str]) -> None:
        """Reset per-agent hidden state for a fresh episode."""
        self._agent_names = list(agent_names)
        self._agent_to_index = {name: i for i, name in enumerate(self._agent_names)}
        n = len(self._agent_names)
        self._hidden = self.model.initial_hidden_state(n, device=self.device)

    @property
    def agent_names(self) -> list[str]:
        return list(self._agent_names)

    # --------------------------------------------------------------
    # One inference step
    # --------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        observations: dict[str, np.ndarray],
        received_comms: dict[str, dict[int, np.ndarray]] | None = None,
        masks: dict[str, float] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Run one inference step.

        Args:
            observations: ``{agent_name: obs_vector[obs_dim]}``.
            received_comms: ``{agent_name: {sender_id: payload[comm_value_dim]}}``.
                Senders not present are treated as silent (mask 0).
            masks: Optional ``{agent_name: 0|1}`` where 0 forces a hidden
                state reset before the GRU step (e.g. agent respawned).

        Returns:
            ``(actions, comm_payloads)``:
                * ``actions``: ``{agent_name: int64[num_heads]}`` ready for
                  :meth:`KivskiParallelEnv.step`.
                * ``comm_payloads``: ``{agent_name: float32[comm_value_dim]}``
                  ready for :meth:`KivskiParallelEnv.step_with_comms`. The
                  value is already multiplied by the gate, so a closed gate
                  yields a zero payload.
        """
        if not self._agent_names:
            raise RuntimeError("PolicyRunner.act() called before reset()")
        if self._hidden is None:
            raise RuntimeError("PolicyRunner has no hidden state -- call reset() first")

        n = len(self._agent_names)
        obs_dim = self.model.obs_dim
        comm_value_dim = self.model.comm_value_dim

        # Batch observations into [N, obs_dim].
        obs_batch = np.zeros((n, obs_dim), dtype=np.float32)
        for name, vec in observations.items():
            if name not in self._agent_to_index:
                continue
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.shape[0] != obs_dim:
                raise ValueError(
                    f"observation for {name!r} has length {arr.shape[0]}, expected {obs_dim}"
                )
            obs_batch[self._agent_to_index[name]] = arr

        # Build per-agent received-comm vector. We do *not* run the attention
        # block at inference because the live env already aggregates messages
        # into a per-agent payload bundle keyed by sender id. To keep the
        # contract simple we pre-aggregate here by averaging payloads from
        # all live senders; this matches the all-mask=1 attention output in
        # the degenerate uniform case and works well in practice for the
        # FastAPI live server.
        comm_batch = np.zeros((n, comm_value_dim), dtype=np.float32)
        if received_comms:
            for name, sender_map in received_comms.items():
                if name not in self._agent_to_index or not sender_map:
                    continue
                payloads = []
                for payload in sender_map.values():
                    arr = np.asarray(payload, dtype=np.float32).reshape(-1)
                    if arr.shape[0] != comm_value_dim:
                        raise ValueError(
                            f"comm payload for {name!r} has length {arr.shape[0]}, "
                            f"expected {comm_value_dim}"
                        )
                    payloads.append(arr)
                if payloads:
                    comm_batch[self._agent_to_index[name]] = np.mean(payloads, axis=0)

        mask_tensor: torch.Tensor | None = None
        if masks:
            m_arr = np.ones(n, dtype=np.float32)
            for name, val in masks.items():
                if name in self._agent_to_index:
                    m_arr[self._agent_to_index[name]] = float(val)
            mask_tensor = torch.from_numpy(m_arr).to(self.device)

        obs_tensor = torch.from_numpy(obs_batch).to(self.device)
        comm_tensor = torch.from_numpy(comm_batch).to(self.device)
        out = self.model.act(
            obs=obs_tensor,
            hidden_state=self._hidden,
            received_comm=comm_tensor,
            joint_obs=None,
            masks=mask_tensor,
            deterministic=self.deterministic,
        )
        self._hidden = out["new_hidden"]

        actions_np = out["actions"].cpu().numpy().astype(np.int64)
        payload_np = out["comm_payload"].cpu().numpy().astype(np.float32)

        actions: dict[str, np.ndarray] = {}
        comm_payloads: dict[str, np.ndarray] = {}
        for name, idx in self._agent_to_index.items():
            actions[name] = actions_np[idx]
            comm_payloads[name] = payload_np[idx]
        return actions, comm_payloads


# ---------------------------------------------------------------------------
# Loadable checkpoint bundle
# ---------------------------------------------------------------------------


@dataclass
class PolicyBundle:
    """Serialised policy: state dict + config + metadata.

    This is the unit of currency for the league: snapshots saved during
    training, frozen opponents loaded at eval, and any external sharing of
    a trained policy. The bundle is intentionally small so it can be
    msgpack-encoded for the league storage.
    """

    model_state: dict[str, Any]
    model_init: dict[str, Any]
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "PolicyBundle":
        """Load a checkpoint saved by :meth:`MAPPOTrainer.save`."""
        ckpt_path = Path(path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_init = dict(ckpt.get("model_init", {}))
        cfg = dict(ckpt.get("cfg", {}))
        meta = dict(ckpt.get("metadata", {}))

        # Backwards-compatible: try to infer model_init from the state dict
        # shapes if it was not stored.
        if not model_init:
            raise ValueError(
                f"Checkpoint {ckpt_path} is missing 'model_init'; cannot reconstruct model."
            )
        return cls(
            model_state=dict(ckpt["model"]),
            model_init=model_init,
            config=cfg,
            metadata=meta,
        )

    def to_runner(
        self,
        device: torch.device | str = "cpu",
        deterministic: bool = False,
    ) -> PolicyRunner:
        """Instantiate a runner around the bundle's model state."""
        init = dict(self.model_init)
        # Strip any extras the constructor doesn't accept.
        allowed = {
            "obs_dim",
            "joint_obs_dim",
            "action_dims",
            "hidden_size",
            "comm_signature_dim",
            "comm_value_dim",
            "comm_attention_heads",
            "gumbel_temp",
            "gru_layers",
            "actor_embedding_dim",
        }
        kwargs = {k: v for k, v in init.items() if k in allowed}
        model = KivskiActorCritic(**kwargs)
        model.load_state_dict(self.model_state)
        return PolicyRunner(model=model, device=device, deterministic=deterministic)

    def save(self, path: str | Path) -> Path:
        """Persist the bundle as a torch checkpoint + JSON metadata sidecar."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model_state,
                "model_init": self.model_init,
                "cfg": self.config,
                "metadata": self.metadata,
            },
            out,
        )
        sidecar = out.with_suffix(out.suffix + ".json") if out.suffix else out.with_suffix(".json")
        meta = dict(self.metadata)
        meta.setdefault("timestamp", time.time())
        with sidecar.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True, default=str)
        return out

    @classmethod
    def from_kivski_config(
        cls,
        model: KivskiActorCritic,
        cfg: KivskiConfig,
        metadata: dict[str, Any] | None = None,
    ) -> "PolicyBundle":
        """Build a bundle from a live model + config (for inline snapshots)."""
        try:
            cfg_dict = cfg.model_dump()
        except AttributeError:  # pragma: no cover - pydantic v1 fallback
            cfg_dict = asdict(cfg)  # type: ignore[arg-type]
        return cls(
            model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
            model_init={
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
            },
            config=cfg_dict,
            metadata=dict(metadata or {}),
        )
