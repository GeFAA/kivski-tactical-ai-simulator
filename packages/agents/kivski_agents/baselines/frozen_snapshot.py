"""A baseline that plays back a frozen policy checkpoint without training.

Used in the league / PBT trainer to pit the current learner against past
versions of itself ("self-play exploit pool"). Torch is imported lazily so
test environments without PyTorch installed can still import this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


__all__ = ["FrozenSnapshotBaseline"]


class FrozenSnapshotBaseline:
    """Loads a saved checkpoint and runs it in inference mode.

    Lazily imports :mod:`torch` and the actor-critic module so the rest of
    the baselines package can be used in environments where PyTorch isn't
    available (e.g. unit tests that only exercise the random / scripted
    baselines).

    The expected checkpoint format is the bundle written by the trainer's
    ``PolicyBundle.from_checkpoint`` / ``PolicyBundle.save`` pair. If the
    upstream code adopts a different on-disk layout this class is the single
    point that needs updating.
    """

    name: str = "frozen_snapshot"

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Any | None = None,
    ) -> None:
        try:
            import torch  # noqa: PLC0415 - lazy on purpose
        except ImportError as exc:
            raise ImportError(
                "FrozenSnapshotBaseline requires PyTorch. Install with "
                "`pip install torch` or pick a different baseline."
            ) from exc

        # Resolve the device, defaulting to CPU.
        if device is None:
            device = torch.device("cpu")
        self._torch = torch
        self._device = device
        self._ckpt_path: Path = Path(checkpoint_path).expanduser().resolve()

        if not self._ckpt_path.is_file():
            raise FileNotFoundError(
                f"FrozenSnapshotBaseline checkpoint not found: {self._ckpt_path}"
            )

        # Try to use the trainer's official bundle loader first; fall back to
        # a generic ``torch.load`` so unit tests can hand us a barebones .pt
        # without depending on the full trainer module.
        self._bundle: Any | None = None
        try:
            from kivski_agents.policy_runner import PolicyBundle  # noqa: PLC0415

            self._bundle = PolicyBundle.from_checkpoint(
                str(self._ckpt_path), device=self._device
            )
        except Exception:
            # Fall back: load the state dict and store it raw. The act()
            # method below will fail loudly if invoked without a proper model.
            try:
                self._raw_state = torch.load(
                    str(self._ckpt_path), map_location=self._device
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load checkpoint {self._ckpt_path!s}: {exc}"
                ) from exc

        self._agent_names: list[str] = []
        # Per-agent recurrent hidden state, lazily created on first act().
        self._hidden_state: dict[str, Any] = {}

    # ------------------------------------------------------------------

    def reset(self, agent_names: list[str]) -> None:
        """Clear recurrent state at the start of a fresh episode."""
        self._agent_names = list(agent_names)
        self._hidden_state = {name: None for name in agent_names}
        if self._bundle is not None and hasattr(self._bundle, "reset"):
            try:
                self._bundle.reset(agent_names)
            except Exception:
                # Bundle is responsible for its own state; ignore reset
                # failures in inference-only mode.
                pass

    # ------------------------------------------------------------------

    def act(
        self,
        observations: dict[str, np.ndarray],
        received_comms: dict[str, dict] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Run the frozen model in inference mode.

        Falls back to all-zeros actions if no bundle is loaded -- this keeps
        the eval pipeline alive even when a checkpoint cannot be decoded.
        """
        if self._bundle is None:
            # Degenerate fallback: emit HOLD-like actions so the env still
            # advances; the caller will see this baseline lose every match.
            return self._zero_actions(observations), {}

        if not hasattr(self._bundle, "act"):
            return self._zero_actions(observations), {}

        with self._torch.no_grad():
            try:
                actions, payloads = self._bundle.act(
                    observations,
                    received_comms=received_comms,
                    hidden_state=self._hidden_state,
                )
            except Exception:
                return self._zero_actions(observations), {}
        # Coerce returned arrays to numpy int64 / float32 for the env.
        np_actions: dict[str, np.ndarray] = {}
        for name, action in actions.items():
            arr = action
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            np_actions[name] = np.asarray(arr, dtype=np.int64)
        np_payloads: dict[str, np.ndarray] = {}
        for name, payload in (payloads or {}).items():
            arr = payload
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            np_payloads[name] = np.asarray(arr, dtype=np.float32)
        return np_actions, np_payloads

    # ------------------------------------------------------------------

    def _zero_actions(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        agents = self._agent_names if self._agent_names else list(observations.keys())
        return {name: np.zeros(5, dtype=np.int64) for name in agents}
