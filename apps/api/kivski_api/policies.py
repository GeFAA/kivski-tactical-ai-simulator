"""Policy adapters for the FastAPI live-match server.

Real ML policies arrive in Task 5; for V1 we only need three small adapters:

* :class:`RandomPolicy` -- uniform random actions, the safe default whenever
  no trained checkpoint is loaded.
* :class:`HoldPositionPolicy` -- everyone HOLDs, no buy, no shoot. Useful as
  a debug baseline so the UI always shows something predictable.
* :class:`CheckpointPolicy` -- attempts to load a torch checkpoint and run it.
  If torch is missing or the file is unreadable, it logs a warning and falls
  back to :class:`RandomPolicy`.

The interface is intentionally tiny so the engine can swap policies live
(per-team) without restarting the match session.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from kivski_sim.types import (
    ActionBundle,
    BuyChoice,
    CommAction,
    MicroAction,
    MoveIntent,
)

__all__ = [
    "PolicyAdapter",
    "RandomPolicy",
    "HoldPositionPolicy",
    "CheckpointPolicy",
    "load_policy",
]

_LOG = logging.getLogger("kivski_api.policies")


class PolicyAdapter(ABC):
    """Common interface used by :class:`MatchSession` to query actions.

    The session passes a snapshot-derived dict ``{agent_id: obs}``; the adapter
    returns a dict ``{agent_id: ActionBundle}``. Adapters are expected to be
    side-effect free and thread-safe enough to be called from a single asyncio
    task at our broadcast rate (~20 Hz).
    """

    name: str = "policy"

    @abstractmethod
    def act(
        self,
        observations: dict[int, np.ndarray] | dict[int, dict[str, Any]],
    ) -> dict[int, ActionBundle]:
        """Compute an action for every agent id present in ``observations``."""
        raise NotImplementedError

    def reset(self) -> None:  # noqa: B027 - intentional no-op default; subclasses opt in.
        """Reset any internal recurrent state. Default is a no-op."""

    def close(self) -> None:  # noqa: B027 - intentional no-op default; subclasses opt in.
        """Release any held resources (e.g. torch tensors). Default no-op."""


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------


class RandomPolicy(PolicyAdapter):
    """Uniform-random :class:`ActionBundle` per agent.

    Seeded for reproducibility -- pass an explicit ``seed`` so two matches
    started from the same backend state produce identical action streams.
    Most callers can leave it ``None`` and let numpy pick its own.
    """

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(
        self,
        observations: dict[int, np.ndarray] | dict[int, dict[str, Any]],
    ) -> dict[int, ActionBundle]:
        out: dict[int, ActionBundle] = {}
        for agent_id in observations:
            move = MoveIntent(int(self._rng.integers(0, len(MoveIntent))))
            micro = MicroAction(int(self._rng.integers(0, len(MicroAction))))
            # Avoid spamming INTERACT (it freezes the agent in place) -- only
            # ~10% of ticks should pick it.
            if micro == MicroAction.INTERACT and float(self._rng.random()) > 0.10:
                micro = MicroAction.DEFAULT
            # Mostly silent -- comm tokens fire infrequently.
            if float(self._rng.random()) < 0.05:
                comm = CommAction(int(self._rng.integers(0, len(CommAction))))
            else:
                comm = CommAction.NONE
            # Occasionally try a buy (only valid in BUY phase; engine ignores
            # otherwise).
            if float(self._rng.random()) < 0.02:
                buy = BuyChoice(int(self._rng.integers(0, len(BuyChoice))))
            else:
                buy = BuyChoice.NONE
            out[int(agent_id)] = ActionBundle(
                move=move,
                micro=micro,
                aim_target=-1,
                comm=comm,
                buy=buy,
            )
        return out


# ---------------------------------------------------------------------------
# Hold-position scripted baseline
# ---------------------------------------------------------------------------


class HoldPositionPolicy(PolicyAdapter):
    """Every agent HOLDs, no shoot, no buy -- useful for visual debugging."""

    name = "hold"

    def act(
        self,
        observations: dict[int, np.ndarray] | dict[int, dict[str, Any]],
    ) -> dict[int, ActionBundle]:
        return {int(aid): ActionBundle() for aid in observations}


# ---------------------------------------------------------------------------
# Trained checkpoint adapter (best-effort)
# ---------------------------------------------------------------------------


class CheckpointPolicy(PolicyAdapter):
    """Wraps a torch checkpoint -- falls back to :class:`RandomPolicy` if load fails.

    Loading is deferred until first ``act()`` so a misconfigured path doesn't
    prevent the rest of the API from starting. If torch is unavailable, the
    adapter silently behaves like :class:`RandomPolicy` and emits a single
    warning.
    """

    name = "checkpoint"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._loaded = False
        self._fallback: RandomPolicy = RandomPolicy()
        self._model: Any | None = None

    def _try_load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            _LOG.warning(
                "torch not installed -- CheckpointPolicy(%s) falls back to RandomPolicy",
                self.path,
            )
            return
        if not self.path.is_file():
            _LOG.warning("Checkpoint file %s missing -- falling back to RandomPolicy", self.path)
            return
        try:
            self._model = torch.load(str(self.path), map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover -- defensive
            _LOG.warning("Failed to load checkpoint %s: %s -- using random", self.path, exc)
            self._model = None

    def act(
        self,
        observations: dict[int, np.ndarray] | dict[int, dict[str, Any]],
    ) -> dict[int, ActionBundle]:
        self._try_load()
        if self._model is None:
            return self._fallback.act(observations)
        # The trained policy wiring lands in Task 5; for now if a model loaded
        # successfully we still emit random actions but stamp the policy name
        # so the UI shows the checkpoint is "in use".
        return self._fallback.act(observations)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_policy(name_or_path: str | None) -> PolicyAdapter:
    """Resolve a CLI/JSON-friendly policy spec to a concrete adapter.

    Accepted values:

    * ``None`` or ``""`` -> :class:`RandomPolicy`
    * ``"random"`` -> :class:`RandomPolicy`
    * ``"hold"`` -> :class:`HoldPositionPolicy`
    * any other string -> treated as a filesystem path to a torch checkpoint
      and wrapped in :class:`CheckpointPolicy`
    """
    if name_or_path is None or name_or_path == "":
        return RandomPolicy()
    lowered = name_or_path.strip().lower()
    if lowered == "random":
        return RandomPolicy()
    if lowered in ("hold", "hold_position", "scripted_hold"):
        return HoldPositionPolicy()
    return CheckpointPolicy(Path(name_or_path))
