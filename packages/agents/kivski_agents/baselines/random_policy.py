"""Uniformly random baseline policy.

Used as the lower-bound sparring partner in the league: any meaningfully
trained policy should crush the random baseline within the first few hundred
episodes. Determinism is preserved by seeding an internal :class:`numpy.random.Generator`
once at construction time -- consecutive ``act`` calls produce a reproducible
sequence given a fixed seed.
"""

from __future__ import annotations

from typing import Any

import numpy as np


__all__ = ["RandomBaseline"]


class RandomBaseline:
    """Random actions uniformly sampled from the env action space.

    The policy treats the action space as a ``MultiDiscrete`` and samples each
    component independently from ``[0, n_i)``. Seeded internally so two
    :class:`RandomBaseline` instances built with the same seed produce
    identical action streams given identical observation sequences.

    Attributes:
        name: Human-readable identifier used in eval reports and Elo books.
    """

    name: str = "random"

    def __init__(self, env_action_space: Any, seed: int = 0) -> None:
        # Save the per-component dim list so we can re-seed without keeping a
        # gymnasium import at module-import time.
        dims_attr = getattr(env_action_space, "nvec", None)
        if dims_attr is None:
            # Fall back to the gymnasium ``MultiDiscrete`` ``nvec`` attribute.
            raise TypeError(
                f"RandomBaseline expects a MultiDiscrete action space, got {type(env_action_space)!r}"
            )
        self._dims: np.ndarray = np.asarray(dims_attr, dtype=np.int64)
        self._seed: int = int(seed)
        self._rng: np.random.Generator = np.random.default_rng(int(seed))
        self._agent_names: list[str] = []

    # ------------------------------------------------------------------

    def reset(self, agent_names: list[str]) -> None:
        """Reset internal state for a fresh episode.

        The RNG is *not* re-seeded here: across-match reproducibility is
        controlled by the initial ``seed`` passed to ``__init__``. If a caller
        wants a fresh RNG per match they can simply re-instantiate.
        """
        self._agent_names = list(agent_names)

    # ------------------------------------------------------------------

    def act(
        self,
        observations: dict[str, np.ndarray],
        received_comms: dict[str, dict] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Sample one MultiDiscrete action per agent.

        Args:
            observations: Dict ``agent_name -> obs_vector``. The vectors are
                ignored -- this is a stateless uniform sampler.
            received_comms: Ignored. Present only to keep the interface
                consistent with the learned policies.

        Returns:
            ``(actions, comm_payloads)`` where ``actions`` is a dict mapping
            each known agent to a length-len(dims) ``np.int64`` array, and
            ``comm_payloads`` is an empty dict (random baseline never sends
            learned messages).
        """
        del received_comms  # unused
        # If reset wasn't called, fall back to keys present in observations.
        agents = self._agent_names if self._agent_names else list(observations.keys())
        actions: dict[str, np.ndarray] = {}
        for name in agents:
            # Independent uniform sample per component.
            sample = np.empty(self._dims.shape[0], dtype=np.int64)
            for i, n in enumerate(self._dims):
                sample[i] = int(self._rng.integers(0, int(n)))
            actions[name] = sample
        return actions, {}
