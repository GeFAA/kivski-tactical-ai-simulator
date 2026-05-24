"""Uniformly random baseline policy.

Used as the lower-bound sparring partner in the league: any meaningfully
trained policy should crush the random baseline within the first few hundred
episodes. Determinism is preserved by seeding an internal :class:`numpy.random.Generator`
once at construction time -- consecutive ``act`` calls produce a reproducible
sequence given a fixed seed.

v0.4 mixed action space:
    * ``move``     -- uniform in ``[-1, 1]^2`` (with ~15% HOLD).
    * ``micro``    -- uniform from 6 categories.
    * ``comm``     -- mostly ``NONE`` with a 5% chance of any random token.
    * ``buy``      -- mostly ``NONE`` with a 2% chance of a random purchase.
    * ``aim_target`` -- uniform over the per-env range.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

__all__ = ["RandomBaseline"]

_NUM_MICRO: int = 6
_NUM_COMM: int = 9
_NUM_BUY: int = 8


class RandomBaseline:
    """Random actions sampled from the env's mixed action space.

    Accepts either:
        * a v0.4 ``spaces.Dict({"move": Box, "discrete": MultiDiscrete})``,
        * a legacy ``spaces.MultiDiscrete`` (treated as the discrete heads).

    Attributes:
        name: Human-readable identifier used in eval reports and Elo books.
    """

    name: str = "random"

    def __init__(self, env_action_space: Any, seed: int = 0) -> None:
        # Pull the discrete-head dimensionality off the space.
        move_low: np.ndarray | None = None
        move_high: np.ndarray | None = None
        if isinstance(env_action_space, spaces.Dict):
            discrete_space = env_action_space.spaces.get("discrete")
            move_space = env_action_space.spaces.get("move")
            if discrete_space is None or move_space is None:
                raise TypeError(
                    "RandomBaseline expects spaces.Dict({'move': Box, 'discrete': MultiDiscrete})"
                )
            self._dims: np.ndarray = np.asarray(discrete_space.nvec, dtype=np.int64)
            move_low = np.asarray(move_space.low, dtype=np.float32)
            move_high = np.asarray(move_space.high, dtype=np.float32)
        else:
            dims_attr = getattr(env_action_space, "nvec", None)
            if dims_attr is None:
                raise TypeError(
                    "RandomBaseline expects spaces.Dict or MultiDiscrete, "
                    f"got {type(env_action_space)!r}"
                )
            self._dims = np.asarray(dims_attr, dtype=np.int64)
        # Default move bounds when only a discrete space was provided.
        if move_low is None:
            move_low = np.array([-1.0, -1.0], dtype=np.float32)
            move_high = np.array([1.0, 1.0], dtype=np.float32)
        self._move_low: np.ndarray = move_low
        self._move_high: np.ndarray = move_high
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
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
        """Sample one mixed action per agent.

        Args:
            observations: Dict ``agent_name -> obs_vector``. The vectors are
                ignored -- this is a stateless uniform sampler.
            received_comms: Ignored. Present only to keep the interface
                consistent with the learned policies.

        Returns:
            ``(actions, comm_payloads)`` where ``actions`` is a dict
            ``agent_name -> {"move": float32[2], "discrete": int64[4]}``,
            and ``comm_payloads`` is an empty dict.
        """
        del received_comms  # unused
        agents = self._agent_names if self._agent_names else list(observations.keys())
        actions: dict[str, dict[str, np.ndarray]] = {}
        for name in agents:
            # Continuous move: uniform in the configured Box. Inject a 15%
            # HOLD rate so the bot occasionally pauses (otherwise it shuffles
            # endlessly which makes random-baseline rollouts hard to read).
            if self._rng.random() < 0.15:
                move = np.zeros(self._move_low.shape, dtype=np.float32)
            else:
                move = self._rng.uniform(
                    low=self._move_low, high=self._move_high
                ).astype(np.float32)
            # Discrete: bias comm/buy toward NONE so the random bot doesn't
            # saturate the message channel or burn money every tick.
            disc = np.empty(self._dims.shape[0], dtype=np.int64)
            for i, n in enumerate(self._dims):
                n = int(n)
                if n == _NUM_COMM:
                    disc[i] = int(self._rng.integers(0, n)) if self._rng.random() < 0.05 else 0
                elif n == _NUM_BUY:
                    disc[i] = int(self._rng.integers(0, n)) if self._rng.random() < 0.02 else 0
                else:
                    disc[i] = int(self._rng.integers(0, n))
            actions[name] = {"move": move, "discrete": disc}
        return actions, {}
