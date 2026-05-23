"""Name -> constructor registry for baselines.

This module is the single look-up point for resolving a baseline by name
(used by the eval CLI, by the trainer's league system, and by external
scripts that want to instantiate a sparring partner without importing each
baseline class explicitly).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kivski_agents.baselines.random_policy import RandomBaseline
from kivski_agents.baselines.scripted import ScriptedHoldBaseline, ScriptedRushBaseline

__all__ = ["BASELINE_REGISTRY", "get_baseline"]


# A constructor takes (env, map_data, seed) and returns a policy instance.
_BaselineFactory = Callable[[Any, Any, int], Any]


BASELINE_REGISTRY: dict[str, _BaselineFactory] = {
    "random": lambda env, map_data, seed: RandomBaseline(env.action_space("agent_0"), seed),
    "scripted_hold": lambda env, map_data, seed: ScriptedHoldBaseline(
        env.action_space("agent_0"), map_data, seed
    ),
    "scripted_rush": lambda env, map_data, seed: ScriptedRushBaseline(
        env.action_space("agent_0"), map_data, seed
    ),
}


def get_baseline(name: str, env: Any, map_data: Any, seed: int = 0) -> Any:
    """Instantiate the baseline registered under ``name``.

    Args:
        name: Registered baseline key (e.g. ``"random"``).
        env: A :class:`kivski_sim.env.KivskiParallelEnv` (used to look up the
            action space).
        map_data: The current :class:`kivski_sim.map_loader.MapData` (used by
            scripted baselines for spatial reasoning).
        seed: Deterministic seed for the baseline's internal RNG.

    Raises:
        ValueError: If ``name`` is not in :data:`BASELINE_REGISTRY`.
    """
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline {name!r}. Available: {sorted(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name](env, map_data, int(seed))
