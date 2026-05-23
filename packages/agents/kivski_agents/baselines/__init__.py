"""Baseline policies used as sparring partners for the learned MAPPO agents.

The baselines all expose the same minimal interface so they are drop-in
replaceable both inside the evaluation runner and inside the league trainer:

    policy.reset(agent_names: list[str]) -> None
    policy.act(observations: dict[str, np.ndarray],
               received_comms: dict[str, dict] | None = None,
               ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]

The returned 2-tuple is ``(actions, comm_payloads)`` where ``comm_payloads``
may be an empty dict for baselines that do not use the TarMAC comm channel.
"""

from __future__ import annotations

from kivski_agents.baselines.random_policy import RandomBaseline
from kivski_agents.baselines.registry import BASELINE_REGISTRY, get_baseline
from kivski_agents.baselines.scripted import ScriptedHoldBaseline, ScriptedRushBaseline

__all__ = [
    "RandomBaseline",
    "ScriptedRushBaseline",
    "ScriptedHoldBaseline",
    "BASELINE_REGISTRY",
    "get_baseline",
]
