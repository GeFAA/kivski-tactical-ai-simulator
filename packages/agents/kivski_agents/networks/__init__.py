"""Neural network building blocks for the Kivski MARL agents.

The package collects the three big components that the MAPPO trainer composes
into a working actor-critic:

* :mod:`kivski_agents.networks.comm` -- TarMAC-style learned communication
  channel (per-agent message encoder, multi-head attention over teammate
  messages, and a Gumbel-Sigmoid broadcast gate).
* :mod:`kivski_agents.networks.actor_critic` -- the observation encoder,
  recurrent GRU core, autoregressive multi-head actor, and a centralised
  critic for CTDE-style PPO updates.

Importing from this package is the canonical way for the rest of the codebase
(``factory.py``, ``mappo.py``, ``policy_runner.py``) to reach into these
modules; the public surface is re-exported below.
"""

from __future__ import annotations

from kivski_agents.networks.actor_critic import (
    ActorHeads,
    KivskiActorCritic,
    ObservationEncoder,
    RecurrentCore,
    ValueHead,
)
from kivski_agents.networks.comm import (
    CommAttention,
    CommEncoder,
    CommGate,
)


__all__ = [
    "ActorHeads",
    "CommAttention",
    "CommEncoder",
    "CommGate",
    "KivskiActorCritic",
    "ObservationEncoder",
    "RecurrentCore",
    "ValueHead",
]
