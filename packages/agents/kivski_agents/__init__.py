"""Kivski multi-agent RL: recurrent MAPPO with TarMAC-style communication."""

from kivski_agents.buffer import RolloutBatch, RolloutBuffer
from kivski_agents.factory import (
    build_model,
    build_trainer,
    default_action_dims,
    infer_joint_obs_dim,
)
from kivski_agents.mappo import MAPPOLoss, MAPPOTrainer
from kivski_agents.networks import (
    ActorHeads,
    CommAttention,
    CommEncoder,
    CommGate,
    KivskiActorCritic,
    ObservationEncoder,
    RecurrentCore,
    ValueHead,
)
from kivski_agents.policy_runner import PolicyBundle, PolicyRunner

__all__ = [
    "ActorHeads",
    "CommAttention",
    "CommEncoder",
    "CommGate",
    "KivskiActorCritic",
    "MAPPOLoss",
    "MAPPOTrainer",
    "ObservationEncoder",
    "PolicyBundle",
    "PolicyRunner",
    "RecurrentCore",
    "RolloutBatch",
    "RolloutBuffer",
    "ValueHead",
    "build_model",
    "build_trainer",
    "default_action_dims",
    "infer_joint_obs_dim",
]


__version__ = "0.1.0"
