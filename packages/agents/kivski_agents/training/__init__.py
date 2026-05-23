"""Training orchestration: vectorised env, rollout collection, league, curriculum, trainer.

This package wires together the MAPPO update loop, self-play / baseline
sparring (the "league"), and the curriculum-style stage progression. The
public surface is small on purpose -- callers usually only touch
:class:`Trainer` (constructed by ``scripts/train.py``); the rest of the
classes are exposed so tests and downstream consumers can inspect / mock
individual pieces.
"""

from kivski_agents.training.curriculum import CurriculumManager
from kivski_agents.training.league import LeagueEntry, LeagueManager, OpponentSampler
from kivski_agents.training.rollout_collector import RolloutCollector
from kivski_agents.training.trainer import Trainer, TrainerConfig
from kivski_agents.training.vec_env import VecEnvWrapper


__all__ = [
    "VecEnvWrapper",
    "RolloutCollector",
    "LeagueManager",
    "LeagueEntry",
    "OpponentSampler",
    "CurriculumManager",
    "Trainer",
    "TrainerConfig",
]
