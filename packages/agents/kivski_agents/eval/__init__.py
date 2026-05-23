"""Evaluation suite: scenarios, head-to-head runner, and Elo/TrueSkill tracking.

The eval surface is intentionally small:

* :class:`ScenarioSpec` describes a stand-alone benchmark (full-pistol round,
  full-buy round, retake situation, ...).
* :class:`EvalRunner` consumes a scenario + two policies and produces an
  :class:`EvalResult` containing per-round and per-match aggregates.
* :class:`EloTracker` / :class:`TrueSkillTracker` book-keep ratings as new
  match results arrive.

All baselines and learned policies share the same minimal interface (``reset``
+ ``act``), so the runner doesn't care whether a player is the random
baseline or a 50M-parameter MAPPO actor.
"""

from __future__ import annotations

from kivski_agents.eval.elo import EloRating, EloTracker, TrueSkillTracker
from kivski_agents.eval.runner import EvalResult, EvalRunner, RoundResult
from kivski_agents.eval.scenarios import ALL_SCENARIOS, EvalScenario, ScenarioSpec, build_scenario


__all__ = [
    "EvalScenario",
    "ScenarioSpec",
    "ALL_SCENARIOS",
    "build_scenario",
    "EvalRunner",
    "EvalResult",
    "RoundResult",
    "EloTracker",
    "EloRating",
    "TrueSkillTracker",
]
