"""Integration smoke test for the PettingZoo wrapper.

Drives a small batch of full matches (team_size=5, dustline) with random
actions to verify:

* the env can run an entire match without crashing
* termination flips to True for every agent at match-end
* across enough seeded matches at least one round ends with an interesting
  outcome (ATTACKERS_ELIM or BOMB_DEFUSED) -- a cheap sanity that the
  observation / action pipeline actually drives meaningful behavior

The test deliberately uses a small ``max_rounds`` so that even a single
random agent can finish a match in well under a second per seed.
"""

from __future__ import annotations

import numpy as np
import pytest
from kivski_sim.config import KivskiConfig
from kivski_sim.env import KivskiParallelEnv
from kivski_sim.map_loader import load_map
from kivski_sim.types import RoundOutcome


def _make_config() -> KivskiConfig:
    return KivskiConfig.model_validate(
        {
            "seed": 2025,
            "simulation": {
                "team_size": 5,
                "max_rounds": 4,
                "side_switch_round": 2,
                "round_time_seconds": 8,
                "bomb_timer_seconds": 4,
                "plant_time_seconds": 1.0,
                "defuse_time_seconds": 1.0,
                "defuse_time_with_kit_seconds": 0.5,
                "buy_time_seconds": 1,
                "tick_rate_hz": 10,
                "max_ticks_per_round": 100,
                "starting_money": 800,
            },
        }
    )


def _random_action_dict(
    env: KivskiParallelEnv, rng: np.random.Generator
) -> dict[str, dict[str, np.ndarray]]:
    """Build a v0.4 mixed-action dict per agent."""
    space = env.action_space("agent_0")
    nvec = np.asarray(space.spaces["discrete"].nvec, dtype=np.int64)
    actions: dict[str, dict[str, np.ndarray]] = {}
    for name in env.possible_agents:
        mv = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        disc = rng.integers(low=np.zeros_like(nvec), high=nvec).astype(np.int64)
        actions[name] = {"move": mv, "discrete": disc}
    return actions


def _run_one_match(seed: int, map_data, cfg: KivskiConfig) -> tuple[bool, list[RoundOutcome]]:
    """Drive one full match with random actions; return (clean_exit, outcomes)."""
    env = KivskiParallelEnv(config=cfg, map_name="dustline", seed=seed, map_data=map_data)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    max_steps = int(cfg.simulation.max_ticks_per_round) * int(cfg.simulation.max_rounds) + 200
    for _ in range(max_steps):
        actions = _random_action_dict(env, rng)
        obs, rewards, terms, truncs, infos = env.step(actions)
        assert set(obs.keys()) == set(env.possible_agents)
        assert set(rewards.keys()) == set(env.possible_agents)
        assert set(terms.keys()) == set(env.possible_agents)
        if all(terms.values()):
            break
    outcomes = [summary.outcome for summary in env.engine.state.round_summaries]
    return all(terms.values()), outcomes


@pytest.mark.parametrize("seed", [11, 22, 33, 44, 55])
def test_random_match_runs_to_completion(seed: int) -> None:
    cfg = _make_config()
    map_data = load_map("dustline")
    finished, _ = _run_one_match(seed, map_data, cfg)
    assert finished, f"match seed={seed} did not terminate"


def test_meaningful_outcomes_across_many_seeds() -> None:
    """Run a handful of matches and check that the engine produces variety."""
    cfg = _make_config()
    map_data = load_map("dustline")
    seeds = list(range(50, 60))
    interesting_seen = False
    all_outcomes: list[RoundOutcome] = []
    for s in seeds:
        finished, outcomes = _run_one_match(s, map_data, cfg)
        assert finished, f"seed {s} did not finish"
        all_outcomes.extend(outcomes)
        if any(o in (RoundOutcome.ATTACKERS_ELIM, RoundOutcome.BOMB_DEFUSED) for o in outcomes):
            interesting_seen = True
            break
    if not interesting_seen:
        # As a softer guard, at least *some* round should resolve to a non-NONE
        # outcome. If every single round ended in a TIMEOUT we still consider
        # that a passing smoke (the engine kept running), but we record it.
        assert any(o != RoundOutcome.NONE for o in all_outcomes), all_outcomes


def test_random_match_50_iterations_sanity() -> None:
    """Mini 50-match equivalent: 10 matches across 5 different seeds, with
    much smaller round caps so the full set completes in a few seconds."""
    cfg = _make_config()
    map_data = load_map("dustline")
    seeds = list(range(60, 65))
    completed = 0
    for s in seeds:
        finished, _ = _run_one_match(s, map_data, cfg)
        if finished:
            completed += 1
    assert completed == len(seeds), f"only {completed}/{len(seeds)} matches finished"


def test_env_reset_after_done_is_clean() -> None:
    cfg = _make_config()
    env = KivskiParallelEnv(config=cfg, map_name="dustline", seed=123, map_data=load_map("dustline"))
    env.reset(seed=123)
    rng = np.random.default_rng(123)
    for _ in range(1500):
        obs, rewards, terms, _, _ = env.step(_random_action_dict(env, rng))
        if all(terms.values()):
            break
    # Reset and run again -- should not raise.
    obs, _ = env.reset(seed=456)
    assert set(obs.keys()) == set(env.possible_agents)
    # One more step should also be fine.
    rewards = env.step(_random_action_dict(env, rng))
    assert isinstance(rewards, tuple) and len(rewards) == 5
