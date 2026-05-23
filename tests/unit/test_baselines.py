"""Unit tests for the baseline policies."""

from __future__ import annotations

import numpy as np
import pytest

from kivski_agents.baselines import (
    BASELINE_REGISTRY,
    RandomBaseline,
    ScriptedHoldBaseline,
    ScriptedRushBaseline,
    get_baseline,
)
from kivski_sim.config import KivskiConfig
from kivski_sim.env import KivskiParallelEnv
from kivski_sim.map_loader import load_map
from kivski_sim.types import BuyChoice, CommAction, MicroAction, MoveIntent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg() -> KivskiConfig:
    """Compact config that keeps tests fast."""
    return KivskiConfig.model_validate(
        {
            "seed": 1234,
            "simulation": {
                "team_size": 2,
                "max_rounds": 4,
                "side_switch_round": 2,
                "round_time_seconds": 6,
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


@pytest.fixture
def env(small_cfg: KivskiConfig) -> KivskiParallelEnv:
    return KivskiParallelEnv(
        config=small_cfg, map_name="dustline", seed=1234, map_data=load_map("dustline")
    )


# ---------------------------------------------------------------------------
# RandomBaseline
# ---------------------------------------------------------------------------


def test_random_baseline_actions_in_space(env: KivskiParallelEnv) -> None:
    """Random baseline's actions must always satisfy the action-space dims."""
    space = env.action_space("agent_0")
    rb = RandomBaseline(space, seed=0)
    obs, _ = env.reset(seed=0)
    rb.reset(list(obs.keys()))
    actions, payloads = rb.act(obs)
    assert payloads == {}
    assert set(actions.keys()) == set(obs.keys())
    for name, act in actions.items():
        assert isinstance(act, np.ndarray)
        assert act.dtype == np.int64
        assert act.shape == (space.nvec.shape[0],)
        for i, dim in enumerate(space.nvec):
            assert 0 <= int(act[i]) < int(dim), (
                f"Agent {name} action[{i}]={int(act[i])} out of range [0,{int(dim)})"
            )


def test_random_baseline_reproducible_with_seed(env: KivskiParallelEnv) -> None:
    """Two RandomBaselines built with the same seed return identical streams."""
    space = env.action_space("agent_0")
    obs, _ = env.reset(seed=0)

    a = RandomBaseline(space, seed=42)
    b = RandomBaseline(space, seed=42)
    a.reset(list(obs.keys()))
    b.reset(list(obs.keys()))

    actions_a, _ = a.act(obs)
    actions_b, _ = b.act(obs)
    for name in actions_a:
        assert np.array_equal(actions_a[name], actions_b[name]), name

    # Different seeds give different streams (statistical check on a small dim).
    c = RandomBaseline(space, seed=999)
    c.reset(list(obs.keys()))
    actions_c, _ = c.act(obs)
    # Probabilistically extremely unlikely to fully match for non-trivial dims.
    any_diff = any(
        not np.array_equal(actions_a[name], actions_c[name]) for name in actions_a
    )
    assert any_diff


# ---------------------------------------------------------------------------
# ScriptedHoldBaseline
# ---------------------------------------------------------------------------


def test_scripted_hold_outputs_valid_actions(env: KivskiParallelEnv) -> None:
    """ScriptedHoldBaseline outputs must satisfy the action space."""
    space = env.action_space("agent_0")
    bot = ScriptedHoldBaseline(space, env.map, seed=0)
    obs, _ = env.reset(seed=0)
    bot.reset(list(obs.keys()))
    actions, payloads = bot.act(obs)
    assert payloads == {}
    for name, act in actions.items():
        assert act.shape == (space.nvec.shape[0],)
        for i, dim in enumerate(space.nvec):
            assert 0 <= int(act[i]) < int(dim), name


# ---------------------------------------------------------------------------
# ScriptedRushBaseline
# ---------------------------------------------------------------------------


def test_scripted_rush_targets_bombsite(env: KivskiParallelEnv) -> None:
    """After reset + a few live steps, rush baseline should move (non-HOLD)."""
    space = env.action_space("agent_0")
    bot = ScriptedRushBaseline(space, env.map, seed=0)
    obs, _ = env.reset(seed=0)
    bot.reset(list(obs.keys()))

    # Step a few times through the BUY phase to reach LIVE.
    actions, _ = bot.act(obs)
    for _ in range(20):
        obs, _, terms, truncs, _ = env.step(actions)
        if all(terms.values()) or all(truncs.values()):
            break
        actions, _ = bot.act(obs)

    # Once in LIVE, at least one agent should be picking a non-HOLD move
    # (i.e. heading toward a bombsite). HOLD = 0; SPRINT = MicroAction.SPRINT
    # we look at the move head only.
    any_movement = any(int(act[0]) != int(MoveIntent.HOLD) for act in actions.values())
    assert any_movement, "Rush baseline should be moving during LIVE phase"


def test_scripted_rush_consensus_targets_set(env: KivskiParallelEnv) -> None:
    """Reset should populate per-agent target_sites with valid A/B picks."""
    space = env.action_space("agent_0")
    bot = ScriptedRushBaseline(space, env.map, seed=7)
    bot.reset(["agent_0", "agent_1"])
    assert set(bot._target_sites.keys()) == {"agent_0", "agent_1"}
    for site in bot._target_sites.values():
        assert site in ("A", "B")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_get_baseline_random(env: KivskiParallelEnv) -> None:
    """The registry returns a RandomBaseline for the 'random' key."""
    pol = get_baseline("random", env, env.map, seed=0)
    assert isinstance(pol, RandomBaseline)
    assert pol.name == "random"


def test_get_baseline_unknown_raises(env: KivskiParallelEnv) -> None:
    """Unknown baseline names raise a ValueError mentioning the available keys."""
    with pytest.raises(ValueError) as excinfo:
        get_baseline("not_a_baseline", env, env.map, seed=0)
    msg = str(excinfo.value)
    assert "not_a_baseline" in msg
    # The error should hint at all the registered keys.
    for k in BASELINE_REGISTRY:
        assert k in msg


def test_registry_contains_expected_keys() -> None:
    """Three core baselines must be registered."""
    assert "random" in BASELINE_REGISTRY
    assert "scripted_hold" in BASELINE_REGISTRY
    assert "scripted_rush" in BASELINE_REGISTRY
