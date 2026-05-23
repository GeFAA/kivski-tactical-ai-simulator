"""Unit tests for the eval suite (scenarios, runner, Elo)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kivski_agents.baselines import RandomBaseline
from kivski_agents.eval import (
    ALL_SCENARIOS,
    EloTracker,
    EvalRunner,
    ScenarioSpec,
    build_scenario,
)
from kivski_sim.config import KivskiConfig
from kivski_sim.types import BombPhase, Phase, Side

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg() -> KivskiConfig:
    """Compact config so eval-smoke tests finish in <1 second."""
    return KivskiConfig.model_validate(
        {
            "seed": 42,
            "simulation": {
                "team_size": 2,
                "max_rounds": 4,
                "side_switch_round": 2,
                "round_time_seconds": 3,
                "bomb_timer_seconds": 3,
                "plant_time_seconds": 0.5,
                "defuse_time_seconds": 0.5,
                "defuse_time_with_kit_seconds": 0.3,
                "buy_time_seconds": 0.5,
                "tick_rate_hz": 10,
                "max_ticks_per_round": 60,
                "starting_money": 800,
            },
        }
    )


# ---------------------------------------------------------------------------
# EloTracker
# ---------------------------------------------------------------------------


def test_elo_expected_score_at_parity_is_05() -> None:
    """Two policies at identical rating must have expected score 0.5 each."""
    tracker = EloTracker()
    tracker.add_policy("a", initial_rating=1000.0)
    tracker.add_policy("b", initial_rating=1000.0)
    assert tracker.expected_score("a", "b") == pytest.approx(0.5)
    assert tracker.expected_score("b", "a") == pytest.approx(0.5)


def test_elo_winner_gains_rating() -> None:
    """A win must move the winner's rating up and the loser's down."""
    tracker = EloTracker(k_factor=32.0)
    tracker.update("a", "b", outcome=1.0)
    assert tracker.ratings["a"].rating > 1000.0
    assert tracker.ratings["b"].rating < 1000.0
    # Symmetric magnitude at parity.
    assert tracker.ratings["a"].rating - 1000.0 == pytest.approx(1000.0 - tracker.ratings["b"].rating)
    # Records updated.
    assert tracker.ratings["a"].wins == 1
    assert tracker.ratings["a"].matches == 1
    assert tracker.ratings["b"].losses == 1
    assert tracker.ratings["b"].matches == 1


def test_elo_draw_does_not_change_ratings_at_parity() -> None:
    """A draw between equal-rated policies leaves ratings unchanged."""
    tracker = EloTracker()
    tracker.update("a", "b", outcome=0.5)
    assert tracker.ratings["a"].rating == pytest.approx(1000.0)
    assert tracker.ratings["b"].rating == pytest.approx(1000.0)
    assert tracker.ratings["a"].draws == 1
    assert tracker.ratings["b"].draws == 1


def test_elo_save_load_roundtrip(tmp_path: Path) -> None:
    """to_json / from_json must preserve all stats."""
    tracker = EloTracker(k_factor=24.0)
    tracker.update("a", "b", outcome=1.0)
    tracker.update("a", "c", outcome=0.5)
    tracker.update("c", "b", outcome=0.0)

    out = tmp_path / "elo.json"
    tracker.to_json(out)

    reloaded = EloTracker.from_json(out)
    assert reloaded.k_factor == 24.0
    assert set(reloaded.ratings.keys()) == set(tracker.ratings.keys())
    for name, rating in tracker.ratings.items():
        r2 = reloaded.ratings[name]
        assert r2.rating == pytest.approx(rating.rating)
        assert r2.matches == rating.matches
        assert r2.wins == rating.wins
        assert r2.draws == rating.draws
        assert r2.losses == rating.losses


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def test_scenario_full_pistol_initializes(small_cfg: KivskiConfig) -> None:
    """The 'full_pistol' scenario should produce a usable env at money=800."""
    spec = ScenarioSpec(name="full_pistol_small", team_size=2, starting_money=800)
    env = build_scenario(spec, small_cfg, seed=0)
    assert env.engine.state.match_outcome.name == "NONE"
    for a in env.engine.state.agents:
        assert a.money == 800


def test_scenario_pre_plant(small_cfg: KivskiConfig) -> None:
    """The retake-style scenario must pre-plant the bomb at the requested site."""
    spec = ScenarioSpec(
        name="retake_smoke",
        team_size=2,
        attackers_alive=1,
        defenders_alive=1,
        bomb_planted=True,
        initial_site="A",
    )
    env = build_scenario(spec, small_cfg, seed=0)
    assert env.engine.state.bomb.phase == BombPhase.PLANTED
    assert env.engine.state.bomb.site == "A"
    assert env.engine.state.phase == Phase.POST_PLANT
    attackers_alive = sum(1 for a in env.engine.state.agents if a.alive and a.side == Side.ATTACKER)
    defenders_alive = sum(1 for a in env.engine.state.agents if a.alive and a.side == Side.DEFENDER)
    assert attackers_alive == 1
    assert defenders_alive == 1


def test_all_built_in_scenarios_have_unique_names() -> None:
    """Sanity: built-in scenarios shouldn't collide on name."""
    names = [s.name for s in ALL_SCENARIOS]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------


def test_eval_runner_random_vs_random_completes(small_cfg: KivskiConfig) -> None:
    """A 1-match Random-vs-Random eval finishes cleanly with sensible aggregates."""
    spec = ScenarioSpec(name="smoke", team_size=2, starting_money=800)
    runner = EvalRunner(spec, small_cfg)

    py = RandomBaseline(runner.env.action_space("agent_0"), seed=1)
    pb = RandomBaseline(runner.env.action_space("agent_0"), seed=2)
    result = runner.run(py, pb, num_matches=1, seed=0)

    assert result.scenario == "smoke"
    assert result.num_matches == 1
    # Exactly one match must resolve into win/draw on either side.
    assert result.yellow_match_wins + result.blue_match_wins + result.draws == 1
    # Winrates are in [0, 1].
    assert 0.0 <= result.yellow_winrate <= 1.0
    assert 0.0 <= result.blue_winrate <= 1.0
    assert result.yellow_winrate + result.blue_winrate == pytest.approx(1.0)
    # Round bookkeeping.
    assert result.avg_rounds_per_match >= 1.0
    assert result.avg_match_duration_ticks >= 1.0
    assert len(result.rounds) >= 1
    # Each round has either "attacker" or "defender" as winning_side.
    for r in result.rounds:
        assert r.winning_side in ("attacker", "defender")


def test_eval_runner_result_serialises(tmp_path: Path, small_cfg: KivskiConfig) -> None:
    """EvalResult.to_dict must JSON-serialise without error."""
    spec = ScenarioSpec(name="json_smoke", team_size=2, starting_money=800)
    runner = EvalRunner(spec, small_cfg)
    py = RandomBaseline(runner.env.action_space("agent_0"), seed=1)
    pb = RandomBaseline(runner.env.action_space("agent_0"), seed=2)
    result = runner.run(py, pb, num_matches=1, seed=0)

    payload = result.to_dict()
    encoded = json.dumps(payload)
    # Round-trips.
    decoded = json.loads(encoded)
    assert decoded["scenario"] == "json_smoke"
    assert decoded["num_matches"] == 1
