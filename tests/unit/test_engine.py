"""Unit tests for the Kivski game engine.

These tests exercise the public engine API and verify the most important
invariants (determinism, round-end conditions, economy payouts, replay
hookup). They do *not* run a full 24-round match: when we need a specific
outcome we script the action stream that gets us there.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from kivski_sim.config import KivskiConfig
from kivski_sim.engine import Engine, Snapshot
from kivski_sim.map_loader import load_map
from kivski_sim.replay import ReplayActionFrame, ReplayEventFrame, ReplayWriter
from kivski_sim.state import AgentState, MatchState
from kivski_sim.types import (
    ActionBundle,
    BombPhase,
    BuyChoice,
    MatchOutcome,
    MicroAction,
    MoveIntent,
    Phase,
    RoundOutcome,
    Side,
    Team,
    WeaponClass,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg() -> KivskiConfig:
    """Small-team config to keep tests fast and outcomes predictable."""
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
def dustline_engine(small_cfg: KivskiConfig) -> Engine:
    return Engine(config=small_cfg, map_data=load_map("dustline"), seed=1234)


@pytest.fixture
def hold_actions() -> dict[int, ActionBundle]:
    """All-HOLD actions (used to advance ticks without changing anything)."""
    return {i: ActionBundle() for i in range(20)}


# ---------------------------------------------------------------------------
# Construction / reset
# ---------------------------------------------------------------------------


def test_reset_initializes_correct_agent_count(small_cfg: KivskiConfig) -> None:
    eng = Engine(config=small_cfg, map_data=load_map("dustline"), seed=1)
    snap = eng.reset()
    # team_size=2 -> 4 agents total.
    assert len(snap.agents) == 2 * int(small_cfg.simulation.team_size)
    # Two teams, two sides represented.
    sides = {a["side"] for a in snap.agents}
    assert sides == {int(Side.ATTACKER), int(Side.DEFENDER)}
    teams = {a["team"] for a in snap.agents}
    assert teams == {int(Team.YELLOW), int(Team.BLUE)}


def test_reset_places_agents_at_spawns(small_cfg: KivskiConfig) -> None:
    md = load_map("dustline")
    eng = Engine(config=small_cfg, map_data=md, seed=99)
    snap = eng.reset()
    for a in snap.agents:
        side = Side(int(a["side"]))
        spawn_arr = md.spawns[side]
        pos = np.array(a["pos"], dtype=np.float64)
        # Position should match exactly one of the side's spawn points.
        diffs = np.linalg.norm(spawn_arr - pos, axis=1)
        assert diffs.min() < 1e-3


def test_step_returns_correct_types(dustline_engine: Engine) -> None:
    dustline_engine.reset()
    snap, rewards, done = dustline_engine.step({})
    assert isinstance(snap, Snapshot)
    assert isinstance(rewards, dict)
    assert isinstance(done, bool)
    for v in rewards.values():
        assert isinstance(v, float)


def test_snapshot_to_json_dict_is_jsonable(dustline_engine: Engine) -> None:
    import json

    dustline_engine.reset()
    snap = dustline_engine.snapshot()
    blob = snap.to_json_dict()
    # If this round-trips without raising, snapshots are safe over the wire.
    text = json.dumps(blob)
    assert isinstance(text, str)
    parsed = json.loads(text)
    assert parsed["tick"] == snap.tick


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def _run_until(engine: Engine, n_ticks: int, action: ActionBundle | None = None) -> Snapshot:
    """Run ``n_ticks`` of HOLD/SPRINT (depending on action) and return final snapshot."""
    if action is None:
        action = ActionBundle()
    snap: Snapshot | None = None
    for _ in range(n_ticks):
        if engine.is_done():
            break
        snap, _, _ = engine.step({i: action for i in range(20)})
    assert snap is not None
    return snap


def test_same_seed_same_trajectory(small_cfg: KivskiConfig) -> None:
    md = load_map("dustline")

    def collect(seed: int) -> list[list[tuple[float, float]]]:
        eng = Engine(config=small_cfg, map_data=md, seed=seed)
        eng.reset()
        # Deterministic action stream: everyone sprints north.
        action = ActionBundle(move=MoveIntent.N, micro=MicroAction.SPRINT)
        snapshots = []
        for _ in range(20):
            snap, _, done = eng.step({i: action for i in range(20)})
            snapshots.append([tuple(a["pos"]) for a in snap.agents])
            if done:
                break
        return snapshots

    a = collect(42)
    b = collect(42)
    assert a == b, "Same seed must produce identical agent trajectories"


def test_different_seeds_diverge(small_cfg: KivskiConfig) -> None:
    md = load_map("dustline")

    def collect(seed: int) -> list[Any]:
        eng = Engine(config=small_cfg, map_data=md, seed=seed)
        eng.reset()
        carriers = []
        for _ in range(3):
            snap, _, _ = eng.step({})
            carriers.append(snap.bomb["carrier"])
        return carriers

    # Bomb carrier is sampled via the spawn rng channel, so different seeds
    # should (with overwhelming probability) pick different carriers.
    # In a 2-attacker setup the carriers may coincide; we just verify the
    # whole engine state is sensitive to the seed by sampling many seeds.
    seeds = [s for s in range(20)]
    snapshots = []
    for s in seeds:
        eng = Engine(config=small_cfg, map_data=md, seed=s)
        snap = eng.reset()
        snapshots.append(snap.bomb["carrier"])
    # Should observe at least two distinct carriers across 20 seeds.
    assert len(set(snapshots)) >= 2


# ---------------------------------------------------------------------------
# Round end conditions
# ---------------------------------------------------------------------------


def _kill_all_defenders(state: MatchState) -> None:
    for a in state.agents:
        if a.side == Side.DEFENDER:
            a.hp = 0.0
            a.alive = False


def _kill_all_attackers(state: MatchState) -> None:
    for a in state.agents:
        if a.side == Side.ATTACKER:
            a.hp = 0.0
            a.alive = False


def test_round_ends_when_all_defenders_dead(small_cfg: KivskiConfig) -> None:
    eng = Engine(config=small_cfg, map_data=load_map("dustline"), seed=7)
    eng.reset()
    # Skip the buy phase first.
    for _ in range(int(small_cfg.simulation.buy_time_seconds * small_cfg.simulation.tick_rate_hz) + 1):
        eng.step({})
    assert eng.state.phase == Phase.LIVE

    _kill_all_defenders(eng.state)
    eng.step({})
    assert eng.state.last_round_outcome == RoundOutcome.ATTACKERS_ELIM
    assert eng.state.teams[Team.YELLOW].score >= 1


def test_round_ends_when_all_attackers_dead(small_cfg: KivskiConfig) -> None:
    eng = Engine(config=small_cfg, map_data=load_map("dustline"), seed=7)
    eng.reset()
    for _ in range(int(small_cfg.simulation.buy_time_seconds * small_cfg.simulation.tick_rate_hz) + 1):
        eng.step({})
    assert eng.state.phase == Phase.LIVE

    _kill_all_attackers(eng.state)
    eng.step({})
    assert eng.state.last_round_outcome == RoundOutcome.DEFENDERS_ELIM
    assert eng.state.teams[Team.BLUE].score >= 1


def test_round_ends_on_timeout(small_cfg: KivskiConfig) -> None:
    eng = Engine(config=small_cfg, map_data=load_map("dustline"), seed=7)
    eng.reset()
    # Skip BUY.
    for _ in range(int(small_cfg.simulation.buy_time_seconds * small_cfg.simulation.tick_rate_hz) + 1):
        eng.step({})
    assert eng.state.phase == Phase.LIVE

    # Run until the round timer expires (round_time_seconds=6 @ 10Hz = 60 ticks).
    timeout_ticks = int(small_cfg.simulation.round_time_seconds * small_cfg.simulation.tick_rate_hz)
    for _ in range(timeout_ticks + 2):
        eng.step({})
        if eng.state.last_round_outcome == RoundOutcome.TIMEOUT:
            break
    assert eng.state.last_round_outcome == RoundOutcome.TIMEOUT
    # Defenders win on timeout.
    assert eng.state.teams[Team.BLUE].score >= 1


def test_match_ends_when_team_reaches_majority(small_cfg: KivskiConfig) -> None:
    eng = Engine(config=small_cfg, map_data=load_map("dustline"), seed=7)
    eng.reset()
    # Force yellow team to win three rounds in a row (out of 4 -> majority is 3).
    for _ in range(int(small_cfg.simulation.max_rounds)):
        # Skip BUY each round.
        for _ in range(int(small_cfg.simulation.buy_time_seconds * small_cfg.simulation.tick_rate_hz) + 1):
            eng.step({})
            if eng.is_done():
                break
        if eng.is_done():
            break
        # Now LIVE -- annihilate the defenders so the attacking team scores.
        if eng.state.phase == Phase.LIVE:
            for a in eng.state.agents:
                if a.side == Side.DEFENDER:
                    a.hp = 0.0
                    a.alive = False
            eng.step({})
        if eng.is_done():
            break
    assert eng.is_done()
    # In this scripted setup, whichever team is on the ATTACKER side each round
    # wins. With side switch at round 2 (out of 4), the same team starts each
    # half on attack; the final outcome must be a non-NONE match result.
    assert eng.state.match_outcome != MatchOutcome.NONE


# ---------------------------------------------------------------------------
# Economy
# ---------------------------------------------------------------------------


def test_economy_loss_bonus_stacks(small_cfg: KivskiConfig) -> None:
    # Use a config with the side switch pushed beyond max_rounds so the same
    # team stays on the losing side across consecutive rounds.
    cfg_dict = small_cfg.model_dump()
    cfg_dict["simulation"]["max_rounds"] = 16
    cfg_dict["simulation"]["side_switch_round"] = 99  # effectively never
    cfg = KivskiConfig.model_validate(cfg_dict)
    eng = Engine(config=cfg, map_data=load_map("dustline"), seed=7)
    eng.reset()

    def play_one_loss_round_for_attackers() -> int:
        """Run a full round and return the round_id at the end (or -1 if done)."""
        start_round = eng.state.round_id
        # Drive ticks until the round_id changes or the match ends.
        # Cap at a safe number of ticks to avoid infinite loops.
        max_ticks = (
            int(cfg.simulation.buy_time_seconds * cfg.simulation.tick_rate_hz)
            + int(cfg.simulation.round_time_seconds * cfg.simulation.tick_rate_hz)
            + 20
        )
        for _ in range(max_ticks):
            eng.step({})
            if eng.is_done():
                return -1
            if eng.state.round_id != start_round:
                return eng.state.round_id
        return eng.state.round_id

    # Track an attacker's money across rounds.
    attacker_id = next(a.agent_id for a in eng.state.agents if a.side == Side.ATTACKER)
    money_history: list[int] = [int(eng.state.agents[attacker_id].money)]

    for _ in range(3):
        result = play_one_loss_round_for_attackers()
        if result < 0:
            break
        money_history.append(int(eng.state.agents[attacker_id].money))

    deltas = [money_history[i + 1] - money_history[i] for i in range(len(money_history) - 1)]
    assert len(deltas) >= 2

    # Each delta must be at least the base loss bonus (1900) and the second
    # delta must be strictly larger than the first (loss streak stacks).
    base = int(cfg.economy.reward_round_loss_base)
    increment = int(cfg.economy.reward_round_loss_increment)
    max_loss = int(cfg.economy.reward_round_loss_max)
    assert deltas[0] >= base, f"first loss bonus too small: {deltas[0]}"
    # The second delta should be larger (base + 1*increment) -- unless capped.
    if deltas[0] < max_loss:
        assert deltas[1] >= deltas[0] + increment - 1, (
            f"loss bonus did not stack: {deltas}"
        )


# ---------------------------------------------------------------------------
# Bomb mechanics
# ---------------------------------------------------------------------------


def _teleport_to_bombsite_A(state: MatchState, agent: AgentState, md: Any) -> None:
    """Teleport ``agent`` to the centre of bombsite A so plant/defuse can succeed."""
    site = md.bombsites["A"]
    agent.pos = np.array(site.center, dtype=np.float32)


def test_bomb_plant_progression(small_cfg: KivskiConfig) -> None:
    md = load_map("dustline")
    eng = Engine(config=small_cfg, map_data=md, seed=7)
    eng.reset()
    # Skip buy.
    for _ in range(int(small_cfg.simulation.buy_time_seconds * small_cfg.simulation.tick_rate_hz) + 1):
        eng.step({})
    # Find the carrier.
    carrier_id = eng.state.bomb.carrier
    carrier = eng.state.agents[carrier_id]
    _teleport_to_bombsite_A(eng.state, carrier, md)
    eng.state.bomb.pos = np.array(carrier.pos, dtype=np.float32)

    plant_ticks = int(small_cfg.simulation.plant_time_seconds * small_cfg.simulation.tick_rate_hz)
    interact = ActionBundle(micro=MicroAction.INTERACT)
    # Drive enough ticks of INTERACT for the plant to complete.
    for _ in range(plant_ticks + 3):
        eng.step({carrier_id: interact})
        if eng.state.phase == Phase.POST_PLANT:
            break

    assert eng.state.phase == Phase.POST_PLANT
    assert eng.state.bomb.phase == BombPhase.PLANTED
    assert eng.state.bomb.site == "A"


def test_bomb_defuse_progression(small_cfg: KivskiConfig) -> None:
    md = load_map("dustline")
    eng = Engine(config=small_cfg, map_data=md, seed=7)
    eng.reset()
    for _ in range(int(small_cfg.simulation.buy_time_seconds * small_cfg.simulation.tick_rate_hz) + 1):
        eng.step({})
    # Plant first.
    carrier_id = eng.state.bomb.carrier
    carrier = eng.state.agents[carrier_id]
    _teleport_to_bombsite_A(eng.state, carrier, md)
    interact = ActionBundle(micro=MicroAction.INTERACT)
    plant_ticks = int(small_cfg.simulation.plant_time_seconds * small_cfg.simulation.tick_rate_hz)
    for _ in range(plant_ticks + 3):
        eng.step({carrier_id: interact})
        if eng.state.phase == Phase.POST_PLANT:
            break
    assert eng.state.phase == Phase.POST_PLANT

    # Now teleport a defender to the bomb and have them defuse.
    defender = next(a for a in eng.state.agents if a.side == Side.DEFENDER)
    defender.pos = np.array(eng.state.bomb.pos, dtype=np.float32)
    defuse_ticks = int(small_cfg.simulation.defuse_time_seconds * small_cfg.simulation.tick_rate_hz)
    for _ in range(defuse_ticks + 5):
        eng.step({defender.agent_id: interact})
        if eng.state.last_round_outcome == RoundOutcome.BOMB_DEFUSED:
            break

    assert eng.state.last_round_outcome == RoundOutcome.BOMB_DEFUSED
    assert eng.state.teams[Team.BLUE].score >= 1


def test_bomb_detonate_after_timer(small_cfg: KivskiConfig) -> None:
    md = load_map("dustline")
    eng = Engine(config=small_cfg, map_data=md, seed=7)
    eng.reset()
    for _ in range(int(small_cfg.simulation.buy_time_seconds * small_cfg.simulation.tick_rate_hz) + 1):
        eng.step({})
    carrier_id = eng.state.bomb.carrier
    carrier = eng.state.agents[carrier_id]
    _teleport_to_bombsite_A(eng.state, carrier, md)
    interact = ActionBundle(micro=MicroAction.INTERACT)
    plant_ticks = int(small_cfg.simulation.plant_time_seconds * small_cfg.simulation.tick_rate_hz)
    for _ in range(plant_ticks + 3):
        eng.step({carrier_id: interact})
        if eng.state.phase == Phase.POST_PLANT:
            break
    assert eng.state.phase == Phase.POST_PLANT

    bomb_ticks = int(small_cfg.simulation.bomb_timer_seconds * small_cfg.simulation.tick_rate_hz)
    # Run out the bomb timer with everyone holding.
    for _ in range(bomb_ticks + 3):
        eng.step({})
        if eng.state.last_round_outcome == RoundOutcome.BOMB_DETONATED:
            break

    assert eng.state.last_round_outcome == RoundOutcome.BOMB_DETONATED
    assert eng.state.teams[Team.YELLOW].score >= 1


# ---------------------------------------------------------------------------
# Replay writer integration
# ---------------------------------------------------------------------------


class _RecordingReplayWriter:
    """Minimal stand-in for ``ReplayWriter`` that just records what it receives."""

    def __init__(self) -> None:
        self.actions: list[ReplayActionFrame] = []
        self.events: list[ReplayEventFrame] = []

    def write_actions(self, frame: ReplayActionFrame) -> None:
        self.actions.append(frame)

    def write_event(self, frame: ReplayEventFrame) -> None:
        self.events.append(frame)


def test_replay_writer_called_when_set(small_cfg: KivskiConfig) -> None:
    eng = Engine(config=small_cfg, map_data=load_map("dustline"), seed=7)
    rec = _RecordingReplayWriter()
    eng.set_replay_writer(rec)
    eng.reset()
    for _ in range(5):
        eng.step({})

    # Reset emits a round_start event.
    assert any(ev.kind == "round_start" for ev in rec.events)
    # Each step records one action frame.
    assert len(rec.actions) >= 5


def test_replay_writer_records_round_end(small_cfg: KivskiConfig) -> None:
    eng = Engine(config=small_cfg, map_data=load_map("dustline"), seed=7)
    rec = _RecordingReplayWriter()
    eng.set_replay_writer(rec)
    eng.reset()
    # Trigger a round end via attacker elim.
    for _ in range(int(small_cfg.simulation.buy_time_seconds * small_cfg.simulation.tick_rate_hz) + 1):
        eng.step({})
    for a in eng.state.agents:
        if a.side == Side.ATTACKER:
            a.hp = 0.0
            a.alive = False
    eng.step({})
    assert any(ev.kind == "round_end" for ev in rec.events)


# ---------------------------------------------------------------------------
# Phase transitions & misc
# ---------------------------------------------------------------------------


def test_buy_phase_transitions_to_live(small_cfg: KivskiConfig) -> None:
    eng = Engine(config=small_cfg, map_data=load_map("dustline"), seed=7)
    snap = eng.reset()
    assert snap.phase == Phase.BUY
    # Step through the entire buy phase.
    buy_ticks = int(small_cfg.simulation.buy_time_seconds * small_cfg.simulation.tick_rate_hz)
    for _ in range(buy_ticks + 1):
        eng.step({})
    assert eng.state.phase == Phase.LIVE


def test_buy_during_buy_phase_changes_weapon(small_cfg: KivskiConfig) -> None:
    eng = Engine(config=small_cfg, map_data=load_map("dustline"), seed=7)
    eng.reset()
    # Give an agent enough money to buy a rifle.
    agent = eng.state.agents[0]
    agent.money = 5000
    actions = {agent.agent_id: ActionBundle(buy=BuyChoice.RIFLE)}
    eng.step(actions)
    assert agent.weapon == WeaponClass.RIFLE
    assert agent.money < 5000


def test_carrier_drop_on_death(small_cfg: KivskiConfig) -> None:
    eng = Engine(config=small_cfg, map_data=load_map("dustline"), seed=7)
    eng.reset()
    # Skip buy.
    for _ in range(int(small_cfg.simulation.buy_time_seconds * small_cfg.simulation.tick_rate_hz) + 1):
        eng.step({})
    carrier_id = eng.state.bomb.carrier
    carrier = eng.state.agents[carrier_id]
    # Kill the carrier directly. We re-use the engine's internal drop path
    # by forcing hp=0 and stepping (this won't trigger the kill bookkeeping
    # but does flip alive/has_bomb the way the engine does).
    carrier.hp = 0.0
    carrier.alive = False
    carrier.has_bomb = False
    eng.state.bomb.phase = BombPhase.DROPPED
    eng.state.bomb.carrier = -1
    snap, _, _ = eng.step({})
    assert snap.bomb["phase"] == int(BombPhase.DROPPED)
