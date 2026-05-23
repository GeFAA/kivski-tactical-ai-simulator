"""Unit tests for the PettingZoo parallel-API wrapper.

These tests pin down the public surface of :class:`KivskiParallelEnv`:

* agent count and naming follow ``2 * team_size`` agents named ``agent_<id>``
* the observation vector length matches the configuration-driven schema
* the action space is a five-component MultiDiscrete
* ``step`` returns a dict per agent for obs / rewards / terminations
* ``reset`` clears the last-known enemy memory
* enemies entering FoV populate that memory
* the engine's outcome rewards reach the corresponding agents
* terminations all flip to True once the match ends
* identical seeds with identical action streams produce identical observation
  sequences (the wrapper must not break engine determinism)
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces
from kivski_sim.config import KivskiConfig
from kivski_sim.env import KivskiParallelEnv, agent_index, agent_name
from kivski_sim.map_loader import load_map
from kivski_sim.obs_decoder import get_observation_dim
from kivski_sim.types import (
    BuyChoice,
    CommAction,
    MicroAction,
    MoveIntent,
    Phase,
    Side,
    Team,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg() -> KivskiConfig:
    """Small-team config so tests run in a few hundred milliseconds."""
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
    return KivskiParallelEnv(config=small_cfg, map_name="dustline", seed=1234, map_data=load_map("dustline"))


def _hold_actions(env: KivskiParallelEnv) -> dict[str, np.ndarray]:
    """All-HOLD action dict for every agent."""
    return {name: np.zeros(5, dtype=np.int64) for name in env.possible_agents}


# ---------------------------------------------------------------------------
# Construction / spaces
# ---------------------------------------------------------------------------


def test_env_creates_correct_agent_count(small_cfg: KivskiConfig) -> None:
    env = KivskiParallelEnv(config=small_cfg, map_name="dustline", seed=1, map_data=load_map("dustline"))
    n = 2 * int(small_cfg.simulation.team_size)
    assert len(env.possible_agents) == n
    assert env.possible_agents == [f"agent_{i}" for i in range(n)]
    # After reset the live "agents" list mirrors possible_agents.
    obs, infos = env.reset(seed=42)
    assert set(obs.keys()) == set(env.possible_agents)
    assert set(infos.keys()) == set(env.possible_agents)
    assert env.agents == env.possible_agents


def test_observation_space_dimension(env: KivskiParallelEnv, small_cfg: KivskiConfig) -> None:
    dim = get_observation_dim(small_cfg)
    assert env.observation_dim == dim
    obs, _ = env.reset(seed=0)
    for name, vec in obs.items():
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert vec.shape == (dim,), f"{name} obs shape {vec.shape} != ({dim},)"
        space = env.observation_space(name)
        assert isinstance(space, spaces.Box)
        assert space.shape == (dim,)


def test_action_space_is_multidiscrete(env: KivskiParallelEnv) -> None:
    space = env.action_space("agent_0")
    assert isinstance(space, spaces.MultiDiscrete)
    # [move, micro, comm, buy, aim_target]
    nvec = np.asarray(space.nvec)
    assert nvec.shape == (5,)
    assert int(nvec[0]) == len(MoveIntent)
    assert int(nvec[1]) == len(MicroAction)
    assert int(nvec[2]) == len(CommAction)
    assert int(nvec[3]) == len(BuyChoice)
    # aim_target = 2*team_size + 1 (no-target + 2*team_size-1 others + self).
    expected_aim = 2 * 2 + 1  # team_size=2 in the fixture
    assert int(nvec[4]) == expected_aim


# ---------------------------------------------------------------------------
# Step return shape
# ---------------------------------------------------------------------------


def test_step_returns_correct_keys(env: KivskiParallelEnv) -> None:
    env.reset(seed=0)
    actions = _hold_actions(env)
    obs, rewards, terms, truncs, infos = env.step(actions)
    expected = set(env.possible_agents)
    assert set(obs.keys()) == expected
    assert set(rewards.keys()) == expected
    assert set(terms.keys()) == expected
    assert set(truncs.keys()) == expected
    assert set(infos.keys()) == expected
    for name in env.possible_agents:
        assert isinstance(rewards[name], float)
        assert isinstance(terms[name], bool)
        assert isinstance(truncs[name], bool)
        assert infos[name]["round_id"] == env.engine.state.round_id


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_reset_clears_last_known(env: KivskiParallelEnv) -> None:
    env.reset(seed=0)
    # Manually inject a stale "last known" entry to make sure reset wipes it.
    name = "agent_0"
    mem = env._memory[name]
    from kivski_sim.env import _LastKnownEnemy

    mem.last_known[99] = _LastKnownEnemy(
        enemy_id=99,
        last_pos=np.zeros(2, dtype=np.float32),
        last_tick=0,
        last_weapon=env.engine.state.agents[0].weapon,
        was_alive=True,
        last_distance=5.0,
    )
    assert 99 in env._memory[name].last_known

    env.reset(seed=0)
    assert env._memory[name].last_known == {}


def test_last_known_updates_when_enemy_in_fov(small_cfg: KivskiConfig) -> None:
    """Place two opposing agents in open space and verify last_known fills."""
    env = KivskiParallelEnv(config=small_cfg, map_name="dustline", seed=2025, map_data=load_map("dustline"))
    env.reset(seed=2025)
    # The dustline map has an open corridor along x=30 in the centre.
    open_pos_a = np.array([30.0, 19.0], dtype=np.float32)
    open_pos_b = np.array([30.0, 21.0], dtype=np.float32)
    state = env.engine.state
    state.agents[0].pos = open_pos_a.copy()
    state.agents[0].facing = float(np.pi / 2)  # facing +y -> toward defender
    state.agents[1].pos = np.array([30.0, 18.0], dtype=np.float32)
    state.agents[2].pos = open_pos_b.copy()
    state.agents[3].pos = np.array([30.0, 22.0], dtype=np.float32)

    # Force LIVE phase so the engine actually evaluates FoV during step.
    state.phase = Phase.LIVE
    state.phase_ticks_remaining = 50

    actions = _hold_actions(env)
    env.step(actions)
    mem_attacker = env._memory["agent_0"]
    # Defenders are at known ids 2 and 3; expect at least one to be discovered.
    assert any(eid in mem_attacker.last_known for eid in (2, 3)), (
        f"agent_0 should have seen at least one defender; last_known={list(mem_attacker.last_known)}"
    )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def test_round_outcome_assigns_rewards_correctly(small_cfg: KivskiConfig) -> None:
    """When the engine ends a round the wrapper must forward its +/-1 rewards."""
    env = KivskiParallelEnv(config=small_cfg, map_name="dustline", seed=7, map_data=load_map("dustline"))
    env.reset(seed=7)
    # Disable shaping so only the outcome reward survives.
    env.set_shaping_factor(0.0)
    # Capture the side mapping *before* the engine reshuffles agents at round end.
    state = env.engine.state
    side_by_id = {int(a.agent_id): a.side for a in state.agents}
    # Force a round end by killing all defenders directly (simulate engine).
    state.phase = Phase.LIVE
    state.phase_ticks_remaining = 50
    for a in state.agents:
        if a.side == Side.DEFENDER:
            a.hp = 0.0
            a.alive = False
    actions = _hold_actions(env)
    _, rewards, _terms, _, _ = env.step(actions)
    # Attackers get +1, defenders get -1 from the engine's _end_round path,
    # routed through the wrapper -- side mapping captured pre-step.
    attacker_rewards = [rewards[agent_name(aid)] for aid, side in side_by_id.items() if side == Side.ATTACKER]
    defender_rewards = [rewards[agent_name(aid)] for aid, side in side_by_id.items() if side == Side.DEFENDER]
    assert all(r >= 1.0 for r in attacker_rewards)
    assert all(r <= -1.0 for r in defender_rewards)


def test_terminations_when_match_over(small_cfg: KivskiConfig) -> None:
    env = KivskiParallelEnv(config=small_cfg, map_name="dustline", seed=99, map_data=load_map("dustline"))
    env.reset(seed=99)
    # Push the yellow team to the win threshold (needed = max_rounds//2 + 1).
    needed = int(small_cfg.simulation.max_rounds) // 2 + 1
    env.engine.state.teams[Team.YELLOW].score = needed - 1
    # Force a single round end by wiping the defenders.
    state = env.engine.state
    state.phase = Phase.LIVE
    state.phase_ticks_remaining = 50
    for a in state.agents:
        if a.side == Side.DEFENDER:
            a.hp = 0.0
            a.alive = False
    _, _, terms, _, _ = env.step(_hold_actions(env))
    assert all(terms.values()), terms
    assert env.engine.state.match_outcome.value != 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_with_seed(small_cfg: KivskiConfig) -> None:
    """Two envs seeded identically and driven by identical actions agree."""
    map_data = load_map("dustline")
    env_a = KivskiParallelEnv(config=small_cfg, map_name="dustline", seed=2026, map_data=map_data)
    env_b = KivskiParallelEnv(config=small_cfg, map_name="dustline", seed=2026, map_data=map_data)
    obs_a0, _ = env_a.reset(seed=2026)
    obs_b0, _ = env_b.reset(seed=2026)
    for name in env_a.possible_agents:
        np.testing.assert_allclose(obs_a0[name], obs_b0[name])

    # Deterministic action stream: cycle through a handful of MultiDiscrete vectors.
    rng = np.random.default_rng(0)
    n_agents = len(env_a.possible_agents)
    nvec = np.asarray(env_a.action_space("agent_0").nvec)
    action_stream: list[dict[str, np.ndarray]] = []
    for _ in range(15):
        step_actions: dict[str, np.ndarray] = {}
        for i in range(n_agents):
            vals = (rng.integers(low=np.zeros_like(nvec), high=nvec)).astype(np.int64)
            step_actions[agent_name(i)] = vals
        action_stream.append(step_actions)

    for actions in action_stream:
        obs_a, r_a, term_a, _, _ = env_a.step(actions)
        obs_b, r_b, term_b, _, _ = env_b.step(actions)
        for name in env_a.possible_agents:
            np.testing.assert_allclose(obs_a[name], obs_b[name], atol=1e-6, err_msg=name)
            assert r_a[name] == pytest.approx(r_b[name])
            assert term_a[name] == term_b[name]
        if all(term_a.values()):
            break


# ---------------------------------------------------------------------------
# Utility / sanity
# ---------------------------------------------------------------------------


def test_agent_name_round_trip() -> None:
    for i in range(10):
        assert agent_index(agent_name(i)) == i


def test_render_returns_snapshot(env: KivskiParallelEnv) -> None:
    env.reset(seed=0)
    snap = env.render()
    assert hasattr(snap, "agents")
    assert hasattr(snap, "bomb")


def test_env_frame_skip(small_cfg: KivskiConfig) -> None:
    """A frame-skipped env must call engine.step() exactly N times per env.step()."""
    map_data = load_map("dustline")
    env_skip = KivskiParallelEnv(
        config=small_cfg,
        map_name="dustline",
        seed=11,
        map_data=map_data,
        frame_skip=4,
    )
    env_skip.reset(seed=11)
    assert env_skip.frame_skip == 4
    # Force LIVE phase so the engine tick rate is uncomplicated by buy phases.
    env_skip.engine.state.phase = Phase.LIVE
    env_skip.engine.state.phase_ticks_remaining = 500
    pre_tick = int(env_skip.engine.state.tick)
    env_skip.step(_hold_actions(env_skip))
    post_tick = int(env_skip.engine.state.tick)
    # Engine should have advanced by frame_skip ticks (assuming no early
    # match termination, which a HOLD action cannot trigger this fast).
    assert post_tick - pre_tick == 4, f"expected 4 inner ticks, got {post_tick - pre_tick}"


def test_env_frame_skip_default_one(small_cfg: KivskiConfig) -> None:
    """Default frame_skip is 1 unless overridden -- live viewer compatibility."""
    map_data = load_map("dustline")
    env_default = KivskiParallelEnv(
        config=small_cfg, map_name="dustline", seed=11, map_data=map_data
    )
    env_default.reset(seed=11)
    assert env_default.frame_skip == int(getattr(small_cfg.simulation, "frame_skip", 1) or 1)


def test_env_frame_skip_rewards_accumulate(small_cfg: KivskiConfig) -> None:
    """Rewards from N inner ticks must sum into the returned reward."""
    map_data = load_map("dustline")
    env_skip = KivskiParallelEnv(
        config=small_cfg, map_name="dustline", seed=22, map_data=map_data, frame_skip=3
    )
    env_default = KivskiParallelEnv(
        config=small_cfg, map_name="dustline", seed=22, map_data=map_data, frame_skip=1
    )
    env_skip.reset(seed=22)
    env_default.reset(seed=22)
    # Force LIVE and a long horizon so neither env terminates early.
    for env in (env_skip, env_default):
        env.engine.state.phase = Phase.LIVE
        env.engine.state.phase_ticks_remaining = 500

    actions = _hold_actions(env_skip)
    _, rewards_skip, _, _, _ = env_skip.step(actions)
    sum_default = {name: 0.0 for name in env_default.possible_agents}
    for _ in range(3):
        _, r_step, _, _, _ = env_default.step(actions)
        for name, r in r_step.items():
            sum_default[name] += float(r)
    # Both runs operate on the same engine seed + identical actions, so the
    # frame-skipped sum should match the explicit step-by-step accumulation.
    for name in env_skip.possible_agents:
        assert rewards_skip[name] == pytest.approx(sum_default[name], abs=1e-6), name


def test_env_set_curriculum_stage_gates_features(small_cfg: KivskiConfig) -> None:
    """Disabling the ``survive`` feature must zero the survival shaping reward."""
    map_data = load_map("dustline")
    env_with = KivskiParallelEnv(
        config=small_cfg, map_name="dustline", seed=33, map_data=map_data, frame_skip=1
    )
    env_with.reset(seed=33)
    env_with.engine.state.phase = Phase.LIVE
    env_with.engine.state.phase_ticks_remaining = 50
    # Reset prev snapshot so reward delta-comparison is clean.
    env_with.set_shaping_factor(1.0)
    env_without = KivskiParallelEnv(
        config=small_cfg, map_name="dustline", seed=33, map_data=map_data, frame_skip=1
    )
    env_without.reset(seed=33)
    env_without.engine.state.phase = Phase.LIVE
    env_without.engine.state.phase_ticks_remaining = 50
    env_without.set_shaping_factor(1.0)
    # Disable everything *except* "survive" on the gated env -- both should
    # then produce identical per-step reward because HOLD doesn't deal/take
    # damage or trigger any of the other buckets.
    env_with.set_curriculum_stage("survive_only", ["survive"])

    actions = _hold_actions(env_with)
    _, r_gated, _, _, _ = env_with.step(actions)
    _, r_open, _, _, _ = env_without.step(actions)
    # All alive agents got survival per tick on both envs -> equal totals.
    for name in env_with.possible_agents:
        assert r_gated[name] == pytest.approx(r_open[name], abs=1e-6), name

    # Now disable "survive" on the gated env and verify the survival bonus
    # disappears entirely (HOLDing alive agents would otherwise net a tiny
    # positive reward).
    env_with.set_curriculum_stage("killshoot", ["damage_dealt"])
    _, r_no_survive, _, _, _ = env_with.step(actions)
    for name in env_with.possible_agents:
        # HOLD + no damage + no plant -> reward should be 0 when survive is gated.
        assert abs(float(r_no_survive[name])) < 1e-6, (name, r_no_survive[name])


def test_step_with_comms_payload_in_info(env: KivskiParallelEnv) -> None:
    env.reset(seed=0)
    # Force LIVE so that the engine actually broadcasts comm messages
    # (during BUY all action fields except `buy` are ignored).
    env.engine.state.phase = Phase.LIVE
    env.engine.state.phase_ticks_remaining = 50
    actions = _hold_actions(env)
    # Make agent 0 emit a comm message with a payload; teammates should see it.
    actions["agent_0"] = np.array([0, 0, int(CommAction.PING_LOCATION), 0, 0], dtype=np.int64)
    payload = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    _obs, _r, _t, _tr, infos = env.step_with_comms(actions, {"agent_0": payload})
    # Teammates of agent_0 are agent_1 (small team_size=2; both on YELLOW).
    receiver_info = infos["agent_1"]
    assert isinstance(receiver_info["comm_messages"], dict)
    assert isinstance(receiver_info["comm_attention_mask"], np.ndarray)
    assert 0 in receiver_info["comm_messages"], receiver_info["comm_messages"]
    np.testing.assert_allclose(receiver_info["comm_messages"][0], payload)
    # The attention mask should mark sender id 0.
    assert receiver_info["comm_attention_mask"][0] == pytest.approx(1.0)
