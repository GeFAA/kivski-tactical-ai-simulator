"""End-to-end smoke tests for the Kivski Tactical AI Simulator.

These tests exercise the full stack -- engine -> env -> baselines -> trainer ->
checkpoint round-trip -- to catch integration regressions before they bite.

They are intentionally lightweight: tiny network sizes, 2 envs, 1-2 episodes.
The point is to verify wiring, not learning quality.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pettingzoo = pytest.importorskip("pettingzoo")

from kivski_sim.config import load_config
from kivski_sim.map_loader import load_map
from kivski_sim.env import KivskiParallelEnv
from kivski_agents.baselines import get_baseline
from kivski_agents.eval.runner import EvalRunner
from kivski_agents.eval.scenarios import ALL_SCENARIOS


def test_environment_runs_a_full_match_with_random_policies():
    cfg = load_config("configs/default.yaml")
    map_data = load_map("dustline")
    env = KivskiParallelEnv(cfg, "dustline", seed=11)
    py = get_baseline("random", env, map_data, seed=1)
    pb = get_baseline("random", env, map_data, seed=2)

    obs, _infos = env.reset(seed=11)
    py.reset(list(obs.keys()))
    pb.reset(list(obs.keys()))

    # Upper bound includes BUY phase per round (buy_time_seconds * tick_rate_hz)
    # plus an extra round-worth of slack to be safe against bookkeeping ticks.
    tick_rate = int(cfg.simulation.tick_rate_hz)
    buy_ticks = int(cfg.simulation.buy_time_seconds) * tick_rate
    per_round_max = int(cfg.simulation.max_ticks_per_round) + buy_ticks
    upper_bound = per_round_max * int(cfg.simulation.max_rounds) + 200

    steps = 0
    while True:
        y_actions, _y_pl = py.act(obs)
        b_actions, _b_pl = pb.act(obs)
        merged = {}
        merged.update(y_actions)
        merged.update(b_actions)
        # Last writer wins when an agent is in both dicts; baselines write all agent ids,
        # so the two dicts overlap. That's fine -- we just need a valid action per agent.
        obs, _r, terminations, truncations, _i = env.step(merged)
        steps += 1
        if all(terminations.values()) or all(truncations.values()):
            break
        assert steps < upper_bound, (
            f"match exceeded upper bound: steps={steps} > {upper_bound} "
            f"(per-round max {per_round_max} * {cfg.simulation.max_rounds} rounds + slack)"
        )

    env.close()
    assert steps > 0


def test_eval_runner_random_vs_scripted_completes():
    cfg = load_config("configs/default.yaml")
    spec = next(s for s in ALL_SCENARIOS if s.name == "default_5v5")
    runner = EvalRunner(spec, cfg, map_name="dustline")
    py = get_baseline("random", runner.env, runner.map_data, seed=42)
    pb = get_baseline("scripted_rush", runner.env, runner.map_data, seed=43)
    result = runner.run(py, pb, num_matches=1, seed=99)
    assert result.num_matches == 1
    assert result.yellow_match_wins + result.blue_match_wins + result.draws == 1
    assert result.rounds, "expected at least one RoundResult in the eval result"


def test_full_training_smoke_one_update(tmp_path: Path, monkeypatch):
    """Tiny end-to-end: build trainer, run two updates, verify checkpoint round-trip."""
    from kivski_agents.training.trainer import Trainer, TrainerConfig
    from kivski_agents.telemetry import NoOpSink

    cfg = load_config("configs/default.yaml")

    tcfg = TrainerConfig(
        total_episodes=2,
        rollout_steps=16,
        num_envs=2,
        checkpoint_every=1,
        eval_every=10_000,         # don't run eval in smoke
        snapshot_every=10_000,     # don't snapshot in smoke
        log_dir=tmp_path / "logs",
        checkpoint_dir=tmp_path / "ckpt",
        device=torch.device("cpu"),
        map_name="dustline",
        run_name="smoke",
    )
    tcfg.log_dir.mkdir(parents=True, exist_ok=True)
    tcfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(cfg, tcfg, telemetry=NoOpSink())
    trainer.train()

    # At least one checkpoint should have landed
    saved = list(tcfg.checkpoint_dir.glob("*.pt"))
    assert saved, f"no checkpoint saved in {tcfg.checkpoint_dir}"

    # Round-trip: load the bundle, run inference for one step
    from kivski_agents.policy_runner import PolicyBundle

    bundle = PolicyBundle.from_checkpoint(saved[0])
    runner = bundle.to_runner(torch.device("cpu"))

    env = KivskiParallelEnv(cfg, "dustline", seed=123)
    obs, _ = env.reset(seed=123)
    runner.reset(list(obs.keys()))
    actions, _comms = runner.act(obs, received_comms={k: {} for k in obs})
    assert set(actions.keys()) == set(obs.keys())
    env.close()


def test_replay_format_round_trips(tmp_path: Path):
    from kivski_sim.replay import (
        ReplayHeader,
        ReplayActionFrame,
        ReplayEventFrame,
        ReplayReader,
        ReplayWriter,
    )

    path = tmp_path / "smoke.replay"
    hdr = ReplayHeader(seed=7, map_name="dustline", team_size=5, kivski_version="0.1.0")
    with ReplayWriter(path, hdr) as w:
        w.write_actions(ReplayActionFrame(tick=0, actions=[{"agent_id": 0, "move": 1, "micro": 0}]))
        w.write_event(ReplayEventFrame(tick=0, kind="round_start", data={"round_id": 0}))
        w.write_actions(ReplayActionFrame(tick=1, actions=[{"agent_id": 0, "move": 2, "micro": 1}]))
        w.write_event(ReplayEventFrame(tick=12, kind="kill", data={"attacker": 0, "victim": 5}))

    r = ReplayReader(path)
    assert r.header.seed == 7
    assert r.header.map_name == "dustline"
    assert sum(1 for _ in r.iter_actions()) == 2
    r2 = ReplayReader(path)
    assert sum(1 for _ in r2.iter_events()) == 2


def test_api_health_and_match_lifecycle_smoke():
    """Tiny synchronous round-trip against the FastAPI app via TestClient."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    from kivski_api.app import create_app

    app = create_app()
    client = fastapi_testclient.TestClient(app)

    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"

    r = client.get("/api/maps")
    assert r.status_code == 200
    assert "dustline" in r.json().get("maps", [])

    r = client.post("/api/match/new", json={"seed": 5, "map": "dustline"})
    assert r.status_code in (200, 201)
    match_id = r.json()["match_id"]

    r = client.delete(f"/api/match/{match_id}")
    assert r.status_code in (200, 204)
