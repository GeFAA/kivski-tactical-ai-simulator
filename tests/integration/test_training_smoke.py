"""Integration smoke tests for the MAPPO training loop.

These tests exercise the trainer's wiring -- vec env + rollout collector +
MAPPO update + league sampling + curriculum -- without depending on a real
telemetry sink or a long training run. The tests are deliberately tiny
(``team_size=2``, ``max_rounds=2``, 2 envs, 16 rollout steps) so they
complete in a few seconds on CPU.

Skip the whole module if PyTorch is not importable, since the trainer
hard-depends on it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


from kivski_agents.metrics import EpisodeStats  # noqa: E402
from kivski_agents.telemetry import NoOpSink  # noqa: E402
from kivski_agents.training.auto_tune import (  # noqa: E402
    detect_optimal_num_envs,
    detect_optimal_workers,
    envs_per_worker_split,
)
from kivski_agents.training.curriculum import CurriculumManager  # noqa: E402
from kivski_agents.training.parallel_vec_env import (  # noqa: E402
    SubprocVecEnv,
    ThreadedVecEnv,
    make_vec_env,
)
from kivski_agents.training.trainer import Trainer, TrainerConfig  # noqa: E402
from kivski_agents.training.vec_env import VecEnvWrapper  # noqa: E402
from kivski_sim.config import KivskiConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture: tiny config + trainer
# ---------------------------------------------------------------------------


def _smoke_cfg() -> KivskiConfig:
    return KivskiConfig.model_validate(
        {
            "seed": 7,
            "simulation": {
                "team_size": 2,
                "max_rounds": 2,
                "side_switch_round": 999,
                "round_time_seconds": 5,
                "bomb_timer_seconds": 3,
                "plant_time_seconds": 1.0,
                "defuse_time_seconds": 1.0,
                "defuse_time_with_kit_seconds": 0.5,
                "buy_time_seconds": 1,
                "tick_rate_hz": 10,
                "max_ticks_per_round": 100,
                "starting_money": 1000,
            },
            "ml": {
                "hidden_size": 32,
                "gru_layers": 1,
                "comm_attention_heads": 2,
                "comm_embedding_dim": 16,
                "ppo_epochs": 1,
                "minibatch_size": 32,
                "learning_rate": 1e-4,
                "entropy_coef": 0.005,
                "value_coef": 0.5,
                "gae_lambda": 0.9,
                "gamma": 0.98,
                "max_grad_norm": 0.5,
            },
            "training": {
                "num_envs": 2,
                "rollout_steps": 16,
                "total_episodes": 2,
                "checkpoint_every_episodes": 100,
                "eval_every_episodes": 100,
                "curriculum": {"enabled": False},
            },
            "league": {
                "population_size": 2,
                "snapshot_every_episodes": 100,
                "exploit_fraction": 0.0,
                "random_fraction": 1.0,
                "scripted_fraction": 0.0,
            },
            "reward_shaping": {"enabled": True, "decay_after_episodes": 1000},
            "telemetry": {"backend": "none"},
        }
    )


def _make_trainer(tmp_path: Path, total_episodes: int = 2) -> Trainer:
    cfg = _smoke_cfg()
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tcfg = TrainerConfig(
        total_episodes=int(total_episodes),
        rollout_steps=16,
        num_envs=2,
        checkpoint_every=10_000,
        eval_every=10_000,
        snapshot_every=10_000,
        log_dir=log_dir,
        checkpoint_dir=ckpt_dir,
        device=torch.device("cpu"),
        map_name="dustline",
        run_name="smoke-test",
        eval_matches=1,
        print_every=10_000,
    )
    trainer = Trainer(cfg, tcfg, telemetry=NoOpSink())
    return trainer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trainer_constructs(tmp_path: Path) -> None:
    """Build a trainer with tiny settings; no rollouts."""
    trainer = _make_trainer(tmp_path)
    assert trainer.episode_count == 0
    assert trainer.update_step == 0
    assert trainer.env_steps == 0
    # The vec env should have the right shape.
    assert trainer.vec_env.num_envs == 2
    assert len(trainer.vec_env.envs) == 2
    assert trainer.vec_env.obs_dim > 0
    # The buffer should be empty.
    assert trainer.buffer.step == 0
    # The league roster should have main + baselines.
    assert "main" in trainer.league.roster
    for bname in ("random", "scripted_hold", "scripted_rush"):
        assert bname in trainer.league.roster
    trainer.vec_env.close()


def test_one_rollout_step(tmp_path: Path) -> None:
    """Run one rollout collection; assert buffer is populated."""
    from kivski_agents.training.rollout_collector import RolloutCollector

    trainer = _make_trainer(tmp_path)
    opponent = trainer.league.sample_opponent(trainer._rng)
    collector = RolloutCollector(
        vec_env=trainer.vec_env,
        training_runner=trainer.training_runner,
        opponent_sampler=opponent,
        buffer=trainer.buffer,
        cfg=trainer.active_cfg,
        device=trainer.device,
    )
    result = collector.collect(T=8)
    assert result.buffer.step == 8
    assert result.total_env_steps == 8 * trainer.vec_env.num_envs
    # Buffer rewards should be finite, masks 0 or 1.
    assert torch.isfinite(result.buffer.rewards).all()
    assert ((result.buffer.masks == 0) | (result.buffer.masks == 1)).all()
    # Comm usage counts should sum to a non-zero count.
    assert sum(result.comm_usage.counts.values()) > 0
    # Last value tensor shape.
    assert result.last_value.shape == (trainer.vec_env.num_envs,)
    trainer.vec_env.close()


def test_one_update_step(tmp_path: Path) -> None:
    """Collect rollouts and run one MAPPO update; assert finite loss."""
    from kivski_agents.training.rollout_collector import RolloutCollector

    trainer = _make_trainer(tmp_path)
    opponent = trainer.league.sample_opponent(trainer._rng)
    collector = RolloutCollector(
        vec_env=trainer.vec_env,
        training_runner=trainer.training_runner,
        opponent_sampler=opponent,
        buffer=trainer.buffer,
        cfg=trainer.active_cfg,
        device=trainer.device,
    )
    result = collector.collect(T=16)
    trainer.buffer.compute_advantages(
        last_value=result.last_value,
        gamma=float(trainer.active_cfg.ml.gamma),
        gae_lambda=float(trainer.active_cfg.ml.gae_lambda),
    )
    loss = trainer.mappo.update(trainer.buffer)
    assert loss.update_count > 0
    # No NaNs in any loss term.
    import math

    assert math.isfinite(float(loss.policy_loss))
    assert math.isfinite(float(loss.value_loss))
    assert math.isfinite(float(loss.entropy))
    assert math.isfinite(float(loss.kl))
    trainer.vec_env.close()


def test_curriculum_disabled_returns_base_config() -> None:
    """When curriculum is off, ``current_config`` is just ``base_cfg``."""
    cfg = _smoke_cfg()
    cm = CurriculumManager(cfg)
    assert cm.enabled is False
    # No mutation on the base config object.
    out = cm.current_config()
    assert out.simulation.team_size == cfg.simulation.team_size
    assert out.simulation.max_rounds == cfg.simulation.max_rounds
    # Advance is a no-op when disabled.
    flipped = cm.advance(episodes_done_in_stage=10)
    assert flipped is False
    assert cm.state.episodes_in_stage == 0


def test_curriculum_enabled_advances() -> None:
    """Enable a tiny curriculum and check stage flipping."""
    cfg_raw = _smoke_cfg().model_dump()
    cfg_raw["training"]["curriculum"] = {
        "enabled": True,
        "stages": [
            {"name": "tiny_1", "team_size": 2, "max_rounds": 2, "use_economy": False, "episodes": 1},
            {"name": "tiny_2", "team_size": 2, "max_rounds": 3, "use_economy": True, "episodes": 1},
        ],
    }
    cfg = KivskiConfig.model_validate(cfg_raw)
    cm = CurriculumManager(cfg)
    assert cm.enabled is True
    assert cm.current_stage_name == "tiny_1"
    flipped = cm.advance(1)
    assert flipped is True
    assert cm.current_stage_name == "tiny_2"
    flipped = cm.advance(1)
    assert flipped is True
    assert cm.finished is True


def test_trainer_full_iteration(tmp_path: Path) -> None:
    """Drive the trainer for a short loop and assert metadata moves forward."""
    trainer = _make_trainer(tmp_path, total_episodes=1)
    # Manually drive a single "train iteration" so the test doesn't need to
    # wait for several full matches to finish.
    initial_update = trainer.update_step
    from kivski_agents.training.rollout_collector import RolloutCollector

    opponent = trainer.league.sample_opponent(trainer._rng)
    collector = RolloutCollector(
        vec_env=trainer.vec_env,
        training_runner=trainer.training_runner,
        opponent_sampler=opponent,
        buffer=trainer.buffer,
        cfg=trainer.active_cfg,
        device=trainer.device,
    )
    result = collector.collect(T=16)
    trainer.buffer.compute_advantages(
        last_value=result.last_value,
        gamma=float(trainer.active_cfg.ml.gamma),
        gae_lambda=float(trainer.active_cfg.ml.gae_lambda),
    )
    trainer.mappo.update(trainer.buffer)
    trainer.update_step += 1
    trainer.env_steps += result.total_env_steps
    for stats in result.episode_stats:
        assert isinstance(stats, EpisodeStats)
        trainer.league.update_elo(
            "main", outcome=0.5 if stats.winner == "draw" else 1.0 if stats.winner == "yellow" else 0.0
        )
    assert trainer.update_step == initial_update + 1
    assert trainer.env_steps > 0
    trainer.vec_env.close()


# ---------------------------------------------------------------------------
# Auto-tuning helpers
# ---------------------------------------------------------------------------


def test_auto_num_envs_returns_reasonable() -> None:
    """``detect_optimal_num_envs`` should return >= 8 and respect overrides."""
    n = detect_optimal_num_envs(None)
    assert n >= 8
    assert n <= 64
    # Explicit request wins.
    assert detect_optimal_num_envs(7) == 7
    assert detect_optimal_num_envs(128) == 128
    # Zero / negative requests fall back to auto-detection.
    assert detect_optimal_num_envs(0) >= 8


def test_auto_workers_returns_reasonable() -> None:
    """``detect_optimal_workers`` should respect the env count cap."""
    n = detect_optimal_workers(16)
    assert 1 <= n <= 16
    assert detect_optimal_workers(1) == 1
    assert detect_optimal_workers(0) == 1


def test_envs_per_worker_split_balanced() -> None:
    """Split must sum to ``num_envs`` and be balanced within +/- 1."""
    split = envs_per_worker_split(10, 3)
    assert split == [4, 3, 3]
    assert sum(envs_per_worker_split(7, 4)) == 7
    assert sum(envs_per_worker_split(32, 8)) == 32


# ---------------------------------------------------------------------------
# Threaded vec env (cheap, always safe to run)
# ---------------------------------------------------------------------------


def test_threaded_vec_env_basic(tmp_path: Path) -> None:
    """Reset + a few steps with the threaded backend should not crash and
    should produce the same observation shapes as the sync wrapper."""
    cfg = _smoke_cfg()
    ve = ThreadedVecEnv(num_envs=4, cfg=cfg, map_name="dustline", base_seed=11, num_workers=2)
    step = ve.reset()
    assert set(step.observations.keys()) == set(ve.agent_names)
    for arr in step.observations.values():
        assert arr.shape == (4, ve.obs_dim)
    # Build dummy actions and step a handful of times.
    rng = np.random.default_rng(0)
    actions = {name: rng.integers(0, 2, size=(4, ve.n_heads)).astype(np.int64) for name in ve.agent_names}
    for _ in range(5):
        out = ve.step(actions)
        for arr in out.observations.values():
            assert arr.shape == (4, ve.obs_dim)
            assert np.isfinite(arr).all()
    ve.close()


def test_make_vec_env_factory_sync_passthrough(tmp_path: Path) -> None:
    """``make_vec_env`` with kind=sync should return the original wrapper."""
    cfg = _smoke_cfg()
    ve = make_vec_env(num_envs=2, cfg=cfg, map_name="dustline", base_seed=3, kind="sync")
    assert isinstance(ve, VecEnvWrapper)
    ve.close()


def test_make_vec_env_invalid_kind_raises() -> None:
    """Unknown ``kind`` values raise a clean ValueError, not a workers panic."""
    cfg = _smoke_cfg()
    with pytest.raises(ValueError):
        make_vec_env(num_envs=2, cfg=cfg, map_name="dustline", base_seed=3, kind="nonsense")


# ---------------------------------------------------------------------------
# Subproc vec env (skipped on CI because Windows ``spawn`` is heavy and
# multiprocessing-in-CI can deadlock under -p no:cacheprovider).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="multiprocessing.spawn can hang on CI runners; covered locally instead",
)
def test_subproc_vec_env_basic() -> None:
    """Spawn 2 workers hosting 4 envs total; reset + step + close cleanly."""
    cfg = _smoke_cfg()
    ve = SubprocVecEnv(num_envs=4, cfg=cfg, map_name="dustline", base_seed=99, num_workers=2)
    try:
        step = ve.reset()
        assert set(step.observations.keys()) == set(ve.agent_names)
        for arr in step.observations.values():
            assert arr.shape == (4, ve.obs_dim)
        rng = np.random.default_rng(123)
        actions = {name: rng.integers(0, 2, size=(4, ve.n_heads)).astype(np.int64) for name in ve.agent_names}
        for _ in range(5):
            out = ve.step(actions)
            for arr in out.observations.values():
                assert arr.shape == (4, ve.obs_dim)
                assert np.isfinite(arr).all()
    finally:
        ve.close()


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="multiprocessing.spawn can hang on CI runners; covered locally instead",
)
def test_subproc_vec_env_fallback_to_sync(monkeypatch) -> None:
    """If ``SubprocVecEnv`` raises during spawn, the factory must fall back."""
    cfg = _smoke_cfg()

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated spawn failure")

    monkeypatch.setattr("kivski_agents.training.parallel_vec_env.SubprocVecEnv.__init__", _boom, raising=True)
    ve = make_vec_env(num_envs=2, cfg=cfg, map_name="dustline", base_seed=7, kind="subproc")
    assert isinstance(ve, VecEnvWrapper), f"expected VecEnvWrapper fallback, got {type(ve)}"
    ve.close()
