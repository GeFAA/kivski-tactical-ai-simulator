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

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


from kivski_agents.metrics import EpisodeStats  # noqa: E402
from kivski_agents.telemetry import NoOpSink  # noqa: E402
from kivski_agents.training.curriculum import CurriculumManager  # noqa: E402
from kivski_agents.training.trainer import Trainer, TrainerConfig  # noqa: E402
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
        trainer.league.update_elo("main", outcome=0.5 if stats.winner == "draw" else 1.0 if stats.winner == "yellow" else 0.0)
    assert trainer.update_step == initial_update + 1
    assert trainer.env_steps > 0
    trainer.vec_env.close()
