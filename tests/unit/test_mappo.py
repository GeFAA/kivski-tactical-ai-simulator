"""Unit tests for the MAPPO buffer + trainer.

These tests use synthetic transitions instead of a live env so the suite
runs in well under a second. The behavioural smoke check
(``test_trainer_update_reduces_loss_on_synthetic_data``) overfits the
trainer on a tiny, repeated batch and asserts that *something* improves --
the goal is to flush out shape bugs, NaNs, or gradient-killing modules,
not to validate convergence on real data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from kivski_agents.buffer import RolloutBatch, RolloutBuffer
from kivski_agents.mappo import MAPPOTrainer
from kivski_agents.networks.actor_critic import KivskiActorCritic
from kivski_sim.config import MLConfig

# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------


T = 5
N_ENVS = 3
N_AGENTS = 4
OBS_DIM = 16
JOINT_OBS_DIM = N_AGENTS * OBS_DIM
HIDDEN = 16
COMM_VAL = 8
COMM_SIG = 8
COMM_HEADS = 2
ACTION_DIMS = [6, 9, 8, 2 * N_AGENTS + 1]  # v0.4 discrete heads only
N_HEADS = len(ACTION_DIMS)


# ---------------------------------------------------------------------------
# Buffer fixtures
# ---------------------------------------------------------------------------


def _make_buffer(t: int = T) -> RolloutBuffer:
    return RolloutBuffer(
        T=t,
        N_envs=N_ENVS,
        n_agents=N_AGENTS,
        obs_dim=OBS_DIM,
        joint_obs_dim=JOINT_OBS_DIM,
        n_heads=N_HEADS,
        hidden_size=HIDDEN,
        comm_value_dim=COMM_VAL,
        device="cpu",
        n_teammates=N_AGENTS - 1,
        gru_layers=1,
    )


def _add_step(buf: RolloutBuffer, step: int, seed: int = 0) -> None:
    g = torch.Generator()
    g.manual_seed(seed)
    buf.add(
        step=step,
        observations=torch.randn(N_ENVS, N_AGENTS, OBS_DIM, generator=g),
        joint_obs=torch.randn(N_ENVS, JOINT_OBS_DIM, generator=g),
        actions=torch.zeros(N_ENVS, N_AGENTS, N_HEADS, dtype=torch.int64),
        log_probs=torch.randn(N_ENVS, N_AGENTS, generator=g),
        value=torch.randn(N_ENVS, generator=g),
        rewards=torch.randn(N_ENVS, N_AGENTS, generator=g),
        masks=torch.ones(N_ENVS, N_AGENTS),
        hidden_states=torch.zeros(1, N_ENVS, N_AGENTS, HIDDEN),
        received_comms=torch.zeros(N_ENVS, N_AGENTS, COMM_VAL),
        comm_masks=torch.zeros(N_ENVS, N_AGENTS, N_AGENTS - 1),
    )


# ---------------------------------------------------------------------------
# Buffer tests
# ---------------------------------------------------------------------------


def test_buffer_collects_correctly() -> None:
    torch.manual_seed(0)
    buf = _make_buffer()
    for step in range(T):
        _add_step(buf, step, seed=step)
    assert buf.step == T
    assert buf.observations.shape == (T, N_ENVS, N_AGENTS, OBS_DIM)
    assert buf.actions.shape == (T, N_ENVS, N_AGENTS, N_HEADS)
    assert buf.log_probs.shape == (T, N_ENVS, N_AGENTS)
    assert buf.values.shape == (T, N_ENVS)
    assert buf.rewards.shape == (T, N_ENVS, N_AGENTS)
    assert buf.masks.shape == (T, N_ENVS, N_AGENTS)
    assert buf.hidden_states.shape == (T, 1, N_ENVS, N_AGENTS, HIDDEN)
    assert buf.received_comms.shape == (T, N_ENVS, N_AGENTS, COMM_VAL)
    assert torch.isfinite(buf.observations).all()


def test_buffer_rejects_wrong_shape() -> None:
    buf = _make_buffer()
    bad_actions = torch.zeros(N_ENVS, N_AGENTS, N_HEADS + 1, dtype=torch.int64)
    with pytest.raises(ValueError):
        buf.add(
            step=0,
            observations=torch.zeros(N_ENVS, N_AGENTS, OBS_DIM),
            joint_obs=torch.zeros(N_ENVS, JOINT_OBS_DIM),
            actions=bad_actions,
            log_probs=torch.zeros(N_ENVS, N_AGENTS),
            value=torch.zeros(N_ENVS),
            rewards=torch.zeros(N_ENVS, N_AGENTS),
            masks=torch.ones(N_ENVS, N_AGENTS),
            hidden_states=torch.zeros(1, N_ENVS, N_AGENTS, HIDDEN),
            received_comms=torch.zeros(N_ENVS, N_AGENTS, COMM_VAL),
            comm_masks=torch.zeros(N_ENVS, N_AGENTS, N_AGENTS - 1),
        )


def test_buffer_compute_advantages_runs() -> None:
    torch.manual_seed(0)
    buf = _make_buffer()
    for step in range(T):
        _add_step(buf, step, seed=step + 100)
    last_value = torch.zeros(N_ENVS)
    buf.compute_advantages(last_value, gamma=0.99, gae_lambda=0.95)
    assert buf.returns.shape == (T, N_ENVS)
    assert buf.advantages.shape == (T, N_ENVS, N_AGENTS)
    assert torch.isfinite(buf.returns).all()
    assert torch.isfinite(buf.advantages).all()


def test_buffer_minibatch_iter_yields_right_size() -> None:
    torch.manual_seed(0)
    buf = _make_buffer()
    for step in range(T):
        _add_step(buf, step, seed=step + 200)
    buf.compute_advantages(torch.zeros(N_ENVS), gamma=0.99, gae_lambda=0.95)

    minibatch_size = 7
    total = T * N_ENVS * N_AGENTS
    seen = 0
    n_batches = 0
    for batch in buf.minibatch_iter(minibatch_size, shuffle=False):
        assert isinstance(batch, RolloutBatch)
        bs = batch.observations.shape[0]
        assert bs <= minibatch_size
        assert batch.observations.shape == (bs, OBS_DIM)
        assert batch.joint_observations.shape == (bs, JOINT_OBS_DIM)
        # v0.4 mixed-action minibatch.
        assert isinstance(batch.actions, dict)
        assert batch.actions["move"].shape == (bs, 2)
        assert batch.actions["discrete"].shape == (bs, N_HEADS)
        assert batch.old_log_probs.shape == (bs,)
        assert batch.old_values.shape == (bs,)
        assert batch.returns.shape == (bs,)
        assert batch.advantages.shape == (bs,)
        assert batch.masks.shape == (bs,)
        assert batch.hidden_states.shape == (1, bs, HIDDEN)
        assert batch.received_comms.shape == (bs, COMM_VAL)
        seen += bs
        n_batches += 1
    assert seen == total
    assert n_batches >= 1


def test_buffer_clear_resets_step() -> None:
    buf = _make_buffer()
    for step in range(T):
        _add_step(buf, step, seed=step + 300)
    assert buf.step == T
    buf.clear()
    assert buf.step == 0


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------


def _build_model() -> KivskiActorCritic:
    return KivskiActorCritic(
        obs_dim=OBS_DIM,
        joint_obs_dim=JOINT_OBS_DIM,
        action_dims=ACTION_DIMS,
        hidden_size=HIDDEN,
        comm_signature_dim=COMM_SIG,
        comm_value_dim=COMM_VAL,
        comm_attention_heads=COMM_HEADS,
        gumbel_temp=1.0,
        gru_layers=1,
        actor_embedding_dim=8,
    )


def _ml_config(**overrides: object) -> MLConfig:
    defaults: dict[str, object] = {
        "algo": "mappo",
        "hidden_size": HIDDEN,
        "gru_layers": 1,
        "comm_attention_heads": COMM_HEADS,
        "comm_embedding_dim": COMM_VAL * 2,
        "gumbel_temperature": 1.0,
        "ppo_clip": 0.2,
        "ppo_epochs": 2,
        "minibatch_size": 8,
        "learning_rate": 1e-3,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "max_grad_norm": 0.5,
    }
    defaults.update(overrides)
    return MLConfig(**defaults)


def _populate_buffer_via_model(model: KivskiActorCritic, buf: RolloutBuffer) -> None:
    """Fill ``buf`` using the model so logged log-probs are coherent."""
    g = torch.Generator()
    g.manual_seed(42)
    for step in range(buf.T):
        obs = torch.randn(N_ENVS, N_AGENTS, OBS_DIM, generator=g)
        joint = obs.reshape(N_ENVS, N_AGENTS * OBS_DIM)
        h0 = model.initial_hidden_state(N_ENVS * N_AGENTS)
        flat_obs = obs.reshape(N_ENVS * N_AGENTS, OBS_DIM)
        recv = torch.zeros(N_ENVS * N_AGENTS, COMM_VAL)
        with torch.no_grad():
            out = model.act(flat_obs, h0, recv, joint_obs=None)
        # v0.4 mixed actions: pack the buffer entry as a dict.
        move_actions = out["move_actions"].reshape(N_ENVS, N_AGENTS, model.continuous_move_dim)
        disc_actions = out["discrete_actions"].reshape(N_ENVS, N_AGENTS, N_HEADS)
        actions = {"move": move_actions, "discrete": disc_actions}
        log_probs = out["log_probs"].reshape(N_ENVS, N_AGENTS)
        # Value is per-env via the centralised joint observation.
        value_out = model.value_head(joint).detach().squeeze(-1)
        buf.add(
            step=step,
            observations=obs,
            joint_obs=joint,
            actions=actions,
            log_probs=log_probs,
            value=value_out,
            rewards=torch.randn(N_ENVS, N_AGENTS, generator=g) * 0.1,
            masks=torch.ones(N_ENVS, N_AGENTS),
            hidden_states=torch.zeros(1, N_ENVS, N_AGENTS, HIDDEN),
            received_comms=torch.zeros(N_ENVS, N_AGENTS, COMM_VAL),
            comm_masks=torch.zeros(N_ENVS, N_AGENTS, N_AGENTS - 1),
        )
    buf.compute_advantages(
        last_value=torch.zeros(N_ENVS),
        gamma=0.99,
        gae_lambda=0.95,
    )


def test_trainer_update_returns_metrics() -> None:
    torch.manual_seed(0)
    model = _build_model()
    cfg = _ml_config()
    trainer = MAPPOTrainer(model, cfg, device="cpu")
    buf = _make_buffer()
    _populate_buffer_via_model(model, buf)

    metrics = trainer.update(buf)
    assert metrics.update_count > 0
    # Every diagnostic must be a finite float.
    for name in ("policy_loss", "value_loss", "entropy", "kl", "grad_norm", "clip_fraction"):
        v = getattr(metrics, name)
        assert isinstance(v, float)
        assert v == v  # not NaN
        assert v != float("inf") and v != -float("inf")


def test_trainer_update_reduces_loss_on_synthetic_data() -> None:
    """Overfit-style smoke test: total loss should decrease after several updates."""
    torch.manual_seed(0)
    model = _build_model()
    cfg = _ml_config(ppo_epochs=1, minibatch_size=16, learning_rate=3e-3, entropy_coef=0.0)
    trainer = MAPPOTrainer(model, cfg, device="cpu")
    buf = _make_buffer(t=T)
    _populate_buffer_via_model(model, buf)

    initial_metrics = trainer.update(buf)
    initial_total = initial_metrics.policy_loss + cfg.value_coef * initial_metrics.value_loss

    final_total = initial_total
    for _ in range(10):
        # Refresh advantages against the current value head so the target is consistent.
        buf.compute_advantages(
            last_value=torch.zeros(N_ENVS),
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )
        metrics = trainer.update(buf)
        final_total = metrics.policy_loss + cfg.value_coef * metrics.value_loss

    # Either the combined loss dropped OR the value-loss component dropped.
    # Both are acceptable signs that gradients are flowing correctly.
    assert (final_total < initial_total) or (metrics.value_loss < initial_metrics.value_loss), (
        f"loss did not decrease: initial={initial_total:.4f}, final={final_total:.4f}"
    )


def test_trainer_save_load_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _build_model()
    cfg = _ml_config()
    trainer = MAPPOTrainer(model, cfg, device="cpu")

    # Run one update so the optimizer has internal state.
    buf = _make_buffer()
    _populate_buffer_via_model(model, buf)
    trainer.update(buf)

    ckpt = tmp_path / "ckpt.pt"
    out_path = trainer.save(ckpt, metadata={"episodes": 17, "tag": "test"})
    assert out_path.exists()
    sidecar = out_path.with_suffix(out_path.suffix + ".json")
    assert sidecar.exists()

    # Build a fresh trainer and load.
    fresh_model = _build_model()
    fresh_trainer = MAPPOTrainer(fresh_model, cfg, device="cpu")
    meta = fresh_trainer.load(ckpt)
    assert meta.get("episodes") == 17
    assert meta.get("tag") == "test"

    # Models must produce the same outputs on the same input.
    torch.manual_seed(123)
    obs = torch.randn(2, OBS_DIM)
    recv = torch.zeros(2, COMM_VAL)
    h0 = fresh_model.initial_hidden_state(2)

    model.eval()
    fresh_model.eval()
    out_a = model.act(obs, h0, recv, deterministic=True)
    out_b = fresh_model.act(obs, h0, recv, deterministic=True)
    assert torch.equal(out_a["discrete_actions"], out_b["discrete_actions"])
    assert torch.allclose(out_a["move_actions"], out_b["move_actions"], atol=1e-5)
    assert torch.allclose(out_a["log_probs"], out_b["log_probs"], atol=1e-5)
