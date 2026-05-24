"""Unit tests for the actor-critic and TarMAC comm network components.

These tests use deliberately small synthetic shapes (``obs_dim=64``,
``n_agents=4``, ``batch=2``) instead of the production sizes so the suite
stays fast on CPU. The point is to pin down tensor shapes, mask
semantics, and the "no NaN" invariant -- the heavier behavioural checks
live in :mod:`tests.unit.test_mappo`.
"""

from __future__ import annotations

import math

import pytest
import torch
from kivski_agents.networks.actor_critic import (
    ActorHeads,
    KivskiActorCritic,
    ObservationEncoder,
    RecurrentCore,
    ValueHead,
)
from kivski_agents.networks.comm import CommAttention, CommEncoder, CommGate

# ---------------------------------------------------------------------------
# Shared test sizes
# ---------------------------------------------------------------------------


OBS_DIM = 64
JOINT_OBS_DIM = 4 * OBS_DIM  # 4 agents on a team
HIDDEN = 32
N_AGENTS = 4
BATCH = 2
COMM_SIG = 16
COMM_VAL = 16
COMM_HEADS = 2
ACTION_DIMS = [6, 9, 8, 2 * N_AGENTS + 1]  # v0.4 discrete heads (micro, comm, buy, aim)


def _seeded_tensor(*shape: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randn(*shape, generator=g)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_observation_encoder_output_shape() -> None:
    torch.manual_seed(0)
    encoder = ObservationEncoder(OBS_DIM, hidden_size=HIDDEN)
    obs = _seeded_tensor(BATCH, OBS_DIM, seed=1)
    out = encoder(obs)
    assert out.shape == (BATCH, HIDDEN)
    assert torch.isfinite(out).all()


def test_observation_encoder_rejects_wrong_dim() -> None:
    encoder = ObservationEncoder(OBS_DIM, hidden_size=HIDDEN)
    with pytest.raises(ValueError):
        encoder(torch.randn(BATCH, OBS_DIM + 1))


def test_recurrent_core_output_shape_per_step() -> None:
    torch.manual_seed(0)
    core = RecurrentCore(input_dim=HIDDEN + COMM_VAL, hidden_size=HIDDEN, num_layers=1)
    h0 = core.initial_hidden(BATCH)
    inp = _seeded_tensor(BATCH, HIDDEN + COMM_VAL, seed=2)
    out, h1 = core(inp, h0)
    assert out.shape == (BATCH, HIDDEN)
    assert h1.shape == (1, BATCH, HIDDEN)
    assert torch.isfinite(out).all()
    assert torch.isfinite(h1).all()


def test_recurrent_core_output_shape_sequence() -> None:
    torch.manual_seed(0)
    core = RecurrentCore(input_dim=HIDDEN + COMM_VAL, hidden_size=HIDDEN, num_layers=1)
    T = 5
    h0 = core.initial_hidden(BATCH)
    inp = _seeded_tensor(T, BATCH, HIDDEN + COMM_VAL, seed=3)
    masks = torch.ones(T, BATCH)
    out, h1 = core(inp, h0, masks=masks)
    assert out.shape == (T, BATCH, HIDDEN)
    assert h1.shape == (1, BATCH, HIDDEN)
    assert torch.isfinite(out).all()


def test_recurrent_core_mask_resets_hidden() -> None:
    """A 0-mask should zero the prior hidden state for that batch row."""
    torch.manual_seed(0)
    core = RecurrentCore(input_dim=HIDDEN + COMM_VAL, hidden_size=HIDDEN, num_layers=1)
    inp = _seeded_tensor(BATCH, HIDDEN + COMM_VAL, seed=4)
    h_nonzero = torch.ones(1, BATCH, HIDDEN)
    masks_zero = torch.zeros(BATCH)
    # Step with zero hidden via mask should match stepping with literally zeros.
    out_a, _ = core(inp, h_nonzero, masks=masks_zero)
    out_b, _ = core(inp, torch.zeros_like(h_nonzero))
    assert torch.allclose(out_a, out_b)


# ---------------------------------------------------------------------------
# Actor heads
# ---------------------------------------------------------------------------


def test_actor_heads_sample_shapes() -> None:
    torch.manual_seed(0)
    head = ActorHeads(hidden_size=HIDDEN, action_dims=ACTION_DIMS, embedding_dim=8)
    hidden = _seeded_tensor(BATCH, HIDDEN, seed=5)
    actions, log_probs, entropy = head.sample(hidden, deterministic=False)
    # v0.4: mixed dict {"move": float[B, D], "discrete": int64[B, num_heads]}
    assert isinstance(actions, dict)
    move = actions["move"]
    disc = actions["discrete"]
    assert move.shape == (BATCH, 2)
    assert move.dtype == torch.float32
    assert disc.shape == (BATCH, len(ACTION_DIMS))
    assert disc.dtype == torch.int64
    assert log_probs.shape == (BATCH,)
    assert entropy.shape == (BATCH,)
    # Move must be inside the [-1, 1] box.
    assert float(move.detach().min()) >= -1.0 - 1e-6
    assert float(move.detach().max()) <= 1.0 + 1e-6
    # Per-head range check on the discrete cascade.
    for i, n_cat in enumerate(ACTION_DIMS):
        assert (disc[:, i] >= 0).all()
        assert (disc[:, i] < n_cat).all()
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy).all()


def test_actor_heads_sample_deterministic_is_argmax() -> None:
    torch.manual_seed(0)
    head = ActorHeads(hidden_size=HIDDEN, action_dims=ACTION_DIMS, embedding_dim=8)
    hidden = _seeded_tensor(BATCH, HIDDEN, seed=6)
    a1, _, _ = head.sample(hidden, deterministic=True)
    a2, _, _ = head.sample(hidden, deterministic=True)
    assert torch.equal(a1["discrete"], a2["discrete"])
    assert torch.allclose(a1["move"], a2["move"])


def test_actor_heads_evaluate_log_prob_finite() -> None:
    torch.manual_seed(0)
    head = ActorHeads(hidden_size=HIDDEN, action_dims=ACTION_DIMS, embedding_dim=8)
    hidden = _seeded_tensor(BATCH, HIDDEN, seed=7)
    actions = {
        "move": torch.zeros(BATCH, 2),
        "discrete": torch.stack(
            [torch.randint(low=0, high=n_cat, size=(BATCH,), dtype=torch.int64) for n_cat in ACTION_DIMS],
            dim=1,
        ),
    }
    log_probs, entropy = head.evaluate(hidden, actions)
    assert log_probs.shape == (BATCH,)
    assert entropy.shape == (BATCH,)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy).all()


def test_actor_heads_evaluate_matches_sample_log_prob() -> None:
    """Re-evaluating logged actions reproduces the same discrete log-prob.

    The move head is stochastic, so for the assertion to hold we re-evaluate
    the sampled move *and* discrete actions through ``evaluate`` and compare
    the joint log-prob -- it must equal the log-prob computed during
    sampling (modulo float noise).
    """
    torch.manual_seed(0)
    head = ActorHeads(hidden_size=HIDDEN, action_dims=ACTION_DIMS, embedding_dim=8)
    hidden = _seeded_tensor(BATCH, HIDDEN, seed=8)
    actions, log_probs, _ = head.sample(hidden, deterministic=False)
    # ``actions["move"]`` is the *clamped* sample; the sampled log-prob used
    # the pre-clamp raw sample, so we accept a slightly larger tolerance.
    re_log_probs, _ = head.evaluate(hidden, actions)
    assert torch.allclose(log_probs, re_log_probs, atol=1e-3)


# ---------------------------------------------------------------------------
# Value head
# ---------------------------------------------------------------------------


def test_value_head_output_shape() -> None:
    torch.manual_seed(0)
    head = ValueHead(JOINT_OBS_DIM, hidden_size=HIDDEN)
    joint = _seeded_tensor(BATCH, JOINT_OBS_DIM, seed=9)
    out = head(joint)
    assert out.shape == (BATCH, 1)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Comm modules
# ---------------------------------------------------------------------------


def test_comm_encoder_output_shapes() -> None:
    torch.manual_seed(0)
    enc = CommEncoder(input_dim=HIDDEN, signature_dim=COMM_SIG, value_dim=COMM_VAL)
    h = _seeded_tensor(BATCH, HIDDEN, seed=10)
    sig, val = enc(h)
    assert sig.shape == (BATCH, COMM_SIG)
    assert val.shape == (BATCH, COMM_VAL)
    # Signature should be (approximately) unit-norm.
    norms = sig.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_comm_gate_train_vs_eval() -> None:
    torch.manual_seed(0)
    gate = CommGate(input_dim=HIDDEN)
    h = _seeded_tensor(BATCH, HIDDEN, seed=11)
    gate.train()
    logits, soft = gate(h, temperature=1.0)
    assert logits.shape == (BATCH, 1)
    assert soft.shape == (BATCH, 1)
    assert (soft >= 0).all() and (soft <= 1).all()
    gate.eval()
    logits2, hard = gate(h)
    assert logits2.shape == (BATCH, 1)
    assert ((hard == 0) | (hard == 1)).all()


def test_comm_attention_respects_mask() -> None:
    """All-zero mask => exactly-zero aggregation (no spurious NaN)."""
    torch.manual_seed(0)
    attn = CommAttention(signature_dim=COMM_SIG, value_dim=COMM_VAL, num_heads=COMM_HEADS)
    q = torch.randn(BATCH, COMM_SIG)
    sigs = torch.randn(BATCH, N_AGENTS - 1, COMM_SIG)
    vals = torch.randn(BATCH, N_AGENTS - 1, COMM_VAL)
    mask = torch.zeros(BATCH, N_AGENTS - 1)
    agg, weights = attn(q, sigs, vals, mask)
    assert agg.shape == (BATCH, COMM_VAL)
    assert weights.shape == (BATCH, N_AGENTS - 1)
    assert torch.isfinite(agg).all()
    assert torch.isfinite(weights).all()
    # When the mask is all-zero the aggregated message is the linear-layer
    # mapping of a zero vector, which equals the bias. The unfiltered
    # attention WEIGHTS must be all-zero (no NaN from masked softmax).
    assert torch.allclose(weights, torch.zeros_like(weights))


def test_comm_attention_partial_mask_normalisation() -> None:
    """Weights for the live teammates must sum to 1 along the last axis."""
    torch.manual_seed(0)
    attn = CommAttention(signature_dim=COMM_SIG, value_dim=COMM_VAL, num_heads=COMM_HEADS)
    q = torch.randn(BATCH, COMM_SIG)
    sigs = torch.randn(BATCH, N_AGENTS - 1, COMM_SIG)
    vals = torch.randn(BATCH, N_AGENTS - 1, COMM_VAL)
    mask = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    _, weights = attn(q, sigs, vals, mask)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    # Masked entries must be exactly zero.
    assert weights[0, 1].item() == pytest.approx(0.0)
    assert weights[1, 2].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Top-level KivskiActorCritic
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


def test_full_model_act_runs_without_error() -> None:
    torch.manual_seed(0)
    model = _build_model()
    obs = _seeded_tensor(BATCH, OBS_DIM, seed=12)
    received = _seeded_tensor(BATCH, COMM_VAL, seed=13)
    joint = _seeded_tensor(BATCH, JOINT_OBS_DIM, seed=14)
    h0 = model.initial_hidden_state(BATCH)
    out = model.act(obs, h0, received, joint_obs=joint)
    # v0.4 mixed-action layout.
    assert isinstance(out["actions"], dict)
    assert out["actions"]["move"].shape == (BATCH, 2)
    assert out["actions"]["discrete"].shape == (BATCH, len(ACTION_DIMS))
    assert out["move_actions"].shape == (BATCH, 2)
    assert out["discrete_actions"].shape == (BATCH, len(ACTION_DIMS))
    assert out["log_probs"].shape == (BATCH,)
    assert out["entropy"].shape == (BATCH,)
    assert out["new_hidden"].shape == (1, BATCH, HIDDEN)
    assert out["comm_signature"].shape == (BATCH, COMM_SIG)
    assert out["comm_value"].shape == (BATCH, COMM_VAL)
    assert out["comm_gate"].shape == (BATCH, 1)
    assert out["comm_payload"].shape == (BATCH, COMM_VAL)
    assert out["value"].shape == (BATCH, 1)
    # Check finiteness on tensor values (skip the dict).
    for key, t in out.items():
        if torch.is_tensor(t):
            assert torch.isfinite(t).all(), f"NaN/Inf in {key!r}"


def test_full_model_act_without_joint_obs_omits_value() -> None:
    torch.manual_seed(0)
    model = _build_model()
    obs = _seeded_tensor(BATCH, OBS_DIM, seed=15)
    received = _seeded_tensor(BATCH, COMM_VAL, seed=16)
    h0 = model.initial_hidden_state(BATCH)
    out = model.act(obs, h0, received, joint_obs=None)
    assert "value" not in out


def test_full_model_evaluate_runs_without_error() -> None:
    torch.manual_seed(0)
    model = _build_model()
    obs = _seeded_tensor(BATCH, OBS_DIM, seed=17)
    received = _seeded_tensor(BATCH, COMM_VAL, seed=18)
    joint = _seeded_tensor(BATCH, JOINT_OBS_DIM, seed=19)
    h0 = model.initial_hidden_state(BATCH)
    # Use sampled actions to feed back into evaluate.
    sample = model.act(obs, h0, received, joint_obs=joint)
    actions = sample["actions"]
    out = model.evaluate(obs, h0, received, actions, joint_obs=joint)
    assert out["log_probs"].shape == (BATCH,)
    assert out["entropy"].shape == (BATCH,)
    assert out["value"].shape == (BATCH, 1)
    assert out["new_hidden"].shape == (1, BATCH, HIDDEN)
    for key, t in out.items():
        assert torch.isfinite(t).all(), f"NaN/Inf in {key!r}"


def test_full_model_initial_hidden_state_shape() -> None:
    model = _build_model()
    h = model.initial_hidden_state(BATCH)
    assert h.shape == (1, BATCH, HIDDEN)
    assert torch.allclose(h, torch.zeros_like(h))


def test_full_model_log_prob_within_expected_range() -> None:
    """log_prob is a sum across heads -> bounded by sum(log(1/n_cat))."""
    torch.manual_seed(0)
    model = _build_model()
    obs = _seeded_tensor(BATCH, OBS_DIM, seed=20)
    received = _seeded_tensor(BATCH, COMM_VAL, seed=21)
    h0 = model.initial_hidden_state(BATCH)
    out = model.act(obs, h0, received)
    log_p = out["log_probs"]
    # Upper bound: 0 (one-hot dist). Lower bound: very negative.
    assert (log_p <= 1e-3).all()
    # Sanity floor against pathological NaNs.
    log_p_min_theoretical = -sum(math.log(n) * 3 for n in ACTION_DIMS) - 100
    assert (log_p > log_p_min_theoretical).all()
