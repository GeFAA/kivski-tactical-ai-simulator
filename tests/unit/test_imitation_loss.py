"""Tests for the BC imitation loss + dataset."""

from __future__ import annotations

import torch

from kivski_agents.imitation.dataset import ImitationDataset
from kivski_agents.imitation.loss import bc_loss


def _fake_demos(n: int = 8, obs_dim: int = 160, joint_obs_dim: int = 800, comm_dim: int = 32, hidden_size: int = 384) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randn(n, obs_dim),
        "joint_obs": torch.randn(n, joint_obs_dim),
        "received_comm": torch.randn(n, comm_dim),
        "move": torch.rand(n, 2) * 2 - 1,
        "discrete": torch.randint(0, 5, (n, 4)),
        "hidden_state": torch.zeros(n, hidden_size),
    }


def test_dataset_len_and_getitem_keys() -> None:
    ds = ImitationDataset(_fake_demos(n=4))
    assert len(ds) == 4
    item = ds[0]
    assert set(item.keys()) == {"obs", "joint_obs", "received_comm", "move", "discrete", "hidden_state"}
    assert item["obs"].dim() == 1


def test_bc_loss_returns_scalar_and_breakdown() -> None:
    n = 4
    move_mean = torch.zeros(n, 2, requires_grad=True)
    move_log_std = torch.zeros(n, 2, requires_grad=True)
    discrete_logits = [torch.zeros(n, k, requires_grad=True) for k in (6, 9, 8, 9)]
    net_out = {"move_mean": move_mean, "move_log_std": move_log_std, "discrete_logits": discrete_logits}
    demo_move = torch.rand(n, 2) * 2 - 1
    demo_discrete = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4], [0, 0, 0, 0], [5, 8, 7, 8]])
    total, breakdown = bc_loss(net_out, demo_move, demo_discrete)
    assert total.dim() == 0
    assert total.item() > 0
    assert "move_nll" in breakdown
    assert "discrete_ce_per_head" in breakdown
    assert len(breakdown["discrete_ce_per_head"]) == 4
    total.backward()  # gradients flow


def test_bc_loss_perfect_discrete_prediction_is_low() -> None:
    n = 2
    move_mean = torch.zeros(n, 2)
    move_log_std = torch.zeros(n, 2)
    # Discrete head with 6 categories; all weight on category 0
    one_hot = torch.zeros(n, 6)
    one_hot[:, 0] = 30.0
    discrete_logits = [one_hot, torch.zeros(n, 9), torch.zeros(n, 8), torch.zeros(n, 9)]
    net_out = {"move_mean": move_mean, "move_log_std": move_log_std, "discrete_logits": discrete_logits}
    demo_discrete = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
    demo_move = torch.zeros(n, 2)
    _total, breakdown = bc_loss(net_out, demo_move, demo_discrete)
    assert breakdown["discrete_ce_per_head"][0] < 0.01
