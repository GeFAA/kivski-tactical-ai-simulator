"""Tests for the live viewer's :mod:`kivski_api.policies` adapters.

Covers the v0.3.0 surface:

* :func:`load_policy` resolves the new named shortcuts
  (``random`` / ``hold`` / ``scripted_rush`` / ``scripted_hold`` /
  ``latest`` / ``best``).
* :func:`load_latest_checkpoint_policy` returns a :class:`RandomPolicy`
  when ``models/checkpoints`` is empty.
* :func:`list_recommended_policies` always exposes the three deterministic
  baselines and adds ``latest`` / ``best`` entries when checkpoints are
  available.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from kivski_api import policies as pol
from kivski_api.policies import (
    CheckpointPolicy,
    HoldPositionPolicy,
    RandomPolicy,
    ScriptedPolicy,
    list_recommended_policies,
    load_latest_checkpoint_policy,
    load_policy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_ckpt_dir(monkeypatch, tmp_path: Path) -> Path:
    """Point the policies module at a *clean* tmp ``models/checkpoints``."""
    root = tmp_path / "models" / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pol, "checkpoints_dir", lambda: root)
    # Also make sure no league state from the repo leaks in.
    monkeypatch.setattr(pol, "_league_state_paths", lambda: [])
    return root


@pytest.fixture()
def populated_ckpt_dir(monkeypatch, tmp_path: Path) -> Path:
    """Point at a ``models/checkpoints`` directory containing two .pt stubs."""
    root = tmp_path / "models" / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    older = root / "run_alpha_ep_1000.pt"
    newer = root / "run_alpha_ep_5000.pt"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    # Force a deterministic mtime order so "latest" picks ``newer``.
    import os
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_700_000_100, 1_700_000_100))
    monkeypatch.setattr(pol, "checkpoints_dir", lambda: root)
    monkeypatch.setattr(pol, "_league_state_paths", lambda: [])
    return root


# ---------------------------------------------------------------------------
# load_policy named shortcuts
# ---------------------------------------------------------------------------


def test_load_policy_named_random_returns_random_policy() -> None:
    assert isinstance(load_policy("random"), RandomPolicy)
    # None / empty string fall back to random.
    assert isinstance(load_policy(None), RandomPolicy)
    assert isinstance(load_policy(""), RandomPolicy)


def test_load_policy_named_hold_returns_hold_policy() -> None:
    assert isinstance(load_policy("hold"), HoldPositionPolicy)
    assert isinstance(load_policy("hold_position"), HoldPositionPolicy)


def test_load_policy_named_scripted_returns_scripted_policy() -> None:
    rush = load_policy("scripted_rush")
    assert isinstance(rush, ScriptedPolicy)
    assert rush.name == "scripted_rush"
    hold = load_policy("scripted_hold")
    assert isinstance(hold, ScriptedPolicy)
    assert hold.name == "scripted_hold"


def test_load_policy_named_random_scripted_acts_produces_actions() -> None:
    """End-to-end smoke: each named policy must produce one ActionBundle per agent."""
    obs = {0: {"alive": True}, 1: {"alive": True}, 2: {"alive": False}}
    for spec in ("random", "hold", "scripted_rush", "scripted_hold"):
        adapter = load_policy(spec)
        out = adapter.act(obs)
        assert set(out.keys()) == set(obs.keys()), spec


# ---------------------------------------------------------------------------
# Latest / best checkpoint resolution
# ---------------------------------------------------------------------------


def test_load_latest_checkpoint_returns_random_when_empty(empty_ckpt_dir: Path) -> None:
    adapter = load_latest_checkpoint_policy()
    assert isinstance(adapter, RandomPolicy)


def test_load_latest_checkpoint_picks_newest_when_present(populated_ckpt_dir: Path) -> None:
    adapter = load_latest_checkpoint_policy()
    assert isinstance(adapter, CheckpointPolicy)
    # Should resolve to the newest (run_alpha_ep_5000.pt).
    assert adapter.path.stem == "run_alpha_ep_5000"


def test_load_policy_latest_keyword(populated_ckpt_dir: Path) -> None:
    adapter = load_policy("latest")
    assert isinstance(adapter, CheckpointPolicy)
    assert adapter.path.stem == "run_alpha_ep_5000"


def test_load_policy_best_falls_back_to_latest_without_league_state(
    populated_ckpt_dir: Path,
) -> None:
    """No league_state.json -> ``best`` resolves to the latest checkpoint."""
    adapter = load_policy("best")
    assert isinstance(adapter, CheckpointPolicy)
    assert adapter.path.stem == "run_alpha_ep_5000"


def test_load_policy_unknown_treated_as_checkpoint(populated_ckpt_dir: Path) -> None:
    """A name matching a file under models/checkpoints loads that checkpoint."""
    adapter = load_policy("run_alpha_ep_1000")
    assert isinstance(adapter, CheckpointPolicy)
    assert adapter.path.stem == "run_alpha_ep_1000"


# ---------------------------------------------------------------------------
# Recommended-policies surface (powers the A/B comparison UI)
# ---------------------------------------------------------------------------


def test_list_recommended_policies_always_includes_baselines(empty_ckpt_dir: Path) -> None:
    opts = list_recommended_policies()
    ids = {o["id"] for o in opts}
    assert "random" in ids
    assert "scripted_rush" in ids
    assert "scripted_hold" in ids
    # With no checkpoints, latest/best must NOT appear.
    assert "latest" not in ids
    assert "best" not in ids


def test_list_recommended_policies_adds_latest_when_present(populated_ckpt_dir: Path) -> None:
    opts = list_recommended_policies()
    ids = {o["id"] for o in opts}
    assert "latest" in ids
    latest_entry = next(o for o in opts if o["id"] == "latest")
    assert "run_alpha_ep_5000" in latest_entry["name"]
