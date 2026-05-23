"""Tests for the per-round auto-reload of trained-checkpoint policies.

These cover the two-layer mechanism added in v0.6:

* :func:`latest_checkpoint_path` (already covered by ``test_api_policies``
  for the happy path) is the discovery primitive; here we re-prove the
  mtime ordering so changes to either layer can't silently regress the
  swap behaviour.
* :meth:`MatchSession._maybe_hot_swap_policy` is the consumer that
  hot-swaps the adapter when a newer ``.pt`` shows up between two
  round-ends. The tests here drive the helper directly (no engine /
  asyncio scheduling involved) so a green run pins the behaviour even
  when the broader run_loop changes.
* :meth:`SessionRegistry.create_match` honours the per-side
  ``auto_reload_*`` flag *only* when the matching policy is
  checkpoint-backed; a request to auto-reload a random side must be
  silently normalised to False (otherwise the API echo would claim the
  feature is active while no swap can ever fire).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from kivski_api import policies as pol
from kivski_api import session as session_module
from kivski_api.policies import (
    CheckpointPolicy,
    RandomPolicy,
    latest_checkpoint_path,
)
from kivski_api.session import SessionRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ckpt_dir(monkeypatch, tmp_path: Path) -> Path:
    """Point the checkpoint discovery helpers at a fresh tmp directory.

    Patches both the policies module *and* the binding the session module
    captured at import time (``from kivski_api.policies import
    latest_checkpoint_path``) so the hot-swap helper reads from our
    sandbox instead of the real ``models/checkpoints``.
    """
    root = tmp_path / "models" / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pol, "checkpoints_dir", lambda: root)
    monkeypatch.setattr(pol, "_league_state_paths", lambda: [])

    def _resolve_latest() -> Path | None:
        candidates = sorted(
            root.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    monkeypatch.setattr(pol, "latest_checkpoint_path", _resolve_latest)
    # session.py captured the symbol at import time; patch its module
    # binding too so the hot-swap helper picks up the sandbox.
    monkeypatch.setattr(session_module, "latest_checkpoint_path", _resolve_latest)
    return root


def _write_dummy_ckpt(root: Path, name: str, mtime: float) -> Path:
    """Drop a placeholder ``.pt`` file with a deterministic mtime.

    The contents don't matter for the hot-swap path because
    :class:`CheckpointPolicy` defers ``torch.load`` until its first
    ``act()`` call -- a swap only needs the file to exist on disk.
    """
    path = root / f"{name}.pt"
    path.write_bytes(b"dummy")
    import os

    os.utime(path, (mtime, mtime))
    return path


def _make_minimal_match(
    monkeypatch,
    *,
    policy_yellow: str | None = None,
    policy_blue: str | None = None,
    auto_reload_yellow: bool = False,
    auto_reload_blue: bool = False,
):
    """Build a real MatchSession by routing through SessionRegistry.

    Going through the registry exercises the same wiring the live
    /api/match/new endpoint uses, so the tests assert the integrated
    behaviour rather than a hand-rolled stub.
    """
    reg = SessionRegistry()
    session = reg.create_match(
        map_name="dustline",
        seed=42,
        policy_yellow=policy_yellow,
        policy_blue=policy_blue,
        auto_reload_yellow=auto_reload_yellow,
        auto_reload_blue=auto_reload_blue,
    )
    return reg, session


# ---------------------------------------------------------------------------
# latest_checkpoint_path -- the discovery primitive
# ---------------------------------------------------------------------------


def test_latest_checkpoint_returns_newest_by_mtime(ckpt_dir: Path) -> None:
    _write_dummy_ckpt(ckpt_dir, "older", mtime=1_700_000_000.0)
    newer = _write_dummy_ckpt(ckpt_dir, "newer", mtime=1_700_000_100.0)
    assert latest_checkpoint_path() == newer


def test_latest_checkpoint_returns_none_when_dir_empty(ckpt_dir: Path) -> None:
    assert latest_checkpoint_path() is None


# ---------------------------------------------------------------------------
# _maybe_hot_swap_policy
# ---------------------------------------------------------------------------


def test_hot_swap_no_op_when_flag_off(ckpt_dir: Path, monkeypatch) -> None:
    """auto_reload_yellow=False must never replace the policy adapter."""
    _write_dummy_ckpt(ckpt_dir, "ep_0001", mtime=1_700_000_000.0)
    _, session = _make_minimal_match(
        monkeypatch,
        policy_yellow="latest",
        policy_blue="random",
        auto_reload_yellow=False,
        auto_reload_blue=False,
    )
    initial = session.policy_yellow
    # Even with a newer ckpt available, the flag-off side must not swap.
    _write_dummy_ckpt(ckpt_dir, "ep_0002", mtime=1_700_000_500.0)

    swapped = asyncio.run(session._maybe_hot_swap_policy("yellow"))
    assert swapped is False
    assert session.policy_yellow is initial


def test_hot_swap_replaces_policy_when_newer_ckpt_available(ckpt_dir: Path, monkeypatch) -> None:
    """A newer .pt on disk plus auto_reload=True must hot-swap the adapter."""
    initial_ckpt = _write_dummy_ckpt(ckpt_dir, "ep_0001", mtime=1_700_000_000.0)
    _, session = _make_minimal_match(
        monkeypatch,
        policy_yellow="latest",
        policy_blue="random",
        auto_reload_yellow=True,
        auto_reload_blue=False,
    )
    # Sanity-check the initial wiring: yellow got the checkpoint adapter
    # pointed at the file we just wrote.
    assert isinstance(session.policy_yellow, CheckpointPolicy)
    assert session._loaded_policy_path_yellow == str(initial_ckpt)
    initial = session.policy_yellow

    # Drop a newer checkpoint and trigger one cycle of the hot-swap.
    newer = _write_dummy_ckpt(ckpt_dir, "ep_0002", mtime=1_700_000_500.0)
    swapped = asyncio.run(session._maybe_hot_swap_policy("yellow"))

    assert swapped is True
    assert session.policy_yellow is not initial
    assert isinstance(session.policy_yellow, CheckpointPolicy)
    assert session.policy_yellow.path == newer
    assert session._loaded_policy_path_yellow == str(newer)
    assert session.policy_yellow_name == f"checkpoint:{newer.stem}"


def test_hot_swap_idempotent_when_same_path(ckpt_dir: Path, monkeypatch) -> None:
    """If the on-disk latest matches what's already loaded, no swap fires."""
    _write_dummy_ckpt(ckpt_dir, "ep_0001", mtime=1_700_000_000.0)
    _, session = _make_minimal_match(
        monkeypatch,
        policy_yellow="latest",
        policy_blue="random",
        auto_reload_yellow=True,
        auto_reload_blue=False,
    )
    before = session.policy_yellow

    # No file changes happened: hot_swap must observe identity and bail.
    swapped = asyncio.run(session._maybe_hot_swap_policy("yellow"))
    assert swapped is False
    assert session.policy_yellow is before


def test_hot_swap_no_op_when_no_checkpoint_on_disk(ckpt_dir: Path, monkeypatch) -> None:
    """An empty checkpoint dir is allowed; helper must short-circuit."""
    # Set up a session that *thinks* it's tracking a ckpt path.
    _, session = _make_minimal_match(
        monkeypatch,
        policy_yellow="random",
        policy_blue="random",
        auto_reload_yellow=True,
        auto_reload_blue=False,
    )
    # Force the flag on even though the side runs RandomPolicy so we can
    # exercise the "no ckpt on disk" guard in isolation. (In normal use
    # create_match would have already normalised this to False.)
    session.auto_reload_yellow = True
    swapped = asyncio.run(session._maybe_hot_swap_policy("yellow"))
    assert swapped is False


def test_hot_swap_broadcasts_policy_reload_event(ckpt_dir: Path, monkeypatch) -> None:
    """A successful swap must emit a ``policy_reload`` WS event for the UI."""
    _write_dummy_ckpt(ckpt_dir, "ep_0001", mtime=1_700_000_000.0)
    _, session = _make_minimal_match(
        monkeypatch,
        policy_yellow="latest",
        policy_blue="random",
        auto_reload_yellow=True,
        auto_reload_blue=False,
    )

    captured: list[dict] = []

    async def _capture(payload: dict) -> None:
        captured.append(payload)

    # Replace the real broadcaster with one that just records calls --
    # no WebSocket plumbing needed for the unit test.
    session._broadcast_event = _capture  # type: ignore[method-assign]
    newer = _write_dummy_ckpt(ckpt_dir, "ep_0002", mtime=1_700_000_500.0)

    asyncio.run(session._maybe_hot_swap_policy("yellow"))

    assert len(captured) == 1
    frame = captured[0]
    assert frame["type"] == "policy_reload"
    assert frame["match_id"] == session.id
    data = frame["data"]
    assert data["side"] == "yellow"
    assert data["name"] == newer.stem
    assert data["path"] == str(newer)


# ---------------------------------------------------------------------------
# Registry: normalisation of auto_reload flag vs the resolved policy kind
# ---------------------------------------------------------------------------


def test_create_match_normalises_flag_for_non_checkpoint_side(ckpt_dir: Path, monkeypatch) -> None:
    """auto_reload on a Random side must be silently dropped to False."""
    _write_dummy_ckpt(ckpt_dir, "ep_0001", mtime=1_700_000_000.0)
    _, session = _make_minimal_match(
        monkeypatch,
        policy_yellow="random",
        policy_blue="latest",
        auto_reload_yellow=True,
        auto_reload_blue=True,
    )
    # Random side: flag normalised away.
    assert session.auto_reload_yellow is False
    assert isinstance(session.policy_yellow, RandomPolicy)
    # Checkpoint side: flag kept, loaded-path seeded.
    assert session.auto_reload_blue is True
    assert isinstance(session.policy_blue, CheckpointPolicy)
    assert session._loaded_policy_path_blue is not None
