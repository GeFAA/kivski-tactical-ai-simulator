"""Integration tests for the per-round auto-reload API surface.

Two layers under test:

1. ``POST /api/match/new`` honours the ``auto_reload_yellow`` /
   ``auto_reload_blue`` flags, echoing them back in the response so the
   frontend can render its indicator from the authoritative server state.
2. The WebSocket session forwards a ``policy_reload`` event when the
   helper fires. We mock the discovery primitive
   (``latest_checkpoint_path``) so the test doesn't need a real trainer
   writing actual ``.pt`` files into the repo.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from kivski_api import session as session_module
from kivski_api.app import create_app
from kivski_api.session import REGISTRY


@pytest.fixture()
def client() -> Iterator[TestClient]:
    app = create_app()
    with TestClient(app) as c:
        yield c
    REGISTRY.sessions.clear()


@dataclass
class _CkptSandbox:
    """Tmp ``models/checkpoints`` dir plus a ``publish`` helper.

    Returned by the ``ckpt_sandbox`` fixture so tests can write
    ``sandbox.publish("ep_0050")`` to drop a fresh checkpoint that the
    monkeypatched discovery primitive will return next time it's asked.
    """

    root: Path
    publish: Callable[[str], Path]


@pytest.fixture()
def ckpt_sandbox(monkeypatch, tmp_path: Path) -> _CkptSandbox:
    """Sandbox the on-disk checkpoint discovery to a tmp dir."""
    root = tmp_path / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    state: dict[str, Path | None] = {"latest": None}

    def _resolve_latest() -> Path | None:
        return state["latest"]

    # Both the session module (uses the captured binding) *and* the
    # policies module (called via load_policy("latest") -> factory) need
    # to point at our resolver.
    from kivski_api import policies as pol

    monkeypatch.setattr(pol, "latest_checkpoint_path", _resolve_latest)
    monkeypatch.setattr(session_module, "latest_checkpoint_path", _resolve_latest)
    # Make the sandbox the only checkpoints dir to keep load_policy
    # consistent with our resolver when it tries to wrap the file.
    monkeypatch.setattr(pol, "checkpoints_dir", lambda: root)
    monkeypatch.setattr(pol, "_league_state_paths", lambda: [])

    def _publish(name: str) -> Path:
        path = root / f"{name}.pt"
        path.write_bytes(b"dummy")
        state["latest"] = path
        return path

    return _CkptSandbox(root=root, publish=_publish)


# ---------------------------------------------------------------------------
# REST: flag round-trip + normalisation
# ---------------------------------------------------------------------------


def test_new_match_echoes_auto_reload_flags(client: TestClient, ckpt_sandbox: _CkptSandbox) -> None:
    """The response must surface the *effective* auto_reload flags."""
    initial = ckpt_sandbox.publish("ep_0010")
    assert initial.is_file()
    resp = client.post(
        "/api/match/new",
        json={
            "map": "dustline",
            "policy_yellow": "latest",
            "policy_blue": "latest",
            "auto_reload_yellow": True,
            "auto_reload_blue": True,
            "autostart": False,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["auto_reload_yellow"] is True
    assert body["auto_reload_blue"] is True
    # Echoed policy names should reflect the resolved checkpoint stem.
    assert body["policy_yellow_name"] == "checkpoint:ep_0010"
    assert body["policy_blue_name"] == "checkpoint:ep_0010"


def test_new_match_normalises_auto_reload_for_random_side(
    client: TestClient, ckpt_sandbox: _CkptSandbox
) -> None:
    """auto_reload on a Random side must be normalised to False in the response."""
    # No checkpoint exists; policy_blue=random.
    resp = client.post(
        "/api/match/new",
        json={
            "map": "dustline",
            "policy_yellow": "random",
            "policy_blue": "random",
            "auto_reload_yellow": True,
            "auto_reload_blue": True,
            "autostart": False,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The request asked for True but the server dropped them because
    # neither side is checkpoint-backed.
    assert body["auto_reload_yellow"] is False
    assert body["auto_reload_blue"] is False


def test_snapshot_endpoint_surfaces_auto_reload(client: TestClient, ckpt_sandbox: _CkptSandbox) -> None:
    """``GET /api/match/{id}/snapshot`` must return the auto_reload flags."""
    ckpt_sandbox.publish("ep_0010")
    create = client.post(
        "/api/match/new",
        json={
            "map": "dustline",
            "policy_yellow": "latest",
            "policy_blue": "random",
            "auto_reload_yellow": True,
            "auto_reload_blue": True,
            "autostart": False,
        },
    )
    assert create.status_code == 200, create.text
    mid = create.json()["match_id"]
    snap = client.get(f"/api/match/{mid}/snapshot")
    assert snap.status_code == 200
    body = snap.json()
    assert body["auto_reload_yellow"] is True
    # Blue was random -> normalised to False even though request asked True.
    assert body["auto_reload_blue"] is False


# ---------------------------------------------------------------------------
# Hot-swap end-to-end via the registry (no asyncio loop required)
# ---------------------------------------------------------------------------


def test_session_hot_swap_fires_after_publishing_new_checkpoint(
    client: TestClient, ckpt_sandbox: _CkptSandbox
) -> None:
    """Drive the hot-swap helper directly on the session created via the API."""
    initial = ckpt_sandbox.publish("ep_0010")
    create = client.post(
        "/api/match/new",
        json={
            "map": "dustline",
            "policy_yellow": "latest",
            "policy_blue": "latest",
            "auto_reload_yellow": True,
            "auto_reload_blue": True,
            "autostart": False,
        },
    )
    assert create.status_code == 200, create.text
    mid = create.json()["match_id"]
    session = REGISTRY.get_match(mid)
    assert session is not None
    assert session._loaded_policy_path_yellow == str(initial)

    # Publish a "newer" checkpoint and run one cycle of the helper.
    newer = ckpt_sandbox.publish("ep_0050")
    swapped = asyncio.run(session._maybe_hot_swap_policy("yellow"))
    assert swapped is True
    assert session._loaded_policy_path_yellow == str(newer)
    assert session.policy_yellow_name == "checkpoint:ep_0050"


# ---------------------------------------------------------------------------
# WebSocket: policy_reload event surface
# ---------------------------------------------------------------------------


def test_websocket_receives_policy_reload_event(client: TestClient, ckpt_sandbox: _CkptSandbox) -> None:
    """A live WS subscriber sees the ``policy_reload`` frame on swap."""
    initial = ckpt_sandbox.publish("ep_0010")
    assert initial.is_file()
    create = client.post(
        "/api/match/new",
        json={
            "map": "dustline",
            "policy_yellow": "latest",
            "policy_blue": "latest",
            "auto_reload_yellow": True,
            "auto_reload_blue": True,
            "autostart": False,
        },
    )
    assert create.status_code == 200, create.text
    mid = create.json()["match_id"]

    with client.websocket_connect(f"/ws/match/{mid}") as ws:
        # Drain the initial map_info + snapshot frames.
        first = ws.receive_json()
        assert first["type"] == "map_info"
        second = ws.receive_json()
        assert second["type"] == "snapshot"

        # Publish a newer checkpoint and trigger the hot-swap from the
        # server side. The helper runs on the same event loop the WS
        # endpoint is using; we have to drive it via the FastAPI app's
        # loop. Easiest path: call the helper directly on the session
        # via the registry -- it broadcasts to ``session.subscribers``
        # which includes our test WS.
        session = REGISTRY.get_match(mid)
        assert session is not None
        newer = ckpt_sandbox.publish("ep_0050")
        # asyncio.run won't work because the TestClient's app already
        # owns a running loop. Schedule on its loop instead.
        from anyio.from_thread import start_blocking_portal

        with start_blocking_portal() as portal:
            portal.call(session._maybe_hot_swap_policy, "yellow")

        # Read frames until we see the policy_reload event (snapshots
        # may interleave because the tick loop is still emitting).
        found = None
        for _ in range(20):
            frame = ws.receive_json()
            if frame.get("type") == "policy_reload":
                found = frame
                break
        assert found is not None, "policy_reload event never arrived"
        assert found["data"]["side"] == "yellow"
        assert found["data"]["name"] == newer.stem
