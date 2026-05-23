"""Integration tests for the FastAPI HTTP surface.

We use :class:`fastapi.testclient.TestClient` (synchronous, backed by httpx)
so the tests stay readable and don't require an asyncio test runner. The
WebSocket endpoint has a basic smoke test that opens a connection, receives
the initial map-info + snapshot frames, and closes cleanly.

Every test creates and tears down its own match so state doesn't leak across
the suite.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from kivski_api.app import create_app
from kivski_api.session import REGISTRY


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """Fresh app + TestClient per test, with global REGISTRY cleanup."""
    app = create_app()
    with TestClient(app) as c:
        yield c
    # Cleanup any matches the test created.
    REGISTRY.sessions.clear()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_endpoint_ok(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "kivski_version" in body


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------


def test_list_maps_includes_dustline(client: TestClient) -> None:
    resp = client.get("/api/maps")
    assert resp.status_code == 200
    body = resp.json()
    assert "maps" in body
    assert "dustline" in body["maps"]


def test_get_map_returns_json(client: TestClient) -> None:
    resp = client.get("/api/maps/dustline")
    assert resp.status_code == 200
    body = resp.json()
    # Sanity-check the schema shape.
    assert body["name"] == "dustline"
    assert "spawns" in body
    assert "bombsites" in body
    assert "width" in body and "height" in body


def test_get_map_404(client: TestClient) -> None:
    resp = client.get("/api/maps/does_not_exist")
    assert resp.status_code == 404


def test_get_map_rejects_traversal(client: TestClient) -> None:
    resp = client.get("/api/maps/..%2Fevil")
    # FastAPI may normalise or 400 -- accept either rejection style.
    assert resp.status_code in (400, 404)


# ---------------------------------------------------------------------------
# Match lifecycle
# ---------------------------------------------------------------------------


def _new_match(client: TestClient, **overrides) -> str:
    body = {
        "seed": 42,
        "map": "dustline",
        "config": None,
        "policy_yellow": "random",
        "policy_blue": "hold",
        "autostart": False,
    }
    body.update(overrides)
    resp = client.post("/api/match/new", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()["match_id"]


def test_create_match_returns_id(client: TestClient) -> None:
    mid = _new_match(client)
    assert isinstance(mid, str) and len(mid) > 0
    assert mid in REGISTRY.sessions


def test_reset_match(client: TestClient) -> None:
    mid = _new_match(client)
    resp = client.post(f"/api/match/{mid}/reset")
    assert resp.status_code == 200
    assert resp.json() == {"reset": True}


def test_pause_resume_match(client: TestClient) -> None:
    mid = _new_match(client)
    p = client.post(f"/api/match/{mid}/pause")
    assert p.status_code == 200 and p.json() == {"paused": True}
    r = client.post(f"/api/match/{mid}/resume")
    assert r.status_code == 200 and r.json() == {"paused": False}


def test_set_speed_match(client: TestClient) -> None:
    mid = _new_match(client)
    resp = client.post(f"/api/match/{mid}/speed", params={"multiplier": 2.5})
    assert resp.status_code == 200
    assert resp.json()["speed"] == pytest.approx(2.5)


def test_snapshot_endpoint(client: TestClient) -> None:
    mid = _new_match(client)
    resp = client.get(f"/api/match/{mid}/snapshot")
    assert resp.status_code == 200
    body = resp.json()
    assert body["match_id"] == mid
    snap = body["data"]
    assert "tick" in snap and "agents" in snap and "bomb" in snap
    assert len(snap["agents"]) == 10  # 5v5 default


def test_delete_match(client: TestClient) -> None:
    mid = _new_match(client)
    resp = client.delete(f"/api/match/{mid}")
    assert resp.status_code == 204
    # Subsequent operations must 404.
    follow = client.post(f"/api/match/{mid}/pause")
    assert follow.status_code == 404


def test_unknown_match_404(client: TestClient) -> None:
    resp = client.post("/api/match/does_not_exist/pause")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Checkpoints (read-only sanity)
# ---------------------------------------------------------------------------


def test_list_checkpoints(client: TestClient) -> None:
    resp = client.get("/api/checkpoints")
    assert resp.status_code == 200
    body = resp.json()
    assert "checkpoints" in body
    assert isinstance(body["checkpoints"], list)


def test_load_missing_checkpoint_404(client: TestClient) -> None:
    resp = client.post("/api/checkpoints/missing/load")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Training status (no job yet)
# ---------------------------------------------------------------------------


def test_training_status_idle(client: TestClient) -> None:
    # Ensure no leftover jobs from previous tests.
    REGISTRY.training.clear()
    resp = client.get("/api/training/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["running"] is False
    assert body["job_id"] is None


def test_training_stop_when_idle_404(client: TestClient) -> None:
    REGISTRY.training.clear()
    resp = client.post("/api/training/stop")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# WebSocket smoke test
# ---------------------------------------------------------------------------


def test_websocket_initial_frames(client: TestClient) -> None:
    mid = _new_match(client, autostart=False)
    with client.websocket_connect(f"/ws/match/{mid}") as ws:
        first = ws.receive_json()
        assert first["type"] == "map_info"
        assert first["match_id"] == mid
        second = ws.receive_json()
        assert second["type"] == "snapshot"
        # Round-trip ping/pong.
        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        # We may receive an intervening snapshot before the pong arrives;
        # accept either ordering by reading up to a few messages.
        for _ in range(5):
            if pong.get("type") == "pong":
                break
            pong = ws.receive_json()
        assert pong["type"] == "pong"
        assert "ts" in pong


def test_websocket_unknown_match(client: TestClient) -> None:
    with client.websocket_connect("/ws/match/nope") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"
