"""Live WebSocket endpoint for streaming match snapshots.

Wire protocol (JSON messages, both directions):

Server -> client::

    {"type": "map_info", "match_id": ..., "data": {...}}    -- sent once on connect
    {"type": "snapshot", "match_id": ..., "data": {...}}    -- per tick
    {"type": "match_done", "match_id": ...}                 -- when engine.is_done()
    {"type": "pong", "ts": <unix-seconds>}                  -- response to ping
    {"type": "error", "detail": "..."}                      -- recoverable error

Client -> server::

    {"type": "ping"}
    {"type": "set_selected_agent", "agent_id": <int>}
    {"type": "pause"} / {"type": "resume"}
    {"type": "set_speed", "multiplier": <float>}

The endpoint is reconnect-safe: when the socket closes for any reason we
remove it from the subscriber set so future broadcasts skip it cleanly.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from kivski_sim.utils import now_unix

from kivski_api.session import REGISTRY

router = APIRouter(tags=["websocket"])
_LOG = logging.getLogger("kivski_api.ws")


@router.websocket("/ws/match/{match_id}")
async def match_websocket(websocket: WebSocket, match_id: str) -> None:
    """Subscribe to a match's live snapshot stream."""
    session = REGISTRY.get_match(match_id)
    if session is None:
        # We have to accept before we can close cleanly with a code.
        await websocket.accept()
        await websocket.send_json({"type": "error", "detail": f"match '{match_id}' not found"})
        await websocket.close(code=4404)
        return

    await websocket.accept()
    session.subscribers.add(websocket)
    _LOG.info("WS connected to match %s (subscribers=%d)", match_id, len(session.subscribers))

    try:
        # Initial frames: map info + immediate snapshot.
        await websocket.send_json(session.map_info_payload())
        await websocket.send_json(
            {
                "type": "snapshot",
                "match_id": session.id,
                "paused": session.paused,
                "speed": session.speed,
                "selected_agent": session.selected_agent,
                "data": session.engine.snapshot().to_json_dict(),
            }
        )

        # Ensure the tick loop is running -- the match may have been created
        # paused or had its loop end on a previous match-done.
        session.start()

        while True:
            msg = await websocket.receive_json()
            await _handle_client_message(websocket, session, msg)

    except WebSocketDisconnect:
        _LOG.info("WS disconnected from match %s", match_id)
    except Exception:
        _LOG.exception("WS error for match %s", match_id)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        session.subscribers.discard(websocket)


async def _handle_client_message(
    websocket: WebSocket,
    session: Any,
    msg: dict[str, Any],
) -> None:
    """Apply one client-sent control message."""
    if not isinstance(msg, dict):
        await websocket.send_json({"type": "error", "detail": "message must be a JSON object"})
        return
    kind = msg.get("type")
    if kind == "ping":
        await websocket.send_json({"type": "pong", "ts": now_unix()})
    elif kind == "set_selected_agent":
        agent_id = msg.get("agent_id")
        session.set_selected_agent(int(agent_id) if agent_id is not None else None)
        await websocket.send_json(
            {"type": "ack", "for": "set_selected_agent", "selected_agent": session.selected_agent}
        )
    elif kind == "pause":
        session.paused = True
        await websocket.send_json({"type": "ack", "for": "pause", "paused": True})
    elif kind == "resume":
        session.paused = False
        session.start()
        await websocket.send_json({"type": "ack", "for": "resume", "paused": False})
    elif kind == "set_speed":
        mult = msg.get("multiplier", 1.0)
        try:
            session.set_speed(float(mult))
        except (TypeError, ValueError):
            await websocket.send_json({"type": "error", "detail": "invalid multiplier"})
            return
        await websocket.send_json({"type": "ack", "for": "set_speed", "speed": session.speed})
    else:
        await websocket.send_json({"type": "error", "detail": f"unknown message type: {kind!r}"})
