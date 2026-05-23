"""Match lifecycle endpoints (create / reset / pause / resume / speed / snapshot / delete)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from kivski_api.session import REGISTRY

router = APIRouter(prefix="/api/match", tags=["match"])
_LOG = logging.getLogger("kivski_api.match")


class NewMatchBody(BaseModel):
    seed: int | None = None
    map: str = Field(default="dustline")
    config: str | None = None
    policy_yellow: str | None = None
    policy_blue: str | None = None
    autostart: bool = Field(
        default=True,
        description="Begin the tick loop immediately; otherwise it stays paused until /resume.",
    )


def _require_session(match_id: str) -> Any:
    session = REGISTRY.get_match(match_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"match '{match_id}' not found")
    return session


@router.post("/new")
async def new_match(body: NewMatchBody) -> dict[str, Any]:
    """Create a new match session and (optionally) start its tick loop."""
    try:
        session = REGISTRY.create_match(
            map_name=body.map,
            seed=body.seed,
            config_path=body.config,
            policy_yellow=body.policy_yellow,
            policy_blue=body.policy_blue,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover -- catch-all for config/typing issues
        raise HTTPException(status_code=400, detail=f"could not create match: {exc}") from exc

    if not body.autostart:
        session.paused = True
    session.start()
    return {
        "match_id": session.id,
        "map": body.map,
        "seed": body.seed,
        "policy_yellow": body.policy_yellow,
        "policy_blue": body.policy_blue,
        "paused": session.paused,
    }


@router.post("/{match_id}/reset")
async def reset_match(match_id: str) -> dict[str, bool]:
    session = _require_session(match_id)
    session.reset()
    return {"reset": True}


@router.post("/{match_id}/pause")
async def pause_match(match_id: str) -> dict[str, bool]:
    session = _require_session(match_id)
    session.paused = True
    return {"paused": True}


@router.post("/{match_id}/resume")
async def resume_match(match_id: str) -> dict[str, bool]:
    session = _require_session(match_id)
    session.paused = False
    session.start()  # idempotent restart if loop has exited
    return {"paused": False}


@router.post("/{match_id}/speed")
async def set_speed(
    match_id: str,
    multiplier: float = Query(..., gt=0.0, le=16.0, description="Tick speed multiplier."),
) -> dict[str, float]:
    session = _require_session(match_id)
    session.set_speed(multiplier)
    return {"speed": session.speed}


@router.get("/{match_id}/snapshot")
async def get_snapshot(match_id: str) -> dict[str, Any]:
    """One-shot JSON snapshot, useful for polling clients without a WebSocket."""
    session = _require_session(match_id)
    return {
        "match_id": match_id,
        "paused": session.paused,
        "speed": session.speed,
        "selected_agent": session.selected_agent,
        "data": session.engine.snapshot().to_json_dict(),
    }


@router.delete("/{match_id}", status_code=204)
async def delete_match(match_id: str) -> Response:
    session = REGISTRY.get_match(match_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"match '{match_id}' not found")
    await REGISTRY.remove_match(match_id)
    return Response(status_code=204)
