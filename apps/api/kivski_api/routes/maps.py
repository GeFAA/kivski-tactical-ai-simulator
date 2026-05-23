"""Map listing + raw JSON download endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from kivski_sim.map_loader import list_maps

router = APIRouter(prefix="/api/maps", tags=["maps"])


def _maps_dir() -> Path:
    """Locate ``packages/maps`` regardless of CWD or editable-install layout."""
    here = Path(__file__).resolve()
    # apps/api/kivski_api/routes/maps.py -> ... -> repo root
    for parent in here.parents:
        candidate = parent / "packages" / "maps"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("packages/maps directory not found")


@router.get("")
async def list_available_maps() -> dict[str, list[str]]:
    """Return ``{"maps": [...]}`` -- alphabetically sorted map names."""
    return {"maps": list_maps()}


@router.get("/{name}")
async def get_map(name: str) -> dict:
    """Return the raw JSON contents of the named map file.

    We deliberately serve the raw JSON (not the parsed :class:`MapData`) so
    the frontend can ingest polygons / spawns directly without a second
    serialisation pass.
    """
    # Defensive: don't let a path-traversal slip through.
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="invalid map name")
    try:
        root = _maps_dir()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    path = root / f"{name}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"map '{name}' not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"malformed map json: {exc}") from exc
