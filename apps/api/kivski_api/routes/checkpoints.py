"""Checkpoint listing / load / delete endpoints.

Checkpoints live in ``models/checkpoints/`` as ``<name>.pt`` (or ``.ckpt``)
files. An optional sidecar ``<name>.json`` provides metadata (episodes,
timestamp, hyperparameters). The currently "loaded" checkpoint is tracked
in-memory on :data:`session.REGISTRY` for V1 -- nothing is persisted across
API restarts.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response

from kivski_api.policies import list_recommended_policies
from kivski_api.session import REGISTRY

router = APIRouter(prefix="/api/checkpoints", tags=["checkpoints"])
_LOG = logging.getLogger("kivski_api.checkpoints")

_VALID_EXTS = {".pt", ".ckpt"}


def _checkpoints_dir() -> Path:
    """Resolve ``models/checkpoints``, creating it on first access."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "models" / "checkpoints"
        if candidate.is_dir():
            return candidate
    # Fallback: create the conventional location at the repo root.
    fallback = here.parents[3] / "models" / "checkpoints"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _entry_for(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    meta: dict[str, Any] = {}
    if sidecar.is_file():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    return {
        "name": path.stem,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "episodes": int(meta.get("episodes", 0)) if "episodes" in meta else None,
        "timestamp": meta.get("timestamp"),
        "metadata": meta,
        "loaded": REGISTRY.loaded_checkpoint == path.stem,
    }


@router.get("")
async def list_checkpoints() -> dict[str, Any]:
    """List all checkpoints with optional sidecar metadata."""
    root = _checkpoints_dir()
    entries = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() in _VALID_EXTS and path.is_file():
            entries.append(_entry_for(path))
    return {"checkpoints": entries, "loaded": REGISTRY.loaded_checkpoint}


@router.get("/recommended")
async def recommended_policies() -> dict[str, Any]:
    """A/B comparison-mode policy picker.

    Returns the curated list of "interesting" opponents the user can pit
    against each other in a live viewer match: deterministic baselines
    (random / scripted_rush / scripted_hold) plus, when available,
    ``latest`` and ``best`` checkpoint shortcuts.
    """
    return {"options": list_recommended_policies()}


def _find_checkpoint(name: str) -> Path:
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="invalid checkpoint name")
    root = _checkpoints_dir()
    for ext in _VALID_EXTS:
        candidate = root / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    raise HTTPException(status_code=404, detail=f"checkpoint '{name}' not found")


@router.post("/{name}/load")
async def load_checkpoint(name: str) -> dict[str, Any]:
    """Mark ``name`` as the currently active checkpoint (in-memory only)."""
    path = _find_checkpoint(name)
    REGISTRY.loaded_checkpoint = path.stem
    _LOG.info("Loaded checkpoint %s", path)
    return {"loaded": True, "name": path.stem, "path": str(path)}


@router.delete("/{name}", status_code=204)
async def delete_checkpoint(name: str) -> Response:
    """Delete the checkpoint file and its sidecar JSON (if present)."""
    path = _find_checkpoint(name)
    try:
        path.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"unable to delete: {exc}") from exc
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        with contextlib.suppress(OSError):
            sidecar.unlink()
    if REGISTRY.loaded_checkpoint == path.stem:
        REGISTRY.loaded_checkpoint = None
    return Response(status_code=204)
