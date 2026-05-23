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


def _load_sidecar(path: Path) -> dict[str, Any]:
    """Best-effort load of the .json sidecar.

    The MAPPO trainer writes ``<name>.pt.json``; an older convention used
    ``<name>.json``. Try both, return whichever parses first.
    """
    for sidecar in (
        path.with_suffix(path.suffix + ".json"),
        path.with_suffix(".json"),
    ):
        if sidecar.is_file():
            try:
                return json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return {}


def _entry_for(path: Path, *, kind: str | None = None) -> dict[str, Any]:
    meta = _load_sidecar(path)
    # Best entries get an explicit ``kind`` flag so the UI can render a
    # crown icon or "BEST" badge without re-parsing the sidecar.
    entry_kind = kind or str(meta.get("kind") or "checkpoint")
    return {
        "name": path.stem,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "episodes": int(meta.get("episode", meta.get("episodes", 0)))
        if ("episode" in meta or "episodes" in meta)
        else None,
        "timestamp": meta.get("timestamp"),
        "metadata": meta,
        "loaded": REGISTRY.loaded_checkpoint == path.stem,
        "kind": entry_kind,
    }


@router.get("")
async def list_checkpoints() -> dict[str, Any]:
    """List all checkpoints with optional sidecar metadata.

    Scans both the shared ``models/checkpoints/`` root (where ``best.pt``
    lives after best-promotion) and every per-run subdirectory created by
    the trainer (``models/checkpoints/<run_name>/main_ep_*.pt``). The
    ``best`` entry is always listed first when it exists so the UI's
    default selection lands on the highest-quality option.
    """
    root = _checkpoints_dir()
    entries: list[dict[str, Any]] = []

    # 1) Best.pt as a virtual top-priority entry.
    best_path = root / "best.pt"
    if best_path.is_file():
        entries.append(_entry_for(best_path, kind="best"))

    # 2) Top-level files (legacy + manually placed checkpoints).
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _VALID_EXTS:
            continue
        if path.name == "best.pt":
            continue  # already added above
        entries.append(_entry_for(path))

    # 3) Per-run checkpoints, newest first.
    sub_entries: list[dict[str, Any]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        for path in sub.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in _VALID_EXTS:
                continue
            sub_entries.append(_entry_for(path))
    sub_entries.sort(
        key=lambda e: (e.get("timestamp") or 0.0, e.get("name") or ""),
        reverse=True,
    )
    entries.extend(sub_entries)
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
    # Direct hits at the root (best.pt, manually placed files).
    for ext in _VALID_EXTS:
        candidate = root / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    # Per-run subdirectories: ``models/checkpoints/<run>/<name>.pt``.
    for ext in _VALID_EXTS:
        matches = list(root.rglob(f"{name}{ext}"))
        matches = [m for m in matches if m.is_file()]
        if matches:
            # Return newest hit so a duplicated name resolves to the
            # most-recent run rather than an arbitrary alphabetical pick.
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]
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
