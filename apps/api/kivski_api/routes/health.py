"""Liveness / version endpoint."""

from __future__ import annotations

from fastapi import APIRouter

import kivski_sim

import kivski_api

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Return server status plus the API and simulator versions."""
    return {
        "status": "ok",
        "version": kivski_api.__version__,
        "kivski_version": kivski_sim.__version__,
    }
