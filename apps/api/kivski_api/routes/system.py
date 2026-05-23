"""Host-level introspection endpoint.

Returns the host's CPU / memory utilisation, platform + Python identity, and
the versions of Kivski + PyTorch / CUDA availability. Consumed by the frontend
``SystemInfo`` panel which polls every few seconds.

The endpoint is intentionally cheap and best-effort: every optional probe is
wrapped in a ``try`` so a missing or partially-installed dependency cannot
take the panel down. ``psutil.cpu_percent`` uses a 0.1 s sample window which
is short enough not to block the event loop noticeably under any of our
broadcast rates.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from typing import Any

import kivski_sim
import psutil
from fastapi import APIRouter

import kivski_api

router = APIRouter(prefix="/api/system", tags=["system"])

# Process start time captured at import so the `uptime_s` field is meaningful
# from the very first request rather than always reading zero.
_PROCESS_STARTED_AT = time.time()


def _torch_info() -> dict[str, Any]:
    """Return ``{torch_version, cuda_available, cuda_device}`` -- best effort."""
    info: dict[str, Any] = {
        "torch_version": None,
        "cuda_available": False,
        "cuda_device": None,
    }
    try:
        import torch  # type: ignore[import-not-found]

        info["torch_version"] = str(torch.__version__)
        try:
            info["cuda_available"] = bool(torch.cuda.is_available())
            if info["cuda_available"]:
                # device 0 is the conventional default
                info["cuda_device"] = str(torch.cuda.get_device_name(0))
        except Exception:
            # torch present but CUDA probe blew up -- leave defaults
            info["cuda_available"] = False
            info["cuda_device"] = None
    except Exception:
        # torch not importable -- leave defaults
        pass
    return info


@router.get("/info")
async def system_info() -> dict[str, Any]:
    """One-shot snapshot of the host environment.

    Keys are deliberately snake_case for consistency with every other
    Kivski endpoint. The frontend client translates to camelCase.
    """
    vm = psutil.virtual_memory()
    try:
        load_avg = list(psutil.getloadavg()) if hasattr(psutil, "getloadavg") else None
    except (OSError, AttributeError):
        # Windows historically had no loadavg; skip silently.
        load_avg = None

    info = _torch_info()
    return {
        "cpu_count": os.cpu_count(),
        # interval=0.1 yields a meaningful single-shot reading without
        # storing per-process state across requests.
        "cpu_percent": float(psutil.cpu_percent(interval=0.1)),
        "memory_total_gb": vm.total / 1e9,
        "memory_used_gb": vm.used / 1e9,
        "memory_percent": float(vm.percent),
        "load_avg": load_avg,
        "platform": platform.platform(),
        "python": sys.version,
        "kivski_api_version": kivski_api.__version__,
        "kivski_sim_version": kivski_sim.__version__,
        "torch_version": info["torch_version"],
        "cuda_available": bool(info["cuda_available"]),
        "cuda_device": info["cuda_device"],
        "uptime_s": float(time.time() - _PROCESS_STARTED_AT),
        "pid": os.getpid(),
    }
