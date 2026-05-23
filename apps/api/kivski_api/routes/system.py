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
    """Return torch + CUDA introspection -- best effort, never raises.

    On CUDA-capable systems also surfaces the compute capability and
    memory totals for the active device so the frontend ``SystemInfo``
    panel can show a one-line GPU summary ("RTX 4070 SUPER, 12 GB, sm_89").
    Memory usage is sampled per request via ``torch.cuda.memory_allocated``
    which is cheap and reflects the live training process when the API
    and trainer share the same CUDA context (i.e. both ran in-process or
    via the launcher's hosted trainer subprocess that imports the same
    venv torch).
    """
    info: dict[str, Any] = {
        "torch_version": None,
        "cuda_available": False,
        "cuda_device": None,
        "cuda_compute_capability": None,
        "gpu_total_memory_gb": None,
        "gpu_used_memory_gb": None,
        "gpu_reserved_memory_gb": None,
    }
    try:
        import torch  # type: ignore[import-not-found]

        info["torch_version"] = str(torch.__version__)
        try:
            info["cuda_available"] = bool(torch.cuda.is_available())
            if info["cuda_available"]:
                # device 0 is the conventional default
                info["cuda_device"] = str(torch.cuda.get_device_name(0))
                try:
                    major, minor = torch.cuda.get_device_capability(0)
                    info["cuda_compute_capability"] = f"{int(major)}.{int(minor)}"
                except Exception:
                    info["cuda_compute_capability"] = None
                try:
                    props = torch.cuda.get_device_properties(0)
                    info["gpu_total_memory_gb"] = float(props.total_memory) / 1e9
                except Exception:
                    info["gpu_total_memory_gb"] = None
                try:
                    info["gpu_used_memory_gb"] = float(torch.cuda.memory_allocated(0)) / 1e9
                    info["gpu_reserved_memory_gb"] = float(torch.cuda.memory_reserved(0)) / 1e9
                except Exception:
                    info["gpu_used_memory_gb"] = None
                    info["gpu_reserved_memory_gb"] = None
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
        "cuda_compute_capability": info["cuda_compute_capability"],
        "gpu_total_memory_gb": info["gpu_total_memory_gb"],
        "gpu_used_memory_gb": info["gpu_used_memory_gb"],
        "gpu_reserved_memory_gb": info["gpu_reserved_memory_gb"],
        "uptime_s": float(time.time() - _PROCESS_STARTED_AT),
        "pid": os.getpid(),
    }
