"""Cloud-sync endpoints for pulling checkpoints from a private Hugging Face Hub repo.

A remote cloud GPU pushes ``main_ep_*.pt`` checkpoints + ``.pt.json`` sidecars
to a private HF Hub repo. This module exposes three thin endpoints so the
local viewer can:

  * inspect the latest cloud training run without touching disk,
  * pull the most recent checkpoint into ``models/checkpoints/cloud/``,
  * pull-and-load it for the active viewer match in a single click.

The optional dependency ``huggingface_hub`` is lazy-imported inside each
handler so the module still imports cleanly on machines that haven't
installed it. The :func:`_hf_config` helper centralises env-var lookup
(``HF_TOKEN`` + ``KIVSKI_HF_REPO``) and returns ``None`` whenever either
is missing, which the handlers translate to ``configured=false`` (status)
or a 503 (pull endpoints).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from kivski_api.session import REGISTRY

router = APIRouter(prefix="/api/cloud", tags=["cloud"])
_LOG = logging.getLogger("kivski_api.cloud")

_VALID_EXTS = {".pt", ".ckpt"}


# ---------- helpers ----------


def _checkpoints_root() -> Path:
    """Resolve ``models/checkpoints``, mirroring routes/checkpoints.py."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "models" / "checkpoints"
        if candidate.is_dir():
            return candidate
    fallback = here.parents[3] / "models" / "checkpoints"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _cloud_dir() -> Path:
    """``models/checkpoints/cloud/`` — created on first use."""
    d = _checkpoints_root() / "cloud"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hf_config() -> tuple[str, str] | None:
    """Return (token, repo_id) when both env vars are set; otherwise None."""
    token = os.getenv("HF_TOKEN")
    repo = os.getenv("KIVSKI_HF_REPO")
    if not token or not repo:
        return None
    return token, repo


def _last_pull_marker() -> Path:
    return _cloud_dir() / ".last_pull"


def _read_last_pull() -> float | None:
    marker = _last_pull_marker()
    if not marker.is_file():
        return None
    try:
        return float(marker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _write_last_pull(ts: float) -> None:
    try:
        _last_pull_marker().write_text(f"{ts:.3f}\n", encoding="utf-8")
    except OSError:
        _LOG.warning("could not write last_pull marker", exc_info=True)


def _pick_latest(files: list[str]) -> str | None:
    """Pick the newest checkpoint from a list of HF-Hub file paths.

    ``files`` contains every path in the repo; we filter to those under
    ``checkpoints/`` ending in ``.pt`` / ``.ckpt``. The "newest" choice
    relies on the trainer's ``main_ep_<N>.pt`` naming — sort by the
    embedded episode number, falling back to lexical order so an
    unconventional name still resolves deterministically.
    """
    import re

    candidates = [
        f
        for f in files
        if f.startswith("checkpoints/")
        and any(f.lower().endswith(ext) for ext in _VALID_EXTS)
    ]
    if not candidates:
        return None

    def _key(p: str) -> tuple[int, str]:
        m = re.search(r"ep[_-]?(\d+)", p, re.IGNORECASE)
        return (int(m.group(1)) if m else -1, p)

    candidates.sort(key=_key, reverse=True)
    return candidates[0]


def _load_local_sidecar(local_pt_path: Path) -> dict[str, Any]:
    sidecar = local_pt_path.with_suffix(local_pt_path.suffix + ".json")
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


# ---------- status ----------


@router.get("/status")
async def cloud_status() -> dict[str, Any]:
    """Return the cached configured-state + latest-checkpoint summary."""
    cfg = _hf_config()
    last_pull = _read_last_pull()
    if cfg is None:
        return {
            "configured": False,
            "repo_id": None,
            "last_pull": last_pull,
            "latest_checkpoint": None,
            "metrics_summary": None,
        }
    token, repo_id = cfg

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        return {
            "configured": False,
            "repo_id": repo_id,
            "last_pull": last_pull,
            "latest_checkpoint": None,
            "metrics_summary": None,
        }

    api = HfApi()
    latest_entry: dict[str, Any] | None = None
    metrics_summary: dict[str, Any] | None = None
    try:
        files = api.list_repo_files(repo_id=repo_id, token=token)
        latest = _pick_latest(list(files))
        if latest is not None:
            name = Path(latest).stem
            sidecar_path = latest + ".json"
            metadata: dict[str, Any] = {}
            uploaded_at: float | None = None
            size_bytes = 0
            try:
                sidecar_local = hf_hub_download(
                    repo_id=repo_id,
                    filename=sidecar_path,
                    token=token,
                )
                metadata = json.loads(
                    Path(sidecar_local).read_text(encoding="utf-8")
                )
            except Exception:  # noqa: BLE001 -- sidecar is optional
                metadata = {}

            ts_raw = metadata.get("timestamp")
            if isinstance(ts_raw, (int, float)):
                uploaded_at = float(ts_raw)
            elif isinstance(ts_raw, str):
                try:
                    uploaded_at = float(ts_raw)
                except ValueError:
                    uploaded_at = None

            try:
                info = api.repo_info(repo_id=repo_id, token=token, files_metadata=True)
                for sibling in getattr(info, "siblings", []) or []:
                    if getattr(sibling, "rfilename", None) == latest:
                        size_bytes = int(getattr(sibling, "size", 0) or 0)
                        break
            except Exception:  # noqa: BLE001 -- size is best-effort
                pass

            latest_entry = {
                "name": name,
                "size_bytes": size_bytes,
                "uploaded_at": uploaded_at,
                "metadata": metadata,
            }
            episode = metadata.get("episode") or metadata.get("episodes")
            total_env_steps = metadata.get("total_env_steps") or metadata.get(
                "env_steps"
            )
            score = metadata.get("score") or metadata.get("winrate_vs_random")
            if any(v is not None for v in (episode, total_env_steps, score)):
                metrics_summary = {
                    "episode": int(episode) if isinstance(episode, (int, float)) else 0,
                    "total_env_steps": int(total_env_steps)
                    if isinstance(total_env_steps, (int, float))
                    else 0,
                    "score": float(score) if isinstance(score, (int, float)) else 0.0,
                }
    except Exception as exc:  # noqa: BLE001 -- surface to UI as configured-but-empty
        _LOG.warning("cloud_status: HF API call failed: %s", exc)
        return {
            "configured": True,
            "repo_id": repo_id,
            "last_pull": last_pull,
            "latest_checkpoint": None,
            "metrics_summary": None,
            "error": str(exc),
        }

    return {
        "configured": True,
        "repo_id": repo_id,
        "last_pull": last_pull,
        "latest_checkpoint": latest_entry,
        "metrics_summary": metrics_summary,
    }


# ---------- pull ----------


def _do_pull() -> dict[str, Any]:
    """Shared pull implementation -- raises HTTPException on error."""
    cfg = _hf_config()
    if cfg is None:
        raise HTTPException(
            status_code=503,
            detail="cloud sync not configured (set HF_TOKEN + KIVSKI_HF_REPO)",
        )
    token, repo_id = cfg

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="huggingface_hub is not installed on the backend",
        ) from exc

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, token=token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"HF list_repo_files failed: {exc}"
        ) from exc

    latest = _pick_latest(list(files))
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail=f"no checkpoint files found under checkpoints/ in {repo_id}",
        )

    cloud_dir = _cloud_dir()
    name = Path(latest).stem
    try:
        local_pt = hf_hub_download(
            repo_id=repo_id,
            filename=latest,
            token=token,
            local_dir=str(cloud_dir),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"HF download failed: {exc}"
        ) from exc

    # Best-effort sidecar download.
    sidecar_remote = latest + ".json"
    if sidecar_remote in files:
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=sidecar_remote,
                token=token,
                local_dir=str(cloud_dir),
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("sidecar download failed: %s", exc)

    now = time.time()
    _write_last_pull(now)

    local_path = Path(local_pt)
    metadata = _load_local_sidecar(local_path)
    return {
        "name": name,
        "path": str(local_path),
        "metadata": metadata,
        "pulled_at": now,
    }


@router.post("/pull")
async def cloud_pull() -> dict[str, Any]:
    """Download the latest checkpoint + sidecar into ``models/checkpoints/cloud/``."""
    return _do_pull()


@router.post("/pull-and-load")
async def cloud_pull_and_load() -> dict[str, Any]:
    """Pull the latest cloud checkpoint, then mark it as the active one.

    Mirrors :func:`kivski_api.routes.checkpoints.load_checkpoint` -- the
    V1 registry only tracks the bare ``stem``, not the full path, so we
    write the same field after a successful download.
    """
    result = _do_pull()
    name = result["name"]
    REGISTRY.loaded_checkpoint = name
    _LOG.info("cloud pull-and-load -> %s", result["path"])
    return {"name": name, "loaded": True, "path": result["path"]}
