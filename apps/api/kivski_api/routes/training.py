"""Training-process manager endpoints.

The API can spawn ``python -m scripts.train`` as a child process, monitor it,
and terminate it. The process is tracked in :data:`session.REGISTRY.training`
keyed by a generated ``job_id``. We deliberately stick to subprocess (rather
than threading the trainer in-process) because the trainer is CPU-/GPU-heavy
and would otherwise starve the event loop.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from kivski_sim.utils import now_unix
from pydantic import BaseModel, Field

from kivski_api.session import REGISTRY, TrainingJob

router = APIRouter(prefix="/api/training", tags=["training"])
_LOG = logging.getLogger("kivski_api.training")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[4]


def _log_dir() -> Path:
    p = _repo_root() / "models" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


class StartTrainingBody(BaseModel):
    config: str | None = Field(default=None, description="Path to config YAML")
    episodes: int | None = Field(default=None, ge=1, description="Override total episodes")
    checkpoint: str | None = Field(default=None, description="Resume from checkpoint name or path")


@router.post("/start")
async def start_training(body: StartTrainingBody) -> dict[str, Any]:
    """Start a child training process. Returns the new ``job_id`` immediately."""
    # Only allow one training job at a time -- prevents accidental GPU contention.
    for existing in REGISTRY.training.values():
        if existing.is_running():
            raise HTTPException(
                status_code=409,
                detail=f"a training job ({existing.job_id}) is already running",
            )

    job_id = uuid.uuid4().hex[:12]
    log_path = _log_dir() / f"train-{job_id}.log"

    cmd: list[str] = [sys.executable, "-m", "scripts.train"]
    config_path = body.config or "configs/default.yaml"
    cmd.extend(["--config", config_path])
    if body.episodes is not None:
        cmd.extend(["--episodes", str(int(body.episodes))])
    if body.checkpoint:
        cmd.extend(["--resume", str(body.checkpoint)])

    _LOG.info("Launching training: %s", " ".join(cmd))

    try:
        log_fh = log_path.open("wb")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"cannot open log file: {exc}") from exc

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_repo_root()),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=(os.name != "nt"),
        )
    except OSError as exc:
        log_fh.close()
        raise HTTPException(status_code=500, detail=f"failed to spawn trainer: {exc}") from exc

    job = TrainingJob(
        job_id=job_id,
        config_path=config_path,
        log_path=log_path,
        started_at=now_unix(),
        pid=proc.pid,
        process=proc,
        episodes=body.episodes,
        resume_from=body.checkpoint,
    )
    REGISTRY.register_training(job)

    return {"job_id": job_id, "pid": proc.pid, "started": True, "log_path": str(log_path)}


@router.post("/stop")
async def stop_training() -> dict[str, Any]:
    """SIGTERM the most-recent running training job."""
    job = REGISTRY.latest_training()
    if job is None or not job.is_running() or job.process is None:
        raise HTTPException(status_code=404, detail="no running training job")
    try:
        if os.name == "nt":
            # SIGTERM isn't really supported on Windows; .terminate() does the
            # equivalent (TerminateProcess).
            job.process.terminate()
        else:
            job.process.send_signal(signal.SIGTERM)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"failed to signal: {exc}") from exc
    # Give it a brief moment to flush logs.
    try:
        rc = job.process.wait(timeout=2.0)
        job.exit_code = int(rc) if rc is not None else None
    except subprocess.TimeoutExpired:
        # Caller can poll /status to see final exit code.
        pass
    return {"stopped": True, "exit_code": job.exit_code}


@router.get("/status")
async def training_status() -> dict[str, Any]:
    """Latest job snapshot + tail of its log."""
    job = REGISTRY.latest_training()
    if job is None:
        return {
            "running": False,
            "job_id": None,
            "pid": None,
            "started_at": 0.0,
            "log_tail": [],
        }
    return {
        "running": job.is_running(),
        "job_id": job.job_id,
        "pid": job.pid,
        "started_at": job.started_at,
        "exit_code": job.exit_code,
        "config_path": job.config_path,
        "episodes": job.episodes,
        "resume_from": job.resume_from,
        "log_tail": job.tail_log(50),
        "log_path": str(job.log_path),
    }


@router.get("/log")
async def training_log() -> StreamingResponse:
    """Stream the entire log file as ``text/plain`` (one-shot, not tailed)."""
    job = REGISTRY.latest_training()
    if job is None:
        raise HTTPException(status_code=404, detail="no training job")
    log_path = job.log_path

    def _iter() -> Any:
        if not log_path.is_file():
            return
        with log_path.open("rb") as fh:
            while True:
                chunk = fh.read(8192)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(_iter(), media_type="text/plain; charset=utf-8")
