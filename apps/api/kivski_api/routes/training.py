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
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from kivski_sim.utils import now_unix
from pydantic import BaseModel, Field

from kivski_api.session import REGISTRY, TrainingJob
from kivski_api.training_clock import get_clock

# Valid checkpoint extensions we'll auto-resume from. ``.pt`` is the
# convention used by the MAPPO trainer; ``.ckpt`` is kept for forward
# compatibility with any external tools that write that suffix.
_VALID_CKPT_EXTS: tuple[str, ...] = (".pt", ".ckpt")

try:  # Optional dependency — only used to broadcast initial training_status.
    from kivski_api.metrics_broadcaster import broadcast_training_status_to_all
except Exception:  # pragma: no cover - circular import safety
    broadcast_training_status_to_all = None  # type: ignore[assignment]

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


def _configs_dir() -> Path:
    """Resolve the repo-root ``configs/`` directory."""
    return _repo_root() / "configs"


def _ckpt_root() -> Path:
    """Repo-rooted ``models/checkpoints`` directory.

    The trainer writes per-run checkpoints under
    ``models/checkpoints/<run_name>/main_ep_*.pt`` and (after best-promotion)
    a shared ``models/checkpoints/best.pt``. Auto-resume needs to search
    the whole tree.
    """
    p = _repo_root() / "models" / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_resumable_checkpoint() -> Path | None:
    """Pick the most-recently-modified ``.pt`` / ``.ckpt`` to resume from.

    Prefers ``best.pt`` when it exists (highest quality), otherwise falls
    back to the newest ``main_ep_*.pt`` anywhere under
    ``models/checkpoints/``. Returns ``None`` when nothing is on disk so
    a fresh run starts cleanly.
    """
    root = _ckpt_root()
    # Prefer the curated best checkpoint when available.
    best = root / "best.pt"
    if best.is_file():
        return best
    # Otherwise scan recursively for the newest per-episode checkpoint.
    candidates: list[Path] = []
    for ext in _VALID_CKPT_EXTS:
        candidates.extend(root.rglob(f"*{ext}"))
    candidates = [p for p in candidates if p.is_file() and p.name != "best.pt"]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


@router.get("/resume-target")
async def get_resume_target() -> dict[str, Any]:
    """Surface what the next ``POST /api/training/start`` would auto-resume from.

    Used by the frontend to decide whether to render the "Resumes from
    <path>" hint next to the Start button. Returns ``available: false``
    plus ``path: null`` when no checkpoint exists.
    """
    auto = _find_resumable_checkpoint()
    if auto is None:
        return {"available": False, "path": None, "name": None}
    return {"available": True, "path": str(auto), "name": auto.stem}


@router.get("/configs")
async def list_training_configs() -> list[dict[str, Any]]:
    """List available trainer-config YAML files under ``configs/``.

    Each entry is ``{id, name, description}`` so the frontend dropdown
    can show a friendly label and use ``id`` as the relative path
    passed to ``/api/training/start``.
    """
    root = _configs_dir()
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.yaml")):
        rel = path.relative_to(_repo_root()).as_posix()
        out.append(
            {
                "id": rel,
                "name": path.stem,
                "description": f"{path.stat().st_size} bytes",
            },
        )
    return out


class StartTrainingBody(BaseModel):
    config: str | None = Field(default=None, description="Path to config YAML")
    episodes: int | None = Field(default=None, ge=1, description="Override total episodes")
    checkpoint: str | None = Field(default=None, description="Resume from checkpoint name or path")
    # When True, never auto-detect a checkpoint -- always start from
    # scratch. Useful for sweeps / regression runs that must not be
    # contaminated by previously-trained weights. Default False keeps
    # the "nothing is ever lost" auto-save promise.
    fresh_start: bool = Field(
        default=False,
        description="Skip auto-resume detection and always start from scratch.",
    )


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
    # Pin a run name so we know exactly which models/logs/<run>/metrics.jsonl
    # the trainer will write into — that path is what the broadcaster tails.
    run_name = f"api-{time.strftime('%Y%m%d-%H%M%S')}-{job_id}"
    metrics_jsonl_path = _repo_root() / "models" / "logs" / run_name / "metrics.jsonl"

    cmd: list[str] = [sys.executable, "-m", "scripts.train", "train"]
    config_path = body.config or "configs/default.yaml"
    cmd.extend(["--config", config_path])
    if body.episodes is not None:
        cmd.extend(["--episodes", str(int(body.episodes))])

    # Resume logic: explicit body.checkpoint always wins; otherwise we
    # auto-detect the newest on-disk checkpoint so a crashed/restarted
    # run picks up where it left off. ``fresh_start=true`` opts out.
    resolved_resume: str | None = None
    if body.checkpoint:
        resolved_resume = str(body.checkpoint)
        _LOG.info("[kivski-api] resuming from %s (explicit)", resolved_resume)
    elif not body.fresh_start:
        auto = _find_resumable_checkpoint()
        if auto is not None:
            resolved_resume = str(auto)
            _LOG.info("[kivski-api] resuming from %s (auto)", resolved_resume)
    if resolved_resume is not None:
        cmd.extend(["--resume", resolved_resume])
    # Force "all" telemetry so the JSONL sink is always present (CSV stays
    # for analytics, JSONL is what powers the live viewer).
    cmd.extend(["--run-name", run_name, "--telemetry", "all"])

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
        # Persist the *resolved* path (explicit or auto) so /api/training/status
        # can surface what we actually resumed from, not just what the user
        # typed.
        resume_from=resolved_resume,
        run_name=run_name,
        metrics_jsonl_path=metrics_jsonl_path,
    )
    REGISTRY.register_training(job)

    # Push an initial training_status frame so the UI flips to "running"
    # immediately instead of waiting for the first metrics record.
    if broadcast_training_status_to_all is not None:
        try:
            await broadcast_training_status_to_all(
                {
                    "running": True,
                    "episode": 0,
                    "totalEpisodes": int(body.episodes or 0),
                }
            )
        except Exception:
            _LOG.exception("failed to broadcast initial training_status")

    return {
        "job_id": job_id,
        "pid": proc.pid,
        "started": True,
        "log_path": str(log_path),
        "run_name": run_name,
        "metrics_jsonl_path": str(metrics_jsonl_path),
        # Expose the auto-resume decision so the UI can show
        # "Resumes from <path>" instead of a blind "Started".
        "resumed_from": resolved_resume,
    }


@router.post("/stop")
async def stop_training() -> dict[str, Any]:
    """SIGTERM the most-recent running training job."""
    job = REGISTRY.latest_training()
    if job is None or not job.is_running() or job.process is None:
        raise HTTPException(status_code=404, detail="no running training job")
    # Mark the stop as intentional so the watchdog won't auto-restart it.
    job.stop_requested = True
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
    # Notify any live viewers that the trainer has stopped so the UI flips
    # the pill back to "idle" without having to poll /status.
    if broadcast_training_status_to_all is not None:
        try:
            await broadcast_training_status_to_all(
                {
                    "running": False,
                    "episode": 0,
                    "totalEpisodes": int(job.episodes or 0),
                }
            )
        except Exception:
            _LOG.exception("failed to broadcast final training_status")
    return {"stopped": True, "exit_code": job.exit_code}


def _clock_snapshot(now: float, running_job: TrainingJob | None) -> dict[str, Any]:
    """Return clock snapshot (wall-clock totals + simulated game-time aggregate).

    Ticks the persistent clock first so a polling client sees an
    up-to-date total even between background lifespan ticks. The
    in-process clock is thread-safe so multiple concurrent /status
    requests can't double-count.

    The returned dict carries both the legacy wall-clock fields
    (``total_seconds``, ``current_session_seconds``) and the new
    aggregated simulated-time fields (``total_simulated_seconds``,
    ``current_session_simulated_seconds``, plus diagnostic ``num_envs``
    / ``frame_skip`` / ``tick_dt`` of the running session). Frontends
    that haven't been updated simply ignore the extra keys.
    """
    clock = get_clock()
    clock.tick(now, running_job is not None)
    session_started = running_job.started_at if running_job is not None else None
    base = clock.to_dict(session_started_at=session_started, now_unix=now)
    sim = clock.compute_total_simulated_seconds(
        current_run_name=(running_job.run_name if running_job is not None else None),
    )
    return {**base, **sim}


@router.get("/status")
async def training_status() -> dict[str, Any]:
    """Latest job snapshot + tail of its log + last crash reason.

    The ``last_crash_reason`` field carries the parsed CRASH_REASON.txt
    the trainer wrote on an unrecoverable error (typically
    ``incompatible_checkpoint``). The frontend renders a red warning
    banner whenever this is non-null so the user knows why auto-restart
    was suppressed.

    Also returns ``training_clock_total_seconds`` (cumulative across all
    runs on this machine) and ``current_session_seconds`` (since the
    running trainer started). Frontend's polling /status loop uses these
    to render the "Training 2h 15m" pill + the drawer counters.
    """
    job = REGISTRY.latest_training()
    crash_reason: dict[str, Any] | None = None
    # Latest job's own reason wins; fall back to the watchdog's global
    # latest so users still see something after the job dict turns over.
    if job is not None and job.last_crash_reason:
        crash_reason = job.last_crash_reason
    elif REGISTRY.watchdog is not None:
        crash_reason = REGISTRY.watchdog.last_crash_reason

    now = now_unix()
    running_job = job if (job is not None and job.is_running()) else None
    clock = _clock_snapshot(now, running_job)

    if job is None:
        return {
            "running": False,
            "job_id": None,
            "pid": None,
            "started_at": 0.0,
            "log_tail": [],
            "last_crash_reason": crash_reason,
            "training_clock_total_seconds": clock["total_seconds"],
            "current_session_seconds": clock["current_session_seconds"],
            "total_simulated_seconds": clock["total_simulated_seconds"],
            "current_session_simulated_seconds": clock["current_session_simulated_seconds"],
            "total_env_steps": clock["total_env_steps"],
            "current_session_env_steps": clock["current_session_env_steps"],
            "current_session_num_envs": clock["current_session_num_envs"],
            "current_session_frame_skip": clock["current_session_frame_skip"],
            "current_session_tick_dt": clock["current_session_tick_dt"],
            "runs_scanned": clock["runs_scanned"],
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
        "last_crash_reason": crash_reason,
        "training_clock_total_seconds": clock["total_seconds"],
        "current_session_seconds": clock["current_session_seconds"],
        "total_simulated_seconds": clock["total_simulated_seconds"],
        "current_session_simulated_seconds": clock["current_session_simulated_seconds"],
        "total_env_steps": clock["total_env_steps"],
        "current_session_env_steps": clock["current_session_env_steps"],
        "current_session_num_envs": clock["current_session_num_envs"],
        "current_session_frame_skip": clock["current_session_frame_skip"],
        "current_session_tick_dt": clock["current_session_tick_dt"],
        "runs_scanned": clock["runs_scanned"],
    }


@router.get("/clock")
async def training_clock() -> dict[str, Any]:
    """Return the persistent training-time clock.

    ``total_seconds`` survives backend restarts (persisted to
    ``models/logs/training_clock.json``). ``current_session_seconds`` is
    derived from the running trainer's ``started_at`` and is 0 when
    no trainer is currently live.

    The simulated-game-time fields aggregate **agent-experienced**
    seconds across every recorded run on disk:

      ``total_simulated_seconds`` = Σ_run env_steps × frame_skip × tick_dt
      ``current_session_simulated_seconds`` = same, but only for the
      currently-running trainer process.

    This is what the user actually cares about: with 48 parallel envs ×
    4 frame-skip × 10 Hz a single wall-clock hour produces ~7700 hours
    of agent game-time, and the user wants to see that big number.
    """
    job = REGISTRY.latest_training()
    running_job = job if (job is not None and job.is_running()) else None
    now = now_unix()
    snap = _clock_snapshot(now, running_job)
    return {
        "total_seconds": snap["total_seconds"],
        "current_session_seconds": snap["current_session_seconds"],
        "running": running_job is not None,
        "session_started_at": (
            float(running_job.started_at) if running_job is not None else 0.0
        ),
        "total_simulated_seconds": snap["total_simulated_seconds"],
        "current_session_simulated_seconds": snap["current_session_simulated_seconds"],
        "total_env_steps": snap["total_env_steps"],
        "current_session_env_steps": snap["current_session_env_steps"],
        "current_session_num_envs": snap["current_session_num_envs"],
        "current_session_frame_skip": snap["current_session_frame_skip"],
        "current_session_tick_dt": snap["current_session_tick_dt"],
        "runs_scanned": snap["runs_scanned"],
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
