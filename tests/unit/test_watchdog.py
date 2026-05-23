"""Tests for the :class:`TrainingWatchdog` restart cascade fix.

These guard the v0.5.1 behaviour: a crashing trainer subprocess gets at
most ``_WATCHDOG_MAX_RESTARTS`` auto-restarts per rolling
``_WATCHDOG_RESTART_WINDOW_SECONDS`` window, and an
``incompatible_checkpoint`` CRASH_REASON triggers a restart *without*
``--resume`` (then immediately stops auto-restarting the resulting fresh
process if it also dies).

We avoid actually spawning trainer subprocesses by injecting a fake
``subprocess.Popen`` substitute via monkeypatch; the watchdog only cares
about the ``poll()`` / ``pid`` / ``returncode`` surface.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from kivski_api import session as session_module
from kivski_api.session import (
    _WATCHDOG_MAX_RESTARTS,
    SessionRegistry,
    TrainingJob,
    TrainingWatchdog,
)

# ---------------------------------------------------------------------------
# Fake subprocess.Popen
# ---------------------------------------------------------------------------


class _FakeProcess:
    """Stand-in for :class:`subprocess.Popen` that exits with the given code."""

    def __init__(self, pid: int, returncode: int) -> None:
        self.pid = int(pid)
        self._returncode = int(returncode)

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:  # pragma: no cover - unused
        return self._returncode

    def terminate(self) -> None:  # pragma: no cover - unused
        pass

    def send_signal(self, _sig: int) -> None:  # pragma: no cover - unused
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_crashed_job(
    *,
    job_id: str = "alpha",
    exit_code: int = 1,
    run_dir: Path | None = None,
) -> TrainingJob:
    """Build a TrainingJob whose subprocess already exited."""
    metrics_path = (run_dir / "metrics.jsonl") if run_dir is not None else None
    return TrainingJob(
        job_id=job_id,
        config_path="configs/fast.yaml",
        log_path=Path("nonexistent.log"),
        started_at=0.0,
        pid=99999,
        process=_FakeProcess(pid=99999, returncode=exit_code),
        episodes=10,
        resume_from=None,
        run_name="test-run",
        metrics_jsonl_path=metrics_path,
    )


def _patch_popen(monkeypatch: pytest.MonkeyPatch, spawned: list[list[str]]) -> None:
    """Replace subprocess.Popen so _restart_job records the cmd without spawning."""
    import subprocess as _subprocess

    def _fake_popen(cmd: list[str], **_kwargs: Any) -> _FakeProcess:
        spawned.append(list(cmd))
        return _FakeProcess(pid=12345, returncode=0)

    monkeypatch.setattr(_subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(session_module.subprocess, "Popen", _fake_popen)


def _patch_routes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, resume: Path | None) -> None:
    """Replace the lazy-imported route helpers so _restart_job stays self-contained."""
    from kivski_api.routes import training as training_routes

    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(training_routes, "_log_dir", lambda: log_dir)
    monkeypatch.setattr(training_routes, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        training_routes, "_find_resumable_checkpoint", lambda: resume
    )


# ---------------------------------------------------------------------------
# Tests: rolling 10-min budget
# ---------------------------------------------------------------------------


def test_watchdog_stops_after_3_restarts_in_10min(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """4 simulated crashes in quick succession -> exactly 3 spawns."""
    spawned: list[list[str]] = []
    _patch_popen(monkeypatch, spawned)
    _patch_routes(monkeypatch, tmp_path, resume=None)

    reg = SessionRegistry()
    wd = TrainingWatchdog(reg)

    # Simulate 4 distinct crashed jobs landing in the registry. The
    # watchdog should restart the first 3 and refuse the 4th because the
    # rolling counter hit the budget.
    for i in range(4):
        job = _make_crashed_job(job_id=f"crash-{i}")
        reg.register_training(job)
        wd._check_once()

    assert len(spawned) == _WATCHDOG_MAX_RESTARTS == 3


def test_watchdog_handles_each_crash_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A crashed job stays ``handled`` -- repeat polls never re-spawn it.

    This is the *actual* bug that filled the disk: pre-fix, every 10s
    poll re-spawned the same crashed parent until the budget exploded
    or the host OOM'd. The ``handled`` flag stops that.
    """
    spawned: list[list[str]] = []
    _patch_popen(monkeypatch, spawned)
    _patch_routes(monkeypatch, tmp_path, resume=None)

    reg = SessionRegistry()
    wd = TrainingWatchdog(reg)

    job = _make_crashed_job(job_id="only-crash")
    reg.register_training(job)

    wd._check_once()  # first sighting -> restart
    wd._check_once()  # second poll -> should NOT spawn anything new
    wd._check_once()  # third poll -> still nothing

    assert len(spawned) == 1
    assert job.handled is True


def test_watchdog_skips_resume_on_incompatible_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A CRASH_REASON with category=incompatible_checkpoint -> no --resume."""
    spawned: list[list[str]] = []
    _patch_popen(monkeypatch, spawned)
    fake_resume = tmp_path / "models" / "checkpoints" / "best.pt"
    fake_resume.parent.mkdir(parents=True, exist_ok=True)
    fake_resume.write_bytes(b"")
    _patch_routes(monkeypatch, tmp_path, resume=fake_resume)

    # Drop a CRASH_REASON.txt that flags the failure as incompatible.
    run_dir = tmp_path / "models" / "logs" / "test-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    crash_path = run_dir / "CRASH_REASON.txt"
    crash_path.write_text(
        "category: incompatible_checkpoint\n"
        f"source_checkpoint: {fake_resume}\n"
        "run_name: test-run\n"
        "episode: 0\n"
        "\n"
        "Checkpoint best.pt is incompatible with current config:\n"
        "  model.hidden_size: ckpt=64 != current=256\n",
        encoding="utf-8",
    )

    reg = SessionRegistry()
    wd = TrainingWatchdog(reg)
    job = _make_crashed_job(job_id="incompat", run_dir=run_dir)
    reg.register_training(job)

    wd._check_once()
    # We *did* spawn one restart -- but without a --resume flag.
    assert len(spawned) == 1
    cmd = spawned[0]
    assert "--resume" not in cmd
    # ...and the parsed reason is exposed for the UI.
    assert job.last_crash_reason is not None
    assert job.last_crash_reason["category"] == "incompatible_checkpoint"
    assert wd.last_crash_reason is not None
    assert wd.last_crash_reason["category"] == "incompatible_checkpoint"


def test_watchdog_respects_stop_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """User-initiated /api/training/stop must NEVER trigger an auto-restart."""
    spawned: list[list[str]] = []
    _patch_popen(monkeypatch, spawned)
    _patch_routes(monkeypatch, tmp_path, resume=None)

    reg = SessionRegistry()
    wd = TrainingWatchdog(reg)

    job = _make_crashed_job(job_id="user-stop", exit_code=-15)
    job.stop_requested = True
    reg.register_training(job)

    wd._check_once()
    assert spawned == []


def test_watchdog_window_resets_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Restart timestamps older than the rolling window are pruned.

    We can't easily fast-forward 10 real minutes in a unit test, so we
    poke the internal list directly instead -- this is the same code
    path ``_check_once`` walks every tick.
    """
    spawned: list[list[str]] = []
    _patch_popen(monkeypatch, spawned)
    _patch_routes(monkeypatch, tmp_path, resume=None)

    reg = SessionRegistry()
    wd = TrainingWatchdog(reg)
    # Pre-load the rolling list with 3 ancient timestamps that should
    # all be pruned on the next _check_once().
    wd._recent_restarts = [0.0, 1.0, 2.0]

    job = _make_crashed_job(job_id="post-window")
    reg.register_training(job)
    wd._check_once()
    # Old entries got pruned (well below now-600s) and this fresh crash
    # consumed only one slot.
    assert len(spawned) == 1
    assert all(ts > 100.0 for ts in wd._recent_restarts)


# ---------------------------------------------------------------------------
# Lifecycle smoke (asyncio start/stop)
# ---------------------------------------------------------------------------


def test_watchdog_lifecycle_start_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """start() schedules a task and registers itself on the registry; stop() awaits."""
    reg = SessionRegistry()
    wd = TrainingWatchdog(reg)

    async def _drive() -> None:
        await wd.start()
        assert reg.watchdog is wd
        await wd.stop()

    asyncio.run(_drive())
