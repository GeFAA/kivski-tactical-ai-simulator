"""Unit tests for :mod:`kivski_api.training_clock`.

The clock is small but load-bearing -- the user-facing "Total trained
5h 30m" number must survive crashes, restarts, and back-to-back ticks
without inflating from a long sleep. These tests guard the trickier
edge cases (atomic write, sleep-clamp, idle vs running attribution).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kivski_api.training_clock import TrainingClock


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_clock_starts_empty(tmp_path: Path) -> None:
    """A fresh clock on a missing file starts at zero and writes on first tick."""
    p = tmp_path / "clock.json"
    clk = TrainingClock(p)
    assert clk.total_seconds == 0.0

    clk.tick(now_unix=1000.0, training_running=False)
    assert p.is_file()
    data = _read(p)
    assert data["total_trained_seconds"] == 0.0
    assert data["last_update"] == 1000.0


def test_clock_accumulates_only_when_running(tmp_path: Path) -> None:
    """Idle ticks update last_update but never inflate the total."""
    p = tmp_path / "clock.json"
    clk = TrainingClock(p)
    clk.tick(1000.0, training_running=False)
    clk.tick(1030.0, training_running=False)
    assert clk.total_seconds == 0.0

    clk.tick(1050.0, training_running=True)
    # last_update before this tick was 1030 -> +20s while running
    assert clk.total_seconds == pytest.approx(20.0)

    clk.tick(1080.0, training_running=False)
    # Idle tick doesn't add anything even though last_update advanced.
    assert clk.total_seconds == pytest.approx(20.0)


def test_clock_clamps_long_gap(tmp_path: Path) -> None:
    """A multi-hour gap (laptop sleep, watchdog stall) is clamped to 60 s."""
    p = tmp_path / "clock.json"
    clk = TrainingClock(p)
    clk.tick(1000.0, training_running=True)
    # 1 hour later — should NOT add 3600 s.
    clk.tick(1000.0 + 3600.0, training_running=True)
    assert clk.total_seconds == pytest.approx(60.0)


def test_clock_persists_across_instances(tmp_path: Path) -> None:
    """Re-opening the same file recovers the prior total."""
    p = tmp_path / "clock.json"
    clk1 = TrainingClock(p)
    clk1.tick(1000.0, training_running=False)
    clk1.tick(1010.0, training_running=True)
    assert clk1.total_seconds == pytest.approx(10.0)

    clk2 = TrainingClock(p)
    assert clk2.total_seconds == pytest.approx(10.0)
    # And it keeps growing from the loaded last_update.
    clk2.tick(1015.0, training_running=True)
    assert clk2.total_seconds == pytest.approx(15.0)


def test_to_dict_session_seconds(tmp_path: Path) -> None:
    """`to_dict` returns 0 session when started_at is None, otherwise the delta."""
    p = tmp_path / "clock.json"
    clk = TrainingClock(p)
    clk.tick(1000.0, training_running=True)

    idle = clk.to_dict()
    assert idle["current_session_seconds"] == 0.0

    running = clk.to_dict(session_started_at=900.0, now_unix=1000.0)
    assert running["current_session_seconds"] == pytest.approx(100.0)


def test_clock_recovers_from_corrupt_file(tmp_path: Path) -> None:
    """A garbage JSON file is treated as 'starting fresh', never raises."""
    p = tmp_path / "clock.json"
    p.write_text("not json at all", encoding="utf-8")
    clk = TrainingClock(p)
    assert clk.total_seconds == 0.0
    # Subsequent ticks rewrite the file with a valid payload.
    clk.tick(2000.0, training_running=False)
    data = _read(p)
    assert data["total_trained_seconds"] == 0.0


# ---------------------------------------------------------------------------
# Aggregated simulated-time accounting
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts as one JSON object per line — matches TelemetrySink."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_compute_simulated_seconds_reads_env_steps_from_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scanner pulls max env_steps + cadence from the trainer's metrics.jsonl."""
    # Pretend our tmp dir is the repo root so the aggregator scans
    # tmp_path/models/logs/* (and not the real machine).
    logs = tmp_path / "models" / "logs"
    run = logs / "run-a"
    _write_jsonl(
        run / "metrics.jsonl",
        [
            {"step": 1, "live/episode": 1.0},
            {
                "step": 1,
                "train/env_steps": 1000.0,
                "train/num_envs": 32.0,
                "train/frame_skip": 4.0,
                "train/tick_dt": 0.1,
            },
            {
                "step": 2,
                "train/env_steps": 5000.0,
                "train/num_envs": 32.0,
                "train/frame_skip": 4.0,
                "train/tick_dt": 0.1,
            },
        ],
    )
    monkeypatch.setattr(
        "kivski_api.training_clock._logs_root", lambda: logs
    )
    monkeypatch.setattr(
        "kivski_api.training_clock._checkpoints_root",
        lambda: tmp_path / "models" / "checkpoints",
    )

    clk = TrainingClock(tmp_path / "clock.json")
    result = clk.compute_total_simulated_seconds()
    assert result["runs_scanned"] == 1
    assert result["total_env_steps"] == pytest.approx(5000.0)
    # 5000 env_steps × 4 frame_skip × 0.1 tick_dt = 2000 sim seconds
    assert result["total_simulated_seconds"] == pytest.approx(2000.0)


def test_compute_simulated_seconds_falls_back_to_checkpoint_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When metrics.jsonl has no env_steps, scanner reads .pt.json sidecars."""
    logs = tmp_path / "models" / "logs"
    ckpts = tmp_path / "models" / "checkpoints"
    # metrics.jsonl with no env_steps key (mimics a pre-May-2026 run).
    _write_jsonl(
        logs / "run-b" / "metrics.jsonl",
        [{"step": 1, "live/episode": 50.0}],
    )
    (ckpts / "run-b").mkdir(parents=True, exist_ok=True)
    (ckpts / "run-b" / "main_ep_5.pt.json").write_text(
        json.dumps({"env_steps": 12345}), encoding="utf-8"
    )
    (ckpts / "run-b" / "main_ep_10.pt.json").write_text(
        json.dumps({"env_steps": 54321}), encoding="utf-8"
    )
    monkeypatch.setattr("kivski_api.training_clock._logs_root", lambda: logs)
    monkeypatch.setattr("kivski_api.training_clock._checkpoints_root", lambda: ckpts)

    clk = TrainingClock(tmp_path / "clock.json")
    result = clk.compute_total_simulated_seconds()
    assert result["total_env_steps"] == pytest.approx(54321.0)
    # Fallback uses _FALLBACK_FRAME_SKIP=4 and _FALLBACK_TICK_DT=0.1.
    assert result["total_simulated_seconds"] == pytest.approx(54321.0 * 4.0 * 0.1)


def test_compute_simulated_seconds_falls_back_to_hparams_x_train_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When neither JSONL env_steps nor sidecars exist, derive from hparams × max(train/step)."""
    logs = tmp_path / "models" / "logs"
    run = logs / "run-c"
    _write_jsonl(
        run / "metrics.jsonl",
        [
            {"step": 5, "train/step": 5.0, "train/episode": 50.0},
            {"step": 12, "train/step": 12.0, "train/episode": 110.0},
        ],
    )
    (run / "hparams.json").write_text(
        json.dumps(
            {
                "num_envs": 32,
                "rollout_steps": 256,
                "tick_rate_hz": 10,
                "frame_skip": 4,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("kivski_api.training_clock._logs_root", lambda: logs)
    monkeypatch.setattr(
        "kivski_api.training_clock._checkpoints_root",
        lambda: tmp_path / "models" / "checkpoints",
    )

    clk = TrainingClock(tmp_path / "clock.json")
    result = clk.compute_total_simulated_seconds()
    # 12 update_steps × 256 rollout × 32 envs = 98304 env_steps
    assert result["total_env_steps"] == pytest.approx(12.0 * 256.0 * 32.0)
    # × 4 frame_skip × 0.1 tick_dt
    expected = 12.0 * 256.0 * 32.0 * 4.0 * 0.1
    assert result["total_simulated_seconds"] == pytest.approx(expected)


def test_compute_simulated_seconds_sums_across_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple runs are aggregated into a single total."""
    logs = tmp_path / "models" / "logs"
    for name, env_steps in (("run-a", 1000.0), ("run-b", 4000.0), ("run-c", 500.0)):
        _write_jsonl(
            logs / name / "metrics.jsonl",
            [
                {
                    "step": 1,
                    "train/env_steps": env_steps,
                    "train/num_envs": 16.0,
                    "train/frame_skip": 2.0,
                    "train/tick_dt": 0.2,
                }
            ],
        )
    monkeypatch.setattr("kivski_api.training_clock._logs_root", lambda: logs)
    monkeypatch.setattr(
        "kivski_api.training_clock._checkpoints_root",
        lambda: tmp_path / "models" / "checkpoints",
    )

    clk = TrainingClock(tmp_path / "clock.json")
    result = clk.compute_total_simulated_seconds()
    assert result["runs_scanned"] == 3
    assert result["total_env_steps"] == pytest.approx(5500.0)
    # 5500 × 2 × 0.2 = 2200
    assert result["total_simulated_seconds"] == pytest.approx(2200.0)


def test_compute_simulated_seconds_persists_run_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-run cache survives clock-instance restarts (saved to clock.json)."""
    logs = tmp_path / "models" / "logs"
    _write_jsonl(
        logs / "run-a" / "metrics.jsonl",
        [
            {
                "step": 1,
                "train/env_steps": 1000.0,
                "train/num_envs": 8.0,
                "train/frame_skip": 1.0,
                "train/tick_dt": 0.1,
            }
        ],
    )
    monkeypatch.setattr("kivski_api.training_clock._logs_root", lambda: logs)
    monkeypatch.setattr(
        "kivski_api.training_clock._checkpoints_root",
        lambda: tmp_path / "models" / "checkpoints",
    )
    clock_path = tmp_path / "clock.json"

    clk1 = TrainingClock(clock_path)
    res1 = clk1.compute_total_simulated_seconds()
    assert res1["total_env_steps"] == pytest.approx(1000.0)

    # Re-open: cache should be hydrated from disk even before scanning.
    clk2 = TrainingClock(clock_path)
    assert "run-a" in clk2.runs
    assert clk2.runs["run-a"]["max_env_steps"] == pytest.approx(1000.0)


def test_compute_simulated_seconds_current_session_breakout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Current run's contribution surfaces separately for the UI."""
    logs = tmp_path / "models" / "logs"
    for name, env_steps in (("run-old", 1000.0), ("run-now", 5000.0)):
        _write_jsonl(
            logs / name / "metrics.jsonl",
            [
                {
                    "step": 1,
                    "train/env_steps": env_steps,
                    "train/num_envs": 32.0,
                    "train/frame_skip": 4.0,
                    "train/tick_dt": 0.1,
                }
            ],
        )
    monkeypatch.setattr("kivski_api.training_clock._logs_root", lambda: logs)
    monkeypatch.setattr(
        "kivski_api.training_clock._checkpoints_root",
        lambda: tmp_path / "models" / "checkpoints",
    )

    clk = TrainingClock(tmp_path / "clock.json")
    result = clk.compute_total_simulated_seconds(current_run_name="run-now")
    assert result["total_env_steps"] == pytest.approx(6000.0)
    assert result["current_session_env_steps"] == pytest.approx(5000.0)
    assert result["current_session_num_envs"] == pytest.approx(32.0)
    assert result["current_session_frame_skip"] == pytest.approx(4.0)
    # 5000 × 4 × 0.1 = 2000
    assert result["current_session_simulated_seconds"] == pytest.approx(2000.0)
