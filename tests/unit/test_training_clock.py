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
