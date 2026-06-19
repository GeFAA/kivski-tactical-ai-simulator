"""Unit tests for kivski_agents.telemetry and friends.

The TensorBoard and W&B backends are only exercised when their optional
dependencies are importable; we ``pytest.skip`` otherwise so the suite
stays green on a stock dev environment that only has CSV available.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pytest
from kivski_agents.metrics import (
    CommUsageStats,
    EpisodeStats,
    TrainStepMetrics,
    comm_usage_to_dict,
    episode_stats_to_dict,
    train_metrics_to_dict,
)
from kivski_agents.run_naming import (
    generate_run_name,
    latest_run_name,
    list_runs,
)
from kivski_agents.telemetry import (
    CSVSink,
    MultiSink,
    NoOpSink,
    TelemetrySink,
    make_sink,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _read_metrics_rows(run_dir: Path) -> list[dict[str, str]]:
    """Return all rows from ``metrics.csv`` as dicts."""
    with (run_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class _RecordingSink(TelemetrySink):
    """In-memory sink that records every call -- handy for MultiSink tests."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.dicts: list[tuple[dict[str, float], int]] = []
        self.texts: list[tuple[str, str, int]] = []
        self.hparams: list[dict[str, Any]] = []
        self.closed: bool = False
        self.flushed: int = 0

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self.scalars.append((key, value, step))

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        self.dicts.append((dict(metrics), step))

    def log_text(self, key: str, text: str, step: int) -> None:
        self.texts.append((key, text, step))

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        self.hparams.append(dict(hparams))

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# CSVSink
# ---------------------------------------------------------------------------


def test_csv_sink_writes_scalars(tmp_path: Path) -> None:
    sink = CSVSink(tmp_path, "run01", flush_every_seconds=0.0)
    sink.log_scalar("loss", 1.0, step=0)
    sink.log_scalar("loss", 0.5, step=1)
    sink.log_scalar("reward", 12.0, step=1)
    sink.close()

    rows = _read_metrics_rows(tmp_path / "run01")
    assert len(rows) == 3
    keys = [r["key"] for r in rows]
    assert keys == ["loss", "loss", "reward"]
    # Values should round-trip as plain floats.
    assert [float(r["value"]) for r in rows] == [1.0, 0.5, 12.0]
    assert [int(r["step"]) for r in rows] == [0, 1, 1]


def test_csv_sink_writes_dict(tmp_path: Path) -> None:
    sink = CSVSink(tmp_path, "run02", flush_every_seconds=0.0)
    metrics = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.5}
    sink.log_dict(metrics, step=7)
    sink.close()

    rows = _read_metrics_rows(tmp_path / "run02")
    assert len(rows) == 4
    seen = {r["key"]: float(r["value"]) for r in rows}
    assert seen == metrics
    assert all(int(r["step"]) == 7 for r in rows)


def test_csv_sink_writes_text_and_hparams(tmp_path: Path) -> None:
    sink = CSVSink(tmp_path, "run03", flush_every_seconds=0.0)
    sink.log_text("notes", "hello world", step=0)
    sink.log_hyperparams({"lr": 3e-4, "gamma": 0.99})
    sink.close()

    text_rows = list(csv.DictReader((tmp_path / "run03" / "text.csv").open("r", encoding="utf-8")))
    assert len(text_rows) == 1
    assert text_rows[0]["text"] == "hello world"

    hp_rows = list(csv.DictReader((tmp_path / "run03" / "hparams.csv").open("r", encoding="utf-8")))
    seen = {r["key"]: r["value"] for r in hp_rows}
    assert seen == {"lr": "0.0003", "gamma": "0.99"}


def test_csv_sink_close_is_idempotent(tmp_path: Path) -> None:
    sink = CSVSink(tmp_path, "runX", flush_every_seconds=0.0)
    sink.log_scalar("a", 1.0, 0)
    sink.close()
    # Second close must not raise and must not corrupt the file.
    sink.close()
    rows = _read_metrics_rows(tmp_path / "runX")
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# NoOpSink
# ---------------------------------------------------------------------------


def test_noop_sink_no_errors() -> None:
    sink = NoOpSink()
    sink.log_scalar("x", 1.0, 0)
    sink.log_dict({"a": 1.0}, 0)
    sink.log_text("k", "v", 0)
    sink.log_hyperparams({"lr": 1e-3})
    sink.flush()
    sink.close()
    # All calls are silent no-ops; nothing to assert beyond "no exception".


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_make_sink_csv_returns_csv(tmp_path: Path) -> None:
    # v0.2: `make_sink("csv")` returns a MultiSink wrapping CSVSink + JSONLSink
    # because the live API broadcaster tails the JSONL feed (CSV stays for
    # offline analytics).
    sink = make_sink("csv", tmp_path, "run-csv")
    assert isinstance(sink, MultiSink), f"expected MultiSink, got {type(sink).__name__}"
    inner_types = {type(s) for s in sink.sinks}
    assert CSVSink in inner_types, f"CSV not in {inner_types}"
    sink.close()


def test_make_sink_none_returns_noop(tmp_path: Path) -> None:
    sink = make_sink("none", tmp_path, "run-none")
    assert isinstance(sink, NoOpSink)


def test_make_sink_is_case_insensitive(tmp_path: Path) -> None:
    sink = make_sink("CSV", tmp_path, "run-CSV")
    assert isinstance(sink, MultiSink)
    inner_types = {type(s) for s in sink.sinks}
    assert CSVSink in inner_types
    sink.close()


def test_make_sink_unknown_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_sink("not_a_backend", tmp_path, "run")


def test_make_sink_all_returns_multisink(tmp_path: Path) -> None:
    sink = make_sink("all", tmp_path, "run-all")
    assert isinstance(sink, MultiSink)
    # CSV is mandatory in "all" mode.
    assert any(isinstance(s, CSVSink) for s in sink.sinks)
    sink.close()


# ---------------------------------------------------------------------------
# MultiSink
# ---------------------------------------------------------------------------


def test_multi_sink_dispatches_to_all() -> None:
    a = _RecordingSink()
    b = _RecordingSink()
    multi = MultiSink([a, b])

    multi.log_scalar("loss", 0.42, step=3)
    multi.log_dict({"r": 1.0, "g": 2.0}, step=4)
    multi.log_text("info", "hello", step=5)
    multi.log_hyperparams({"lr": 1e-3})
    multi.flush()
    multi.close()

    for child in (a, b):
        assert child.scalars == [("loss", 0.42, 3)]
        assert child.dicts == [({"r": 1.0, "g": 2.0}, 4)]
        assert child.texts == [("info", "hello", 5)]
        assert child.hparams == [{"lr": 1e-3}]
        assert child.flushed == 1
        assert child.closed is True


def test_multi_sink_isolates_failures() -> None:
    class _BrokenSink(_RecordingSink):
        def log_scalar(self, key: str, value: float, step: int) -> None:
            raise RuntimeError("disk on fire")

    bad = _BrokenSink()
    good = _RecordingSink()
    multi = MultiSink([bad, good])

    multi.log_scalar("loss", 0.1, step=0)
    # The good sink must have received the call despite the bad one blowing up.
    assert good.scalars == [("loss", 0.1, 0)]
    # The error must have been captured for inspection.
    assert len(multi.errors) == 1
    assert multi.errors[0][0] == "_BrokenSink"


# ---------------------------------------------------------------------------
# Optional backends (skipped if dependency is missing)
# ---------------------------------------------------------------------------


def test_make_sink_tensorboard(tmp_path: Path) -> None:
    try:
        import torch.utils.tensorboard  # noqa: F401
    except ImportError:
        pytest.skip("PyTorch/TensorBoard not installed")

    from kivski_agents.telemetry import TensorBoardSink

    sink = make_sink("tensorboard", tmp_path, "tb-run")
    try:
        assert isinstance(sink, TensorBoardSink)
        sink.log_scalar("loss", 1.0, step=0)
        sink.log_dict({"a": 1.0, "b": 2.0}, step=1)
        sink.log_text("notes", "hello", step=0)
    finally:
        sink.close()
    # SummaryWriter creates an events file in the run dir.
    assert any((tmp_path / "tb-run").iterdir())


def test_make_sink_wandb_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        import wandb  # noqa: F401
    except ImportError:
        pytest.skip("wandb not installed")

    from kivski_agents.telemetry import WandbSink

    # Force offline mode so we don't try to talk to wandb.ai.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("KIVSKI_WANDB_PROJECT", "kivski-tests")

    sink = make_sink("wandb", tmp_path, "wb-run")
    try:
        assert isinstance(sink, WandbSink)
        sink.log_scalar("loss", 0.5, step=0)
    finally:
        sink.close()


# ---------------------------------------------------------------------------
# Run naming
# ---------------------------------------------------------------------------


def test_run_name_format() -> None:
    name = generate_run_name()
    # Anchored regex: prefix-YYYYMMDD-HHMMSS-shortuid8
    assert re.fullmatch(r"kivski-\d{8}-\d{6}-[0-9a-f]{8}", name)


def test_run_name_custom_prefix() -> None:
    name = generate_run_name("eval")
    assert name.startswith("eval-")
    assert re.fullmatch(r"eval-\d{8}-\d{6}-[0-9a-f]{8}", name)


def test_run_name_rejects_bad_prefix() -> None:
    with pytest.raises(ValueError):
        generate_run_name("")
    with pytest.raises(ValueError):
        generate_run_name("bad prefix")


def test_list_and_latest_runs(tmp_path: Path) -> None:
    # Empty dir -> empty list / None
    assert list_runs(tmp_path) == []
    assert latest_run_name(tmp_path) is None

    # Create three runs in known mtime order.
    import os as _os
    import time as _time

    names = ["run-a", "run-b", "run-c"]
    base = _time.time()
    for i, n in enumerate(names):
        d = tmp_path / n
        d.mkdir()
        _os.utime(d, (base + i, base + i))

    listed = list_runs(tmp_path)
    assert listed == names  # oldest first
    assert latest_run_name(tmp_path) == "run-c"


def test_list_runs_missing_dir(tmp_path: Path) -> None:
    """list_runs on a non-existent dir returns []."""
    assert list_runs(tmp_path / "does-not-exist") == []


# ---------------------------------------------------------------------------
# Metrics dataclasses
# ---------------------------------------------------------------------------


def test_episode_stats_to_dict_keys() -> None:
    s = EpisodeStats(
        episode=3,
        match_done=True,
        yellow_score=13,
        blue_score=11,
        winner="yellow",
        total_rounds=24,
        avg_round_duration_ticks=520.5,
        total_survivors=120,
        total_deaths=118,
        bombs_planted=14,
        bombs_defused=6,
        bombs_detonated=8,
        total_rewards_yellow=42.0,
        total_rewards_blue=-3.0,
        timestamp=1_700_000_000.0,
    )
    d = episode_stats_to_dict(s)
    assert d["episode/episode"] == 3.0
    assert d["episode/winner_code"] == 1.0
    assert d["episode/match_done"] == 1.0
    assert d["episode/yellow_score"] == 13.0
    # Every value must be a finite float (sink-safe).
    for v in d.values():
        assert isinstance(v, float)


def test_train_metrics_to_dict_keys() -> None:
    m = TrainStepMetrics(
        step=100,
        episode=5,
        policy_loss=0.12,
        value_loss=0.34,
        entropy=1.23,
        kl_divergence=0.01,
        explained_variance=0.85,
        grad_norm=0.4,
        learning_rate=3e-4,
        advantage_mean=0.0,
        advantage_std=1.0,
        fps=850.0,
    )
    d = train_metrics_to_dict(m)
    assert d["train/policy_loss"] == 0.12
    assert d["train/learning_rate"] == pytest.approx(3e-4)
    assert d["train/fps"] == 850.0


def test_comm_usage_to_dict() -> None:
    stats = CommUsageStats(
        counts={0: 10, 1: 5, 3: 20},
        entropy=0.97,
        mean_payload_norm=1.42,
    )
    d = comm_usage_to_dict(stats)
    assert d["comm/entropy"] == pytest.approx(0.97)
    assert d["comm/total_messages"] == 35.0
    assert d["comm/count/0"] == 10.0
    assert d["comm/count/1"] == 5.0
    assert d["comm/count/3"] == 20.0
    assert "comm/count/2" not in d  # missing actions not synthesised
