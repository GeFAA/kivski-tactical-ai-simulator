"""Unit tests for kivski_sim.replay."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import msgpack
import pytest

from kivski_sim.replay import (
    REPLAY_FORMAT_VERSION,
    ReplayActionFrame,
    ReplayEventFrame,
    ReplayHeader,
    ReplayReader,
    ReplayVersionError,
    ReplayWriter,
)


def _make_header() -> ReplayHeader:
    return ReplayHeader(
        version=REPLAY_FORMAT_VERSION,
        seed=20260523,
        config_hash="abc1234567890def",
        map_name="dustline",
        team_size=5,
        created_at=1_700_000_000.0,
        kivski_version="0.1.0",
    )


def test_roundtrip_header(tmp_path: Path) -> None:
    path = tmp_path / "match.kreplay"
    header = _make_header()
    with ReplayWriter(path, header) as w:
        pass  # only header
    r = ReplayReader(path)
    assert r.header.seed == header.seed
    assert r.header.config_hash == header.config_hash
    assert r.header.map_name == header.map_name
    assert r.header.team_size == header.team_size
    assert r.header.version == REPLAY_FORMAT_VERSION
    assert r.header.kivski_version == header.kivski_version


def test_roundtrip_actions(tmp_path: Path) -> None:
    path = tmp_path / "match.kreplay"
    frames = [
        ReplayActionFrame(
            tick=t,
            actions=[
                {
                    "agent_id": i,
                    "move": (t + i) % 9,
                    "micro": (t + i) % 6,
                    "comm": (t * 3 + i) % 9,
                    "comm_payload": [0.1 * j for j in range(4)],
                    "buy": 0,
                    "aim_target": -1,
                }
                for i in range(10)
            ],
        )
        for t in range(5)
    ]
    with ReplayWriter(path, _make_header()) as w:
        for f in frames:
            w.write_actions(f)

    r = ReplayReader(path)
    read = list(r.iter_actions())
    assert len(read) == 5
    for a, b in zip(frames, read, strict=True):
        assert a.tick == b.tick
        assert len(a.actions) == len(b.actions)
        for src, dst in zip(a.actions, b.actions, strict=True):
            assert src["agent_id"] == dst["agent_id"]
            assert src["move"] == dst["move"]
            assert src["micro"] == dst["micro"]
            assert src["comm"] == dst["comm"]
            assert src["buy"] == dst["buy"]
            assert src["aim_target"] == dst["aim_target"]
            # Float list survives the round-trip.
            assert [round(x, 6) for x in src["comm_payload"]] == [
                round(x, 6) for x in dst["comm_payload"]
            ]


def test_roundtrip_events(tmp_path: Path) -> None:
    path = tmp_path / "match.kreplay"
    events = [
        ReplayEventFrame(tick=0, kind="round_start", data={"round_id": 1}),
        ReplayEventFrame(tick=42, kind="kill", data={"attacker": 3, "victim": 7}),
        ReplayEventFrame(tick=100, kind="plant", data={"site": "A", "by": 3}),
        ReplayEventFrame(tick=180, kind="defuse", data={"by": 8}),
        ReplayEventFrame(tick=181, kind="round_end", data={"winner": "defender"}),
    ]
    with ReplayWriter(path, _make_header()) as w:
        for e in events:
            w.write_event(e)

    r = ReplayReader(path)
    read = list(r.iter_events())
    assert [(e.tick, e.kind, e.data) for e in read] == [
        (e.tick, e.kind, e.data) for e in events
    ]


def test_mixed_actions_and_events(tmp_path: Path) -> None:
    """Iterators must filter cleanly even when both stream types are interleaved."""
    path = tmp_path / "match.kreplay"
    with ReplayWriter(path, _make_header()) as w:
        w.write_event(ReplayEventFrame(tick=0, kind="round_start", data={"r": 1}))
        w.write_actions(ReplayActionFrame(tick=0, actions=[{"agent_id": 0}]))
        w.write_event(ReplayEventFrame(tick=1, kind="kill", data={"v": 2}))
        w.write_actions(ReplayActionFrame(tick=1, actions=[{"agent_id": 0}]))

    r = ReplayReader(path)
    actions = list(r.iter_actions())
    events = list(r.iter_events())
    assert [a.tick for a in actions] == [0, 1]
    assert [e.kind for e in events] == ["round_start", "kill"]


def test_unknown_version_raises(tmp_path: Path) -> None:
    """Reading a file whose header advertises an unknown version must raise."""
    path = tmp_path / "bogus.kreplay"
    # Manually craft a header with a bogus version, bypassing the writer's
    # automatic version stamping.
    fake_header = asdict(_make_header())
    fake_header["version"] = REPLAY_FORMAT_VERSION + 999
    with path.open("wb") as fh:
        fh.write(msgpack.packb({"t": "H", "p": fake_header}, use_bin_type=True))

    with pytest.raises(ReplayVersionError):
        ReplayReader(path)


def test_writer_force_stamps_current_version(tmp_path: Path) -> None:
    """Even if the caller passes a stale header.version, the writer overwrites it."""
    path = tmp_path / "match.kreplay"
    stale = _make_header()
    stale.version = 0  # nonsense
    with ReplayWriter(path, stale) as _:
        pass
    r = ReplayReader(path)
    assert r.header.version == REPLAY_FORMAT_VERSION
