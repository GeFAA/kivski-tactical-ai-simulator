"""Compact streaming replay format based on msgpack.

A replay file is a sequence of msgpack-encoded records:

* the first record is the :class:`ReplayHeader` (tagged ``"H"``);
* every subsequent record is either an action frame (tag ``"A"``) or an event
  frame (tag ``"E"``).

Observations are intentionally **not** stored: the engine is deterministic, so
given the seed, config-hash and the action stream, the full match can be
re-simulated bit-exactly.

The streaming format means writers can append frames as the match progresses
without holding everything in memory, and readers can iterate without loading
the whole file at once.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import IO, Any

import msgpack

__all__ = [
    "REPLAY_FORMAT_VERSION",
    "ReplayActionFrame",
    "ReplayEventFrame",
    "ReplayHeader",
    "ReplayReader",
    "ReplayWriter",
    "ReplayVersionError",
]


REPLAY_FORMAT_VERSION: int = 1

# Per-record tags used by the streaming format. Keep them single-character to
# minimise the serialized footprint.
_TAG_HEADER = "H"
_TAG_ACTION = "A"
_TAG_EVENT = "E"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReplayHeader:
    version: int = REPLAY_FORMAT_VERSION
    seed: int = 0
    config_hash: str = ""
    map_name: str = ""
    team_size: int = 5
    created_at: float = 0.0  # unix timestamp
    kivski_version: str = "0.1.0"


@dataclass
class ReplayActionFrame:
    tick: int
    # One dict per agent: {agent_id, move, micro, comm, comm_payload, buy, aim_target}
    actions: list[dict] = field(default_factory=list)


@dataclass
class ReplayEventFrame:
    tick: int
    kind: str  # "round_start" | "round_end" | "plant" | "defuse" | "detonate" | "kill"
    data: dict = field(default_factory=dict)


class ReplayVersionError(ValueError):
    """Raised when a replay file uses an unsupported format version."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pack(tag: str, payload: dict[str, Any]) -> bytes:
    """Serialize one tagged record."""
    return msgpack.packb({"t": tag, "p": payload}, use_bin_type=True)


def _action_frame_to_dict(frame: ReplayActionFrame) -> dict[str, Any]:
    return {"tick": int(frame.tick), "actions": frame.actions}


def _event_frame_to_dict(frame: ReplayEventFrame) -> dict[str, Any]:
    return {"tick": int(frame.tick), "kind": str(frame.kind), "data": frame.data}


def _dict_to_header(d: dict[str, Any]) -> ReplayHeader:
    valid = {f.name for f in fields(ReplayHeader)}
    filtered = {k: v for k, v in d.items() if k in valid}
    return ReplayHeader(**filtered)


def _dict_to_action(d: dict[str, Any]) -> ReplayActionFrame:
    return ReplayActionFrame(tick=int(d["tick"]), actions=list(d.get("actions", [])))


def _dict_to_event(d: dict[str, Any]) -> ReplayEventFrame:
    return ReplayEventFrame(
        tick=int(d["tick"]),
        kind=str(d["kind"]),
        data=dict(d.get("data", {})),
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class ReplayWriter:
    """Streaming replay writer.

    Use as a context manager or call :meth:`close` explicitly.
    """

    def __init__(self, path: str | Path, header: ReplayHeader) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[bytes] | None = self._path.open("wb")
        self._closed = False
        # Force-stamp the running format version so a stale header literal
        # cannot smuggle in an out-of-date number.
        header.version = REPLAY_FORMAT_VERSION
        self._fh.write(_pack(_TAG_HEADER, asdict(header)))

    # ------------------------------------------------------------------

    def write_actions(self, frame: ReplayActionFrame) -> None:
        self._assert_open()
        assert self._fh is not None
        self._fh.write(_pack(_TAG_ACTION, _action_frame_to_dict(frame)))

    def write_event(self, frame: ReplayEventFrame) -> None:
        self._assert_open()
        assert self._fh is not None
        self._fh.write(_pack(_TAG_EVENT, _event_frame_to_dict(frame)))

    def close(self) -> None:
        if self._closed:
            return
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
        self._closed = True

    # ------------------------------------------------------------------

    def __enter__(self) -> ReplayWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _assert_open(self) -> None:
        if self._closed or self._fh is None:
            raise RuntimeError("ReplayWriter is closed")


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class ReplayReader:
    """Streaming replay reader.

    The header is parsed eagerly so version mismatches surface immediately. The
    action / event streams are iterated lazily.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"replay file not found: {self._path}")
        self._header = self._load_header()

    # ------------------------------------------------------------------

    @property
    def header(self) -> ReplayHeader:
        return self._header

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------

    def iter_actions(self) -> Iterator[ReplayActionFrame]:
        for tag, payload in self._iter_records(skip_header=True):
            if tag == _TAG_ACTION:
                yield _dict_to_action(payload)

    def iter_events(self) -> Iterator[ReplayEventFrame]:
        for tag, payload in self._iter_records(skip_header=True):
            if tag == _TAG_EVENT:
                yield _dict_to_event(payload)

    # ------------------------------------------------------------------

    def _load_header(self) -> ReplayHeader:
        records = self._iter_records(skip_header=False)
        try:
            tag, payload = next(records)
        except StopIteration as exc:  # empty file
            raise ValueError(f"replay file is empty: {self._path}") from exc
        if tag != _TAG_HEADER:
            raise ValueError(f"replay file does not start with header (tag={tag!r}): {self._path}")
        header = _dict_to_header(payload)
        if header.version != REPLAY_FORMAT_VERSION:
            raise ReplayVersionError(
                f"unsupported replay version {header.version}; this build expects {REPLAY_FORMAT_VERSION}"
            )
        return header

    def _iter_records(self, *, skip_header: bool) -> Iterator[tuple[str, dict]]:
        with self._path.open("rb") as fh:
            unpacker = msgpack.Unpacker(fh, raw=False, strict_map_key=False)
            first = True
            for rec in unpacker:
                if not isinstance(rec, dict) or "t" not in rec or "p" not in rec:
                    raise ValueError(f"malformed replay record: {rec!r}")
                tag = rec["t"]
                payload = rec["p"]
                if first and skip_header:
                    first = False
                    continue
                first = False
                yield tag, payload
