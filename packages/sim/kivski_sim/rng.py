"""Deterministic multi-channel RNG management.

Each subsystem of the simulator (combat, spawning, buy-noise, sound, comm init,
drop) draws random numbers from its own named channel. A channel is a
``numpy.random.Generator`` (PCG64) seeded from ``hashlib.blake2b`` of the
combination ``(seed, channel_name)``. This guarantees:

1. **Reproducibility**: two ``RngHub``s with the same seed produce identical
   sequences on the same channel.
2. **Isolation**: re-seeding a sub-game (e.g. re-rolling combat dice) does not
   pollute unrelated channels like ``spawn`` or ``buy_noise``.
3. **Replay-safety**: ``snapshot()``/``restore()`` round-trips the bit-exact
   state of every channel so a replay can resume mid-match.

The known channel names are exposed via :data:`KNOWN_CHANNELS` so that callers
can pre-warm channels at construction time and keep replay snapshots stable.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Final

import numpy as np

__all__ = ["KNOWN_CHANNELS", "RngHub"]


KNOWN_CHANNELS: Final[tuple[str, ...]] = (
    "combat",
    "spawn",
    "buy_noise",
    "sound",
    "comm_init",
    "drop",
)


def _derive_seed(seed: int, name: str) -> int:
    """Derive a 128-bit seed from ``(seed, name)`` using BLAKE2b.

    Using BLAKE2b ensures we get a high-quality, deterministic seed that is
    independent across channel names even when the master seed is small. We
    return a 128-bit integer because PCG64 accepts seeds up to 128 bits and
    using the full width avoids any practical collision risk between channels.
    """
    payload = struct.pack(">q", int(seed) & 0xFFFFFFFFFFFFFFFF) + name.encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(digest, "big", signed=False)


class RngHub:
    """Multi-channel deterministic RNG.

    Channels are created lazily on first access via :meth:`channel` and cached
    thereafter. Channel state can be serialized with :meth:`snapshot` and
    restored with :meth:`restore`, which is used by the replay system.
    """

    __slots__ = ("_seed", "_channels")

    def __init__(self, seed: int) -> None:
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int, got {type(seed).__name__}")
        self._seed: int = int(seed)
        self._channels: dict[str, np.random.Generator] = {}

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def seed(self) -> int:
        return self._seed

    def channel(self, name: str) -> np.random.Generator:
        """Return the deterministic ``Generator`` for ``name``.

        Cached -- the first call creates a fresh PCG64-backed Generator seeded
        with ``blake2b(seed, name)``; subsequent calls return the same object
        so that consumers see a continuous sequence.
        """
        if not name:
            raise ValueError("channel name must be a non-empty string")
        gen = self._channels.get(name)
        if gen is None:
            bit_gen = np.random.PCG64(_derive_seed(self._seed, name))
            gen = np.random.Generator(bit_gen)
            self._channels[name] = gen
        return gen

    def channels(self) -> tuple[str, ...]:
        """Return the names of all currently materialized channels."""
        return tuple(self._channels.keys())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop every materialized channel.

        After ``reset()`` the next :meth:`channel` call rebuilds the generator
        from the original seed, so the sequence starts over identically.
        """
        self._channels.clear()

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Serialize current channel state for replay storage.

        Returns a plain dict (msgpack-friendly) containing the master seed and
        each channel's bit-generator state. The state dict from NumPy is
        retained as-is (it is JSON/msgpack-serializable since NumPy 1.17).
        """
        return {
            "seed": self._seed,
            "channels": {
                name: gen.bit_generator.state for name, gen in self._channels.items()
            },
        }

    def restore(self, snap: dict) -> None:
        """Restore the hub from a snapshot produced by :meth:`snapshot`.

        Channels not present in the snapshot are dropped. Channels present in
        the snapshot but not currently materialized are rebuilt.
        """
        if "seed" not in snap or "channels" not in snap:
            raise ValueError("snapshot is missing required keys 'seed'/'channels'")
        self._seed = int(snap["seed"])
        self._channels.clear()
        for name, state in snap["channels"].items():
            bit_gen = np.random.PCG64(_derive_seed(self._seed, name))
            bit_gen.state = state
            self._channels[name] = np.random.Generator(bit_gen)
