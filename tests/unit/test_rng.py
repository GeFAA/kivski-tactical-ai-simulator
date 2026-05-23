"""Unit tests for kivski_sim.rng.RngHub."""

from __future__ import annotations

import numpy as np
from kivski_sim.rng import KNOWN_CHANNELS, RngHub


def test_same_seed_same_sequence() -> None:
    """Two hubs created with the same seed must produce identical draws."""
    a = RngHub(seed=42)
    b = RngHub(seed=42)
    for ch in ("combat", "spawn"):
        seq_a = a.channel(ch).standard_normal(8)
        seq_b = b.channel(ch).standard_normal(8)
        np.testing.assert_array_equal(seq_a, seq_b)


def test_different_channels_independent() -> None:
    """Drawing from one channel must not advance another."""
    hub = RngHub(seed=2026)
    # Snapshot 'spawn' before exercising 'combat'.
    spawn_before = hub.channel("spawn").bit_generator.state
    # Exercise combat heavily.
    _ = hub.channel("combat").integers(0, 100, size=1000)
    spawn_after = hub.channel("spawn").bit_generator.state
    assert spawn_before == spawn_after, "spawn channel state changed after combat draws"

    # And two different channels with the same seed must produce different
    # sequences (otherwise the per-channel derivation is broken).
    combat_seq = RngHub(seed=7).channel("combat").integers(0, 1 << 30, size=16)
    spawn_seq = RngHub(seed=7).channel("spawn").integers(0, 1 << 30, size=16)
    assert not np.array_equal(combat_seq, spawn_seq)


def test_snapshot_restore_recovers_state() -> None:
    """A snapshot taken mid-sequence must let us reproduce the tail."""
    hub = RngHub(seed=123)
    # Burn some values across multiple channels so we have non-initial state.
    _ = hub.channel("combat").standard_normal(5)
    _ = hub.channel("spawn").integers(0, 100, size=3)
    _ = hub.channel("sound").random(4)

    snap = hub.snapshot()
    expected_combat = hub.channel("combat").standard_normal(10)
    expected_spawn = hub.channel("spawn").integers(0, 100, size=10)
    expected_sound = hub.channel("sound").random(10)

    # Build a fresh hub from the snapshot and replay the tail.
    restored = RngHub(seed=999)  # deliberately different seed -- restore overrides
    restored.restore(snap)
    np.testing.assert_array_equal(restored.channel("combat").standard_normal(10), expected_combat)
    np.testing.assert_array_equal(restored.channel("spawn").integers(0, 100, size=10), expected_spawn)
    np.testing.assert_array_equal(restored.channel("sound").random(10), expected_sound)
    assert restored.seed == 123


def test_reset_returns_to_initial_sequence() -> None:
    """After reset() the same channel must replay its opening sequence."""
    hub = RngHub(seed=11)
    first = hub.channel("combat").integers(0, 1 << 20, size=4).copy()
    hub.reset()
    after = hub.channel("combat").integers(0, 1 << 20, size=4)
    np.testing.assert_array_equal(first, after)


def test_known_channels_all_resolvable() -> None:
    """Every name in KNOWN_CHANNELS must yield a usable Generator."""
    hub = RngHub(seed=1)
    for ch in KNOWN_CHANNELS:
        gen = hub.channel(ch)
        assert isinstance(gen, np.random.Generator)
        # Sanity: produces a finite float
        assert np.isfinite(gen.random())
