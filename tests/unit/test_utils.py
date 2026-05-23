"""Unit tests for kivski_sim.utils."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from kivski_sim.utils import (
    angle_diff,
    clamp,
    ensure_dir,
    hash_config,
    lerp,
    now_unix,
    softmax_np,
    write_json_atomic,
)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def test_angle_diff_handles_wrap() -> None:
    # Exact zero diff
    assert angle_diff(0.0, 0.0) == pytest.approx(0.0)
    # Across +/- pi boundary -- 359deg vs 1deg should be -2deg, not +358deg.
    assert angle_diff(math.radians(1), math.radians(359)) == pytest.approx(
        math.radians(2), abs=1e-9
    )
    # The reverse must flip sign.
    assert angle_diff(math.radians(359), math.radians(1)) == pytest.approx(
        math.radians(-2), abs=1e-9
    )
    # Multi-wrap input: 5pi vs 0 is equivalent to pi vs 0 -> pi (or -pi).
    diff = angle_diff(5.0 * math.pi, 0.0)
    assert abs(abs(diff) - math.pi) < 1e-9
    # Always within [-pi, pi].
    for a in np.linspace(-50, 50, 41):
        for b in np.linspace(-50, 50, 41):
            d = angle_diff(float(a), float(b))
            assert -math.pi - 1e-9 <= d <= math.pi + 1e-9


def test_clamp_in_range() -> None:
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10
    # Endpoints are inclusive.
    assert clamp(0, 0, 10) == 0
    assert clamp(10, 0, 10) == 10
    # Float values work too.
    assert clamp(1.5, 1.0, 2.0) == pytest.approx(1.5)
    # Inverted bounds are an error.
    with pytest.raises(ValueError):
        clamp(0, 10, 0)


def test_lerp_basic() -> None:
    assert lerp(0.0, 10.0, 0.0) == 0.0
    assert lerp(0.0, 10.0, 1.0) == 10.0
    assert lerp(0.0, 10.0, 0.5) == 5.0
    # Extrapolation allowed.
    assert lerp(0.0, 10.0, -0.5) == -5.0
    assert lerp(0.0, 10.0, 1.5) == 15.0


def test_softmax_sums_to_one() -> None:
    x = np.array([1.0, 2.0, 3.0])
    out = softmax_np(x)
    assert out.sum() == pytest.approx(1.0)
    # Monotonic preservation: larger input -> larger output.
    assert out[0] < out[1] < out[2]

    # Batched along axis -1.
    x2 = np.array([[1.0, 2.0, 3.0], [10.0, 0.0, -5.0]])
    out2 = softmax_np(x2, axis=-1)
    np.testing.assert_allclose(out2.sum(axis=-1), np.ones(2), atol=1e-12)

    # Numeric stability: huge inputs do not overflow.
    big = np.array([1000.0, 1001.0, 1002.0])
    out_big = softmax_np(big)
    assert np.all(np.isfinite(out_big))
    assert out_big.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Time / hashing
# ---------------------------------------------------------------------------


def test_now_unix_is_float_and_recent() -> None:
    t = now_unix()
    assert isinstance(t, float)
    # Should be roughly 'now' (after the year 2020).
    assert t > 1_577_836_800.0


def test_hash_config_deterministic() -> None:
    cfg_a = {"a": 1, "b": [1, 2, 3], "c": {"x": True, "y": "z"}}
    # Same content but with reordered keys must hash to the same value.
    cfg_b = {"c": {"y": "z", "x": True}, "b": [1, 2, 3], "a": 1}
    h_a = hash_config(cfg_a)
    h_b = hash_config(cfg_b)
    assert h_a == h_b
    # Hash format: 16 lowercase hex chars (BLAKE2b digest_size=8).
    assert len(h_a) == 16
    int(h_a, 16)  # parseable as hex

    # Different content -> different hash.
    cfg_c = dict(cfg_a)
    cfg_c["a"] = 2
    assert hash_config(cfg_c) != h_a


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def test_ensure_dir_creates_recursive(tmp_path: Path) -> None:
    target = tmp_path / "deeply" / "nested" / "subdir"
    out = ensure_dir(target)
    assert out.exists() and out.is_dir()
    # Idempotent.
    out2 = ensure_dir(target)
    assert out2 == out


def test_write_json_atomic_writes_correctly(tmp_path: Path) -> None:
    target = tmp_path / "out" / "config.json"
    data = {"seed": 42, "items": [1, 2, 3], "nested": {"k": "v"}}
    write_json_atomic(target, data)

    assert target.exists()
    # The intermediate .tmp file must be gone.
    assert not target.with_suffix(target.suffix + ".tmp").exists()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == data


def test_write_json_atomic_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "config.json"
    write_json_atomic(target, {"v": 1})
    write_json_atomic(target, {"v": 2})
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == {"v": 2}
