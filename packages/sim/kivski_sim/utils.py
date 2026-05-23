"""Small, dependency-light utilities shared across the simulator.

Everything in here is meant to be safe to import from any module without
triggering large transitive imports.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "angle_diff",
    "clamp",
    "ensure_dir",
    "hash_config",
    "lerp",
    "now_unix",
    "softmax_np",
    "write_json_atomic",
]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def angle_diff(a: float, b: float) -> float:
    """Shortest signed angular difference ``a - b``, normalised to ``[-pi, pi]``.

    Useful for computing turn deltas where the inputs are unbounded radians
    that may wrap multiple times.
    """
    d = (float(a) - float(b)) % (2.0 * math.pi)
    if d > math.pi:
        d -= 2.0 * math.pi
    return d


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp ``x`` into the closed interval ``[lo, hi]``.

    ``lo`` must be ``<= hi``; we don't auto-swap because that almost always
    indicates a caller bug.
    """
    if lo > hi:
        raise ValueError(f"clamp: lo ({lo}) must be <= hi ({hi})")
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between ``a`` and ``b`` at parameter ``t``.

    ``t`` is **not** clamped -- callers can request extrapolation by passing
    values outside ``[0, 1]``.
    """
    return float(a) + (float(b) - float(a)) * float(t)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax for NumPy arrays."""
    arr = np.asarray(x, dtype=np.float64)
    shifted = arr - np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Time / hashing
# ---------------------------------------------------------------------------


def now_unix() -> float:
    """Current wall-clock time as a unix timestamp (float seconds)."""
    return time.time()


def hash_config(cfg_dict: dict) -> str:
    """Stable 16-hex-char BLAKE2b fingerprint of a config dict.

    Two configurations with the same logical content always hash to the same
    string regardless of key ordering, because we serialize with
    ``sort_keys=True``.
    """
    blob = json.dumps(cfg_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create ``path`` (and parents) if it does not already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json_atomic(path: str | os.PathLike[str], data: Any) -> None:
    """Write ``data`` as JSON to ``path`` atomically.

    The data is written to a sibling ``.tmp`` file, ``fsync``'d, and only then
    ``os.replace``'d into place. This means readers will either see the old
    file or the complete new one -- never a partial write.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    payload = json.dumps(data, indent=2, sort_keys=True, default=str)
    # Use a low-level fd so we can fsync deterministically.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                # Some platforms / mock filesystems don't support fsync.
                pass
    except Exception:
        # Best-effort cleanup of the temp file.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, target)
