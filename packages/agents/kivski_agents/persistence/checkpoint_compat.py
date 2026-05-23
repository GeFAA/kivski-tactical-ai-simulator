"""Checkpoint compatibility metadata + validation.

Why this exists
---------------

The trainer and the API both ``torch.load`` checkpoints written by past
runs. Past v0.5.0 we hit a real footgun: a smoke run produced
``models/checkpoints/best.pt`` with ``hidden_size=64`` and the next
training session (with ``hidden_size=256`` from a different YAML) tried
to ``load_state_dict`` it. PyTorch raised a ``RuntimeError: size
mismatch ...``, the trainer process crashed, the API watchdog kept
respawning it, and the disk filled with broken ``watchdog-*`` log
directories until the host went OOM.

The fix lives here. Every checkpoint now carries a small
``metadata`` blob describing the architecture (``hidden_size``,
``comm_value_dim``, ``gru_layers``) and the environment shape
(``obs_dim``, ``n_heads``, ``team_size``). Before any
``load_state_dict`` we compare those numbers against what the *current*
trainer/runner is built to handle. If they disagree we raise
:class:`CheckpointIncompatibleError` -- a *non-restartable* error that
the API watchdog knows to never retry.

Backwards compat: old checkpoints without ``metadata`` get a single
warning ("checkpoint lacks metadata, attempting load anyway"); the
caller is expected to fall back to a guarded ``torch.load`` /
``load_state_dict`` and convert any ``RuntimeError("size mismatch")``
into :class:`CheckpointIncompatibleError` so the watchdog short-circuit
still triggers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "CheckpointIncompatibleError",
    "build_compat_metadata",
    "check_compat",
    "load_blob_with_compat",
    "load_sidecar_metadata",
    "write_sidecar_metadata",
]

_LOG = logging.getLogger("kivski_agents.persistence.checkpoint_compat")

# Public schema version for the ``metadata`` blob. Bump when fields change
# in a breaking way so the loader can refuse to interpret an unknown
# layout instead of guessing.
COMPAT_SCHEMA_VERSION = 1

# Fields that *must* match between a checkpoint's saved arch and the
# currently-built model. ``comm_signature_dim`` is implied by
# ``comm_value_dim`` in the default factory (they're symmetric) so we
# don't strictly need it; we still record it for diagnostics.
_REQUIRED_ARCH_KEYS: tuple[str, ...] = (
    "hidden_size",
    "comm_value_dim",
    "gru_layers",
)
_REQUIRED_ENV_KEYS: tuple[str, ...] = (
    "obs_dim",
    "n_heads",
    "team_size",
)


class CheckpointIncompatibleError(Exception):
    """Raised when a checkpoint's saved arch/env shape can't be loaded.

    This is intentionally NOT a transient error. The API watchdog
    catches it via the per-job ``CRASH_REASON.txt`` artefact and refuses
    to auto-resume from the offending checkpoint -- otherwise a
    persistently-incompatible ``best.pt`` would cause an infinite
    restart cascade.
    """

    # A short machine-friendly tag used by CRASH_REASON.txt parsing.
    category: str = "incompatible_checkpoint"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_compat_metadata(
    *,
    model_arch: dict[str, Any],
    env_shape: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the metadata blob saved alongside every checkpoint.

    ``model_arch`` and ``env_shape`` are sliced to the required keys so
    callers can pass a broader dict (e.g. the full ``_model_init_dict``
    or a ``vec_env`` summary) without leaking fields the validator
    doesn't care about. Extra keys land in the top-level ``extra``
    bucket -- they aren't validated but are preserved for diagnostics.
    """
    arch = {k: _as_int_or_none(model_arch.get(k)) for k in _REQUIRED_ARCH_KEYS}
    env = {k: _as_int_or_none(env_shape.get(k)) for k in _REQUIRED_ENV_KEYS}
    meta: dict[str, Any] = {
        "schema_version": int(COMPAT_SCHEMA_VERSION),
        "model_arch": arch,
        "env_shape": env,
        "kivski_version": "0.4.0",
        "timestamp": float(time.time()),
    }
    if extra:
        meta["extra"] = dict(extra)
    return meta


def _as_int_or_none(value: Any) -> int | None:
    """Coerce ``value`` to an int when possible; ``None`` otherwise.

    Used so a partial metadata blob (e.g. an old checkpoint missing a
    field) doesn't crash the validator -- it just won't compare that
    field, falling back to a soft load.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def check_compat(
    saved: dict[str, Any],
    expected: dict[str, Any],
    *,
    source: str = "<checkpoint>",
) -> None:
    """Compare a saved metadata blob against the currently-expected shape.

    Raises :class:`CheckpointIncompatibleError` listing every mismatch.
    Missing fields on *either* side are skipped (so a partial blob still
    validates whatever it can). The ``source`` string is embedded in the
    error message so the user knows which checkpoint failed.
    """
    if not saved:
        # Caller will have already emitted the "no metadata" warning.
        return
    saved_arch = dict(saved.get("model_arch", {}) or {})
    saved_env = dict(saved.get("env_shape", {}) or {})
    expected_arch = dict(expected.get("model_arch", {}) or {})
    expected_env = dict(expected.get("env_shape", {}) or {})

    mismatches: list[str] = []
    for key in _REQUIRED_ARCH_KEYS:
        a = _as_int_or_none(saved_arch.get(key))
        b = _as_int_or_none(expected_arch.get(key))
        if a is None or b is None:
            continue
        if a != b:
            mismatches.append(f"model.{key}: ckpt={a} != current={b}")
    for key in _REQUIRED_ENV_KEYS:
        a = _as_int_or_none(saved_env.get(key))
        b = _as_int_or_none(expected_env.get(key))
        if a is None or b is None:
            continue
        if a != b:
            mismatches.append(f"env.{key}: ckpt={a} != current={b}")

    if mismatches:
        msg = (
            f"Checkpoint {source!s} is incompatible with current config:\n  "
            + "\n  ".join(mismatches)
            + "\nDelete this checkpoint or start a new run."
        )
        raise CheckpointIncompatibleError(msg)


def load_blob_with_compat(
    ckpt_path: str | Path,
    expected: dict[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint blob and validate its metadata against ``expected``.

    Returns the raw blob dict (whatever ``torch.save`` wrote) so the
    caller can inspect ``state_dict`` / ``optimizer`` / ``model_init``
    etc. without re-loading.

    Behaviour matrix:

    * Blob has ``metadata`` -> validate via :func:`check_compat`. On
      mismatch raise :class:`CheckpointIncompatibleError`.
    * Blob has no ``metadata`` -> log a soft warning and return the blob
      anyway. Caller must still handle the eventual ``RuntimeError`` from
      ``load_state_dict`` and translate it into
      :class:`CheckpointIncompatibleError`.
    """
    path = Path(ckpt_path)
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(blob, dict):
        # Lone state dict -- can't validate. Wrap into a dict-shape so
        # callers see a uniform return type.
        _LOG.warning(
            "checkpoint %s is a bare state_dict (no metadata); "
            "attempting load anyway",
            path,
        )
        return {"model": blob, "metadata": {}}
    meta = blob.get("metadata") or {}
    if not meta or "model_arch" not in meta:
        _LOG.warning(
            "checkpoint %s lacks compat metadata; attempting load anyway "
            "(any size mismatch will surface as CheckpointIncompatibleError)",
            path,
        )
        return blob
    check_compat(meta, expected, source=path.name)
    return blob


# ---------------------------------------------------------------------------
# Sidecar JSON helpers
# ---------------------------------------------------------------------------


def _sidecar_path_for(path: Path) -> Path:
    """Return the ``<name>.pt.json`` sidecar path next to ``path``."""
    return path.with_suffix(path.suffix + ".json") if path.suffix else path.with_suffix(".json")


def write_sidecar_metadata(ckpt_path: str | Path, metadata: dict[str, Any]) -> Path:
    """Persist ``metadata`` next to ``ckpt_path`` as JSON (``.pt.json``).

    Used as a torch-free read path -- ops people can `cat` this file to
    sanity-check a checkpoint without spinning up Python.
    """
    p = Path(ckpt_path)
    sidecar = _sidecar_path_for(p)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True, default=str)
    return sidecar


def load_sidecar_metadata(ckpt_path: str | Path) -> dict[str, Any] | None:
    """Read the torch-free JSON sidecar if present, else ``None``."""
    sidecar = _sidecar_path_for(Path(ckpt_path))
    if not sidecar.is_file():
        return None
    try:
        with sidecar.open("r", encoding="utf-8") as fh:
            return dict(json.load(fh))
    except (OSError, json.JSONDecodeError):
        return None
