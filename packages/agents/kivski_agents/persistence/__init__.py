"""Persistence utilities shared between trainer, API, and runners.

This package collects everything related to writing/reading checkpoints
in a way that survives architecture changes, including the
:class:`CheckpointIncompatibleError` raised when a saved checkpoint
doesn't match the currently-built model/env.
"""

from __future__ import annotations

from kivski_agents.persistence.checkpoint_compat import (
    CheckpointIncompatibleError,
    build_compat_metadata,
    check_compat,
    load_blob_with_compat,
    load_sidecar_metadata,
    write_sidecar_metadata,
)

__all__ = [
    "CheckpointIncompatibleError",
    "build_compat_metadata",
    "check_compat",
    "load_blob_with_compat",
    "load_sidecar_metadata",
    "write_sidecar_metadata",
]
