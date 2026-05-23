"""Tests for the checkpoint compatibility-check pipeline.

These guard the v0.5.1 fix for the "OOM-from-respawn-cascade" bug:

* :class:`MAPPOTrainer.save` must write ``model_arch`` / ``env_shape``
  into the checkpoint blob and a sidecar JSON.
* :class:`MAPPOTrainer.load` must raise
  :class:`CheckpointIncompatibleError` when the saved arch differs from
  the currently-built model -- *before* the underlying ``size mismatch``
  ``RuntimeError`` would normally fire from PyTorch.
* Backwards compat: a checkpoint without metadata still attempts the
  load (with a warning) and only fails if the state_dict actually
  doesn't match.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from kivski_agents.factory import build_model, build_trainer, default_action_dims, infer_joint_obs_dim
from kivski_agents.mappo import MAPPOTrainer
from kivski_agents.persistence import (
    CheckpointIncompatibleError,
    build_compat_metadata,
    check_compat,
    load_blob_with_compat,
)
from kivski_sim.config import KivskiConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_trainer(
    *,
    hidden_size: int = 32,
    team_size: int = 3,
    obs_dim: int = 24,
) -> MAPPOTrainer:
    """Build the smallest viable MAPPOTrainer for compat tests."""
    cfg = KivskiConfig()
    # Force a known-small hidden size so we can vary it per test.
    cfg = cfg.model_copy(update={"ml": cfg.ml.model_copy(update={"hidden_size": hidden_size})})
    cfg = cfg.model_copy(update={"simulation": cfg.simulation.model_copy(update={"team_size": team_size})})
    action_dims = default_action_dims(team_size)
    joint_obs_dim = infer_joint_obs_dim(obs_dim, team_size)
    model = build_model(
        cfg=cfg,
        obs_dim=obs_dim,
        joint_obs_dim=joint_obs_dim,
        action_dims=action_dims,
        device="cpu",
    )
    return build_trainer(model, cfg, device="cpu")


# ---------------------------------------------------------------------------
# Save: blob + sidecar contain compat metadata
# ---------------------------------------------------------------------------


def test_save_includes_metadata(tmp_path: Path) -> None:
    """save() must embed model_arch + env_shape in both blob and sidecar JSON."""
    trainer = _make_trainer(hidden_size=32, team_size=3, obs_dim=24)
    path = tmp_path / "ckpt.pt"
    trainer.save(
        path,
        metadata={"episode": 42, "run_name": "test"},
        env_shape={"obs_dim": 24, "n_heads": 5, "team_size": 3},
    )

    # 1) The torch blob carries metadata with the required fields.
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob["metadata"]
    assert meta["schema_version"] == 1
    assert meta["model_arch"]["hidden_size"] == 32
    assert "comm_value_dim" in meta["model_arch"]
    assert "gru_layers" in meta["model_arch"]
    assert meta["env_shape"] == {"obs_dim": 24, "n_heads": 5, "team_size": 3}
    # User-supplied keys are preserved.
    assert meta["episode"] == 42
    assert meta["run_name"] == "test"

    # 2) The sidecar JSON is torch-free and has the same fields.
    sidecar = path.with_suffix(path.suffix + ".json")
    assert sidecar.is_file()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["model_arch"]["hidden_size"] == 32
    assert data["env_shape"]["team_size"] == 3


# ---------------------------------------------------------------------------
# Load: incompatible arch raises CheckpointIncompatibleError
# ---------------------------------------------------------------------------


def test_load_raises_on_arch_mismatch(tmp_path: Path) -> None:
    """A checkpoint with hidden=32 must NOT load into a hidden=64 trainer."""
    small = _make_trainer(hidden_size=32, team_size=3, obs_dim=24)
    path = tmp_path / "ckpt.pt"
    small.save(path, env_shape={"obs_dim": 24, "n_heads": 5, "team_size": 3})

    # New trainer with a different hidden_size -> must refuse.
    big = _make_trainer(hidden_size=64, team_size=3, obs_dim=24)
    with pytest.raises(CheckpointIncompatibleError) as excinfo:
        big.load(path, env_shape={"obs_dim": 24, "n_heads": 5, "team_size": 3})
    msg = str(excinfo.value)
    assert "hidden_size" in msg
    assert "32" in msg and "64" in msg


def test_load_raises_on_env_mismatch(tmp_path: Path) -> None:
    """team_size or obs_dim mismatches must also raise."""
    a = _make_trainer(hidden_size=32, team_size=3, obs_dim=24)
    path = tmp_path / "ckpt.pt"
    a.save(path, env_shape={"obs_dim": 24, "n_heads": 5, "team_size": 3})

    # Same model arch but different team_size: must refuse before
    # state_dict load.
    b = _make_trainer(hidden_size=32, team_size=3, obs_dim=24)
    with pytest.raises(CheckpointIncompatibleError):
        b.load(path, env_shape={"obs_dim": 24, "n_heads": 5, "team_size": 5})


# ---------------------------------------------------------------------------
# Load: matching arch round-trips fine
# ---------------------------------------------------------------------------


def test_load_succeeds_on_match(tmp_path: Path) -> None:
    """Save then load with the same shape -> no error, weights round-trip."""
    a = _make_trainer(hidden_size=32, team_size=3, obs_dim=24)
    path = tmp_path / "ckpt.pt"
    a.save(path, env_shape={"obs_dim": 24, "n_heads": 5, "team_size": 3})

    b = _make_trainer(hidden_size=32, team_size=3, obs_dim=24)
    meta = b.load(path, env_shape={"obs_dim": 24, "n_heads": 5, "team_size": 3})
    assert isinstance(meta, dict)
    assert meta["model_arch"]["hidden_size"] == 32

    # And the weights actually copied over.
    for k, va in a.model.state_dict().items():
        vb = b.model.state_dict()[k]
        assert torch.allclose(va, vb)


# ---------------------------------------------------------------------------
# Load: backwards-compat for old checkpoints without metadata
# ---------------------------------------------------------------------------


def test_load_backwards_compat_no_metadata(tmp_path: Path) -> None:
    """A pre-v0.5.1 checkpoint (no metadata) still loads when shapes match.

    The save path is bypassed -- we write a raw torch blob without the
    ``metadata`` key to simulate an old artefact. Load must succeed
    silently (with a warning) when the state_dict happens to match, and
    must raise CheckpointIncompatibleError translating the size-mismatch
    RuntimeError when it doesn't.
    """
    src = _make_trainer(hidden_size=32, team_size=3, obs_dim=24)
    path = tmp_path / "legacy.pt"
    # Hand-write a legacy blob with NO metadata key.
    torch.save(
        {
            "model": src.model.state_dict(),
            "optimizer": src.optimizer.state_dict(),
            # NB: no metadata, no model_init, no cfg -- realistic old format.
        },
        path,
    )

    # Same shape -> load works (warning only).
    dst_same = _make_trainer(hidden_size=32, team_size=3, obs_dim=24)
    dst_same.load(path)

    # Different shape -> size mismatch is translated.
    dst_other = _make_trainer(hidden_size=64, team_size=3, obs_dim=24)
    with pytest.raises(CheckpointIncompatibleError) as excinfo:
        dst_other.load(path)
    assert "state_dict" in str(excinfo.value) or "size mismatch" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# load_blob_with_compat helper standalone
# ---------------------------------------------------------------------------


def test_load_blob_with_compat_raises(tmp_path: Path) -> None:
    """The standalone helper used by PolicyBundle.from_checkpoint."""
    src = _make_trainer(hidden_size=32, team_size=3, obs_dim=24)
    path = tmp_path / "ckpt.pt"
    src.save(path, env_shape={"obs_dim": 24, "n_heads": 5, "team_size": 3})

    expected = {
        "model_arch": {
            "hidden_size": 64,
            "comm_value_dim": 16,
            "gru_layers": 1,
        },
        "env_shape": {"obs_dim": 24, "n_heads": 5, "team_size": 3},
    }
    with pytest.raises(CheckpointIncompatibleError) as excinfo:
        load_blob_with_compat(path, expected)
    assert "hidden_size" in str(excinfo.value)


def test_check_compat_skips_missing_fields() -> None:
    """When a field is absent on either side it's skipped, not flagged."""
    saved = {"model_arch": {"hidden_size": 64}}
    expected = {"model_arch": {"comm_value_dim": 16, "hidden_size": 64}}
    # No raise -> hidden_size matches, comm_value_dim is missing on saved.
    check_compat(saved, expected)


def test_build_compat_metadata_coerces_int_or_none() -> None:
    """Stringy inputs are coerced to int; unparseable values become None."""
    meta = build_compat_metadata(
        model_arch={"hidden_size": "256", "comm_value_dim": None, "gru_layers": 2},
        env_shape={"obs_dim": 48, "n_heads": "5", "team_size": "five"},
    )
    assert meta["model_arch"]["hidden_size"] == 256
    assert meta["model_arch"]["comm_value_dim"] is None
    assert meta["model_arch"]["gru_layers"] == 2
    assert meta["env_shape"]["obs_dim"] == 48
    assert meta["env_shape"]["n_heads"] == 5
    assert meta["env_shape"]["team_size"] is None  # "five" can't be int()'d
