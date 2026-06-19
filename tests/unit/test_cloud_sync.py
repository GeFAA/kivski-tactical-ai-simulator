"""Smoke tests for HFHubSyncer -- no real HF network calls.

These guard the v0.5/v0.6 best-effort cloud-sync contract: a missing token,
missing repo, missing huggingface_hub dep, or any HF API error must surface
into ``last_push_ok`` / ``last_error`` but never crash the trainer thread.
"""

from __future__ import annotations

import time
from pathlib import Path

from kivski_agents.cloud_sync import HFHubSyncer, maybe_build_syncer_from_env


def test_syncer_disabled_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("KIVSKI_HF_REPO", raising=False)
    assert maybe_build_syncer_from_env() is None


def test_syncer_disabled_with_only_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake_hf_xxx")
    monkeypatch.delenv("KIVSKI_HF_REPO", raising=False)
    assert maybe_build_syncer_from_env() is None


def test_syncer_disabled_with_only_repo(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("KIVSKI_HF_REPO", "user/repo")
    assert maybe_build_syncer_from_env() is None


def test_syncer_enabled_with_both(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake_hf_xxx")
    monkeypatch.setenv("KIVSKI_HF_REPO", "user/repo")
    s = maybe_build_syncer_from_env()
    assert s is not None
    assert s.repo_id == "user/repo"
    assert s.enabled is True
    s.shutdown(timeout=1)


def test_syncer_handles_hf_api_failure_gracefully(tmp_path: Path):
    """If huggingface_hub is missing OR the HF API raises (network/auth/etc),
    the syncer must record the error and keep the trainer alive."""
    s = HFHubSyncer("user/repo", hf_token="fake_token_xxx")
    fake_ckpt = tmp_path / "fake.pt"
    fake_ckpt.write_bytes(b"x")
    s.push_checkpoint(fake_ckpt)

    # Wait briefly for the background upload attempt to fail.
    for _ in range(20):
        time.sleep(0.25)
        if s.last_push_ok is False:
            break

    # Drain pending work so we don't race the assertion.
    s.shutdown(timeout=5)
    assert s.last_push_ok is False, "expected HF failure to be recorded"
    assert s.last_error is not None


def test_syncer_shutdown_is_idempotent():
    """Double-shutdown must not raise -- the API server may call it twice on
    graceful stop + atexit."""
    s = HFHubSyncer("user/repo", hf_token="fake")
    s.shutdown(timeout=1)
    s.shutdown(timeout=1)  # second call should be a no-op


def test_disabled_syncer_is_no_op(tmp_path: Path):
    """A syncer constructed with enabled=False must silently ignore push_*."""
    s = HFHubSyncer("user/repo", hf_token=None, enabled=False)
    assert s.enabled is False
    # Should not raise even though the file doesn't exist and there's no token.
    s.push_checkpoint(tmp_path / "missing.pt")
    s.push_metrics(tmp_path / "missing.csv", "run-x")
    s.push_clock(tmp_path / "missing.json")
    s.shutdown(timeout=1)
