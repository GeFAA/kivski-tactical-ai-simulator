"""Hugging Face Hub sync for cloud-trained checkpoints + metrics.

The :class:`HFHubSyncer` runs uploads on a single background thread so the
training loop is never blocked by network I/O or HF rate limits. Failures are
captured into ``last_error`` and ``last_push_ok`` but never raised back to the
caller -- cloud sync is strictly best-effort.

Typical use::

    syncer = maybe_build_syncer_from_env()
    if syncer:
        syncer.push_checkpoint(ckpt_path, metadata_path)
        ...
        syncer.shutdown()

Env vars:
    HF_TOKEN          -- Hugging Face access token (write scope).
    KIVSKI_HF_REPO    -- target ``user/repo`` for the private model repo.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

__all__ = ["HFHubSyncer", "maybe_build_syncer_from_env"]

logger = logging.getLogger(__name__)


class HFHubSyncer:
    """Best-effort background sync of checkpoints/metrics to a private HF repo."""

    def __init__(
        self,
        repo_id: str,
        hf_token: str | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.repo_id: str = str(repo_id)
        self._token: str | None = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
        self._enabled: bool = bool(enabled)
        self._last_push_ok: bool = True
        self._last_error: str | None = None
        self._lock = threading.Lock()
        # Single-worker pool serialises uploads so we never trip HF rate limits.
        self._executor: ThreadPoolExecutor | None = None
        self._pending: list[Future] = []

        if self._enabled and not self._token:
            logger.warning(
                "HFHubSyncer: no HF token provided (constructor arg or HF_TOKEN env). Disabling cloud sync."
            )
            self._enabled = False

        if self._enabled:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-sync")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_push_ok(self) -> bool:
        return self._last_push_ok

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def enabled(self) -> bool:
        return self._enabled

    def push_checkpoint(self, ckpt_path: Path, metadata_path: Path | None = None) -> None:
        """Upload a checkpoint .pt (and optional sidecar) to ``checkpoints/``."""
        if not self._enabled:
            return
        ckpt = Path(ckpt_path)
        meta = Path(metadata_path) if metadata_path is not None else None
        self._submit(self._upload_checkpoint, ckpt, meta)

    def push_metrics(self, metrics_csv: Path, run_name: str) -> None:
        """Upload the metrics CSV to ``runs/{run_name}/metrics.csv``."""
        if not self._enabled:
            return
        csv = Path(metrics_csv)
        self._submit(self._upload_metrics, csv, str(run_name))

    def push_clock(self, clock_json: Path) -> None:
        """Upload the aggregated training clock JSON to ``clock.json``."""
        if not self._enabled:
            return
        clock = Path(clock_json)
        self._submit(self._upload_clock, clock)

    def shutdown(self, timeout: float = 30.0) -> None:
        """Wait for pending uploads to finish, then close the executor."""
        if self._executor is None:
            return
        executor = self._executor
        pending = list(self._pending)
        self._executor = None
        self._pending = []
        for fut in pending:
            try:
                fut.result(timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                self._record_error(f"shutdown wait failed: {exc!r}")
        try:
            executor.shutdown(wait=True, cancel_futures=False)
        except Exception as exc:  # noqa: BLE001
            self._record_error(f"executor shutdown failed: {exc!r}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _submit(self, fn, *args) -> None:
        if self._executor is None:
            return
        try:
            fut = self._executor.submit(self._wrap, fn, *args)
        except RuntimeError as exc:
            # Executor already shut down.
            self._record_error(f"submit failed: {exc!r}")
            return
        self._pending.append(fut)
        # Garbage-collect completed futures so the list doesn't grow unbounded.
        self._pending = [f for f in self._pending if not f.done()]

    def _wrap(self, fn, *args) -> None:
        try:
            fn(*args)
            with self._lock:
                self._last_push_ok = True
                self._last_error = None
        except Exception as exc:  # noqa: BLE001 - never let sync crash training
            self._record_error(f"{fn.__name__} failed: {exc!r}")

    def _record_error(self, msg: str) -> None:
        with self._lock:
            self._last_push_ok = False
            self._last_error = msg
        logger.warning("HFHubSyncer: %s", msg)

    def _hf_api(self):
        """Lazy-import the HF SDK so a missing dep doesn't break the trainer."""
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed. Install with"
                " `pip install 'kivski[cloud]'` or `pip install huggingface_hub`."
            ) from exc
        return HfApi(token=self._token)

    def _upload_checkpoint(self, ckpt: Path, meta: Path | None) -> None:
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        api = self._hf_api()
        api.upload_file(
            path_or_fileobj=str(ckpt),
            path_in_repo=f"checkpoints/{ckpt.name}",
            repo_id=self.repo_id,
            repo_type="model",
            token=self._token,
        )
        if meta is not None and meta.is_file():
            api.upload_file(
                path_or_fileobj=str(meta),
                path_in_repo=f"checkpoints/{meta.name}",
                repo_id=self.repo_id,
                repo_type="model",
                token=self._token,
            )

    def _upload_metrics(self, csv: Path, run_name: str) -> None:
        if not csv.is_file():
            raise FileNotFoundError(f"metrics csv not found: {csv}")
        api = self._hf_api()
        api.upload_file(
            path_or_fileobj=str(csv),
            path_in_repo=f"runs/{run_name}/metrics.csv",
            repo_id=self.repo_id,
            repo_type="model",
            token=self._token,
        )

    def _upload_clock(self, clock: Path) -> None:
        if not clock.is_file():
            raise FileNotFoundError(f"clock json not found: {clock}")
        api = self._hf_api()
        api.upload_file(
            path_or_fileobj=str(clock),
            path_in_repo="clock.json",
            repo_id=self.repo_id,
            repo_type="model",
            token=self._token,
        )


def maybe_build_syncer_from_env() -> HFHubSyncer | None:
    """Construct an :class:`HFHubSyncer` from env vars, or return ``None``.

    Requires both ``HF_TOKEN`` and ``KIVSKI_HF_REPO`` to be set.
    """
    token = os.environ.get("HF_TOKEN")
    repo = os.environ.get("KIVSKI_HF_REPO")
    if not token or not repo:
        return None
    return HFHubSyncer(repo_id=repo, hf_token=token, enabled=True)
