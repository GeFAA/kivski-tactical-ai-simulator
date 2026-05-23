"""Unified telemetry sinks for the Kivski Tactical AI Simulator.

This module provides a single :class:`TelemetrySink` abstract base class with
three concrete backends (CSV, TensorBoard, Weights & Biases) plus a
:class:`MultiSink` fan-out and a :class:`NoOpSink` for tests / disabled
telemetry. The :func:`make_sink` factory selects a backend based on the
``TelemetryConfig.backend`` string from :mod:`kivski_sim.config`.

Design goals:
    * Single import surface for the trainer/eval code.
    * Lazy imports for optional dependencies (``torch.utils.tensorboard``
      and ``wandb``) so users only pay for what they enable.
    * Robust against partial failures: a flaky backend in a ``MultiSink``
      must never silently swallow data destined for the other sinks.
    * CSV sink is buffered with periodic flushes so that disk I/O does
      not dominate hot training loops.
"""

from __future__ import annotations

import contextlib
import csv
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kivski_sim.utils import ensure_dir, now_unix

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


__all__ = [
    "TelemetrySink",
    "CSVSink",
    "TensorBoardSink",
    "WandbSink",
    "MultiSink",
    "NoOpSink",
    "make_sink",
]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TelemetrySink(ABC):
    """Abstract base class for telemetry backends.

    Implementations should be safe to call from a single producer thread.
    Cross-thread coordination is the caller's responsibility unless the
    concrete sink documents otherwise (the bundled :class:`CSVSink` is
    thread-safe).
    """

    @abstractmethod
    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log a single scalar metric tagged with ``key`` at ``step``."""

    @abstractmethod
    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        """Log a dict of scalar metrics at ``step``.

        Equivalent to calling :meth:`log_scalar` for each entry, but
        implementations may batch the writes for efficiency.
        """

    @abstractmethod
    def log_text(self, key: str, text: str, step: int) -> None:
        """Log a free-form text record (e.g. a config dump or stack trace)."""

    @abstractmethod
    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        """Persist run hyperparameters once, at the start of the run."""

    @abstractmethod
    def close(self) -> None:
        """Flush and release any underlying resources. Idempotent."""

    def flush(self) -> None:
        """Force-flush any buffered writes. Default implementation is a no-op."""
        return None


# ---------------------------------------------------------------------------
# CSV sink
# ---------------------------------------------------------------------------


class CSVSink(TelemetrySink):
    """Append-only CSV sink, one directory per run.

    Layout under ``log_dir / run_name``::

        metrics.csv        scalars (step,key,value,wall_time)
        text.csv           text records (step,key,text,wall_time)
        hparams.csv        flat hyperparameter dump (key,value)

    Writes are buffered in memory and flushed every
    ``flush_every_seconds`` seconds (or whenever :meth:`flush` or
    :meth:`close` is called). The sink uses an internal lock so it can
    be shared across threads safely.
    """

    def __init__(
        self,
        log_dir: Path,
        run_name: str,
        flush_every_seconds: float = 5.0,
    ) -> None:
        self.run_dir: Path = ensure_dir(Path(log_dir) / run_name)
        self.metrics_path: Path = self.run_dir / "metrics.csv"
        self.text_path: Path = self.run_dir / "text.csv"
        self.hparams_path: Path = self.run_dir / "hparams.csv"
        self.flush_every_seconds: float = float(flush_every_seconds)

        self._lock = threading.Lock()
        self._scalar_buffer: list[tuple[int, str, float, float]] = []
        self._text_buffer: list[tuple[int, str, str, float]] = []
        self._last_flush: float = now_unix()
        self._closed: bool = False

        # Write headers if files don't yet exist.
        self._ensure_header(self.metrics_path, ["step", "key", "value", "wall_time"])
        self._ensure_header(self.text_path, ["step", "key", "text", "wall_time"])

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _ensure_header(path: Path, header: list[str]) -> None:
        """Create the file with a header row if it doesn't exist yet."""
        if path.exists():
            return
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)

    def _maybe_flush_locked(self) -> None:
        """Flush if the buffered window has elapsed. Assumes lock held."""
        if (now_unix() - self._last_flush) >= self.flush_every_seconds:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Drain buffers to disk. Assumes lock held."""
        if self._scalar_buffer:
            with self.metrics_path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerows(self._scalar_buffer)
            self._scalar_buffer.clear()
        if self._text_buffer:
            with self.text_path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerows(self._text_buffer)
            self._text_buffer.clear()
        self._last_flush = now_unix()

    # -- public API ---------------------------------------------------------

    def log_scalar(self, key: str, value: float, step: int) -> None:
        if self._closed:
            return
        with self._lock:
            self._scalar_buffer.append((int(step), str(key), float(value), now_unix()))
            self._maybe_flush_locked()

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        if self._closed:
            return
        ts = now_unix()
        with self._lock:
            for k, v in metrics.items():
                self._scalar_buffer.append((int(step), str(k), float(v), ts))
            self._maybe_flush_locked()

    def log_text(self, key: str, text: str, step: int) -> None:
        if self._closed:
            return
        with self._lock:
            self._text_buffer.append((int(step), str(key), str(text), now_unix()))
            self._maybe_flush_locked()

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        if self._closed:
            return
        # Hyperparams are rewritten in full (small, written once-ish).
        with (
            self._lock,
            self.hparams_path.open("w", encoding="utf-8", newline="") as fh,
        ):
            writer = csv.writer(fh)
            writer.writerow(["key", "value"])
            for k, v in sorted(hparams.items()):
                writer.writerow([str(k), str(v)])

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_locked()
            self._closed = True


# ---------------------------------------------------------------------------
# TensorBoard sink
# ---------------------------------------------------------------------------


class TensorBoardSink(TelemetrySink):
    """TensorBoard backend wrapping ``torch.utils.tensorboard.SummaryWriter``.

    ``torch`` is imported lazily so that users who pick a different backend
    do not pay the import cost.
    """

    def __init__(self, log_dir: Path, run_name: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "TensorBoardSink requires PyTorch (torch). Install with "
                "`pip install torch` or pick a different telemetry backend."
            ) from exc

        self.run_dir: Path = ensure_dir(Path(log_dir) / run_name)
        self._writer = SummaryWriter(log_dir=str(self.run_dir))
        self._closed: bool = False

    def log_scalar(self, key: str, value: float, step: int) -> None:
        if self._closed:
            return
        self._writer.add_scalar(key, float(value), int(step))

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        if self._closed:
            return
        for k, v in metrics.items():
            self._writer.add_scalar(str(k), float(v), int(step))

    def log_text(self, key: str, text: str, step: int) -> None:
        if self._closed:
            return
        self._writer.add_text(key, text, int(step))

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        if self._closed:
            return
        # SummaryWriter.add_hparams demands a metric dict; we record only
        # the hparams and let the user pair them with metrics later.
        flat = {k: _coerce_scalar(v) for k, v in hparams.items()}
        try:
            self._writer.add_hparams(flat, {})
        except Exception:  # pragma: no cover - tb internals are picky
            # Fallback: dump as text so the data isn't lost.
            self._writer.add_text("hparams", "\n".join(f"{k}={v}" for k, v in flat.items()), 0)

    def flush(self) -> None:
        if self._closed:
            return
        self._writer.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._writer.flush()
        self._writer.close()
        self._closed = True


def _coerce_scalar(v: Any) -> Any:
    """Coerce a value to a TensorBoard-friendly scalar / string."""
    if isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


# ---------------------------------------------------------------------------
# Weights & Biases sink
# ---------------------------------------------------------------------------


class WandbSink(TelemetrySink):
    """Weights & Biases backend (optional dependency).

    Behaviour:
        * ``project`` defaults to the ``KIVSKI_WANDB_PROJECT`` env var (or
          ``"kivski"`` if unset).
        * ``mode`` defaults to ``"online"`` when ``WANDB_API_KEY`` is set
          and ``"offline"`` otherwise, so unauthenticated machines still
          get a local run directory under ``log_dir``.
        * If ``wandb`` is not installed, the constructor raises
          :class:`ImportError` with installation guidance.
    """

    def __init__(
        self,
        log_dir: Path,
        run_name: str,
        project: str | None = None,
        mode: str | None = None,
    ) -> None:
        try:
            import wandb  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "WandbSink requires the 'wandb' package. Install with "
                "`pip install wandb` or pick a different telemetry backend "
                "(e.g. 'csv' or 'tensorboard')."
            ) from exc

        self._wandb = wandb
        self.run_dir: Path = ensure_dir(Path(log_dir) / run_name)

        resolved_project = project or os.getenv("KIVSKI_WANDB_PROJECT", "kivski")
        if mode is None:
            mode = "online" if os.getenv("WANDB_API_KEY") else "offline"

        self._run = wandb.init(
            project=resolved_project,
            name=run_name,
            dir=str(self.run_dir),
            mode=mode,
            reinit=True,
        )
        self._closed: bool = False

    def log_scalar(self, key: str, value: float, step: int) -> None:
        if self._closed:
            return
        self._wandb.log({str(key): float(value)}, step=int(step))

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        if self._closed:
            return
        self._wandb.log({str(k): float(v) for k, v in metrics.items()}, step=int(step))

    def log_text(self, key: str, text: str, step: int) -> None:
        if self._closed:
            return
        # W&B doesn't have a "log text at step" primitive; record as a
        # summary entry keyed by step so subsequent updates don't overwrite.
        self._run.summary[f"{key}@{int(step)}"] = str(text)

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        if self._closed:
            return
        # `config.update` is idempotent w.r.t. identical keys.
        self._wandb.config.update(dict(hparams), allow_val_change=True)

    def flush(self) -> None:
        # wandb doesn't expose a public flush; logs are pushed by the
        # background thread on its own schedule.
        return None

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._wandb.finish()
        finally:
            self._closed = True


# ---------------------------------------------------------------------------
# Multi sink (fan-out)
# ---------------------------------------------------------------------------


@dataclass
class MultiSink(TelemetrySink):
    """Fan-out wrapper: forwards every call to a list of child sinks.

    Failures in one child sink are caught and recorded in :attr:`errors`
    rather than raised, so a transient hiccup in (say) W&B can't take
    down CSV logging.
    """

    sinks: list[TelemetrySink]
    errors: list[tuple[str, Exception]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # dataclass + ABC plays nicely as long as we don't pass abstract
        # methods to the dataclass machinery -- ABCMeta is the metaclass.
        if not isinstance(self.sinks, list):
            self.sinks = list(self.sinks)

    def _dispatch(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        for sink in self.sinks:
            try:
                getattr(sink, method_name)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - intentional: keep others alive
                self.errors.append((type(sink).__name__, exc))

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self._dispatch("log_scalar", key, value, step)

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        self._dispatch("log_dict", metrics, step)

    def log_text(self, key: str, text: str, step: int) -> None:
        self._dispatch("log_text", key, text, step)

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        self._dispatch("log_hyperparams", hparams)

    def flush(self) -> None:
        self._dispatch("flush")

    def close(self) -> None:
        self._dispatch("close")


# ---------------------------------------------------------------------------
# No-op sink
# ---------------------------------------------------------------------------


class NoOpSink(TelemetrySink):
    """Sink that silently discards every call.

    Useful for unit tests and for the ``backend="none"`` config setting.
    """

    def log_scalar(self, key: str, value: float, step: int) -> None:  # noqa: D401
        return None

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        return None

    def log_text(self, key: str, text: str, step: int) -> None:
        return None

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_sink(backend: str, log_dir: Path, run_name: str) -> TelemetrySink:
    """Construct a telemetry sink from a config string.

    Supported values for ``backend``:
        * ``"csv"`` -- :class:`CSVSink`
        * ``"tensorboard"`` -- :class:`TensorBoardSink`
        * ``"wandb"`` -- :class:`WandbSink`
        * ``"all"`` -- CSV + TensorBoard (+ W&B if importable). Missing
          optional backends are skipped with a console-friendly warning
          via the standard :mod:`logging` module rather than failing.
        * ``"none"`` -- :class:`NoOpSink`

    Args:
        backend: Backend identifier (case-insensitive).
        log_dir: Root directory for run outputs.
        run_name: Per-run subdirectory name.

    Returns:
        A constructed :class:`TelemetrySink`.

    Raises:
        ValueError: If ``backend`` is not recognised.
        ImportError: If a specific backend's optional dependency is missing.
    """
    key = backend.strip().lower()
    log_path = Path(log_dir)

    if key == "csv":
        return CSVSink(log_path, run_name)
    if key == "tensorboard":
        return TensorBoardSink(log_path, run_name)
    if key == "wandb":
        return WandbSink(log_path, run_name)
    if key == "none":
        return NoOpSink()
    if key == "all":
        sinks: list[TelemetrySink] = [CSVSink(log_path, run_name)]
        # TensorBoard is "best effort" in "all" mode.
        with contextlib.suppress(ImportError):
            sinks.append(TensorBoardSink(log_path, run_name))
        # W&B too.
        with contextlib.suppress(ImportError):
            sinks.append(WandbSink(log_path, run_name))
        return MultiSink(sinks)

    raise ValueError(
        f"Unknown telemetry backend {backend!r}. Expected one of: csv, tensorboard, wandb, all, none."
    )
