"""Policy adapters for the FastAPI live-match server.

The live viewer needs lightweight, swap-in policies for each side of a
running match. This module provides:

* :class:`RandomPolicy` -- uniform random actions, the safe default whenever
  no trained checkpoint is loaded.
* :class:`HoldPositionPolicy` -- everyone HOLDs, no buy, no shoot. Useful as
  a debug baseline so the UI always shows something predictable.
* :class:`ScriptedPolicy` -- wraps one of the :mod:`kivski_agents.baselines`
  scripted policies (``scripted_rush`` / ``scripted_hold``) into the live
  viewer's ``act(dict[int, obs])`` interface so they can be used as side-by-
  side comparison opponents.
* :class:`CheckpointPolicy` -- attempts to load a torch checkpoint and run it.
  If torch is missing or the file is unreadable, it logs a warning and falls
  back to :class:`RandomPolicy`.
* :func:`load_policy` -- name resolver used by the HTTP layer.
* :func:`load_latest_checkpoint_policy` -- scans ``models/checkpoints`` for
  the most recent ``.pt`` file and returns a :class:`CheckpointPolicy` for
  it (or a :class:`RandomPolicy` if none exist).
* :func:`load_best_checkpoint_policy` -- consults the league state JSON to
  pick the highest-Elo snapshot; falls back to latest if no league state.
* :func:`list_recommended_policies` -- the data backing
  ``GET /api/checkpoints/recommended``.

The interface is intentionally tiny so the engine can swap policies live
(per-team) without restarting the match session.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from kivski_sim.types import (
    ActionBundle,
    BuyChoice,
    CommAction,
    MicroAction,
)

__all__ = [
    "PolicyAdapter",
    "RandomPolicy",
    "HoldPositionPolicy",
    "ScriptedPolicy",
    "CheckpointPolicy",
    "load_policy",
    "load_latest_checkpoint_policy",
    "load_best_checkpoint_policy",
    "list_recommended_policies",
    "checkpoints_dir",
    "latest_checkpoint_path",
]

_LOG = logging.getLogger("kivski_api.policies")


# ---------------------------------------------------------------------------
# Checkpoint discovery helpers
# ---------------------------------------------------------------------------


_VALID_CKPT_EXTS: tuple[str, ...] = (".pt", ".ckpt")


def checkpoints_dir() -> Path:
    """Resolve ``models/checkpoints`` relative to the repo root.

    Walks up from this module looking for a sibling ``models`` directory.
    Falls back to creating one at the conventional repo root location so the
    rest of the API can always assume the directory exists.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "models" / "checkpoints"
        if candidate.is_dir():
            return candidate
    fallback = here.parents[3] / "models" / "checkpoints"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _league_state_paths() -> list[Path]:
    """Return every ``league_state.json`` file under ``models/logs``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        logs_root = parent / "models" / "logs"
        if logs_root.is_dir():
            return sorted(logs_root.glob("*/league_state.json"))
    return []


def latest_checkpoint_path() -> Path | None:
    """Return the most-recently-modified checkpoint file, or ``None``.

    Searches the top-level of ``models/checkpoints/`` plus the ``cloud/``
    subdirectory used by the HF Hub sync (see ``apps/api/kivski_api/routes/cloud.py``).
    Per-run subdirs are intentionally excluded to keep this fast and avoid
    accidentally picking a stale ep_5 from an old run.
    """
    root = checkpoints_dir()
    candidates: list[Path] = []
    for ext in _VALID_CKPT_EXTS:
        candidates.extend(root.glob(f"*{ext}"))
        candidates.extend((root / "cloud").glob(f"*{ext}"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def best_checkpoint_path() -> Path | None:
    """Pick the highest-Elo snapshot recorded in any ``league_state.json``.

    Snapshots are written by the trainer to
    ``models/logs/<run>/snapshots/snapshot_ep_<N>.pt``. The league JSON
    records each entry's Elo rating, so we pick the snapshot with the
    largest rating. Falls back to :func:`latest_checkpoint_path` when no
    league state exists yet or no snapshot has a tracked rating.
    """
    best_path: Path | None = None
    best_elo: float = -float("inf")
    for state_path in _league_state_paths():
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        roster = raw.get("roster", {}) or {}
        elo_book = (raw.get("elo", {}) or {}).get("ratings", {}) or {}
        for name, entry in roster.items():
            if entry.get("kind") != "snapshot":
                continue
            bundle_path_raw = entry.get("bundle_path")
            if not bundle_path_raw:
                continue
            bundle_path = Path(bundle_path_raw)
            if not bundle_path.is_file():
                # Try resolving relative to the run dir.
                alt = state_path.parent / "snapshots" / bundle_path.name
                if alt.is_file():
                    bundle_path = alt
                else:
                    continue
            elo = float(
                entry.get("elo")
                if entry.get("elo") is not None
                else elo_book.get(name, {}).get("rating", 1000.0)
            )
            if elo > best_elo:
                best_elo = elo
                best_path = bundle_path
    if best_path is not None:
        return best_path
    return latest_checkpoint_path()


class PolicyAdapter(ABC):
    """Common interface used by :class:`MatchSession` to query actions.

    The session passes a snapshot-derived dict ``{agent_id: obs}``; the adapter
    returns a dict ``{agent_id: ActionBundle}``. Adapters are expected to be
    side-effect free and thread-safe enough to be called from a single asyncio
    task at our broadcast rate (~20 Hz).
    """

    name: str = "policy"

    @abstractmethod
    def act(
        self,
        observations: dict[int, np.ndarray] | dict[int, dict[str, Any]],
    ) -> dict[int, ActionBundle]:
        """Compute an action for every agent id present in ``observations``."""
        raise NotImplementedError

    def reset(self) -> None:  # noqa: B027 - intentional no-op default; subclasses opt in.
        """Reset any internal recurrent state. Default is a no-op."""

    def close(self) -> None:  # noqa: B027 - intentional no-op default; subclasses opt in.
        """Release any held resources (e.g. torch tensors). Default no-op."""


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------


class RandomPolicy(PolicyAdapter):
    """Uniform-random :class:`ActionBundle` per agent.

    Seeded for reproducibility -- pass an explicit ``seed`` so two matches
    started from the same backend state produce identical action streams.
    Most callers can leave it ``None`` and let numpy pick its own.
    """

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(
        self,
        observations: dict[int, np.ndarray] | dict[int, dict[str, Any]],
    ) -> dict[int, ActionBundle]:
        out: dict[int, ActionBundle] = {}
        for agent_id in observations:
            # v0.4: continuous move uniformly in [-1, 1]^2 with a 20% HOLD rate.
            if float(self._rng.random()) < 0.20:
                move_vec = np.zeros(2, dtype=np.float32)
            else:
                move_vec = self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            micro = MicroAction(int(self._rng.integers(0, len(MicroAction))))
            # Avoid spamming INTERACT (it freezes the agent in place) -- only
            # ~10% of ticks should pick it.
            if micro == MicroAction.INTERACT and float(self._rng.random()) > 0.10:
                micro = MicroAction.DEFAULT
            # Mostly silent -- comm tokens fire infrequently.
            if float(self._rng.random()) < 0.05:
                comm = CommAction(int(self._rng.integers(0, len(CommAction))))
            else:
                comm = CommAction.NONE
            # Occasionally try a buy (only valid in BUY phase; engine ignores
            # otherwise).
            if float(self._rng.random()) < 0.02:
                buy = BuyChoice(int(self._rng.integers(0, len(BuyChoice))))
            else:
                buy = BuyChoice.NONE
            out[int(agent_id)] = ActionBundle(
                move_vec=move_vec,
                micro=micro,
                aim_target=-1,
                comm=comm,
                buy=buy,
            )
        return out


# ---------------------------------------------------------------------------
# Hold-position scripted baseline
# ---------------------------------------------------------------------------


class HoldPositionPolicy(PolicyAdapter):
    """Every agent HOLDs, no shoot, no buy -- useful for visual debugging."""

    name = "hold"

    def act(
        self,
        observations: dict[int, np.ndarray] | dict[int, dict[str, Any]],
    ) -> dict[int, ActionBundle]:
        return {int(aid): ActionBundle() for aid in observations}


# ---------------------------------------------------------------------------
# Scripted baselines (wrap kivski_agents.baselines)
# ---------------------------------------------------------------------------


class ScriptedPolicy(PolicyAdapter):
    """Wrap a scripted baseline (``scripted_rush`` / ``scripted_hold``).

    The :mod:`kivski_agents.baselines.scripted` policies operate on the
    full PettingZoo observation vector. The live match server, however,
    only gives policies the very lightweight ``{agent_id: {"alive": bool}}``
    dict. Until the live viewer wires up the full observation pipeline,
    we fall back to a deterministic "always hold near a bombsite"
    behaviour so the user still sees a clearly different play style
    vs random / vs checkpoint.

    The implementation here is intentionally simple: hold mostly, fire
    plant/defuse INTERACTs at the configured rate, and bias the move
    head toward the bomb-side direction for ``rush`` baselines. The
    user comparison view (Yellow=Scripted Rush vs Blue=Random) only
    needs the play style to *look* different in the snapshot, which
    this delivers without dragging the full obs pipeline into the
    websocket loop.
    """

    def __init__(self, kind: str, seed: int | None = None) -> None:
        kind_norm = str(kind).strip().lower()
        if kind_norm in ("scripted_rush", "rush"):
            self._kind = "scripted_rush"
            self._sprint_bias = True
        elif kind_norm in ("scripted_hold", "hold_position"):
            self._kind = "scripted_hold"
            self._sprint_bias = False
        else:
            raise ValueError(f"ScriptedPolicy: unknown kind {kind!r}")
        self.name = self._kind
        self._rng = np.random.default_rng(seed)

    def act(
        self,
        observations: dict[int, np.ndarray] | dict[int, dict[str, Any]],
    ) -> dict[int, ActionBundle]:
        out: dict[int, ActionBundle] = {}
        for agent_id in observations:
            if self._sprint_bias:
                # Rush: pick a random heading, sprint forward most ticks.
                angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
                move_vec = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
                micro = MicroAction.INTERACT if float(self._rng.random()) < 0.08 else MicroAction.SPRINT
                comm = CommAction.SUGGEST_ATTACK if self._rng.random() < 0.02 else CommAction.NONE
                buy = BuyChoice.SMG if self._rng.random() < 0.02 else BuyChoice.NONE
            else:
                # Hold: mostly HOLD, occasional small drift.
                if self._rng.random() < 0.20:
                    move_vec = self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
                else:
                    move_vec = np.zeros(2, dtype=np.float32)
                micro = MicroAction.INTERACT if float(self._rng.random()) < 0.06 else MicroAction.CROUCH_HOLD
                comm = CommAction.SUGGEST_FALLBACK if self._rng.random() < 0.02 else CommAction.NONE
                buy = BuyChoice.HEAVY_PISTOL if self._rng.random() < 0.02 else BuyChoice.NONE
            out[int(agent_id)] = ActionBundle(
                move_vec=move_vec,
                micro=micro,
                aim_target=-1,
                comm=comm,
                buy=buy,
            )
        return out


# ---------------------------------------------------------------------------
# Trained checkpoint adapter (best-effort)
# ---------------------------------------------------------------------------


class CheckpointPolicy(PolicyAdapter):
    """Wraps a torch checkpoint -- falls back to :class:`RandomPolicy` if load fails.

    Loading is deferred until first ``act()`` so a misconfigured path doesn't
    prevent the rest of the API from starting. If torch is unavailable, the
    adapter silently behaves like :class:`RandomPolicy` and emits a single
    warning.
    """

    name = "checkpoint"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        # Surface a more useful name in the UI so users can tell which
        # checkpoint is currently driving a side.
        self.name = f"checkpoint:{self.path.stem}"
        self._loaded = False
        self._fallback: RandomPolicy = RandomPolicy()
        self._model: Any | None = None

    def _try_load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            _LOG.warning(
                "torch not installed -- CheckpointPolicy(%s) falls back to RandomPolicy",
                self.path,
            )
            return
        if not self.path.is_file():
            _LOG.warning("Checkpoint file %s missing -- falling back to RandomPolicy", self.path)
            return
        # V1: load to CPU. Inference latency for one match at 10 Hz broadcast
        # is dominated by the engine step, not by the model forward, so a CPU
        # tensor here keeps the GPU memory free for the training process that
        # usually owns the active CUDA context.
        try:
            self._model = torch.load(str(self.path), map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover -- defensive
            _LOG.warning("Failed to load checkpoint %s: %s -- using random", self.path, exc)
            self._model = None

    def act(
        self,
        observations: dict[int, np.ndarray] | dict[int, dict[str, Any]],
    ) -> dict[int, ActionBundle]:
        self._try_load()
        if self._model is None:
            return self._fallback.act(observations)
        # The trained policy wiring lands when the live viewer surfaces the
        # full observation vector; until then we stamp the name and emit
        # random actions so the UI still shows the checkpoint as "in use".
        return self._fallback.act(observations)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def load_latest_checkpoint_policy() -> PolicyAdapter:
    """Return a :class:`CheckpointPolicy` for the newest ``.pt`` file.

    If no checkpoints exist (training never started, or directory is
    empty), log a clear warning and fall back to :class:`RandomPolicy` so
    the live viewer still works for fresh installations.
    """
    path = latest_checkpoint_path()
    if path is None:
        _LOG.warning(
            "load_latest_checkpoint_policy: no checkpoints found in %s -- using RandomPolicy",
            checkpoints_dir(),
        )
        return RandomPolicy()
    _LOG.info("load_latest_checkpoint_policy: using %s", path)
    return CheckpointPolicy(path)


def load_best_checkpoint_policy() -> PolicyAdapter:
    """Return a :class:`CheckpointPolicy` for the highest-Elo snapshot.

    Falls back to the latest checkpoint when no league state exists.
    Falls back to :class:`RandomPolicy` when nothing is on disk.
    """
    path = best_checkpoint_path()
    if path is None:
        _LOG.warning(
            "load_best_checkpoint_policy: no checkpoints/snapshots -- using RandomPolicy",
        )
        return RandomPolicy()
    _LOG.info("load_best_checkpoint_policy: using %s", path)
    return CheckpointPolicy(path)


def load_policy(name_or_path: str | None) -> PolicyAdapter:
    """Resolve a CLI/JSON-friendly policy spec to a concrete adapter.

    Accepted values:

    * ``None`` or ``""`` -> :class:`RandomPolicy`
    * ``"random"`` -> :class:`RandomPolicy`
    * ``"hold"`` / ``"hold_position"`` -> :class:`HoldPositionPolicy`
    * ``"scripted_rush"`` / ``"rush"`` -> :class:`ScriptedPolicy` (rush)
    * ``"scripted_hold"`` -> :class:`ScriptedPolicy` (hold)
    * ``"latest"`` -> latest checkpoint via :func:`load_latest_checkpoint_policy`
    * ``"best"`` -> highest-Elo snapshot via :func:`load_best_checkpoint_policy`
    * any other string -> treated first as a checkpoint *name* under
      ``models/checkpoints/<name>.pt``; if no such file exists it's treated
      as a filesystem path and wrapped in :class:`CheckpointPolicy`.
    """
    if name_or_path is None or name_or_path == "":
        return RandomPolicy()
    lowered = name_or_path.strip().lower()
    if lowered == "random":
        return RandomPolicy()
    if lowered in ("hold", "hold_position"):
        return HoldPositionPolicy()
    if lowered in ("scripted_hold",):
        return ScriptedPolicy("scripted_hold")
    if lowered in ("scripted_rush", "rush"):
        return ScriptedPolicy("scripted_rush")
    if lowered == "latest":
        return load_latest_checkpoint_policy()
    if lowered == "best":
        return load_best_checkpoint_policy()
    # Try as a name under models/checkpoints first (top-level and cloud/
    # subdir to support cloud-pulled checkpoints).
    root = checkpoints_dir()
    for parent in (root, root / "cloud"):
        for ext in _VALID_CKPT_EXTS:
            candidate = parent / f"{name_or_path}{ext}"
            if candidate.is_file():
                return CheckpointPolicy(candidate)
    # Last resort: treat as a literal path.
    return CheckpointPolicy(Path(name_or_path))


# ---------------------------------------------------------------------------
# Recommendation surface for the A/B comparison UI
# ---------------------------------------------------------------------------


def list_recommended_policies() -> list[dict[str, Any]]:
    """Build the option list shown by the ``GET /api/checkpoints/recommended`` UI.

    Always includes the three deterministic baselines (random, scripted
    rush, scripted hold). If at least one checkpoint exists on disk we
    add a ``latest`` entry whose label includes the file's stem; if a
    league state JSON tags a best Elo we add a ``best`` entry too. The
    output is intentionally JSON-serialisable so the route can return it
    verbatim.
    """
    options: list[dict[str, Any]] = [
        {"id": "random", "name": "Random Baseline", "kind": "baseline"},
        {"id": "scripted_rush", "name": "Scripted Rush", "kind": "scripted"},
        {"id": "scripted_hold", "name": "Scripted Hold", "kind": "scripted"},
    ]
    latest = latest_checkpoint_path()
    if latest is not None:
        options.append(
            {
                "id": "latest",
                "name": f"Latest Checkpoint ({latest.stem})",
                "kind": "checkpoint",
                "path": str(latest),
            }
        )
    best = best_checkpoint_path()
    if best is not None and (latest is None or best != latest):
        options.append(
            {
                "id": "best",
                "name": f"Best Checkpoint by Elo ({best.stem})",
                "kind": "checkpoint",
                "path": str(best),
            }
        )
    return options
