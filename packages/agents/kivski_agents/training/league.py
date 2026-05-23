"""League / Population-Based Training opponent pool.

The :class:`LeagueManager` keeps a roster of opponents that the training
policy spars with during rollout collection. A roster always contains:

* ``"main"`` -- a frozen alias of the *current* training policy, used for
  self-play. We don't actually freeze it for V1; instead we sample the
  same model the trainer holds, so it always reflects the latest weights.
* ``"random"``, ``"scripted_rush"``, ``"scripted_hold"`` -- the canonical
  baselines from :mod:`kivski_agents.baselines`.
* ``"snapshot_ep_<N>"`` -- frozen :class:`PolicyBundle` snapshots saved
  periodically by the trainer. We cap the population at
  :attr:`LeagueConfig.population_size` and evict the oldest first.

Sampling is fraction-based per :class:`LeagueConfig`:

    fraction               sampled opponent
    --------               ----------------
    random_fraction        RandomBaseline
    scripted_fraction      ScriptedRushBaseline or ScriptedHoldBaseline (50/50)
    exploit_fraction       a frozen snapshot (weighted toward recent ones)
    rest                   "main" (current training policy)

All exposed opponents follow the standard ``reset(agent_names)`` +
``act(observations, received_comms=None) -> (actions, payloads)``
interface used by the rest of the eval / training stack.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from kivski_sim.config import LeagueConfig
from kivski_sim.env import KivskiParallelEnv
from kivski_sim.map_loader import MapData

from kivski_agents.baselines import BASELINE_REGISTRY, get_baseline
from kivski_agents.eval.elo import EloTracker
from kivski_agents.policy_runner import PolicyBundle, PolicyRunner

__all__ = [
    "LeagueEntry",
    "LeagueManager",
    "OpponentSampler",
    "MainSelfPlayPolicy",
]


# ---------------------------------------------------------------------------
# Roster entry
# ---------------------------------------------------------------------------


@dataclass
class LeagueEntry:
    """One opponent in the league roster."""

    name: str
    bundle_path: Path | None = None  # None for "main" and baselines
    elo: float = 1000.0
    creation_episode: int = 0
    plays: int = 0
    wins: int = 0
    kind: str = "snapshot"  # "main" | "baseline" | "snapshot"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "bundle_path": str(self.bundle_path) if self.bundle_path else None,
            "elo": float(self.elo),
            "creation_episode": int(self.creation_episode),
            "plays": int(self.plays),
            "wins": int(self.wins),
            "kind": str(self.kind),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LeagueEntry:
        return cls(
            name=str(raw.get("name", "?")),
            bundle_path=Path(raw["bundle_path"]) if raw.get("bundle_path") else None,
            elo=float(raw.get("elo", 1000.0)),
            creation_episode=int(raw.get("creation_episode", 0)),
            plays=int(raw.get("plays", 0)),
            wins=int(raw.get("wins", 0)),
            kind=str(raw.get("kind", "snapshot")),
        )


# ---------------------------------------------------------------------------
# Opponent sampler (uniform "act" contract)
# ---------------------------------------------------------------------------


class OpponentSampler:
    """Wraps a chosen opponent into the standard ``reset`` + ``act`` interface.

    The trainer treats every opponent identically -- regardless of whether
    it is a baseline, a frozen snapshot, or the live "main" policy.
    """

    def __init__(self, name: str, policy: Any) -> None:
        self.name: str = str(name)
        self.policy: Any = policy
        # Allow tests / telemetry to recover what was actually sampled.
        self._agent_names: list[str] = []

    def reset(self, agent_names: list[str]) -> None:
        self._agent_names = list(agent_names)
        # Some policies (e.g. very small stub baselines used in tests) may
        # raise here; we silently swallow to keep the trainer alive.
        with contextlib.suppress(Exception):
            self.policy.reset(agent_names)

    def act(
        self,
        observations: dict[str, np.ndarray],
        received_comms: dict[str, dict] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        try:
            return self.policy.act(observations, received_comms=received_comms)
        except TypeError:
            # Some baselines don't accept ``received_comms``.
            return self.policy.act(observations)


# ---------------------------------------------------------------------------
# Live self-play wrapper around the current training model
# ---------------------------------------------------------------------------


class MainSelfPlayPolicy:
    """Adapts a :class:`PolicyRunner` to the baseline ``act`` interface.

    Used when the league samples "main" -- the opponent is the current
    learning policy itself. We deliberately do *not* freeze the weights: the
    training policy keeps updating between rollouts, so each self-play
    collection sees the most recent network.
    """

    name: str = "main"

    def __init__(self, model: torch.nn.Module, device: torch.device | str = "cpu") -> None:
        from kivski_agents.policy_runner import PolicyRunner as _PR  # local to avoid cycles

        self.runner: _PR = _PR(model=model, device=device, deterministic=False)
        self._agent_names: list[str] = []

    def reset(self, agent_names: list[str]) -> None:
        self._agent_names = list(agent_names)
        self.runner.reset(agent_names)

    def act(
        self,
        observations: dict[str, np.ndarray],
        received_comms: dict[str, dict] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        return self.runner.act(observations, received_comms=received_comms)


# ---------------------------------------------------------------------------
# League manager
# ---------------------------------------------------------------------------


class LeagueManager:
    """Manages the population of opponents the training policy sees."""

    def __init__(
        self,
        log_dir: Path,
        cfg: LeagueConfig,
        env: KivskiParallelEnv,
        map_data: MapData,
        device: torch.device | str = "cpu",
        *,
        main_model: torch.nn.Module | None = None,
    ) -> None:
        self.log_dir: Path = Path(log_dir)
        self.cfg: LeagueConfig = cfg
        self.env: KivskiParallelEnv = env
        self.map_data: MapData = map_data
        self.device: torch.device = torch.device(device)
        self._main_model: torch.nn.Module | None = main_model

        self.snapshot_dir: Path = self.log_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.state_path: Path = self.log_dir / "league_state.json"

        # Roster: name -> entry.
        self.roster: dict[str, LeagueEntry] = {}
        # Always-present entries.
        self.roster["main"] = LeagueEntry(name="main", kind="main")
        for bname in ("random", "scripted_rush", "scripted_hold"):
            if bname in BASELINE_REGISTRY:
                self.roster[bname] = LeagueEntry(name=bname, kind="baseline")

        # Pick up any pre-existing snapshots from disk.
        for path in sorted(self.snapshot_dir.glob("snapshot_ep_*.pt")):
            ep = self._parse_episode_from_path(path)
            entry_name = path.stem  # "snapshot_ep_<N>"
            self.roster[entry_name] = LeagueEntry(
                name=entry_name,
                bundle_path=path,
                creation_episode=int(ep),
                kind="snapshot",
            )

        self.elo_tracker: EloTracker = EloTracker()
        for name in self.roster:
            self.elo_tracker.add_policy(name)

    # ------------------------------------------------------------------
    # Roster mutation
    # ------------------------------------------------------------------

    def set_main_model(self, model: torch.nn.Module) -> None:
        """Update the underlying model the ``"main"`` opponent wraps."""
        self._main_model = model

    def add_snapshot(self, bundle_path: Path, episode: int) -> str:
        """Register a frozen snapshot as a new roster entry.

        Args:
            bundle_path: Path to a :class:`PolicyBundle` checkpoint (``.pt``).
            episode: Episode counter at which the snapshot was taken.

        Returns:
            The registered roster name (``"snapshot_ep_<episode>"``).
        """
        name = f"snapshot_ep_{int(episode)}"
        self.roster[name] = LeagueEntry(
            name=name,
            bundle_path=Path(bundle_path),
            creation_episode=int(episode),
            kind="snapshot",
        )
        self.elo_tracker.add_policy(name)
        # Cap the population by evicting the oldest snapshots when we exceed
        # the configured size.
        self._trim_population()
        return name

    def _trim_population(self) -> None:
        """Evict the oldest snapshots if we exceed ``population_size``."""
        snaps = [e for e in self.roster.values() if e.kind == "snapshot"]
        if len(snaps) <= int(self.cfg.population_size):
            return
        snaps.sort(key=lambda e: e.creation_episode)
        excess = len(snaps) - int(self.cfg.population_size)
        for entry in snaps[:excess]:
            # Remove from roster but leave the file on disk (operators may
            # still want to inspect / restore them manually).
            self.roster.pop(entry.name, None)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_opponent(self, rng: np.random.Generator) -> OpponentSampler:
        """Pick an opponent for the next rollout."""
        r = float(rng.random())
        rand_thr = float(max(0.0, self.cfg.random_fraction))
        scripted_thr = rand_thr + float(max(0.0, self.cfg.scripted_fraction))
        exploit_thr = scripted_thr + float(max(0.0, self.cfg.exploit_fraction))

        if r < rand_thr:
            return self._instantiate("random")
        if r < scripted_thr:
            # 50/50 between rush and hold.
            choice = "scripted_rush" if rng.random() < 0.5 else "scripted_hold"
            return self._instantiate(choice)
        if r < exploit_thr:
            snap_name = self._sample_snapshot(rng)
            if snap_name is not None:
                return self._instantiate(snap_name)
            # Fall through to main if no snapshots yet.
        return self._instantiate("main")

    def _sample_snapshot(self, rng: np.random.Generator) -> str | None:
        snaps = [e for e in self.roster.values() if e.kind == "snapshot"]
        if not snaps:
            return None
        # Weight by recency: newest snapshots get higher weight.
        snaps.sort(key=lambda e: e.creation_episode)
        weights = np.linspace(1.0, 3.0, num=len(snaps), dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(snaps), p=weights))
        return snaps[idx].name

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def _instantiate(self, name: str) -> OpponentSampler:
        """Materialise the policy behind a roster entry."""
        if name not in self.roster:
            raise KeyError(f"opponent {name!r} is not in the league roster")
        entry = self.roster[name]

        if entry.kind == "main":
            if self._main_model is None:
                raise RuntimeError("LeagueManager: 'main' opponent requested but no model was set.")
            return OpponentSampler(
                name="main", policy=MainSelfPlayPolicy(model=self._main_model, device=self.device)
            )

        if entry.kind == "baseline":
            # Use the project's standard baseline factory.
            seed = int(entry.creation_episode) ^ 0xA5A5_5A5A
            return OpponentSampler(
                name=name,
                policy=get_baseline(name, self.env, self.map_data, seed=seed),
            )

        # Snapshot: load the frozen bundle into a PolicyRunner-driven adapter.
        if entry.bundle_path is None or not entry.bundle_path.is_file():
            # Fall back to "main" -- a missing snapshot is better than crashing
            # the trainer mid-run.
            return self._instantiate("main")
        bundle = PolicyBundle.from_checkpoint(entry.bundle_path)
        runner = bundle.to_runner(device=self.device, deterministic=False)
        return OpponentSampler(name=name, policy=_RunnerPolicyAdapter(runner=runner, name=name))

    # ------------------------------------------------------------------
    # Elo updates
    # ------------------------------------------------------------------

    def update_elo(self, opponent_name: str, outcome: float) -> None:
        """Record an outcome of "main" vs ``opponent_name``.

        ``outcome`` is 1.0 if main won, 0.0 if opponent won, 0.5 for draw.
        """
        self.elo_tracker.add_policy("main")
        self.elo_tracker.add_policy(opponent_name)
        self.elo_tracker.update("main", opponent_name, float(outcome))
        if opponent_name in self.roster:
            entry = self.roster[opponent_name]
            entry.plays += 1
            if outcome < 0.5:
                # ``opponent_name`` won.
                entry.wins += 1
            # Mirror Elo onto the entry for quick at-a-glance views.
            entry.elo = float(self.elo_tracker.ratings[opponent_name].rating)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Path | None = None) -> Path:
        """Serialise the roster + Elo book to JSON."""
        out = Path(path) if path is not None else self.state_path
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "roster": {name: e.to_dict() for name, e in self.roster.items()},
            "elo": self.elo_tracker.to_dict(),
            "config": {
                "population_size": int(self.cfg.population_size),
                "snapshot_every_episodes": int(self.cfg.snapshot_every_episodes),
                "exploit_fraction": float(self.cfg.exploit_fraction),
                "random_fraction": float(self.cfg.random_fraction),
                "scripted_fraction": float(self.cfg.scripted_fraction),
            },
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return out

    def load_state(self, path: Path | None = None) -> None:
        """Restore the roster + Elo book from a JSON file."""
        src = Path(path) if path is not None else self.state_path
        if not src.is_file():
            return
        raw = json.loads(src.read_text(encoding="utf-8"))
        for name, entry_raw in raw.get("roster", {}).items():
            self.roster[name] = LeagueEntry.from_dict(entry_raw)
        if "elo" in raw:
            self.elo_tracker = EloTracker.from_dict(raw["elo"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_episode_from_path(path: Path) -> int:
        """Extract ``N`` from filenames like ``snapshot_ep_1234.pt``."""
        stem = path.stem  # snapshot_ep_<N>
        try:
            return int(stem.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return 0


# ---------------------------------------------------------------------------
# Internal adapter: PolicyRunner -> baseline-style act(...)
# ---------------------------------------------------------------------------


class _RunnerPolicyAdapter:
    """Glue a :class:`PolicyRunner` into the baseline ``reset`` / ``act`` contract."""

    def __init__(self, runner: PolicyRunner, name: str) -> None:
        self.runner: PolicyRunner = runner
        self.name: str = str(name)

    def reset(self, agent_names: list[str]) -> None:
        self.runner.reset(agent_names)

    def act(
        self,
        observations: dict[str, np.ndarray],
        received_comms: dict[str, dict] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        return self.runner.act(observations, received_comms=received_comms)
