"""Kivski simulation package: deterministic top-down 5v5 bomb-defuse engine."""

try:  # engine arrives in a later task; tolerate its absence so the rest of the
    # package (types, map_loader, geometry, visibility) stays importable.
    from kivski_sim.engine import Engine, EngineConfig, Snapshot  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    Engine = None  # type: ignore[assignment]
    EngineConfig = None  # type: ignore[assignment]
    Snapshot = None  # type: ignore[assignment]

try:  # pettingzoo/gymnasium are optional at the package level so that
    # downstream code that only needs the engine can import kivski_sim
    # without pulling in the full RL stack.
    from kivski_sim.env import KivskiParallelEnv  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    KivskiParallelEnv = None  # type: ignore[assignment]

from kivski_sim.types import (
    ActionBundle,
    AgentId,
    BombPhase,
    BuyChoice,
    CommAction,
    MatchOutcome,
    MicroAction,
    MoveIntent,
    Phase,
    RoundOutcome,
    Side,
    Team,
    WeaponClass,
)

__all__ = [
    "ActionBundle",
    "AgentId",
    "BombPhase",
    "BuyChoice",
    "CommAction",
    "Engine",
    "EngineConfig",
    "KivskiParallelEnv",
    "MatchOutcome",
    "MicroAction",
    "MoveIntent",
    "Phase",
    "RoundOutcome",
    "Side",
    "Snapshot",
    "Team",
    "WeaponClass",
]

__version__ = "0.1.0"
