"""Mutable in-memory state of a single match.

`MatchState` is the canonical truth the `Engine` operates on. It is intentionally
a plain dataclass of NumPy arrays + Python lists so it is fast to clone,
serialize (msgpack) and inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kivski_sim.types import (
    BombPhase,
    BuyChoice,
    MatchOutcome,
    Message,
    Phase,
    RoundOutcome,
    RoundSummary,
    Side,
    SoundEvent,
    Team,
    WeaponClass,
)


@dataclass(slots=True)
class AgentState:
    """Per-agent mutable state. Indexed by `AgentId` (0..N-1)."""

    agent_id: int
    team: Team
    side: Side
    pos: np.ndarray  # shape (2,) float32 -- tile coordinates
    vel: np.ndarray  # shape (2,) float32
    facing: float = 0.0  # radians
    alive: bool = True
    hp: float = 100.0
    armor: float = 0.0
    money: int = 800
    weapon: WeaponClass = WeaponClass.SIDEARM
    secondary: WeaponClass = WeaponClass.KNIFE
    has_bomb: bool = False
    has_defuse_kit: bool = False
    last_shot_tick: int = -1000
    reaction_cooldown: int = 0  # ticks until next aim/fire allowed
    buy_choice_pending: BuyChoice = BuyChoice.NONE
    armor_buy_pending: bool = False
    # Per-round bookkeeping
    damage_done_round: float = 0.0
    damage_taken_round: float = 0.0
    kills_round: int = 0
    deaths_round: int = 0
    assists_round: int = 0
    # Career bookkeeping
    kills_match: int = 0
    deaths_match: int = 0
    money_spent_match: int = 0


@dataclass(slots=True)
class BombState:
    """State of the bomb object on the map."""

    phase: BombPhase = BombPhase.CARRIED
    carrier: int = -1  # AgentId or -1 if not carried
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    plant_progress: float = 0.0  # 0..1
    defuse_progress: float = 0.0
    defuser: int = -1
    time_since_plant: float = 0.0
    site: str | None = None  # "A" | "B" once planted


@dataclass(slots=True)
class TeamState:
    team: Team
    side: Side
    score: int = 0
    consecutive_losses: int = 0  # for loss-bonus stacking
    # Per-round running aggregates for reward shaping / metrics
    map_control_tiles: int = 0


@dataclass(slots=True)
class MatchState:
    """Top-level mutable state. Everything the engine mutates lives here."""

    seed: int
    tick: int = 0
    round_id: int = 0
    phase: Phase = Phase.WARMUP
    phase_ticks_remaining: int = 0

    agents: list[AgentState] = field(default_factory=list)
    teams: dict[Team, TeamState] = field(default_factory=dict)
    bomb: BombState = field(default_factory=BombState)

    sounds: list[SoundEvent] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    round_summaries: list[RoundSummary] = field(default_factory=list)

    last_round_outcome: RoundOutcome = RoundOutcome.NONE
    match_outcome: MatchOutcome = MatchOutcome.NONE

    # --- helpers ---------------------------------------------------------

    def alive_agents(self, side: Side | None = None) -> list[AgentState]:
        if side is None:
            return [a for a in self.agents if a.alive]
        return [a for a in self.agents if a.alive and a.side == side]

    def agents_on_side(self, side: Side) -> list[AgentState]:
        return [a for a in self.agents if a.side == side]

    def agents_on_team(self, team: Team) -> list[AgentState]:
        return [a for a in self.agents if a.team == team]
