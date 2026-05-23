"""Core enums, dataclasses, and protocol types for the Kivski simulator.

Every other module in `kivski_sim` (and downstream `kivski_agents` / `kivski_api`)
imports from here. Keep this file *cheap* (no heavy deps) so it can be safely
imported in any context including the frontend code-gen scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NewType

import numpy as np

AgentId = NewType("AgentId", int)
RoundId = NewType("RoundId", int)
Tick = NewType("Tick", int)


# ---------------------------------------------------------------------------
# Sides, phases, outcomes
# ---------------------------------------------------------------------------


class Side(IntEnum):
    """Which functional side an agent currently plays."""

    ATTACKER = 0  # yellow team in v1
    DEFENDER = 1  # blue team in v1


class Team(IntEnum):
    """Fixed team identity (does not change on side switch)."""

    YELLOW = 0
    BLUE = 1


class Phase(IntEnum):
    """Round-level high-level phase."""

    WARMUP = 0
    BUY = 1
    LIVE = 2
    POST_PLANT = 3
    ROUND_OVER = 4
    MATCH_OVER = 5


class BombPhase(IntEnum):
    """State of the bomb object."""

    CARRIED = 0
    DROPPED = 1
    PLANTING = 2
    PLANTED = 3
    DEFUSING = 4
    DEFUSED = 5
    DETONATED = 6


class RoundOutcome(IntEnum):
    NONE = 0
    ATTACKERS_ELIM = 1  # attackers killed all defenders before plant
    DEFENDERS_ELIM = 2  # defenders killed all attackers
    BOMB_DETONATED = 3
    BOMB_DEFUSED = 4
    TIMEOUT = 5  # round time expired without plant


class MatchOutcome(IntEnum):
    NONE = 0
    YELLOW_WIN = 1
    BLUE_WIN = 2
    DRAW = 3


# ---------------------------------------------------------------------------
# Weapons / equipment
# ---------------------------------------------------------------------------


class WeaponClass(IntEnum):
    """Generic original weapon families (no copyrighted naming)."""

    KNIFE = 0  # baseline melee, always available
    SIDEARM = 1  # cheap starter pistol
    HEAVY_PISTOL = 2
    SMG = 3
    RIFLE = 4
    PRECISION = 5  # long-range / sniper-style
    SHOTGUN = 6


@dataclass(slots=True, frozen=True)
class WeaponStats:
    cls: WeaponClass
    name: str
    cost: int
    damage_per_hit: float
    fire_rate_hz: float  # shots per second
    optimal_range: float  # tiles where accuracy is highest
    max_range: float  # tiles where damage falls off to zero
    armor_penetration: float  # 0..1 multiplier vs armored hp
    accuracy_standing: float
    accuracy_moving: float
    side_restricted: int = -1  # -1 = both, 0 = attacker only, 1 = defender only


# Static weapon catalogue. Keep IDs stable -- they index into observation/action arrays.
WEAPONS: dict[WeaponClass, WeaponStats] = {
    WeaponClass.KNIFE: WeaponStats(
        cls=WeaponClass.KNIFE,
        name="Blade",
        cost=0,
        damage_per_hit=55,
        fire_rate_hz=1.4,
        optimal_range=0.8,
        max_range=1.2,
        armor_penetration=0.5,
        accuracy_standing=0.95,
        accuracy_moving=0.95,
    ),
    WeaponClass.SIDEARM: WeaponStats(
        cls=WeaponClass.SIDEARM,
        name="ZP-9",
        cost=0,
        damage_per_hit=24,
        fire_rate_hz=3.2,
        optimal_range=5,
        max_range=14,
        armor_penetration=0.45,
        accuracy_standing=0.80,
        accuracy_moving=0.45,
    ),
    WeaponClass.HEAVY_PISTOL: WeaponStats(
        cls=WeaponClass.HEAVY_PISTOL,
        name="Kestrel-50",
        cost=700,
        damage_per_hit=42,
        fire_rate_hz=2.0,
        optimal_range=6,
        max_range=18,
        armor_penetration=0.85,
        accuracy_standing=0.78,
        accuracy_moving=0.40,
    ),
    WeaponClass.SMG: WeaponStats(
        cls=WeaponClass.SMG,
        name="Viper-Repeater",
        cost=1500,
        damage_per_hit=18,
        fire_rate_hz=10.5,
        optimal_range=8,
        max_range=22,
        armor_penetration=0.55,
        accuracy_standing=0.80,
        accuracy_moving=0.65,
    ),
    WeaponClass.RIFLE: WeaponStats(
        cls=WeaponClass.RIFLE,
        name="Hex-Rifle",
        cost=2700,
        damage_per_hit=34,
        fire_rate_hz=8.0,
        optimal_range=14,
        max_range=42,
        armor_penetration=0.80,
        accuracy_standing=0.86,
        accuracy_moving=0.30,
    ),
    WeaponClass.PRECISION: WeaponStats(
        cls=WeaponClass.PRECISION,
        name="Talon Marksman",
        cost=4200,
        damage_per_hit=110,
        fire_rate_hz=1.2,
        optimal_range=28,
        max_range=80,
        armor_penetration=0.95,
        accuracy_standing=0.92,
        accuracy_moving=0.05,
    ),
    WeaponClass.SHOTGUN: WeaponStats(
        cls=WeaponClass.SHOTGUN,
        name="Maw-12",
        cost=1100,
        damage_per_hit=70,
        fire_rate_hz=1.4,
        optimal_range=4,
        max_range=10,
        armor_penetration=0.55,
        accuracy_standing=0.85,
        accuracy_moving=0.50,
    ),
}


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------


class MoveIntent(IntEnum):
    """Hierarchical movement: choose a coarse compass direction or hold."""

    HOLD = 0
    N = 1
    NE = 2
    E = 3
    SE = 4
    S = 5
    SW = 6
    W = 7
    NW = 8


MOVE_VECTORS: dict[MoveIntent, tuple[float, float]] = {
    MoveIntent.HOLD: (0.0, 0.0),
    MoveIntent.N: (0.0, -1.0),
    MoveIntent.NE: (0.7071, -0.7071),
    MoveIntent.E: (1.0, 0.0),
    MoveIntent.SE: (0.7071, 0.7071),
    MoveIntent.S: (0.0, 1.0),
    MoveIntent.SW: (-0.7071, 0.7071),
    MoveIntent.W: (-1.0, 0.0),
    MoveIntent.NW: (-0.7071, -0.7071),
}


class MicroAction(IntEnum):
    """Higher-level posture / contextual action."""

    DEFAULT = 0  # normal walk, ready weapon
    CROUCH_HOLD = 1  # crouched, accuracy boost but slow
    PEEK = 2  # shoulder peek -- briefly expose to gather info
    SPRINT = 3  # faster but louder + worse accuracy
    FALL_BACK = 4  # crouch + walk backwards to last cover
    INTERACT = 5  # plant / defuse / pickup if eligible


class CommAction(IntEnum):
    """Discrete communication channel actions (learnable semantics).

    Important: the *meaning* of these is not hardcoded -- they are token ids
    fed through a learned attention-based comm channel (TarMAC-style). The
    labels below are for the live viewer only.
    """

    NONE = 0
    PING_LOCATION = 1
    WARN_DANGER = 2
    REQUEST_SUPPORT = 3
    SUGGEST_ROTATE = 4
    SUGGEST_ATTACK = 5
    SUGGEST_FALLBACK = 6
    CONTACT_ENEMY = 7
    BOMBSITE_CLEAR = 8


class BuyChoice(IntEnum):
    """What to purchase this buy phase. Only valid during Phase.BUY."""

    NONE = 0
    SIDEARM = 1
    HEAVY_PISTOL = 2
    SMG = 3
    SHOTGUN = 4
    RIFLE = 5
    PRECISION = 6
    ARMOR = 7


@dataclass(slots=True)
class ActionBundle:
    """One agent's full action this tick (autoregressive heads collapsed)."""

    move: MoveIntent = MoveIntent.HOLD
    micro: MicroAction = MicroAction.DEFAULT
    aim_target: int = -1  # -1 = no specific target; otherwise agent id
    comm: CommAction = CommAction.NONE
    comm_payload: np.ndarray | None = None  # learned message vector (TarMAC)
    buy: BuyChoice = BuyChoice.NONE


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Vec2:
    x: float
    y: float

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    @staticmethod
    def from_array(arr: np.ndarray) -> Vec2:
        return Vec2(float(arr[0]), float(arr[1]))


@dataclass(slots=True)
class SoundEvent:
    """A noise emitted on the map -- approximate position only."""

    tick: int
    pos: tuple[float, float]
    radius: float
    intensity: float
    source_team: Team
    kind: str = "step"  # step | shot | plant | defuse | bomb_pickup


@dataclass(slots=True)
class Message:
    """A peer-to-peer comm message between teammates."""

    tick: int
    sender: AgentId
    receivers: tuple[AgentId, ...]
    action: CommAction
    payload: np.ndarray | None = None
    pos: tuple[float, float] | None = None  # for PING_LOCATION etc.


@dataclass(slots=True)
class CombatEvent:
    tick: int
    attacker: AgentId
    victim: AgentId
    weapon: WeaponClass
    damage: float
    killed: bool
    distance: float
    through_cover: bool


@dataclass(slots=True)
class RoundSummary:
    round_id: int
    outcome: RoundOutcome
    winning_side: Side
    yellow_score: int
    blue_score: int
    bomb_planted: bool
    bomb_planted_site: str | None
    duration_ticks: int
    survivors_yellow: int
    survivors_blue: int


@dataclass(slots=True)
class FrameMeta:
    """Compact, JSON-friendly per-tick info for the live viewer."""

    tick: int
    round_id: int
    phase: Phase
    bomb_phase: BombPhase
    yellow_score: int
    blue_score: int
    seconds_left: float
    plant_progress: float = 0.0
    defuse_progress: float = 0.0
    events: list[CombatEvent] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    sounds: list[SoundEvent] = field(default_factory=list)
