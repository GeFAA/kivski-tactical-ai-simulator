"""Pure combat helpers extracted from the engine.

All functions here are stateless utilities that consume the typed weapon
stats, the per-tick agent micro context, and the combat config. They are
deterministic given their inputs; randomness is sampled by the caller via
the ``RngHub`` "combat" channel so test seeds remain reproducible.
"""

from __future__ import annotations

import math

import numpy as np

from kivski_sim.config import CombatConfig
from kivski_sim.types import MicroAction, WeaponStats

__all__ = [
    "compute_hit_probability",
    "compute_damage",
    "sample_reaction_time",
]


def _accuracy_modifier_for_attacker(
    micro: MicroAction,
    base_cfg: CombatConfig,
    standing_acc: float,
    moving_acc: float,
) -> float:
    """Return an effective accuracy in [0, 1] for the attacker's posture.

    ``standing_acc`` / ``moving_acc`` come from the weapon stats so e.g. a
    sniper barely moves and shoots while an SMG keeps decent moving accuracy.
    Crouching boosts accuracy further, sprinting heavily penalizes it, and
    fall-back posture (crouch-walk) is in between.
    """
    if micro == MicroAction.CROUCH_HOLD:
        # Lean into the weapon's standing accuracy and add the crouch bonus.
        return float(min(1.0, max(standing_acc, base_cfg.base_accuracy_crouched)))
    if micro == MicroAction.SPRINT:
        return float(max(0.05, moving_acc * 0.4))
    if micro == MicroAction.FALL_BACK:
        return float(max(moving_acc, 0.6 * standing_acc))
    if micro == MicroAction.PEEK:
        # A peek is essentially standing-still for a brief moment.
        return float(standing_acc)
    if micro == MicroAction.INTERACT:
        # Planting/defusing means the agent isn't shooting effectively.
        return float(moving_acc * 0.5)
    # DEFAULT -> ready, slight movement penalty modeled by linear blend.
    return float(0.5 * (standing_acc + moving_acc))


def _distance_falloff(weapon: WeaponStats, distance: float) -> float:
    """Smooth falloff: 1.0 at optimal range, linearly decays past it."""
    if distance <= weapon.optimal_range:
        return 1.0
    if distance >= weapon.max_range:
        return 0.0
    span = max(1e-6, weapon.max_range - weapon.optimal_range)
    return float(max(0.0, 1.0 - (distance - weapon.optimal_range) / span))


def _target_evasion(micro: MicroAction) -> float:
    """Multiplier (<=1) applied to the attacker's accuracy.

    A sprinting target is harder to hit, a peeking target slightly easier
    (briefly exposed), and a planting/defusing target is sitting duck.
    """
    if micro == MicroAction.SPRINT:
        return 0.75
    if micro == MicroAction.PEEK:
        return 1.05
    if micro == MicroAction.INTERACT:
        return 1.20
    if micro == MicroAction.CROUCH_HOLD:
        return 0.92
    return 1.0


def compute_hit_probability(
    weapon: WeaponStats,
    distance: float,
    attacker_micro: MicroAction,
    target_micro: MicroAction,
    base_acc_config: CombatConfig,
    through_cover: bool = False,
) -> float:
    """Compute the probability of landing a hit in a single shot.

    The model is intentionally simple: accuracy * distance_falloff *
    target_evasion * cover_penalty. Each factor sits in [0, 1] and we clamp
    the product back into [0, 1] just to be safe.
    """
    acc = _accuracy_modifier_for_attacker(
        attacker_micro,
        base_acc_config,
        weapon.accuracy_standing,
        weapon.accuracy_moving,
    )
    falloff = _distance_falloff(weapon, float(distance))
    evasion = _target_evasion(target_micro)
    cover = 0.55 if through_cover else 1.0
    p = acc * falloff * evasion * cover
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return float(p)


def compute_damage(
    weapon: WeaponStats,
    distance: float,
    target_armor: float,
    through_cover: bool,
    cover_damage_multiplier: float,
) -> tuple[float, float]:
    """Return ``(damage_to_hp, damage_absorbed_by_armor)``.

    Damage scales with distance (cubic-ish falloff using the same span as
    the hit probability) and is reduced by armor. Armor absorbs a fraction
    of incoming damage proportional to ``1 - weapon.armor_penetration``, and
    is itself worn down by the absorbed amount.
    """
    base = float(weapon.damage_per_hit)
    base *= _distance_falloff(weapon, float(distance)) ** 1.2
    if through_cover:
        base *= float(cover_damage_multiplier)

    if base <= 0.0:
        return 0.0, 0.0

    if target_armor <= 0.0:
        return float(base), 0.0

    # Fraction of damage routed through armor.
    armor_share = max(0.0, 1.0 - float(weapon.armor_penetration))
    armor_dmg = base * armor_share
    hp_dmg = base - armor_dmg

    # Armor cannot absorb more than its remaining value -- spill the rest
    # over into HP damage.
    if armor_dmg > target_armor:
        overflow = armor_dmg - target_armor
        armor_dmg = float(target_armor)
        hp_dmg += overflow

    return float(hp_dmg), float(armor_dmg)


def sample_reaction_time(rng: np.random.Generator, cfg: CombatConfig) -> int:
    """Sample a reaction-time cooldown in ticks from ``[min, max]`` inclusive."""
    lo = int(cfg.reaction_time_min_ticks)
    hi = int(cfg.reaction_time_max_ticks)
    if hi <= lo:
        return lo
    return int(rng.integers(lo, hi + 1))


def shots_per_tick(weapon: WeaponStats, dt: float) -> float:
    """How many shots a weapon would resolve in a single sim tick.

    Useful for high-rate weapons where the engine should resolve more than
    one combat sample per tick (SMG/Rifle). Bounded to a sensible cap so a
    pathological config can't blow up the loop.
    """
    rate = max(0.0, float(weapon.fire_rate_hz))
    return float(min(6.0, rate * max(1e-3, dt)))


def angle_from_to(src: np.ndarray, dst: np.ndarray) -> float:
    """Signed radian angle from ``src`` to ``dst``."""
    return float(math.atan2(float(dst[1] - src[1]), float(dst[0] - src[0])))
