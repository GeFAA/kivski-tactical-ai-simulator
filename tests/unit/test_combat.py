"""Unit tests for ``kivski_sim.combat`` and ``kivski_sim.economy`` helpers."""

from __future__ import annotations

import numpy as np
from kivski_sim.combat import (
    compute_damage,
    compute_hit_probability,
    sample_reaction_time,
    shots_per_tick,
)
from kivski_sim.config import CombatConfig, EconomyConfig
from kivski_sim.economy import ARMOR_COST, apply_buy_choice, kill_reward
from kivski_sim.rng import RngHub
from kivski_sim.state import AgentState
from kivski_sim.types import (
    WEAPONS,
    BuyChoice,
    MicroAction,
    Side,
    Team,
    WeaponClass,
)

# ---------------------------------------------------------------------------
# compute_hit_probability
# ---------------------------------------------------------------------------


def test_hit_probability_decreases_with_distance() -> None:
    weapon = WEAPONS[WeaponClass.RIFLE]
    cfg = CombatConfig()
    near = compute_hit_probability(
        weapon,
        distance=5.0,
        attacker_micro=MicroAction.DEFAULT,
        target_micro=MicroAction.DEFAULT,
        base_acc_config=cfg,
    )
    mid = compute_hit_probability(
        weapon,
        distance=weapon.optimal_range,
        attacker_micro=MicroAction.DEFAULT,
        target_micro=MicroAction.DEFAULT,
        base_acc_config=cfg,
    )
    far = compute_hit_probability(
        weapon,
        distance=weapon.max_range - 1.0,
        attacker_micro=MicroAction.DEFAULT,
        target_micro=MicroAction.DEFAULT,
        base_acc_config=cfg,
    )
    out_of_range = compute_hit_probability(
        weapon,
        distance=weapon.max_range + 5.0,
        attacker_micro=MicroAction.DEFAULT,
        target_micro=MicroAction.DEFAULT,
        base_acc_config=cfg,
    )
    # Within optimal range -> hit prob equals the base at point-blank;
    # beyond max range -> zero.
    assert near >= mid
    assert mid >= far
    assert out_of_range == 0.0


def test_hit_probability_crouch_increases_chance() -> None:
    weapon = WEAPONS[WeaponClass.RIFLE]
    cfg = CombatConfig()
    p_default = compute_hit_probability(
        weapon,
        distance=10.0,
        attacker_micro=MicroAction.DEFAULT,
        target_micro=MicroAction.DEFAULT,
        base_acc_config=cfg,
    )
    p_crouch = compute_hit_probability(
        weapon,
        distance=10.0,
        attacker_micro=MicroAction.CROUCH_HOLD,
        target_micro=MicroAction.DEFAULT,
        base_acc_config=cfg,
    )
    assert p_crouch >= p_default


def test_hit_probability_sprinting_reduces_chance() -> None:
    weapon = WEAPONS[WeaponClass.RIFLE]
    cfg = CombatConfig()
    p_default = compute_hit_probability(
        weapon,
        distance=10.0,
        attacker_micro=MicroAction.DEFAULT,
        target_micro=MicroAction.DEFAULT,
        base_acc_config=cfg,
    )
    p_sprint = compute_hit_probability(
        weapon,
        distance=10.0,
        attacker_micro=MicroAction.SPRINT,
        target_micro=MicroAction.DEFAULT,
        base_acc_config=cfg,
    )
    assert p_sprint < p_default


def test_hit_probability_through_cover_reduced() -> None:
    weapon = WEAPONS[WeaponClass.RIFLE]
    cfg = CombatConfig()
    p_open = compute_hit_probability(
        weapon,
        distance=8.0,
        attacker_micro=MicroAction.DEFAULT,
        target_micro=MicroAction.DEFAULT,
        base_acc_config=cfg,
        through_cover=False,
    )
    p_cover = compute_hit_probability(
        weapon,
        distance=8.0,
        attacker_micro=MicroAction.DEFAULT,
        target_micro=MicroAction.DEFAULT,
        base_acc_config=cfg,
        through_cover=True,
    )
    assert p_cover < p_open


# ---------------------------------------------------------------------------
# compute_damage
# ---------------------------------------------------------------------------


def test_damage_with_armor_reduces_hp() -> None:
    weapon = WEAPONS[WeaponClass.RIFLE]
    hp_no_armor, armor_no = compute_damage(
        weapon, distance=5.0, target_armor=0.0, through_cover=False, cover_damage_multiplier=0.55
    )
    hp_with_armor, armor_with = compute_damage(
        weapon, distance=5.0, target_armor=100.0, through_cover=False, cover_damage_multiplier=0.55
    )
    assert armor_no == 0.0
    assert armor_with > 0.0
    assert hp_with_armor < hp_no_armor


def test_damage_through_cover_reduced() -> None:
    weapon = WEAPONS[WeaponClass.RIFLE]
    hp_open, _ = compute_damage(
        weapon, distance=5.0, target_armor=0.0, through_cover=False, cover_damage_multiplier=0.55
    )
    hp_cover, _ = compute_damage(
        weapon, distance=5.0, target_armor=0.0, through_cover=True, cover_damage_multiplier=0.55
    )
    assert hp_cover < hp_open


def test_damage_zero_at_max_range() -> None:
    weapon = WEAPONS[WeaponClass.RIFLE]
    hp, armor = compute_damage(
        weapon,
        distance=weapon.max_range + 0.1,
        target_armor=0.0,
        through_cover=False,
        cover_damage_multiplier=0.55,
    )
    assert hp == 0.0
    assert armor == 0.0


# ---------------------------------------------------------------------------
# Reaction time
# ---------------------------------------------------------------------------


def test_sample_reaction_time_within_bounds() -> None:
    cfg = CombatConfig()
    rng = RngHub(seed=42).channel("combat")
    samples = [sample_reaction_time(rng, cfg) for _ in range(200)]
    assert all(cfg.reaction_time_min_ticks <= s <= cfg.reaction_time_max_ticks for s in samples)
    # The samples should not all be identical (would mean a deterministic bug).
    assert len(set(samples)) > 1


def test_sample_reaction_time_deterministic() -> None:
    cfg = CombatConfig()
    a = [sample_reaction_time(RngHub(seed=42).channel("combat"), cfg) for _ in range(5)]
    b = [sample_reaction_time(RngHub(seed=42).channel("combat"), cfg) for _ in range(5)]
    assert a == b


def test_shots_per_tick_scales_with_rate() -> None:
    rifle = WEAPONS[WeaponClass.RIFLE]
    sniper = WEAPONS[WeaponClass.PRECISION]
    # 10 Hz tick rate -> dt=0.1, rifle ~ 8 shots/sec -> 0.8 shots/tick;
    # sniper ~ 1.2 shots/sec -> 0.12 shots/tick.
    assert shots_per_tick(rifle, 0.1) > shots_per_tick(sniper, 0.1)


# ---------------------------------------------------------------------------
# Economy: buy
# ---------------------------------------------------------------------------


def _make_agent(money: int = 800, side: Side = Side.ATTACKER) -> AgentState:
    return AgentState(
        agent_id=0,
        team=Team.YELLOW,
        side=side,
        pos=np.zeros(2, dtype=np.float32),
        vel=np.zeros(2, dtype=np.float32),
        money=money,
        weapon=WeaponClass.SIDEARM,
    )


def test_buy_insufficient_funds_fails() -> None:
    eco = EconomyConfig()
    agent = _make_agent(money=100)
    ok = apply_buy_choice(agent, BuyChoice.RIFLE, eco)
    assert not ok
    assert agent.weapon == WeaponClass.SIDEARM
    assert agent.money == 100


def test_buy_deducts_money() -> None:
    eco = EconomyConfig()
    agent = _make_agent(money=5000)
    ok = apply_buy_choice(agent, BuyChoice.RIFLE, eco)
    assert ok
    assert agent.weapon == WeaponClass.RIFLE
    assert agent.money == 5000 - WEAPONS[WeaponClass.RIFLE].cost
    assert agent.money_spent_match == WEAPONS[WeaponClass.RIFLE].cost


def test_buy_armor_sets_full_armor() -> None:
    eco = EconomyConfig()
    agent = _make_agent(money=1000)
    ok = apply_buy_choice(agent, BuyChoice.ARMOR, eco)
    assert ok
    assert agent.armor == 100.0
    assert agent.money == 1000 - ARMOR_COST


def test_buy_armor_when_already_armored_fails() -> None:
    eco = EconomyConfig()
    agent = _make_agent(money=2000)
    apply_buy_choice(agent, BuyChoice.ARMOR, eco)
    # Second purchase is a no-op (armor still at 100).
    ok2 = apply_buy_choice(agent, BuyChoice.ARMOR, eco)
    assert not ok2
    assert agent.armor == 100.0


def test_buy_same_weapon_twice_no_double_charge() -> None:
    eco = EconomyConfig()
    agent = _make_agent(money=5000)
    apply_buy_choice(agent, BuyChoice.RIFLE, eco)
    money_after_first = agent.money
    ok2 = apply_buy_choice(agent, BuyChoice.RIFLE, eco)
    assert not ok2
    assert agent.money == money_after_first


def test_buy_none_is_noop() -> None:
    eco = EconomyConfig()
    agent = _make_agent(money=5000)
    ok = apply_buy_choice(agent, BuyChoice.NONE, eco)
    assert not ok
    assert agent.money == 5000


# ---------------------------------------------------------------------------
# Economy: kill reward
# ---------------------------------------------------------------------------


def test_kill_reward_rifle() -> None:
    eco = EconomyConfig()
    assert kill_reward(WeaponClass.RIFLE, eco) == eco.reward_kill_rifle


def test_kill_reward_sidearm_uses_pistol_bucket() -> None:
    eco = EconomyConfig()
    assert kill_reward(WeaponClass.SIDEARM, eco) == eco.reward_kill_pistol


def test_kill_reward_smg() -> None:
    eco = EconomyConfig()
    assert kill_reward(WeaponClass.SMG, eco) == eco.reward_kill_smg


def test_kill_reward_sniper_low() -> None:
    eco = EconomyConfig()
    assert kill_reward(WeaponClass.PRECISION, eco) < kill_reward(WeaponClass.RIFLE, eco)
