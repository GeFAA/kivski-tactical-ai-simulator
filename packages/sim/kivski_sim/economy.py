"""Economy helpers: buy validation, kill rewards, and round-end payouts.

Pulled out of ``engine.py`` so that the engine remains focused on the per-tick
update loop. All functions mutate state in place (no hidden return values for
caller's bookkeeping) and are deterministic with respect to their arguments.
"""

from __future__ import annotations

from kivski_sim.config import EconomyConfig
from kivski_sim.state import AgentState, MatchState, TeamState
from kivski_sim.types import (
    BuyChoice,
    RoundOutcome,
    Side,
    Team,
    WEAPONS,
    WeaponClass,
)


__all__ = [
    "ARMOR_COST",
    "DEFUSE_KIT_COST",
    "apply_buy_choice",
    "kill_reward",
    "round_end_payouts",
]


ARMOR_COST: int = 650
DEFUSE_KIT_COST: int = 400


_BUY_TO_WEAPON: dict[BuyChoice, WeaponClass] = {
    BuyChoice.SIDEARM: WeaponClass.SIDEARM,
    BuyChoice.HEAVY_PISTOL: WeaponClass.HEAVY_PISTOL,
    BuyChoice.SMG: WeaponClass.SMG,
    BuyChoice.SHOTGUN: WeaponClass.SHOTGUN,
    BuyChoice.RIFLE: WeaponClass.RIFLE,
    BuyChoice.PRECISION: WeaponClass.PRECISION,
}


def kill_reward(weapon: WeaponClass, eco_cfg: EconomyConfig) -> int:
    """Return the cash a killer earns for fragging an enemy with ``weapon``."""
    if weapon == WeaponClass.RIFLE:
        return int(eco_cfg.reward_kill_rifle)
    if weapon in (WeaponClass.SIDEARM, WeaponClass.HEAVY_PISTOL):
        return int(eco_cfg.reward_kill_pistol)
    if weapon == WeaponClass.SMG:
        return int(eco_cfg.reward_kill_smg)
    if weapon == WeaponClass.SHOTGUN:
        return int(eco_cfg.reward_kill_smg)
    if weapon == WeaponClass.PRECISION:
        return int(eco_cfg.reward_kill_sniper)
    if weapon == WeaponClass.KNIFE:
        # Knife kills are flavorful but cheap.
        return int(eco_cfg.reward_kill_pistol)
    return int(eco_cfg.reward_kill_pistol)


def apply_buy_choice(
    agent: AgentState,
    choice: BuyChoice,
    eco_cfg: EconomyConfig,
) -> bool:
    """Validate and apply a buy choice in place.

    Returns ``True`` on a successful purchase, ``False`` if the agent could
    not afford it or the choice is a no-op. ``BuyChoice.NONE`` returns False
    silently.
    """
    if choice == BuyChoice.NONE:
        return False

    if choice == BuyChoice.ARMOR:
        if agent.armor >= 100.0:
            return False
        if agent.money < ARMOR_COST:
            return False
        agent.money -= ARMOR_COST
        agent.armor = 100.0
        agent.money_spent_match += ARMOR_COST
        return True

    weapon_cls = _BUY_TO_WEAPON.get(choice)
    if weapon_cls is None:
        return False

    stats = WEAPONS[weapon_cls]
    # Side-restricted weapons (currently none) would be rejected here.
    if stats.side_restricted != -1 and stats.side_restricted != int(agent.side):
        return False

    cost = int(stats.cost)
    if agent.money < cost:
        return False

    # If the agent already has that primary, do not double-charge.
    if agent.weapon == weapon_cls:
        return False

    agent.money -= cost
    agent.weapon = weapon_cls
    agent.money_spent_match += cost
    return True


def round_end_payouts(
    state: MatchState,
    outcome: RoundOutcome,
    winning_side: Side,
    eco_cfg: EconomyConfig,
) -> None:
    """Pay out money to all agents and update team loss streaks.

    Behavior:

    * Winning team gets the flat ``reward_round_win`` and resets its loss
      streak to zero.
    * Losing team gets ``reward_round_loss_base + streak * increment`` clipped
      to ``reward_round_loss_max`` and increments its streak.
    * If the bomb was planted, every attacker who was alive at plant time
      (approximated as: side==ATTACKER agents still alive when the round
      ends OR who were the carrier/planter) gets the plant bonus.
    * The defuser receives the defuse bonus; all surviving attackers receive
      the detonate bonus when the bomb detonated.
    """
    # Determine winning / losing team via side mapping.
    side_to_team: dict[Side, Team] = {}
    for team_state in state.teams.values():
        side_to_team[team_state.side] = team_state.team

    losing_side = Side.DEFENDER if winning_side == Side.ATTACKER else Side.ATTACKER
    win_team = side_to_team.get(winning_side)
    lose_team = side_to_team.get(losing_side)

    if win_team is not None:
        win_ts: TeamState = state.teams[win_team]
        win_ts.consecutive_losses = 0
        for a in state.agents_on_team(win_team):
            a.money = min(a.money + int(eco_cfg.reward_round_win), 16000)

    if lose_team is not None:
        lose_ts: TeamState = state.teams[lose_team]
        bonus = int(eco_cfg.reward_round_loss_base) + int(eco_cfg.reward_round_loss_increment) * int(
            lose_ts.consecutive_losses
        )
        bonus = min(bonus, int(eco_cfg.reward_round_loss_max))
        for a in state.agents_on_team(lose_team):
            a.money = min(a.money + bonus, 16000)
        lose_ts.consecutive_losses += 1

    # Bomb-related bonuses (independent of round outcome -- planters keep
    # their reward even if defenders defuse).
    bomb = state.bomb
    if bomb.site is not None:
        # If a plant happened, reward every attacker (rough heuristic -- in a
        # full game we'd track who actually contributed).
        for a in state.agents:
            if a.side == Side.ATTACKER:
                a.money = min(a.money + int(eco_cfg.reward_bomb_plant), 16000)

    if outcome == RoundOutcome.BOMB_DEFUSED and bomb.defuser >= 0:
        for a in state.agents:
            if a.agent_id == bomb.defuser:
                a.money = min(a.money + int(eco_cfg.reward_bomb_defuse), 16000)
                break

    if outcome == RoundOutcome.BOMB_DETONATED:
        for a in state.agents:
            if a.side == Side.ATTACKER and a.alive:
                a.money = min(a.money + int(eco_cfg.reward_bomb_detonate), 16000)
