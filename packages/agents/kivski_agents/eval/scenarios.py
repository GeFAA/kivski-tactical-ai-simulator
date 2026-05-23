"""Standardised benchmark scenarios for evaluating Kivski policies.

A scenario is a tightly-specified set of starting conditions that exercises
a particular skill (full-pistol economy management, full-buy aim duel,
2v3 retake, save round, ...). Keeping these as plain dataclasses makes the
eval suite trivially extensible: drop in a new ``ScenarioSpec`` and the
runner picks it up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kivski_sim.config import KivskiConfig
from kivski_sim.env import KivskiParallelEnv
from kivski_sim.map_loader import MapData, load_map
from kivski_sim.state import BombState
from kivski_sim.types import BombPhase, Phase, Side


__all__ = [
    "ScenarioSpec",
    "EvalScenario",
    "ALL_SCENARIOS",
    "build_scenario",
]


# ---------------------------------------------------------------------------
# Scenario specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioSpec:
    """Declarative description of an evaluation scenario.

    Attributes:
        name: Unique short identifier (used in CLI args, output paths, ...).
        team_size: Agents per team (5 for the canonical mode).
        attackers_alive: If set, kill the rest of the attacker side after reset
            so the scenario starts with this many alive attackers. Used for
            retake / clutch scenarios.
        defenders_alive: Same as above for the defending side.
        starting_money: If set, override every agent's money after reset.
        bomb_planted: If True, immediately plant the bomb at ``initial_site``
            and transition the phase to ``POST_PLANT``.
        initial_site: ``"A"`` or ``"B"``; required when ``bomb_planted``
            is True. Otherwise ignored.
        max_matches: Default number of matches to play for this scenario when
            invoked from the CLI without ``--matches``.
    """

    name: str
    team_size: int
    attackers_alive: int | None = None
    defenders_alive: int | None = None
    starting_money: int | None = None
    bomb_planted: bool = False
    initial_site: str | None = None
    max_matches: int = 20


# Backwards-compatible alias matching the older module path used in
# ``kivski_sim.config.EvaluationConfig``.
EvalScenario = ScenarioSpec


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------


ALL_SCENARIOS: list[ScenarioSpec] = [
    ScenarioSpec(name="full_pistol", team_size=5, starting_money=800),
    ScenarioSpec(name="full_buy", team_size=5, starting_money=5000),
    ScenarioSpec(
        name="retake_2v3",
        team_size=5,
        attackers_alive=3,
        defenders_alive=2,
        bomb_planted=True,
        initial_site="A",
    ),
    ScenarioSpec(name="save_round", team_size=5, starting_money=0),
    ScenarioSpec(name="default_5v5", team_size=5, starting_money=2700),
]


# ---------------------------------------------------------------------------
# Scenario instantiation
# ---------------------------------------------------------------------------


def _override_team_size(cfg: KivskiConfig, team_size: int) -> KivskiConfig:
    """Return a copy of ``cfg`` with the simulation team size overridden."""
    raw = cfg.model_dump()
    raw.setdefault("simulation", {})["team_size"] = int(team_size)
    return KivskiConfig.model_validate(raw)


def _kill_excess(env: KivskiParallelEnv, side: Side, keep_alive: int) -> None:
    """Set ``alive = False`` on agents of ``side`` so only ``keep_alive`` remain.

    Agents are killed in deterministic ``agent_id`` order so the choice is
    reproducible across runs.
    """
    state = env.engine.state
    same_side = [a for a in state.agents if a.side == side]
    same_side.sort(key=lambda a: int(a.agent_id))
    keep = max(0, min(int(keep_alive), len(same_side)))
    to_kill = same_side[keep:]
    for a in to_kill:
        a.alive = False
        a.hp = 0.0
        # Reset bomb carrying so we don't lose the bomb to a dead agent.
        if a.has_bomb:
            a.has_bomb = False
            # Drop the bomb at the dead agent's position.
            state.bomb.phase = BombPhase.DROPPED
            state.bomb.carrier = -1
            state.bomb.pos = np.array(a.pos, dtype=np.float32)


def _override_starting_money(env: KivskiParallelEnv, money: int) -> None:
    """Set every alive agent's money. Useful for forcing eco / full-buy rounds."""
    capped = max(0, int(money))
    for a in env.engine.state.agents:
        a.money = capped


def _force_plant(env: KivskiParallelEnv, site_name: str) -> None:
    """Pre-plant the bomb at the centroid of ``site_name`` and transition phases.

    Used for retake / post-plant scenarios. The plant is "free" -- no plant
    animation, no plant sound, no plant reward to the attackers. The bomb
    timer is initialized to the configured full duration so the runner gets a
    deterministic post-plant window.
    """
    state = env.engine.state
    site = env.map.bombsites.get(site_name.upper())
    if site is None:
        raise ValueError(
            f"Cannot pre-plant: bombsite {site_name!r} not present on map "
            f"{env.map.name!r} (available: {sorted(env.map.bombsites)})"
        )
    # Carrier (if any) drops the bomb.
    for a in state.agents:
        if a.has_bomb:
            a.has_bomb = False

    bomb: BombState = state.bomb
    bomb.phase = BombPhase.PLANTED
    bomb.carrier = -1
    bomb.pos = np.array(site.center, dtype=np.float32)
    bomb.plant_progress = 1.0
    bomb.defuse_progress = 0.0
    bomb.defuser = -1
    bomb.time_since_plant = 0.0
    bomb.site = site_name.upper()
    # Transition phase + reset the post-plant timer.
    state.phase = Phase.POST_PLANT
    state.phase_ticks_remaining = int(
        round(env.engine.config.simulation.bomb_timer_seconds
              * env.engine.config.simulation.tick_rate_hz)
    )


def build_scenario(
    spec: ScenarioSpec,
    cfg: KivskiConfig,
    seed: int,
    map_name: str = "dustline",
    *,
    map_data: MapData | None = None,
) -> KivskiParallelEnv:
    """Create a :class:`KivskiParallelEnv` pre-configured for ``spec``.

    Applies the scenario overrides in this order:

    1. Override team size (if it differs from the config) before constructing
       the env so the action / observation spaces are sized correctly.
    2. Reset the env (which runs the engine's normal initialization).
    3. Override starting money on each agent (if requested).
    4. Force-plant the bomb (if requested), which also transitions the phase.
    5. Kill excess agents on either side (if requested) so the scenario starts
       with the desired number of alive players per side.

    Returns:
        A fresh :class:`KivskiParallelEnv` ready for ``env.step(...)``.
    """
    final_cfg = cfg
    if int(spec.team_size) != int(cfg.simulation.team_size):
        final_cfg = _override_team_size(cfg, spec.team_size)

    md = map_data if map_data is not None else load_map(map_name)
    env = KivskiParallelEnv(
        config=final_cfg, map_name=map_name, seed=int(seed), map_data=md
    )
    env.reset(seed=int(seed))

    # Apply scenario overrides after reset so the engine is fully initialized.
    if spec.starting_money is not None:
        _override_starting_money(env, int(spec.starting_money))

    if spec.bomb_planted:
        site = spec.initial_site or "A"
        _force_plant(env, site)

    if spec.attackers_alive is not None:
        _kill_excess(env, Side.ATTACKER, int(spec.attackers_alive))
    if spec.defenders_alive is not None:
        _kill_excess(env, Side.DEFENDER, int(spec.defenders_alive))

    return env
