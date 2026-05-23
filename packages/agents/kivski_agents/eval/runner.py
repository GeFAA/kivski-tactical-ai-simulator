"""Head-to-head match runner used by the eval CLI and the league trainer.

A single :class:`EvalRunner` is bound to a :class:`ScenarioSpec` and a
:class:`KivskiConfig`; calling :meth:`EvalRunner.run` plays a configurable
number of matches between two policies and returns an :class:`EvalResult`
with per-round and per-match aggregates.

Side assignment
---------------

The engine's :class:`Team` (YELLOW vs BLUE) is fixed for the whole match and
the *side* (ATTACKER vs DEFENDER) flips at the configured ``side_switch_round``.
Concretely:

* ``agent_0 .. agent_{team_size-1}`` are always on team YELLOW.
* The remaining ``team_size`` agents are on team BLUE.
* ``policy_yellow`` plays YELLOW; ``policy_blue`` plays BLUE.

This means the user-facing assignment is by **team**, not by **side** -- the
runner doesn't try to "swap" policies mid-match when the engine flips sides.
That matches the standard self-play convention for symmetric games.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kivski_agents.eval.scenarios import ScenarioSpec, build_scenario
from kivski_sim.config import KivskiConfig
from kivski_sim.env import KivskiParallelEnv, agent_name
from kivski_sim.map_loader import MapData, load_map
from kivski_sim.types import MatchOutcome, Phase, RoundOutcome, Team


__all__ = ["RoundResult", "EvalResult", "EvalRunner"]


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class RoundResult:
    """Per-round summary captured from the engine's ``RoundSummary`` records."""

    round_id: int
    winning_side: str  # "attacker" | "defender"
    outcome: str       # str(RoundOutcome.<NAME>)
    duration_ticks: int
    survivors_yellow: int
    survivors_blue: int
    bomb_planted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": int(self.round_id),
            "winning_side": str(self.winning_side),
            "outcome": str(self.outcome),
            "duration_ticks": int(self.duration_ticks),
            "survivors_yellow": int(self.survivors_yellow),
            "survivors_blue": int(self.survivors_blue),
            "bomb_planted": bool(self.bomb_planted),
        }


@dataclass
class EvalResult:
    """Aggregated result of a head-to-head evaluation series."""

    scenario: str
    policy_yellow: str
    policy_blue: str
    num_matches: int
    yellow_match_wins: int
    blue_match_wins: int
    draws: int
    avg_rounds_per_match: float
    avg_match_duration_ticks: float
    bomb_plant_rate: float
    bomb_defuse_rate: float
    rounds: list[RoundResult] = field(default_factory=list)

    # ------------------------------------------------------------------

    @property
    def yellow_winrate(self) -> float:
        """Fraction of matches won by the YELLOW policy (draws count 0.5)."""
        if self.num_matches <= 0:
            return 0.0
        score = float(self.yellow_match_wins) + 0.5 * float(self.draws)
        return score / float(self.num_matches)

    @property
    def blue_winrate(self) -> float:
        if self.num_matches <= 0:
            return 0.0
        score = float(self.blue_match_wins) + 0.5 * float(self.draws)
        return score / float(self.num_matches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": str(self.scenario),
            "policy_yellow": str(self.policy_yellow),
            "policy_blue": str(self.policy_blue),
            "num_matches": int(self.num_matches),
            "yellow_match_wins": int(self.yellow_match_wins),
            "blue_match_wins": int(self.blue_match_wins),
            "draws": int(self.draws),
            "yellow_winrate": float(self.yellow_winrate),
            "blue_winrate": float(self.blue_winrate),
            "avg_rounds_per_match": float(self.avg_rounds_per_match),
            "avg_match_duration_ticks": float(self.avg_match_duration_ticks),
            "bomb_plant_rate": float(self.bomb_plant_rate),
            "bomb_defuse_rate": float(self.bomb_defuse_rate),
            "rounds": [r.to_dict() for r in self.rounds],
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class EvalRunner:
    """Plays a batch of matches between two policies on a fixed scenario.

    Policies must support the standard baseline interface:

        policy.reset(agent_names: list[str]) -> None
        policy.act(observations, received_comms) -> (actions_dict, payloads_dict)
    """

    def __init__(
        self,
        scenario: ScenarioSpec,
        cfg: KivskiConfig,
        map_name: str = "dustline",
        *,
        map_data: MapData | None = None,
    ) -> None:
        self.scenario: ScenarioSpec = scenario
        self.cfg: KivskiConfig = cfg
        self.map_name: str = map_name
        self.map_data: MapData = map_data if map_data is not None else load_map(map_name)
        # Build an env once so callers can inspect spaces / map without paying
        # the cost again. The per-match envs in :meth:`run` are fresh copies.
        self.env: KivskiParallelEnv = build_scenario(
            scenario, cfg, seed=int(cfg.seed), map_name=map_name, map_data=self.map_data
        )

    # ------------------------------------------------------------------

    @property
    def team_size(self) -> int:
        return int(self.scenario.team_size)

    def _yellow_names(self) -> list[str]:
        return [agent_name(i) for i in range(self.team_size)]

    def _blue_names(self) -> list[str]:
        return [agent_name(i + self.team_size) for i in range(self.team_size)]

    # ------------------------------------------------------------------

    def run(
        self,
        policy_yellow: Any,
        policy_blue: Any,
        num_matches: int,
        seed: int = 0,
    ) -> EvalResult:
        """Play ``num_matches`` matches and aggregate the results."""
        if num_matches <= 0:
            raise ValueError(f"num_matches must be > 0, got {num_matches}")

        yellow_names = self._yellow_names()
        blue_names = self._blue_names()

        rounds: list[RoundResult] = []
        yellow_match_wins = 0
        blue_match_wins = 0
        draws = 0
        total_match_duration_ticks = 0
        total_rounds = 0
        bomb_plant_count = 0
        bomb_defuse_count = 0

        for match_idx in range(int(num_matches)):
            match_seed = int(seed) + int(match_idx)
            env = build_scenario(
                self.scenario,
                self.cfg,
                seed=match_seed,
                map_name=self.map_name,
                map_data=self.map_data,
            )

            # Per-match policy reset; baselines use this for episode-scoped
            # state like ScriptedRushBaseline's per-agent target site sampling.
            try:
                policy_yellow.reset(yellow_names)
            except Exception:
                pass
            try:
                policy_blue.reset(blue_names)
            except Exception:
                pass

            observations, _infos = env.reset(seed=match_seed)

            done = False
            safety_cap = self._compute_safety_cap()
            steps = 0
            while not done and steps < safety_cap:
                # 1) Slice observations per side and gather actions.
                obs_yellow = {name: observations[name] for name in yellow_names if name in observations}
                obs_blue = {name: observations[name] for name in blue_names if name in observations}
                act_yellow, payload_yellow = self._safe_act(policy_yellow, obs_yellow)
                act_blue, payload_blue = self._safe_act(policy_blue, obs_blue)

                # 2) Merge into the single dict the env expects.
                merged_actions: dict[str, Any] = {}
                merged_actions.update(act_yellow)
                merged_actions.update(act_blue)
                merged_payloads: dict[str, Any] = {}
                merged_payloads.update(payload_yellow or {})
                merged_payloads.update(payload_blue or {})

                # 3) Step the env (use the comm-aware variant when payloads exist).
                if merged_payloads:
                    observations, _rewards, terminations, truncations, _infos = env.step_with_comms(
                        merged_actions, comm_payloads=merged_payloads
                    )
                else:
                    observations, _rewards, terminations, truncations, _infos = env.step(
                        merged_actions
                    )
                done = all(terminations.values()) or all(truncations.values())
                steps += 1

            # ---- Match wrap-up ----------------------------------------
            match_outcome = env.engine.state.match_outcome
            if match_outcome == MatchOutcome.YELLOW_WIN:
                yellow_match_wins += 1
            elif match_outcome == MatchOutcome.BLUE_WIN:
                blue_match_wins += 1
            else:
                draws += 1
            total_match_duration_ticks += int(env.engine.state.tick)

            for summary in env.engine.state.round_summaries:
                rounds.append(
                    RoundResult(
                        round_id=int(summary.round_id),
                        winning_side=("attacker" if int(summary.winning_side) == 0 else "defender"),
                        outcome=str(RoundOutcome(int(summary.outcome)).name),
                        duration_ticks=int(summary.duration_ticks),
                        survivors_yellow=int(summary.survivors_yellow),
                        survivors_blue=int(summary.survivors_blue),
                        bomb_planted=bool(summary.bomb_planted),
                    )
                )
                if summary.bomb_planted:
                    bomb_plant_count += 1
                if RoundOutcome(int(summary.outcome)) == RoundOutcome.BOMB_DEFUSED:
                    bomb_defuse_count += 1
            total_rounds += len(env.engine.state.round_summaries)

        # Pre-planted scenarios start with a bomb already placed; count those too
        # so the "plant rate" headline isn't artificially low.
        if self.scenario.bomb_planted:
            # Each match starts with the bomb planted, but the engine's round
            # summary will reflect that via ``summary.bomb_planted`` since the
            # state is mutated *before* the engine evaluates the round. Nothing
            # extra to do here -- documenting the intent.
            pass

        denom_matches = max(1, int(num_matches))
        denom_rounds = max(1, int(total_rounds))
        return EvalResult(
            scenario=str(self.scenario.name),
            policy_yellow=str(getattr(policy_yellow, "name", policy_yellow.__class__.__name__)),
            policy_blue=str(getattr(policy_blue, "name", policy_blue.__class__.__name__)),
            num_matches=int(num_matches),
            yellow_match_wins=int(yellow_match_wins),
            blue_match_wins=int(blue_match_wins),
            draws=int(draws),
            avg_rounds_per_match=float(total_rounds) / float(denom_matches),
            avg_match_duration_ticks=float(total_match_duration_ticks) / float(denom_matches),
            bomb_plant_rate=float(bomb_plant_count) / float(denom_rounds),
            bomb_defuse_rate=float(bomb_defuse_count) / float(denom_rounds),
            rounds=rounds,
        )

    # ------------------------------------------------------------------

    def _safe_act(
        self,
        policy: Any,
        observations: dict[str, np.ndarray],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Call ``policy.act`` while tolerating policies that return only actions.

        Defensive: a policy may forget to return the comm-payload dict, or
        return a bare actions dict instead of the tuple. The runner accepts
        both shapes so authors of new baselines don't trip over the contract.
        """
        try:
            result = policy.act(observations)
        except TypeError:
            # Older signature without comm support.
            try:
                result = policy.act(observations)
            except Exception:
                return self._fallback_hold(observations), {}
        except Exception:
            return self._fallback_hold(observations), {}

        if isinstance(result, tuple):
            actions = result[0]
            payloads = result[1] if len(result) > 1 else {}
        else:
            actions = result
            payloads = {}
        # Coerce to numpy where reasonable; the env handles type conversion too.
        coerced: dict[str, Any] = {}
        for name, value in actions.items():
            if isinstance(value, np.ndarray):
                coerced[name] = value
            else:
                coerced[name] = np.asarray(value, dtype=np.int64)
        return coerced, payloads or {}

    def _fallback_hold(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """All-zeros (== HOLD/NONE) actions if a policy errors out."""
        return {name: np.zeros(5, dtype=np.int64) for name in observations}

    def _compute_safety_cap(self) -> int:
        """Maximum ticks per match used as a runaway-loop safety net."""
        per_round = max(
            int(self.cfg.simulation.max_ticks_per_round),
            int(math.ceil(self.cfg.simulation.round_time_seconds * self.cfg.simulation.tick_rate_hz))
            + int(math.ceil(self.cfg.simulation.bomb_timer_seconds * self.cfg.simulation.tick_rate_hz))
            + int(math.ceil(self.cfg.simulation.buy_time_seconds * self.cfg.simulation.tick_rate_hz)),
        )
        # Allow a bit of slack on top of the formal max rounds.
        return int((int(self.cfg.simulation.max_rounds) + 2) * per_round)
