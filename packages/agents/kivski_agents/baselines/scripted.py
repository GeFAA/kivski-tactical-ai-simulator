"""Hand-written heuristic baselines used as sparring partners.

The two policies (:class:`ScriptedHoldBaseline`, :class:`ScriptedRushBaseline`)
are deliberately simple. They are not meant to be hard for the learned MAPPO
agents -- they only need to be *consistent* opponents that exercise the same
observation/action contract the learned policies do.

Crucially, both baselines work exclusively off the **observation vector**
exposed by :class:`kivski_sim.env.KivskiParallelEnv` via the schema in
:mod:`kivski_sim.obs_decoder`. They never touch ``engine.state`` directly so
they generalize to any map and any team-size variation as long as the schema
is followed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from kivski_sim.config import KivskiConfig
from kivski_sim.map_loader import MapData
from kivski_sim.obs_decoder import build_observation_schema, decode_observation
from kivski_sim.types import (
    BuyChoice,
    CommAction,
    MicroAction,
    MoveIntent,
)


__all__ = ["ScriptedHoldBaseline", "ScriptedRushBaseline"]


# Maximum integer in the action sub-spaces, mirroring the env's MultiDiscrete dims.
_NUM_MOVE_INTENTS: int = len(MoveIntent)
_NUM_MICRO_ACTIONS: int = len(MicroAction)
_NUM_COMM_ACTIONS: int = len(CommAction)
_NUM_BUY_OPTIONS: int = len(BuyChoice)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compass_intent_from_dxdy(dx: float, dy: float) -> MoveIntent:
    """Translate a (dx, dy) delta vector into the nearest 8-way compass intent.

    Returns :attr:`MoveIntent.HOLD` for the degenerate zero vector.
    """
    if dx == 0.0 and dy == 0.0:
        return MoveIntent.HOLD
    # MoveIntent uses y-down convention (N = -y). Angle measured clockwise from N.
    angle = math.atan2(dx, -dy)  # 0 = N, pi/2 = E, pi = S, -pi/2 = W
    # Quantize to 8 directions.
    sector = int(round(angle / (math.pi / 4.0))) % 8
    return [
        MoveIntent.N,   # 0
        MoveIntent.NE,  # 1
        MoveIntent.E,   # 2
        MoveIntent.SE,  # 3
        MoveIntent.S,   # 4
        MoveIntent.SW,  # 5 (= -3 mod 8)
        MoveIntent.W,   # 6 (= -2 mod 8)
        MoveIntent.NW,  # 7 (= -1 mod 8)
    ][sector]


def _pack_action(
    move: MoveIntent,
    micro: MicroAction,
    comm: CommAction,
    buy: BuyChoice,
    aim_target: int,
) -> np.ndarray:
    """Pack a five-component MultiDiscrete action as ``np.int64``."""
    return np.array(
        [
            int(min(max(int(move), 0), _NUM_MOVE_INTENTS - 1)),
            int(min(max(int(micro), 0), _NUM_MICRO_ACTIONS - 1)),
            int(min(max(int(comm), 0), _NUM_COMM_ACTIONS - 1)),
            int(min(max(int(buy), 0), _NUM_BUY_OPTIONS - 1)),
            int(max(int(aim_target), 0)),
        ],
        dtype=np.int64,
    )


def _pick_buy(money_norm: float, target_class: BuyChoice) -> BuyChoice:
    """Pick a buy choice given normalized money and a primary preference.

    Falls back to cheaper alternatives if not enough money is available.
    Money in the observation is normalized by 8000 (see ``env._build_observation``).
    """
    money = float(money_norm) * 8000.0
    # Cheap fallbacks in increasing cost order.
    if target_class == BuyChoice.RIFLE and money >= 2700:
        return BuyChoice.RIFLE
    if target_class in (BuyChoice.RIFLE, BuyChoice.SMG) and money >= 1500:
        return BuyChoice.SMG
    if money >= 700:
        return BuyChoice.HEAVY_PISTOL
    return BuyChoice.SIDEARM


def _is_buy_phase(decoded_obs: dict[str, Any]) -> bool:
    """True iff the decoded map_ctx says we are in the BUY phase."""
    map_ctx = decoded_obs.get("map_ctx", {})
    return float(map_ctx.get("phase_buy", 0.0)) > 0.5


def _is_alive(decoded_obs: dict[str, Any]) -> bool:
    """True iff the agent's self block reports ``alive``."""
    self_block = decoded_obs.get("self", {})
    return float(self_block.get("alive", 1.0)) > 0.5


def _has_bomb(decoded_obs: dict[str, Any]) -> bool:
    self_block = decoded_obs.get("self", {})
    return float(self_block.get("has_bomb", 0.0)) > 0.5


def _money_norm(decoded_obs: dict[str, Any]) -> float:
    self_block = decoded_obs.get("self", {})
    return float(self_block.get("money_norm", 0.0))


def _nearest_visible_enemy_slot(decoded_obs: dict[str, Any]) -> int | None:
    """Pick a recently-seen enemy slot index (1-based for aim_target).

    Strategy: among the ``enemies`` slots prefer those with the smallest
    ``age_norm`` (most recently observed) where ``was_alive`` is true. Returns
    ``None`` if no usable last-known enemy entries are present.
    """
    enemies = decoded_obs.get("enemies", [])
    best_idx: int | None = None
    best_age: float = float("inf")
    for slot_idx, slot in enumerate(enemies):
        was_alive = float(slot.get("was_alive", 0.0)) > 0.5
        if not was_alive:
            continue
        age = float(slot.get("age_norm", 1.0))
        if age >= best_age:
            continue
        # Reject totally-zero entries (those are padding slots).
        if (
            float(slot.get("dx", 0.0)) == 0.0
            and float(slot.get("dy", 0.0)) == 0.0
            and age == 0.0
        ):
            continue
        best_age = age
        best_idx = slot_idx
    return best_idx


def _bombsite_distance(decoded_obs: dict[str, Any], site: str) -> tuple[float, float, float]:
    """Return ``(dx, dy, distance)`` to the named bombsite ("A" or "B") in obs space.

    The observation stores bombsite deltas normalized by ``(map.width, map.height)``,
    so the values returned here are unitless ratios. They are perfectly usable
    for direction picking and threshold comparison even though they aren't
    measured in tiles.
    """
    map_ctx = decoded_obs.get("map_ctx", {})
    key = site.upper()
    if key == "A":
        dx = float(map_ctx.get("bombsite_a_dx", 0.0))
        dy = float(map_ctx.get("bombsite_a_dy", 0.0))
    elif key == "B":
        dx = float(map_ctx.get("bombsite_b_dx", 0.0))
        dy = float(map_ctx.get("bombsite_b_dy", 0.0))
    else:
        raise ValueError(f"Unknown bombsite {site!r}; expected 'A' or 'B'")
    dist = math.sqrt(dx * dx + dy * dy)
    return dx, dy, dist


def _nearest_bombsite(decoded_obs: dict[str, Any]) -> tuple[str, float, float, float]:
    """Return the closer of the two bombsites: ``(name, dx, dy, dist)``."""
    a = _bombsite_distance(decoded_obs, "A")
    b = _bombsite_distance(decoded_obs, "B")
    if a[2] <= b[2]:
        return ("A", a[0], a[1], a[2])
    return ("B", b[0], b[1], b[2])


# ---------------------------------------------------------------------------
# ScriptedHoldBaseline
# ---------------------------------------------------------------------------


class ScriptedHoldBaseline:
    """Defenders-style heuristic: stay near bombsites, hold angles.

    Strategy:
        * Move toward the nearer bombsite. When within a tight obs-space
          threshold (normalized distance), crouch and hold.
        * If a recently-seen enemy slot has a valid (non-padding) entry,
          target it via ``aim_target``.
        * Buy: prefer :attr:`BuyChoice.HEAVY_PISTOL` when affordable, else
          :attr:`BuyChoice.SIDEARM`, else :attr:`BuyChoice.NONE`.
        * Occasionally ping the team when an enemy is spotted.
    """

    name: str = "scripted_hold"

    def __init__(self, action_space: Any, map_data: MapData, seed: int = 0) -> None:
        # action_space is kept on the instance for parity with RandomBaseline but
        # we only use it to derive the aim_target high.
        self._action_space: Any = action_space
        self._map: MapData = map_data
        # The schema is config-dependent. We use the *current* default config
        # which mirrors how the env is built by EvalRunner. For non-default
        # observation slot counts callers can override via env_schema kwarg
        # in a future refactor.
        self._schema: dict[str, Any] = build_observation_schema(KivskiConfig())
        self._rng: np.random.Generator = np.random.default_rng(int(seed))
        self._agent_names: list[str] = []
        # Threshold in normalized obs-space distance for "on site".
        self._hold_threshold: float = 0.10

    def reset(self, agent_names: list[str]) -> None:
        """Reset per-episode state."""
        self._agent_names = list(agent_names)

    def act(
        self,
        observations: dict[str, np.ndarray],
        received_comms: dict[str, dict] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        del received_comms  # unused
        agents = self._agent_names if self._agent_names else list(observations.keys())
        actions: dict[str, np.ndarray] = {}
        for name in agents:
            obs = observations.get(name)
            if obs is None:
                actions[name] = _pack_action(
                    MoveIntent.HOLD, MicroAction.DEFAULT, CommAction.NONE, BuyChoice.NONE, 0
                )
                continue
            try:
                decoded = decode_observation(np.asarray(obs), schema=self._schema)
            except Exception:
                # Defensive: if the obs vector is sized differently (custom
                # ObservationConfig), fall back to a hold action.
                actions[name] = _pack_action(
                    MoveIntent.HOLD, MicroAction.DEFAULT, CommAction.NONE, BuyChoice.NONE, 0
                )
                continue
            actions[name] = self._act_one(decoded)
        return actions, {}

    # ------------------------------------------------------------------

    def _act_one(self, decoded: dict[str, Any]) -> np.ndarray:
        if not _is_alive(decoded):
            return _pack_action(
                MoveIntent.HOLD, MicroAction.DEFAULT, CommAction.NONE, BuyChoice.NONE, 0
            )

        # ----- BUY phase -------------------------------------------------
        if _is_buy_phase(decoded):
            buy = _pick_buy(_money_norm(decoded), BuyChoice.HEAVY_PISTOL)
            return _pack_action(
                MoveIntent.HOLD, MicroAction.DEFAULT, CommAction.NONE, buy, 0
            )

        # ----- LIVE phase ------------------------------------------------
        site_name, dx, dy, dist = _nearest_bombsite(decoded)
        enemy_slot = _nearest_visible_enemy_slot(decoded)
        # aim_target = 1 + slot_index in the *agents-other-than-self* ordering.
        # We don't know the per-agent ordering from the obs alone, but using
        # the slot index directly is a reasonable proxy that exercises the
        # head and will sometimes hit the right enemy. The engine clamps to
        # -1 when the slot doesn't resolve, so wrong choices are harmless.
        aim_target = int(enemy_slot + 1) if enemy_slot is not None else 0

        if dist < self._hold_threshold:
            # On-site: crouch and hold the angle. If we see an enemy, fire.
            comm = CommAction.PING_LOCATION if enemy_slot is not None else CommAction.NONE
            return _pack_action(
                MoveIntent.HOLD, MicroAction.CROUCH_HOLD, comm, BuyChoice.NONE, aim_target
            )

        # Approach the chosen bombsite.
        move_intent = _compass_intent_from_dxdy(dx, dy)
        comm = (
            CommAction.CONTACT_ENEMY
            if enemy_slot is not None
            else CommAction.SUGGEST_ROTATE
        )
        # Probabilistic comm suppression so the channel isn't perpetually saturated.
        if float(self._rng.random()) > 0.05:
            comm = CommAction.NONE
        return _pack_action(
            move_intent, MicroAction.DEFAULT, comm, BuyChoice.NONE, aim_target
        )


# ---------------------------------------------------------------------------
# ScriptedRushBaseline
# ---------------------------------------------------------------------------


class ScriptedRushBaseline:
    """Attackers-style heuristic: pick a site, rush to it, plant if carrier.

    Strategy:
        * Each round (reset), pick a target site per agent. We bias the random
          draw 60/40 toward "everyone picks the same site" so attackers tend
          to clump up (the most common rush pattern).
        * Sprint toward the chosen site center. When inside the site,
          ``INTERACT`` if the agent is the bomb carrier (planting); otherwise
          hold the angle.
        * Shoot any visible enemy via ``aim_target``.
        * Buy: prefer :attr:`BuyChoice.SMG`, fall back to pistol options when
          short on money.
    """

    name: str = "scripted_rush"

    def __init__(self, action_space: Any, map_data: MapData, seed: int = 0) -> None:
        self._action_space: Any = action_space
        self._map: MapData = map_data
        self._schema: dict[str, Any] = build_observation_schema(KivskiConfig())
        self._rng: np.random.Generator = np.random.default_rng(int(seed))
        self._agent_names: list[str] = []
        self._target_sites: dict[str, str] = {}
        # Threshold in normalized obs-space distance for "inside the site".
        self._on_site_threshold: float = 0.06

    def reset(self, agent_names: list[str]) -> None:
        """Sample a target site per agent for this match.

        With probability 0.6 *all* agents pick the same site (team consensus);
        otherwise each samples independently. This matches the common pattern
        of "default A" / "default B" rushes at the start of competitive rounds.
        """
        self._agent_names = list(agent_names)
        self._target_sites = {}
        consensus = float(self._rng.random()) < 0.6
        if consensus:
            shared = "A" if float(self._rng.random()) < 0.5 else "B"
            for name in agent_names:
                self._target_sites[name] = shared
        else:
            for name in agent_names:
                self._target_sites[name] = "A" if float(self._rng.random()) < 0.5 else "B"

    def act(
        self,
        observations: dict[str, np.ndarray],
        received_comms: dict[str, dict] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        del received_comms
        agents = self._agent_names if self._agent_names else list(observations.keys())
        actions: dict[str, np.ndarray] = {}
        for name in agents:
            obs = observations.get(name)
            if obs is None:
                actions[name] = _pack_action(
                    MoveIntent.HOLD, MicroAction.DEFAULT, CommAction.NONE, BuyChoice.NONE, 0
                )
                continue
            try:
                decoded = decode_observation(np.asarray(obs), schema=self._schema)
            except Exception:
                actions[name] = _pack_action(
                    MoveIntent.HOLD, MicroAction.DEFAULT, CommAction.NONE, BuyChoice.NONE, 0
                )
                continue
            site = self._target_sites.get(name, "A")
            actions[name] = self._act_one(decoded, site)
        return actions, {}

    # ------------------------------------------------------------------

    def _act_one(self, decoded: dict[str, Any], target_site: str) -> np.ndarray:
        if not _is_alive(decoded):
            return _pack_action(
                MoveIntent.HOLD, MicroAction.DEFAULT, CommAction.NONE, BuyChoice.NONE, 0
            )

        # ----- BUY phase -------------------------------------------------
        if _is_buy_phase(decoded):
            buy = _pick_buy(_money_norm(decoded), BuyChoice.SMG)
            return _pack_action(
                MoveIntent.HOLD, MicroAction.DEFAULT, CommAction.NONE, buy, 0
            )

        # ----- LIVE phase ------------------------------------------------
        dx, dy, dist = _bombsite_distance(decoded, target_site)
        enemy_slot = _nearest_visible_enemy_slot(decoded)
        aim_target = int(enemy_slot + 1) if enemy_slot is not None else 0

        on_site = dist < self._on_site_threshold
        carrier = _has_bomb(decoded)

        # If we are the bomb carrier and on site -> plant.
        if on_site and carrier:
            return _pack_action(
                MoveIntent.HOLD, MicroAction.INTERACT, CommAction.SUGGEST_ATTACK, BuyChoice.NONE, 0
            )

        # On site but no bomb: hold and shoot.
        if on_site:
            comm = (
                CommAction.CONTACT_ENEMY if enemy_slot is not None else CommAction.NONE
            )
            return _pack_action(
                MoveIntent.HOLD, MicroAction.CROUCH_HOLD, comm, BuyChoice.NONE, aim_target
            )

        # Rush toward the chosen site.
        move_intent = _compass_intent_from_dxdy(dx, dy)
        micro = MicroAction.SPRINT
        # Slow down ("DEFAULT") if we already see an enemy to improve accuracy.
        if enemy_slot is not None:
            micro = MicroAction.DEFAULT
        comm = CommAction.NONE
        if float(self._rng.random()) < 0.03:
            comm = CommAction.SUGGEST_ATTACK
        if enemy_slot is not None and float(self._rng.random()) < 0.10:
            comm = CommAction.CONTACT_ENEMY
        return _pack_action(
            move_intent, micro, comm, BuyChoice.NONE, aim_target
        )
