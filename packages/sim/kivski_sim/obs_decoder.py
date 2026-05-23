"""Decode flat observation vectors back into readable nested dicts.

The training loop only ever consumes the *flat* ``np.ndarray`` produced by
:class:`kivski_sim.env.KivskiParallelEnv`. For debugging, inspection in the
viewer, and unit tests, this module reconstructs a human-readable structure
from that vector, plus exposes the per-field schema and a helper for the
observation length so external code (e.g. the policy network factory) can
size its inputs without instantiating an env.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from kivski_sim.config import KivskiConfig
from kivski_sim.types import WeaponClass


__all__ = [
    "OBSERVATION_SECTION_NAMES",
    "build_observation_schema",
    "decode_observation",
    "get_observation_dim",
    "section_widths",
]


# Section names mirror the order in ``KivskiParallelEnv._build_observation``.
OBSERVATION_SECTION_NAMES: tuple[str, ...] = (
    "self",
    "self_pos",
    "teammates",
    "enemies",
    "sounds",
    "messages",
    "map_ctx",
    "team_ctx",
)


# Per-element widths. Re-declared here so this module does not import the env
# (which has heavy deps like pettingzoo). They are validated against the env
# at runtime via the ``test_observation_space_dimension`` test.
_NUM_WEAPONS: int = len(WeaponClass)
_SELF_BLOCK_WIDTH: int = 7 + _NUM_WEAPONS
_SELF_POS_WIDTH: int = 3
_TEAMMATE_SLOT_WIDTH: int = 8
_ENEMY_SLOT_WIDTH: int = 6
_SOUND_SLOT_WIDTH: int = 5
_MESSAGE_SLOT_WIDTH: int = 7
_NUM_PHASES_OBS: int = 4
_MAP_CTX_WIDTH: int = 6 + _NUM_PHASES_OBS
_TEAM_CTX_WIDTH: int = 6


# ---------------------------------------------------------------------------
# Section-width arithmetic
# ---------------------------------------------------------------------------


def section_widths(cfg: KivskiConfig) -> dict[str, int]:
    """Return the byte-width of each observation section."""
    obs_cfg = cfg.agent.observation
    return {
        "self": _SELF_BLOCK_WIDTH,
        "self_pos": _SELF_POS_WIDTH,
        "teammates": _TEAMMATE_SLOT_WIDTH * int(obs_cfg.teammate_slots),
        "enemies": _ENEMY_SLOT_WIDTH * int(obs_cfg.last_known_enemies),
        "sounds": _SOUND_SLOT_WIDTH * int(obs_cfg.sound_event_slots),
        "messages": _MESSAGE_SLOT_WIDTH * int(obs_cfg.received_message_slots),
        "map_ctx": _MAP_CTX_WIDTH,
        "team_ctx": _TEAM_CTX_WIDTH,
    }


def get_observation_dim(cfg: KivskiConfig) -> int:
    """Total length of the flat observation vector for one agent."""
    return sum(section_widths(cfg).values())


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------


def _self_block_fields() -> list[tuple[int, int, str]]:
    """Return (slot_start, slot_end_exclusive, name) tuples for the self block."""
    fields: list[tuple[int, int, str]] = [
        (0, 1, "hp_norm"),
        (1, 2, "armor_norm"),
        (2, 3, "money_norm"),
    ]
    # Weapon one-hot.
    for w in WeaponClass:
        idx = 3 + int(w)
        fields.append((idx, idx + 1, f"weapon_is_{w.name.lower()}"))
    fields.append((3 + _NUM_WEAPONS, 4 + _NUM_WEAPONS, "has_bomb"))
    fields.append((4 + _NUM_WEAPONS, 5 + _NUM_WEAPONS, "alive"))
    return fields


def _self_pos_fields(offset: int) -> list[tuple[int, int, str]]:
    return [
        (offset + 0, offset + 1, "pos_x_norm"),
        (offset + 1, offset + 2, "pos_y_norm"),
        (offset + 2, offset + 3, "facing_norm"),
    ]


def _teammate_slot_fields(offset: int) -> list[tuple[int, int, str]]:
    names = ["alive", "hp_norm", "dx", "dy", "distance", "has_bomb", "weapon_id_norm", "money_norm"]
    return [(offset + i, offset + i + 1, n) for i, n in enumerate(names)]


def _enemy_slot_fields(offset: int) -> list[tuple[int, int, str]]:
    names = ["age_norm", "dx", "dy", "weapon_id_norm", "was_alive", "distance_at_last_obs"]
    return [(offset + i, offset + i + 1, n) for i, n in enumerate(names)]


def _sound_slot_fields(offset: int) -> list[tuple[int, int, str]]:
    names = ["age_norm", "dx", "dy", "intensity", "kind_id_norm"]
    return [(offset + i, offset + i + 1, n) for i, n in enumerate(names)]


def _message_slot_fields(offset: int) -> list[tuple[int, int, str]]:
    names = [
        "age_norm",
        "sender_team_idx_norm",
        "comm_action_norm",
        "dx",
        "dy",
        "has_payload",
        "payload_norm",
    ]
    return [(offset + i, offset + i + 1, n) for i, n in enumerate(names)]


def _map_ctx_fields(offset: int) -> list[tuple[int, int, str]]:
    base = [
        (offset + 0, offset + 1, "bombsite_a_dx"),
        (offset + 1, offset + 2, "bombsite_a_dy"),
        (offset + 2, offset + 3, "bombsite_b_dx"),
        (offset + 3, offset + 4, "bombsite_b_dy"),
        (offset + 4, offset + 5, "time_in_round_norm"),
        (offset + 5, offset + 6, "reserved"),
    ]
    phase_names = ("phase_buy", "phase_live", "phase_post_plant", "phase_round_over")
    for i, n in enumerate(phase_names):
        base.append((offset + 6 + i, offset + 6 + i + 1, n))
    return base


def _team_ctx_fields(offset: int) -> list[tuple[int, int, str]]:
    names = [
        "teammates_alive_frac",
        "enemies_alive_known_frac",
        "bomb_phase_norm",
        "my_team_score_norm",
        "enemy_team_score_norm",
        "consecutive_losses_norm",
    ]
    return [(offset + i, offset + i + 1, n) for i, n in enumerate(names)]


def build_observation_schema(cfg: KivskiConfig) -> dict[str, Any]:
    """Return a nested schema describing each slot in the observation vector.

    The shape is::

        {
            "total_dim": int,
            "sections": {
                "self":     {"start": 0, "end": 14, "fields": [(0, 1, "hp_norm"), ...]},
                "self_pos": {"start": 14, "end": 17, "fields": [...]},
                ...
            },
        }
    """
    widths = section_widths(cfg)
    obs_cfg = cfg.agent.observation
    schema: dict[str, Any] = {"total_dim": sum(widths.values()), "sections": {}}

    cursor = 0
    # ----- Self block ----------------------------------------------------
    schema["sections"]["self"] = {
        "start": cursor,
        "end": cursor + widths["self"],
        "fields": _self_block_fields(),
    }
    cursor += widths["self"]
    # ----- Self pos ------------------------------------------------------
    schema["sections"]["self_pos"] = {
        "start": cursor,
        "end": cursor + widths["self_pos"],
        "fields": _self_pos_fields(cursor),
    }
    cursor += widths["self_pos"]
    # ----- Teammates -----------------------------------------------------
    teammate_section: dict[str, Any] = {
        "start": cursor,
        "end": cursor + widths["teammates"],
        "slots": [],
    }
    for slot in range(int(obs_cfg.teammate_slots)):
        slot_start = cursor + slot * _TEAMMATE_SLOT_WIDTH
        teammate_section["slots"].append(
            {
                "index": slot,
                "start": slot_start,
                "end": slot_start + _TEAMMATE_SLOT_WIDTH,
                "fields": _teammate_slot_fields(slot_start),
            }
        )
    schema["sections"]["teammates"] = teammate_section
    cursor += widths["teammates"]
    # ----- Enemies -------------------------------------------------------
    enemy_section: dict[str, Any] = {
        "start": cursor,
        "end": cursor + widths["enemies"],
        "slots": [],
    }
    for slot in range(int(obs_cfg.last_known_enemies)):
        slot_start = cursor + slot * _ENEMY_SLOT_WIDTH
        enemy_section["slots"].append(
            {
                "index": slot,
                "start": slot_start,
                "end": slot_start + _ENEMY_SLOT_WIDTH,
                "fields": _enemy_slot_fields(slot_start),
            }
        )
    schema["sections"]["enemies"] = enemy_section
    cursor += widths["enemies"]
    # ----- Sounds --------------------------------------------------------
    sound_section: dict[str, Any] = {
        "start": cursor,
        "end": cursor + widths["sounds"],
        "slots": [],
    }
    for slot in range(int(obs_cfg.sound_event_slots)):
        slot_start = cursor + slot * _SOUND_SLOT_WIDTH
        sound_section["slots"].append(
            {
                "index": slot,
                "start": slot_start,
                "end": slot_start + _SOUND_SLOT_WIDTH,
                "fields": _sound_slot_fields(slot_start),
            }
        )
    schema["sections"]["sounds"] = sound_section
    cursor += widths["sounds"]
    # ----- Messages ------------------------------------------------------
    message_section: dict[str, Any] = {
        "start": cursor,
        "end": cursor + widths["messages"],
        "slots": [],
    }
    for slot in range(int(obs_cfg.received_message_slots)):
        slot_start = cursor + slot * _MESSAGE_SLOT_WIDTH
        message_section["slots"].append(
            {
                "index": slot,
                "start": slot_start,
                "end": slot_start + _MESSAGE_SLOT_WIDTH,
                "fields": _message_slot_fields(slot_start),
            }
        )
    schema["sections"]["messages"] = message_section
    cursor += widths["messages"]
    # ----- Map ctx -------------------------------------------------------
    schema["sections"]["map_ctx"] = {
        "start": cursor,
        "end": cursor + widths["map_ctx"],
        "fields": _map_ctx_fields(cursor),
    }
    cursor += widths["map_ctx"]
    # ----- Team ctx ------------------------------------------------------
    schema["sections"]["team_ctx"] = {
        "start": cursor,
        "end": cursor + widths["team_ctx"],
        "fields": _team_ctx_fields(cursor),
    }
    cursor += widths["team_ctx"]

    assert cursor == schema["total_dim"], (cursor, schema["total_dim"])
    return schema


# Backwards-compatible alias requested in the task: ``OBSERVATION_SCHEMA`` for
# a *default* config. Tools that need a config-specific schema should call
# :func:`build_observation_schema` directly.
OBSERVATION_SCHEMA: dict[str, Any] = build_observation_schema(KivskiConfig())


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def decode_observation(
    obs_vector: np.ndarray,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a flat observation vector into a readable nested dict.

    When ``schema`` is ``None`` the default-config schema is used; pass a
    custom one if your env uses non-default ``ObservationConfig`` slot counts.

    The returned dict mirrors the structure produced by
    :func:`build_observation_schema`: top-level section names map to either a
    field dict (single block) or a list of slot dicts.
    """
    if schema is None:
        schema = OBSERVATION_SCHEMA
    arr = np.asarray(obs_vector, dtype=np.float32).ravel()
    out: dict[str, Any] = {}
    for section_name, section in schema["sections"].items():
        if "slots" in section:
            slot_list: list[dict[str, float]] = []
            for slot in section["slots"]:
                fields = slot["fields"]
                slot_list.append({name: float(arr[s]) for s, _e, name in fields})
            out[section_name] = slot_list
        else:
            fields = section["fields"]
            out[section_name] = {name: float(arr[s]) for s, _e, name in fields}
    return out
