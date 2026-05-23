/**
 * Wire-protocol translator: backend (snake_case, int-enum) → frontend
 * (camelCase, string-union) shapes.
 *
 * The FastAPI backend ships `Snapshot.to_json_dict()` payloads with the
 * raw schema defined in :file:`packages/sim/kivski_sim/engine.py`:
 *
 *     {
 *       tick, round_id, phase: int, bomb_phase: int,
 *       yellow_score, blue_score, seconds_left, plant_progress, defuse_progress,
 *       agents: [
 *         { id, team, side, alive, hp, armor, pos: [x,y], facing,
 *           weapon, money, has_bomb, has_defuse_kit,
 *           kills_round, deaths_round }
 *       ],
 *       bomb: { phase, carrier, pos, plant_progress, defuse_progress,
 *               defuser, time_since_plant, site },
 *       events, messages, sounds
 *     }
 *
 * The frontend (see `types.ts`) wants a normalised camelCase shape with
 * string-union enums, `{x,y}` vectors and `weapons[]` arrays. The free
 * functions below do the translation as cheap, pure mappers. They are
 * deliberately defensive so a malformed/partial payload still produces
 * a renderable snapshot rather than throwing.
 */

import type {
  AgentSnapshot,
  BombPhase,
  BombSnapshot,
  CommAction,
  EventItem,
  EventKind,
  MapInfoFrame,
  MatchPhase,
  MatchSnapshot,
  MessageItem,
  Side,
  Team,
  Vec2,
  WeaponKind,
  WeaponState,
} from "./types";

// ---------------------------------------------------------------------------
// Enum mapping tables (mirror packages/sim/kivski_sim/types.py)
// ---------------------------------------------------------------------------

/**
 * `Phase` enum in the backend:
 *     WARMUP=0, BUY=1, LIVE=2, POST_PLANT=3, ROUND_OVER=4, MATCH_OVER=5
 *
 * The frontend has no dedicated `post_plant`/`round_over` strings — both
 * fold into `post_round` which the UI already labels as "Post-Round".
 */
export const PHASE_INT_TO_STRING: Record<number, MatchPhase> = {
  0: "warmup",
  1: "buy",
  2: "live",
  3: "post_round", // POST_PLANT
  4: "post_round", // ROUND_OVER
  5: "match_over",
};

/**
 * `BombPhase` enum in the backend:
 *     CARRIED=0, DROPPED=1, PLANTING=2, PLANTED=3, DEFUSING=4, DEFUSED=5, DETONATED=6
 *
 * The frontend has no `dropped` string — we collapse it onto `none`
 * (bomb visible on the ground but not held / armed).
 */
export const BOMB_PHASE_INT_TO_STRING: Record<number, BombPhase> = {
  0: "carried",
  1: "none", // DROPPED — frontend doesn't have a dedicated state
  2: "planting",
  3: "planted",
  4: "defusing",
  5: "defused",
  6: "exploded", // DETONATED
};

/**
 * `Side` enum in the backend:
 *     ATTACKER=0, DEFENDER=1
 */
export const SIDE_INT_TO_STRING: Record<number, Side> = {
  0: "attacker",
  1: "defender",
};

/**
 * `Team` enum in the backend:
 *     YELLOW=0, BLUE=1
 */
export const TEAM_INT_TO_STRING: Record<number, Team> = {
  0: "yellow",
  1: "blue",
};

/**
 * `WeaponClass` enum in the backend:
 *     KNIFE=0, SIDEARM=1, HEAVY_PISTOL=2, SMG=3, RIFLE=4, PRECISION=5, SHOTGUN=6
 *
 * Mapped to the frontend's coarser `WeaponKind` union.
 */
export const WEAPON_INT_TO_KIND: Record<number, WeaponKind> = {
  0: "knife",
  1: "pistol",
  2: "pistol",
  3: "smg",
  4: "rifle",
  5: "sniper",
  6: "shotgun",
};

/**
 * Human-readable names for each weapon class, mirroring the backend's
 * `WEAPONS[cls].name` field. Used only for the agent inspector tooltip.
 */
export const WEAPON_INT_TO_NAME: Record<number, string> = {
  0: "Blade",
  1: "ZP-9",
  2: "Kestrel-50",
  3: "Viper-Repeater",
  4: "Hex-Rifle",
  5: "Talon Marksman",
  6: "Maw-12",
};

/**
 * `CommAction` enum in the backend (`packages/sim/kivski_sim/types.py`):
 *     NONE=0, PING_LOCATION=1, WARN_DANGER=2, REQUEST_SUPPORT=3,
 *     SUGGEST_ROTATE=4, SUGGEST_ATTACK=5, SUGGEST_FALLBACK=6,
 *     CONTACT_ENEMY=7, BOMBSITE_CLEAR=8
 *
 * The frontend type uses string unions; the backend's `NONE` maps to the
 * `SILENT` literal so the comms styling table (`commActionStyle`) can
 * render an opaque "no-op" entry for diagnostic frames.
 */
export const COMM_ACTION_INT_TO_STRING: Record<number, CommAction> = {
  0: "SILENT",
  1: "PING_LOCATION",
  2: "WARN_DANGER",
  3: "REQUEST_SUPPORT",
  4: "SUGGEST_ROTATE",
  5: "SUGGEST_ATTACK",
  6: "SUGGEST_FALLBACK",
  7: "CONTACT_ENEMY",
  8: "BOMBSITE_CLEAR",
};

/**
 * Human-readable display label for each comm action — used by the inspector
 * and the comms tab when rendering message chips. Kept in step with
 * `commActionStyle` in `event-icons.ts` (label-only mirror).
 */
export const COMM_ACTION_LABEL: Record<CommAction, string> = {
  SILENT: "Silent",
  PING_LOCATION: "Ping Location",
  WARN_DANGER: "Warn Danger",
  REQUEST_SUPPORT: "Request Support",
  SUGGEST_ROTATE: "Suggest Rotate",
  SUGGEST_ATTACK: "Suggest Attack",
  SUGGEST_FALLBACK: "Suggest Fall Back",
  CONTACT_ENEMY: "Enemy Contact",
  BOMBSITE_CLEAR: "Bombsite Clear",
  ACK: "Acknowledge",
};

// ---------------------------------------------------------------------------
// Utility coercion helpers
// ---------------------------------------------------------------------------

const toNumber = (v: unknown, fallback = 0): number => {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  }
  return fallback;
};

const toInt = (v: unknown, fallback = 0): number => Math.trunc(toNumber(v, fallback));

const toBool = (v: unknown, fallback = false): boolean => {
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return v !== 0;
  if (typeof v === "string") return v === "true" || v === "1";
  return fallback;
};

const toVec2 = (v: unknown, fallback: Vec2 = { x: 0, y: 0 }): Vec2 => {
  if (Array.isArray(v) && v.length >= 2) {
    return { x: toNumber(v[0], fallback.x), y: toNumber(v[1], fallback.y) };
  }
  if (typeof v === "object" && v !== null) {
    const obj = v as { x?: unknown; y?: unknown };
    if ("x" in obj && "y" in obj) {
      return { x: toNumber(obj.x, fallback.x), y: toNumber(obj.y, fallback.y) };
    }
  }
  return fallback;
};

/** Stable string id used throughout the frontend (`agent_<int-id>`). */
export const agentIdToString = (rawId: unknown): string => {
  if (typeof rawId === "string") return rawId;
  return `agent_${toInt(rawId, 0)}`;
};

// ---------------------------------------------------------------------------
// Per-agent translator
// ---------------------------------------------------------------------------

interface RawAgent {
  id?: unknown;
  team?: unknown;
  side?: unknown;
  alive?: unknown;
  hp?: unknown;
  armor?: unknown;
  pos?: unknown;
  facing?: unknown;
  weapon?: unknown;
  money?: unknown;
  has_bomb?: unknown;
  has_defuse_kit?: unknown;
  kills_round?: unknown;
  deaths_round?: unknown;
}

interface RawBomb {
  phase?: unknown;
  carrier?: unknown;
  pos?: unknown;
  plant_progress?: unknown;
  defuse_progress?: unknown;
  defuser?: unknown;
  time_since_plant?: unknown;
  site?: unknown;
}

interface RawSnapshot {
  tick?: unknown;
  round_id?: unknown;
  phase?: unknown;
  bomb_phase?: unknown;
  yellow_score?: unknown;
  blue_score?: unknown;
  seconds_left?: unknown;
  plant_progress?: unknown;
  defuse_progress?: unknown;
  agents?: unknown;
  bomb?: unknown;
  events?: unknown;
  messages?: unknown;
  sounds?: unknown;
}

/**
 * Build the synthetic `weapons[]` array from the single weapon int the
 * backend currently ships. The engine doesn't track ammo, so we report
 * sane fixed values that the inspector can render without dividing by
 * zero. The knife is always carried as a fallback secondary.
 */
const weaponsFromInt = (weaponInt: number): WeaponState[] => {
  const primaryKind = WEAPON_INT_TO_KIND[weaponInt] ?? "rifle";
  const isKnife = primaryKind === "knife";
  const primary: WeaponState = {
    kind: primaryKind,
    slot: isKnife ? "knife" : "primary",
    ammoMag: 30,
    ammoReserve: 90,
  };
  if (isKnife) {
    // Only the blade — no need for a duplicate secondary slot.
    return [primary];
  }
  const knife: WeaponState = {
    kind: "knife",
    slot: "knife",
    ammoMag: 0,
    ammoReserve: 0,
  };
  return [primary, knife];
};

/**
 * Decode a single agent. `bombContext` carries the bomb's current phase
 * and carrier/defuser so we can derive `isPlanting` / `isDefusing` —
 * fields that aren't present in the raw per-agent payload.
 */
export function decodeAgentSnapshot(
  raw: RawAgent,
  bombContext: { phase: BombPhase; carrierId: string | null; defuserId: string | null },
): AgentSnapshot {
  const id = agentIdToString(raw.id);
  const sideInt = toInt(raw.side, 0);
  const side: Side = SIDE_INT_TO_STRING[sideInt] ?? "attacker";
  const teamInt = toInt(raw.team, 0);
  const team: Team = TEAM_INT_TO_STRING[teamInt] ?? "yellow";
  const weaponInt = toInt(raw.weapon, 0);
  const hasBomb = toBool(raw.has_bomb, false);
  const hasDefuseKit = toBool(raw.has_defuse_kit, false);
  const isAlive = toBool(raw.alive, true);

  // Derived planting/defusing flags: the bomb only reports its carrier or
  // defuser by id, so any matching alive agent is currently acting on it.
  const isPlanting =
    isAlive &&
    bombContext.phase === "planting" &&
    bombContext.carrierId === id;
  const isDefusing =
    isAlive &&
    bombContext.phase === "defusing" &&
    bombContext.defuserId === id;

  // Friendly display name: agent_3 → "Y-3" / "B-3" so the team prefix is
  // visible everywhere the name surfaces (sidebar, inspector, comms chips).
  const numericTail = id.startsWith("agent_") ? id.slice("agent_".length) : id;
  const teamPrefix = team === "yellow" ? "Y" : "B";
  const friendlyName = `${teamPrefix}-${numericTail}`;
  return {
    id,
    name: friendlyName,
    team,
    side,
    pos: toVec2(raw.pos),
    facing: toNumber(raw.facing, 0),
    hp: toNumber(raw.hp, 0),
    armor: toNumber(raw.armor, 0),
    money: toInt(raw.money, 0),
    isAlive,
    isPlanting,
    isDefusing,
    // V1 backend does not yet have a comm "talking now" flag.
    isTalking: false,
    hasBomb,
    hasDefuseKit,
    weapons: weaponsFromInt(weaponInt),
    activeWeaponIdx: 0,
    kills: toInt(raw.kills_round, 0),
    deaths: toInt(raw.deaths_round, 0),
    // Backend doesn't track per-agent assists yet.
    assists: 0,
  };
}

// ---------------------------------------------------------------------------
// Bomb translator
// ---------------------------------------------------------------------------

/**
 * Translate the bomb sub-payload. The frontend's `BombSnapshot.timer`
 * is the seconds-since-plant (during post-plant) so the UI can show a
 * countdown; before plant we fall back to `seconds_left` of the live
 * phase via the caller.
 */
export function decodeBombSnapshot(raw: RawBomb | undefined): BombSnapshot {
  const r: RawBomb = raw ?? {};
  const phaseInt = toInt(r.phase, 0);
  const phase: BombPhase = BOMB_PHASE_INT_TO_STRING[phaseInt] ?? "none";
  const carrierIdInt = toInt(r.carrier, -1);
  const carrierId = carrierIdInt >= 0 ? agentIdToString(carrierIdInt) : null;
  const siteRaw = typeof r.site === "string" ? r.site : "";
  const siteId = siteRaw.length > 0 ? siteRaw : null;

  // `pos` is always shipped; the bomb is at the carrier's position or at
  // the plant site. We only zero it when the bomb is conceptually missing.
  const pos = toVec2(r.pos);
  const showPos = !(pos.x === 0 && pos.y === 0 && phase === "none" && carrierId === null);

  return {
    pos: showPos ? pos : null,
    phase,
    timer: toNumber(r.time_since_plant, 0),
    carrierId,
    siteId,
  };
}

/**
 * Backend ships `defuser` as an int (`-1` when nobody is defusing).
 * Exposed as a separate helper so the bomb context can be assembled
 * before each agent is decoded.
 */
export function bombDefuserId(raw: RawBomb | undefined): string | null {
  const r: RawBomb = raw ?? {};
  const defuserInt = toInt(r.defuser, -1);
  return defuserInt >= 0 ? agentIdToString(defuserInt) : null;
}

// ---------------------------------------------------------------------------
// Snapshot translator
// ---------------------------------------------------------------------------

/**
 * Decode a raw backend `Snapshot.to_json_dict()` blob into the frontend
 * `MatchSnapshot` shape. Pure function; `mapName` is supplied by the
 * caller since the snapshot itself doesn't carry it.
 *
 * Score mapping: the backend tracks raw team scores. In V1 the YELLOW
 * team starts as ATTACKER and BLUE as DEFENDER. After a side switch the
 * team-to-side mapping flips; we recover the current mapping from the
 * agents' `team`/`side` fields so the UI shows scores aligned with the
 * current attacker/defender colour, not the original team identity.
 */
export function decodeMatchSnapshot(raw: unknown, mapName: string): MatchSnapshot {
  const r: RawSnapshot = (typeof raw === "object" && raw !== null ? raw : {}) as RawSnapshot;
  const phaseInt = toInt(r.phase, 0);
  const phase: MatchPhase = PHASE_INT_TO_STRING[phaseInt] ?? "warmup";

  const rawBomb = (typeof r.bomb === "object" && r.bomb !== null ? r.bomb : {}) as RawBomb;
  const bomb = decodeBombSnapshot(rawBomb);
  const defuserId = bombDefuserId(rawBomb);
  const bombContext = {
    phase: bomb.phase,
    carrierId: bomb.carrierId,
    defuserId,
  };

  const rawAgents: RawAgent[] = Array.isArray(r.agents) ? (r.agents as RawAgent[]) : [];
  const agents: AgentSnapshot[] = rawAgents.map((a) => decodeAgentSnapshot(a, bombContext));

  // Derive team-to-side mapping from the agents so scores stay aligned
  // with the current attacker/defender colour after a side switch.
  let attackerTeam = 0; // 0 = YELLOW, 1 = BLUE
  for (const a of rawAgents) {
    const sideInt = toInt(a.side, 0);
    const teamInt = toInt(a.team, 0);
    if (sideInt === 0) {
      attackerTeam = teamInt;
      break;
    }
  }
  const yellow = toInt(r.yellow_score, 0);
  const blue = toInt(r.blue_score, 0);
  const score =
    attackerTeam === 0
      ? { attacker: yellow, defender: blue }
      : { attacker: blue, defender: yellow };

  return {
    tick: toInt(r.tick, 0),
    serverTs: Date.now(),
    round: toInt(r.round_id, 0),
    phase,
    secondsLeft: toNumber(r.seconds_left, 0),
    score,
    agents,
    bomb,
    mapName,
  };
}

// ---------------------------------------------------------------------------
// map_info translator
// ---------------------------------------------------------------------------

interface RawMapInfo {
  name?: unknown;
  tick_rate_hz?: unknown;
  tickRate?: unknown;
}

/**
 * Decode the `map_info` initial WebSocket frame. We only surface the
 * fields the frontend actually consumes today (map name + optional tick
 * rate); the full geometry is fetched through `loadMap` over REST.
 */
export function decodeMapInfo(raw: unknown): MapInfoFrame {
  const r: RawMapInfo = (typeof raw === "object" && raw !== null ? raw : {}) as RawMapInfo;
  const name = typeof r.name === "string" ? r.name : "dustline";
  const tickRateRaw = r.tickRate ?? r.tick_rate_hz;
  const tickRate = typeof tickRateRaw === "number" ? tickRateRaw : undefined;
  return { mapName: name, tickRate };
}

// ---------------------------------------------------------------------------
// Per-tick comm message translator
// ---------------------------------------------------------------------------

interface RawMessage {
  tick?: unknown;
  sender?: unknown;
  receivers?: unknown;
  action?: unknown;
  payload?: unknown;
  pos?: unknown;
}

/**
 * Decode one entry from a snapshot's `messages[]` buffer (see
 * `engine.Engine._message_to_dict`). The backend ships `payload` as a
 * numpy array which currently can't survive JSON serialization, so the
 * field is usually absent on the wire; if it is present we attempt to
 * parse it as a flat number list.
 *
 * `fallbackTick` is used when the raw payload omits `tick` (older backends).
 */
export function decodeMessageItem(raw: unknown, fallbackTick = 0): MessageItem {
  const r: RawMessage = (typeof raw === "object" && raw !== null ? raw : {}) as RawMessage;
  const tick = toInt(r.tick, fallbackTick);
  const senderId = toInt(r.sender, 0);
  const fromId = agentIdToString(senderId);
  const receivers = Array.isArray(r.receivers) ? (r.receivers as unknown[]) : [];
  const toIds = receivers.map((rid) => agentIdToString(toInt(rid, 0)));
  const actionInt = toInt(r.action, 0);
  const action = COMM_ACTION_INT_TO_STRING[actionInt] ?? "SILENT";
  const actionLabel = COMM_ACTION_LABEL[action];
  const posRaw = r.pos;
  const pos: Vec2 | undefined =
    Array.isArray(posRaw) || (typeof posRaw === "object" && posRaw !== null)
      ? toVec2(posRaw)
      : undefined;
  const payload: number[] | undefined = Array.isArray(r.payload)
    ? (r.payload as unknown[]).map((p) => toNumber(p, 0))
    : undefined;
  const recvCount = toIds.length;
  const text = `${actionLabel} → ${recvCount} teammate${recvCount === 1 ? "" : "s"}`;
  // Random suffix avoids id-collisions when multiple messages share the
  // exact tick/sender/action combo (silent-spam case).
  const id = `msg-${tick}-${senderId}-${actionInt}-${Math.random()
    .toString(36)
    .slice(2, 6)}`;
  return {
    id,
    ts: Date.now(),
    tick,
    fromId,
    toIds,
    text,
    tag: action.toLowerCase().replace(/_/g, "-"),
    action,
    actionLabel,
    pos,
    payload,
  };
}

// ---------------------------------------------------------------------------
// Per-tick combat event translator
// ---------------------------------------------------------------------------

interface RawCombatEvent {
  tick?: unknown;
  attacker?: unknown;
  victim?: unknown;
  weapon?: unknown;
  damage?: unknown;
  killed?: unknown;
  distance?: unknown;
  through_cover?: unknown;
}

/**
 * Decode one entry from a snapshot's `events[]` buffer (combat events
 * only — see `engine.Engine._combat_event_to_dict`). `kind` is either
 * `"kill"` (when the engine flagged the hit as lethal) or `"info"`
 * (non-lethal hit) so the existing chip styles light up correctly.
 */
export function decodeEventItem(raw: unknown, fallbackTick = 0): EventItem {
  const r: RawCombatEvent = (typeof raw === "object" && raw !== null
    ? raw
    : {}) as RawCombatEvent;
  const tick = toInt(r.tick, fallbackTick);
  const attackerId = toInt(r.attacker, 0);
  const victimId = toInt(r.victim, 0);
  const weaponInt = toInt(r.weapon, 0);
  const damage = toNumber(r.damage, 0);
  const distance = toNumber(r.distance, 0);
  const killed = toBool(r.killed, false);
  const throughCover = toBool(r.through_cover, false);

  const actorIdStr = agentIdToString(attackerId);
  const targetIdStr = agentIdToString(victimId);
  const weaponName = WEAPON_INT_TO_NAME[weaponInt] ?? "weapon";
  const kind: EventKind = killed ? "kill" : "info";
  const coverHint = throughCover ? ", through cover" : "";
  const text = killed
    ? `${actorIdStr} killed ${targetIdStr} (${weaponName}, ${distance.toFixed(1)}t${coverHint})`
    : `${actorIdStr} hit ${targetIdStr} for ${Math.round(damage)} HP (${weaponName}, ${distance.toFixed(1)}t${coverHint})`;
  const id = `evt-${tick}-${attackerId}-${victimId}-${Math.random()
    .toString(36)
    .slice(2, 6)}`;
  return {
    id,
    ts: Date.now(),
    tick,
    kind,
    actorId: actorIdStr,
    targetId: targetIdStr,
    text,
  };
}
