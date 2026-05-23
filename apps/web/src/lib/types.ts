/**
 * Frontend mirror of the Python `types.py` schema used by the backend
 * tactical sim. Kept intentionally minimal and `string-union`-typed so
 * we can decode JSON-over-WebSocket frames without a heavy IDL layer.
 *
 * Keep in sync with: packages/sim/types.py
 */

// ---------- Enums (as string unions) ----------

export type Side = "attacker" | "defender";

/**
 * Persistent team identity. The team a player belongs to never changes
 * during a match; what changes is their `Side` (attacker / defender) at
 * the side-switch round. The viewer groups the sidebar by `Team` so the
 * UI is stable across the switch and only labels (and the bullet colour)
 * change to reflect the new role.
 *
 * Mirrors `Team` in :file:`packages/sim/kivski_sim/types.py`
 * (YELLOW = 0, BLUE = 1).
 */
export type Team = "yellow" | "blue";

export type MatchPhase =
  | "warmup"
  | "buy"
  | "live"
  | "post_round"
  | "halftime"
  | "match_over";

export type BombPhase = "carried" | "planting" | "planted" | "defusing" | "exploded" | "defused" | "none";

export type WeaponSlot = "primary" | "secondary" | "knife" | "grenade" | "c4";

export type WeaponKind =
  | "knife"
  | "pistol"
  | "smg"
  | "rifle"
  | "ar"
  | "sniper"
  | "shotgun"
  | "lmg"
  | "grenade"
  | "flash"
  | "smoke"
  | "molotov"
  | "c4";

/**
 * Communication actions emitted by the policy's comm-head. Keep in sync
 * with `packages/marl/comm/actions.py` on the backend.
 */
export type CommAction =
  | "PING_LOCATION"
  | "WARN_DANGER"
  | "REQUEST_SUPPORT"
  | "SUGGEST_ROTATE"
  | "SUGGEST_ATTACK"
  | "SUGGEST_FALLBACK"
  | "CONTACT_ENEMY"
  | "BOMBSITE_CLEAR"
  | "ACK"
  | "SILENT";

/** Outcome of a finished round. */
export type RoundOutcome =
  | "attacker_elim"
  | "defender_elim"
  | "bomb_explode"
  | "bomb_defused"
  | "time_out"
  | "draw";

// ---------- Geometry ----------

export interface Vec2 {
  x: number;
  y: number;
}

// ---------- Domain objects ----------

export interface WeaponState {
  kind: WeaponKind;
  slot: WeaponSlot;
  ammoMag: number;
  ammoReserve: number;
}

export interface AgentSnapshot {
  id: string;
  /** Display name of the agent or AI policy persona. */
  name: string;
  /**
   * Persistent team identity (yellow / blue). Does NOT change at the
   * side-switch round — use this to group the sidebar by team so the
   * UI is stable across switches.
   */
  team: Team;
  /**
   * Current side role (attacker / defender). Flips at the side-switch
   * round. Use this for role-specific labels and the dot/accent colour.
   */
  side: Side;
  pos: Vec2;
  /** Yaw in radians (0 = +x axis, CCW positive). */
  facing: number;
  hp: number;
  armor: number;
  money: number;
  isAlive: boolean;
  isPlanting: boolean;
  isDefusing: boolean;
  /** True if currently broadcasting comms this tick. */
  isTalking: boolean;
  hasBomb: boolean;
  /** True if the defender is carrying a defuse kit (faster defuse). */
  hasDefuseKit: boolean;
  weapons: WeaponState[];
  activeWeaponIdx: number;
  kills: number;
  deaths: number;
  assists: number;
  /** Optional behavior label from the policy (e.g. "rush A", "rotate"). */
  intent?: string | null;
}

export interface BombSnapshot {
  pos: Vec2 | null;
  phase: BombPhase;
  /** Seconds until detonation / defuse completion, depending on phase. */
  timer: number;
  carrierId: string | null;
  /** "A" | "B" | null while not planted. */
  siteId: string | null;
}

export interface MatchSnapshot {
  tick: number;
  /** Server wall-clock in ms since epoch (best-effort). */
  serverTs: number;
  round: number;
  phase: MatchPhase;
  /** Seconds left in the current phase (round, buy, freezetime, ...). */
  secondsLeft: number;
  score: { attacker: number; defender: number };
  agents: AgentSnapshot[];
  bomb: BombSnapshot;
  mapName: string;
}

// ---------- Events / Feed ----------

export type EventKind =
  | "kill"
  | "death"
  | "plant"
  | "defuse"
  | "round_start"
  | "round_end"
  | "bomb_explode"
  | "purchase"
  | "comm"
  | "info"
  | "sound";

export interface EventItem {
  id: string;
  ts: number;
  tick: number;
  kind: EventKind;
  /** Optional involved agent IDs (killer/victim/planter/etc.). */
  actorId?: string;
  targetId?: string;
  /** Short text representation suitable for the feed. */
  text: string;
  side?: Side;
  /** Optional position (used by sound-vis, kill-feed dots, etc.). */
  pos?: Vec2;
  /** Optional outcome label for round_end events. */
  outcome?: RoundOutcome;
}

// ---------- Comms ----------

/**
 * A single message broadcast by an agent on the comm-head.
 *
 * The viewer reads `action` to choose the marker style on the map, and
 * `payload` (a small fixed-size vector of normalized continuous values)
 * to render a mini bar-chart in the comms tab.
 */
export interface MessageItem {
  id: string;
  ts: number;
  tick: number;
  fromId: string;
  toIds: string[];
  /** Free-form short string from agent policy. */
  text: string;
  /** Tag for chip color: "callout" | "rotate" | "request" | etc. */
  tag?: string;
  /** Structured action enum from the comm head. */
  action?: CommAction;
  /** Human-friendly label for the action (mirrors `action`). */
  actionLabel?: string;
  /** Where the message is "about" — pings/warns target a map point. */
  pos?: Vec2;
  /** Continuous payload values in [-1, 1] for the chart preview. */
  payload?: number[];
}

// ---------- Agent inspector ----------

/**
 * Compact per-observer attention vector. Each observer (agent id)
 * attends to its teammates with a weight in [0, 1].
 */
export interface AttentionWeights {
  observerId: string;
  targets: { agentId: string; weight: number }[];
}

/**
 * Legacy/dict-shape for attention used by inspection blobs. Both shapes
 * are accepted by the inspector view.
 */
export interface AttentionWeightsDict {
  /** Per-source-id → weight in [0, 1]. */
  byAgent: Record<string, number>;
  /** Per-feature-channel attention (e.g. "vision", "comm", "memory"). */
  byChannel?: Record<string, number>;
}

export interface AgentInspection {
  agentId: string;
  /** Last action emitted by the policy (legacy single-head form). */
  lastAction?: {
    name: string;
    params: Record<string, number | string | boolean>;
  };
  /** Multi-head action — one entry per policy head. */
  lastHeads?: {
    move?: string;
    micro?: string;
    comm?: string;
    buy?: string;
    aimTarget?: string;
  };
  /** Critic value-estimate at last decision. */
  valueEstimate?: number;
  /** Logged observation (compact summary, not raw tensors). */
  observation?: Record<string, unknown>;
  /** Grouped observation feature vector — group name → flat vector. */
  observationGroups?: Record<string, number[]>;
  attention?: AttentionWeightsDict;
  /** Optional running policy entropy or other diagnostics. */
  diagnostics?: Record<string, number>;
  /** Hidden state preview (first N dims) for the inspector. */
  hiddenStatePreview?: number[];
}

// ---------- Training / metrics ----------

export interface TrainingStatus {
  /** True when the trainer worker is actively running episodes. */
  running: boolean;
  episode: number;
  totalEpisodes: number;
  /** Most recent rolling metrics from the trainer (one value per push). */
  policyLoss?: number;
  valueLoss?: number;
  entropy?: number;
}

export interface MetricsSample {
  /** Episode number this sample belongs to. */
  episode: number;
  /** Windowed win-rate vs. the random baseline in [0, 1]. */
  winrateVsRandom?: number;
  /** Windowed win-rate vs. the scripted baseline in [0, 1]. */
  winrateVsScripted?: number;
  /** Loss diagnostics that arrived with this sample. */
  policyLoss?: number;
  valueLoss?: number;
  entropy?: number;
}

export interface EconomySample {
  tick: number;
  attackerTotal: number;
  defenderTotal: number;
}

export interface RoundResult {
  round: number;
  outcome: RoundOutcome;
  winner: Side | "draw";
}

// ---------- WebSocket frame envelope ----------

/** Initial frame sent by the backend on WebSocket connect with map metadata. */
export interface MapInfoFrame {
  /** Map identifier (e.g. "dustline"). */
  mapName: string;
  /** Optional engine tick rate in Hz if the backend reports it. */
  tickRate?: number;
}

/**
 * Per-side policy hot-swap notification. Emitted by the backend at the
 * end of a round when ``auto_reload_yellow`` / ``auto_reload_blue`` was
 * requested on ``/api/match/new`` *and* a newer checkpoint has appeared
 * on disk since the side's adapter was last (re)loaded.
 *
 * The viewer uses this to flash a transient toast in the policy badge
 * area so the user can see live which snapshot drives each team.
 */
export interface PolicyReloadFrame {
  side: "yellow" | "blue";
  /** Stem of the freshly-loaded checkpoint file (e.g. "snapshot_ep_500"). */
  name: string;
  /** Absolute path to the new checkpoint on the backend host. */
  path?: string;
  /** Adapter name in use *before* the swap (best-effort, may be null). */
  previous?: string | null;
}

export type WSFrame =
  | { type: "snapshot"; data: MatchSnapshot }
  | { type: "event"; data: EventItem }
  | { type: "message"; data: MessageItem }
  | { type: "inspect"; data: AgentInspection }
  | { type: "attention_update"; data: AttentionWeights }
  | { type: "training_status"; data: TrainingStatus }
  | { type: "metrics_sample"; data: MetricsSample }
  | { type: "round_result"; data: RoundResult }
  /**
   * Legacy "hello" frame (kept for backwards compatibility). Real backend
   * sends `map_info` — see :type:`MapInfoFrame`.
   */
  | { type: "hello"; data: { mapName: string; tickRate: number } }
  /** Sent once by the backend on WS connect; carries map metadata. */
  | { type: "map_info"; data: MapInfoFrame }
  /** Sent when the engine reports match complete. */
  | { type: "match_done"; matchId?: string }
  /** Response to a client `ping`. */
  | { type: "pong"; ts: number }
  /** Generic acknowledgement of a control command. */
  | { type: "ack"; for: string; [k: string]: unknown }
  /** Sent after a per-round auto-reload hot-swaps a side's policy adapter. */
  | { type: "policy_reload"; data: PolicyReloadFrame }
  | { type: "error"; data: { message: string } };

// ---------- Map data ----------

export interface MapZone {
  id: string;
  /** Polygon (CCW) in map units. */
  poly: Vec2[];
  kind: "site_a" | "site_b" | "spawn_attacker" | "spawn_defender" | "buy" | "neutral";
  label?: string;
}

export interface MapWall {
  poly: Vec2[];
  /** "wall" = full block, "cover" = partial. */
  kind: "wall" | "cover";
}

export interface MapData {
  name: string;
  width: number;
  height: number;
  /** Pixels per map-unit, for the viewer. */
  pxPerUnit: number;
  walls: MapWall[];
  zones: MapZone[];
}
