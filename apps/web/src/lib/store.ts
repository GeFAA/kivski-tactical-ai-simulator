import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type {
  AgentInspection,
  AgentSnapshot,
  AttentionWeights,
  BombSnapshot,
  EconomySample,
  EventItem,
  MatchPhase,
  MatchSnapshot,
  MessageItem,
  MetricsSample,
  RoundResult,
  TrainingStatus,
} from "./types";

const MAX_EVENTS = 200;
const MAX_MESSAGES = 200;
const MAX_METRICS = 500;
const MAX_ECONOMY = 600;
const MAX_ROUNDS = 60;
const MAX_HEATMAP_POSITIONS = 4000;

// ---------- State shape ----------

/**
 * Per-team policy descriptor for the comparison-match UI. Both fields
 * are stored verbatim from the backend's `/api/match/new` response so
 * the header can show a friendly label without re-translating ids:
 *
 *   - `id`   = policy specifier passed to the backend
 *              ("random", "scripted_rush", "latest", "best",
 *              "checkpoint:<name>", or a raw ckpt path).
 *   - `name` = human-readable label for display.
 *
 * Null fields mean the policy is unknown (e.g. before the first match
 * is created or when the backend doesn't report the field).
 */
export interface PolicyAssignment {
  id: string;
  name: string;
}

interface MatchState {
  tick: number;
  round: number;
  phase: MatchPhase;
  secondsLeft: number;
  score: { attacker: number; defender: number };
  agents: AgentSnapshot[];
  bomb: BombSnapshot;
  mapName: string;
  /** True when WebSocket has at least one frame and is alive. */
  connected: boolean;
  /**
   * Backend match id of the currently subscribed session, or null while
   * the WS handshake hasn't completed yet. REST control commands
   * (pause/resume/speed/reset) target `/api/match/{currentMatchId}/...`,
   * so they must short-circuit until this is set.
   */
  currentMatchId: string | null;
  /**
   * Active policies per team. Populated when `/api/match/new` succeeds
   * (or via the `setCurrentPolicies` action). `null` means the field
   * has not yet been negotiated with the backend.
   */
  currentPolicies: {
    yellow: PolicyAssignment | null;
    blue: PolicyAssignment | null;
  };
  /**
   * Per-side auto-reload flags echoed by the backend on
   * ``/api/match/new``. When True the policy badge renders an extra
   * "auto" suffix so the user can see at a glance that the side will
   * hot-swap every round.
   */
  autoReload: {
    yellow: boolean;
    blue: boolean;
  };
  /**
   * Most-recent ``policy_reload`` event for the transient toast. Set
   * when the backend hot-swaps a side's checkpoint; the header reads
   * ``ts`` to fade the badge after a couple of seconds.
   */
  lastPolicyReload: {
    side: "yellow" | "blue";
    name: string;
    previous: string | null;
    ts: number;
  } | null;
}

export type RightTab = "events" | "inspector" | "comms" | "metrics" | "sys";

/**
 * Top-level UI mode. ``simple`` strips the viewer down to map + score +
 * round-timer + a single "events" hint so non-technical users aren't
 * confronted with debug toggles, inspector panels, and training
 * dashboards. ``advanced`` restores the original power-user layout
 * (right sidebar, training panel, debug overlays, all controls).
 *
 * Persisted in localStorage under ``kivski-ui-mode`` so the choice
 * survives page reloads. Default is ``simple`` for first-time visitors
 * — the gear-icon settings drawer is the affordance for switching.
 */
export type UiMode = "simple" | "advanced";

/** Tabs inside the settings drawer. */
export type SettingsTab = "match" | "training" | "view" | "about";

interface UIState {
  selectedAgentId: string | null;
  rightTab: RightTab;
  showFov: boolean;
  showSound: boolean;
  showComms: boolean;
  showLastKnown: boolean;
  showHeatmap: boolean;
  speed: number;
  paused: boolean;
  /**
   * Monotonically-incrementing token used to force the App-level
   * WebSocket effect to tear down and re-handshake (e.g. after the
   * comparison-match modal POSTs to /api/match/new). The actual
   * value is irrelevant — only changes trigger React's `useEffect`
   * re-run via dependency comparison.
   */
  matchToken: number;
  /** Top-level UI density preset. See ``UiMode`` for details. */
  uiMode: UiMode;
  /** Open/closed state of the gear-icon settings drawer. */
  settingsOpen: boolean;
  /** Last-active tab inside the settings drawer (persisted). */
  settingsTab: SettingsTab;
}

interface InspectionState {
  /** Latest inspection blob per agent (keyed by agentId). */
  byAgent: Record<string, AgentInspection>;
  /**
   * Per-observer attention weights toward teammates.
   * Outer key = observer agent id, inner key = target agent id, value = weight in [0,1].
   */
  attentionWeights: Record<string, Record<string, number>>;
}

interface FeedState {
  eventFeed: EventItem[];
  recentMessages: MessageItem[];
}

interface MetricsState {
  trainingStatus: TrainingStatus;
  /** Time-series of metrics samples, ordered oldest → newest. */
  metricsHistory: MetricsSample[];
  /** Per-tick economy time series for the current match. */
  economyHistory: EconomySample[];
  /** Round outcomes for the timeline & pie. */
  roundResults: RoundResult[];
  /** Downsampled per-side position samples for the heatmap. */
  heatmapPositions: { side: AgentSnapshot["side"]; x: number; y: number }[];
}

interface Actions {
  // Networking-driven
  setConnected: (v: boolean) => void;
  setCurrentMatchId: (id: string | null) => void;
  setMatchSnapshot: (snap: MatchSnapshot) => void;
  pushEvent: (e: EventItem) => void;
  pushMessage: (m: MessageItem) => void;
  setInspection: (insp: AgentInspection) => void;
  setMapName: (name: string) => void;
  setAttentionWeights: (a: AttentionWeights) => void;
  setTrainingStatus: (s: TrainingStatus) => void;
  pushMetricsSample: (s: MetricsSample) => void;
  pushRoundResult: (r: RoundResult) => void;
  /**
   * Update the per-team active policy descriptors. Pass `null` to clear
   * a side back to "unknown" (e.g. when starting a fresh match without
   * a policy override).
   */
  setCurrentPolicies: (
    p: Partial<{
      yellow: PolicyAssignment | null;
      blue: PolicyAssignment | null;
    }>,
  ) => void;
  /**
   * Record the auto-reload flags reported by the backend on the most-
   * recent ``/api/match/new`` response. Drives the "auto" suffix on
   * the policy badges in the header.
   */
  setAutoReload: (p: Partial<{ yellow: boolean; blue: boolean }>) => void;
  /**
   * Record a ``policy_reload`` event from the backend. Updates
   * ``currentPolicies`` for the affected side so the badge label
   * reflects the new checkpoint immediately, and seeds
   * ``lastPolicyReload`` for the transient toast.
   */
  pushPolicyReload: (e: {
    side: "yellow" | "blue";
    name: string;
    previous?: string | null;
  }) => void;

  // UI
  selectAgent: (id: string | null) => void;
  setRightTab: (t: RightTab) => void;
  toggleFov: () => void;
  toggleSound: () => void;
  toggleComms: () => void;
  toggleLastKnown: () => void;
  toggleHeatmap: () => void;
  setSpeed: (s: number) => void;
  setPaused: (p: boolean) => void;
  togglePaused: () => void;
  /**
   * Force the App-level WebSocket subscription to re-handshake
   * (used after `/api/match/new` to switch policies). Simply
   * increments `matchToken` so `useEffect` deps differ.
   */
  setMatchToken: () => void;
  /** Switch between simple (default) and advanced UI density. */
  setUiMode: (mode: UiMode) => void;
  /** Open or close the gear-icon settings drawer. */
  setSettingsOpen: (open: boolean) => void;
  /** Switch the active tab inside the settings drawer. */
  setSettingsTab: (tab: SettingsTab) => void;

  reset: () => void;
}

export type AppState = MatchState &
  UIState &
  InspectionState &
  FeedState &
  MetricsState &
  Actions;

// ---------- Initial state ----------

const initialMatch: MatchState = {
  tick: 0,
  round: 0,
  phase: "warmup",
  secondsLeft: 0,
  score: { attacker: 0, defender: 0 },
  agents: [],
  bomb: { pos: null, phase: "none", timer: 0, carrierId: null, siteId: null },
  mapName: "dustline",
  connected: false,
  currentMatchId: null,
  currentPolicies: { yellow: null, blue: null },
  autoReload: { yellow: false, blue: false },
  lastPolicyReload: null,
};

const initialUI: UIState = {
  selectedAgentId: null,
  rightTab: "events",
  showFov: false,
  showSound: false,
  showComms: true,
  showLastKnown: false,
  showHeatmap: false,
  speed: 1,
  paused: false,
  matchToken: 0,
  uiMode: "simple",
  settingsOpen: false,
  settingsTab: "match",
};

const initialInspection: InspectionState = { byAgent: {}, attentionWeights: {} };
const initialFeed: FeedState = { eventFeed: [], recentMessages: [] };
const initialMetrics: MetricsState = {
  trainingStatus: {
    running: false,
    episode: 0,
    totalEpisodes: 0,
  },
  metricsHistory: [],
  economyHistory: [],
  roundResults: [],
  heatmapPositions: [],
};

// ---------- Heatmap sampling helpers ----------

/** Subsample positions: 1 in every N ticks to keep buffer small. */
const HEATMAP_SAMPLE_EVERY = 4;

const samplePositions = (
  buffer: MetricsState["heatmapPositions"],
  tick: number,
  agents: AgentSnapshot[],
): MetricsState["heatmapPositions"] => {
  if (tick % HEATMAP_SAMPLE_EVERY !== 0) return buffer;
  const next = buffer.slice();
  for (const a of agents) {
    if (!a.isAlive) continue;
    next.push({ side: a.side, x: a.pos.x, y: a.pos.y });
  }
  // Trim from the front (oldest) when over cap.
  if (next.length > MAX_HEATMAP_POSITIONS) {
    next.splice(0, next.length - MAX_HEATMAP_POSITIONS);
  }
  return next;
};

const computeEconomySample = (
  tick: number,
  agents: AgentSnapshot[],
): EconomySample => {
  let attackerTotal = 0;
  let defenderTotal = 0;
  for (const a of agents) {
    if (a.side === "attacker") attackerTotal += a.money;
    else defenderTotal += a.money;
  }
  return { tick, attackerTotal, defenderTotal };
};

// ---------- Store ----------

// Persisted slice of the UI state. We intentionally exclude transient
// per-match selections (selectedAgentId, paused, currentMatchId, ...)
// because they belong to whichever match was live when the tab closed
// and would point at dead data on the next page load. The debug toggles
// + playback speed + active tab are *user preferences* and survive
// across reloads via localStorage.
//
// ``uiMode`` (simple/advanced) and ``settingsTab`` (last-active tab in
// the drawer) are also persisted so a user who switched to Advanced is
// not yanked back to Simple on reload. ``uiMode`` is additionally
// mirrored to a dedicated ``kivski-ui-mode`` key by ``setUiMode`` for
// easy DevTools inspection.
type PersistedUI = Pick<
  AppState,
  | "showFov"
  | "showSound"
  | "showComms"
  | "showLastKnown"
  | "showHeatmap"
  | "speed"
  | "rightTab"
  | "uiMode"
  | "settingsTab"
>;

const PERSIST_STORAGE_KEY = "kivski-ui-state";
// Version 2 introduced uiMode / settingsTab. Migrate by simply dropping
// the old state — defaults are reasonable for the new fields.
const PERSIST_STORAGE_VERSION = 2;

/**
 * Read the dedicated ``kivski-ui-mode`` key (written by ``setUiMode``)
 * so it overrides the bundled persist payload. Lets a user (or a test
 * harness) set the mode explicitly via DevTools without touching the
 * larger blob. Returns ``null`` when the key is absent / invalid.
 */
const readUiModeOverride = (): UiMode | null => {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem("kivski-ui-mode");
    if (raw === "simple" || raw === "advanced") return raw;
    return null;
  } catch {
    return null;
  }
};

export const useStore = create<AppState>()(persist((set) => ({
  ...initialMatch,
  ...initialUI,
  ...initialInspection,
  ...initialFeed,
  ...initialMetrics,

  setConnected: (v) => set({ connected: v }),

  setCurrentMatchId: (id) => set({ currentMatchId: id }),

  setMatchSnapshot: (snap) =>
    set((s) => {
      const newHeatmap = samplePositions(s.heatmapPositions, snap.tick, snap.agents);
      // Throttle economy updates: only when money totals change or every 8 ticks.
      const econ = computeEconomySample(snap.tick, snap.agents);
      const last = s.economyHistory[s.economyHistory.length - 1];
      const shouldPush =
        !last ||
        last.attackerTotal !== econ.attackerTotal ||
        last.defenderTotal !== econ.defenderTotal ||
        snap.tick - last.tick >= 8;
      let economyHistory = s.economyHistory;
      if (shouldPush) {
        economyHistory = [...s.economyHistory, econ];
        if (economyHistory.length > MAX_ECONOMY) {
          economyHistory = economyHistory.slice(-MAX_ECONOMY);
        }
      }
      return {
        tick: snap.tick,
        round: snap.round,
        phase: snap.phase,
        secondsLeft: snap.secondsLeft,
        score: snap.score,
        agents: snap.agents,
        bomb: snap.bomb,
        mapName: snap.mapName,
        heatmapPositions: newHeatmap,
        economyHistory,
      };
    }),

  pushEvent: (e) =>
    set((s) => {
      const next = [e, ...s.eventFeed];
      if (next.length > MAX_EVENTS) next.length = MAX_EVENTS;
      // Auto-capture round_end as a round result if outcome is present.
      let roundResults = s.roundResults;
      if (e.kind === "round_end" && e.outcome) {
        const winner: RoundResult["winner"] =
          e.outcome === "attacker_elim" || e.outcome === "bomb_explode"
            ? "attacker"
            : e.outcome === "defender_elim" || e.outcome === "bomb_defused"
              ? "defender"
              : "draw";
        roundResults = [...s.roundResults, { round: s.round, outcome: e.outcome, winner }];
        if (roundResults.length > MAX_ROUNDS) {
          roundResults = roundResults.slice(-MAX_ROUNDS);
        }
      }
      return { eventFeed: next, roundResults };
    }),

  pushMessage: (m) =>
    set((s) => {
      const next = [m, ...s.recentMessages];
      if (next.length > MAX_MESSAGES) next.length = MAX_MESSAGES;
      return { recentMessages: next };
    }),

  setInspection: (insp) =>
    set((s) => ({
      byAgent: { ...s.byAgent, [insp.agentId]: insp },
    })),

  setMapName: (name) => set({ mapName: name }),

  setCurrentPolicies: (p) =>
    set((s) => ({
      currentPolicies: {
        yellow: p.yellow === undefined ? s.currentPolicies.yellow : p.yellow,
        blue: p.blue === undefined ? s.currentPolicies.blue : p.blue,
      },
    })),

  setAutoReload: (p) =>
    set((s) => ({
      autoReload: {
        yellow: p.yellow === undefined ? s.autoReload.yellow : p.yellow,
        blue: p.blue === undefined ? s.autoReload.blue : p.blue,
      },
    })),

  pushPolicyReload: (e) =>
    set((s) => {
      const id = `checkpoint:${e.name}`;
      const name = id; // mirrors the backend's ``policy_*_name`` convention
      const nextPolicies = {
        ...s.currentPolicies,
        [e.side]: { id, name },
      };
      return {
        currentPolicies: nextPolicies,
        lastPolicyReload: {
          side: e.side,
          name: e.name,
          previous: e.previous ?? null,
          ts: Date.now(),
        },
      };
    }),

  setAttentionWeights: (a) =>
    set((s) => {
      const inner: Record<string, number> = {};
      for (const t of a.targets) inner[t.agentId] = t.weight;
      return {
        attentionWeights: { ...s.attentionWeights, [a.observerId]: inner },
      };
    }),

  setTrainingStatus: (st) =>
    set((s) => ({
      trainingStatus: {
        ...s.trainingStatus,
        ...st,
      },
    })),

  pushMetricsSample: (m) =>
    set((s) => {
      const next = [...s.metricsHistory, m];
      if (next.length > MAX_METRICS) next.splice(0, next.length - MAX_METRICS);
      return { metricsHistory: next };
    }),

  pushRoundResult: (r) =>
    set((s) => {
      const next = [...s.roundResults, r];
      if (next.length > MAX_ROUNDS) next.splice(0, next.length - MAX_ROUNDS);
      return { roundResults: next };
    }),

  selectAgent: (id) => set({ selectedAgentId: id }),
  setRightTab: (t) => set({ rightTab: t }),
  toggleFov: () => set((s) => ({ showFov: !s.showFov })),
  toggleSound: () => set((s) => ({ showSound: !s.showSound })),
  toggleComms: () => set((s) => ({ showComms: !s.showComms })),
  toggleLastKnown: () => set((s) => ({ showLastKnown: !s.showLastKnown })),
  toggleHeatmap: () => set((s) => ({ showHeatmap: !s.showHeatmap })),
  setSpeed: (s) => set({ speed: s }),
  setPaused: (p) => set({ paused: p }),
  togglePaused: () => set((s) => ({ paused: !s.paused })),
  setMatchToken: () => set((s) => ({ matchToken: s.matchToken + 1 })),
  setUiMode: (mode) => {
    set({ uiMode: mode });
    // Mirror the choice to a dedicated, easy-to-discover localStorage
    // key as well so the value is obvious in DevTools and a future
    // settings-import flow can pick it up without parsing the larger
    // ``kivski-ui-state`` blob. Best-effort: a Safari private-mode
    // QuotaExceeded should not break the UI toggle.
    try {
      window.localStorage.setItem("kivski-ui-mode", mode);
    } catch {
      /* ignore */
    }
  },
  setSettingsOpen: (open) => set({ settingsOpen: open }),
  setSettingsTab: (tab) => set({ settingsTab: tab }),

  reset: () =>
    set({
      ...initialMatch,
      ...initialUI,
      ...initialInspection,
      ...initialFeed,
      ...initialMetrics,
    }),
}), {
  name: PERSIST_STORAGE_KEY,
  version: PERSIST_STORAGE_VERSION,
  storage: createJSONStorage(() => localStorage),
  // Whitelist exactly the UI-preference fields we want to survive a
  // reload. Anything not listed here is dropped from the persisted
  // payload, which means the next page load reads fresh defaults for
  // match data, inspection state, metrics history, etc.
  partialize: (state): PersistedUI => ({
    showFov: state.showFov,
    showSound: state.showSound,
    showComms: state.showComms,
    showLastKnown: state.showLastKnown,
    showHeatmap: state.showHeatmap,
    speed: state.speed,
    rightTab: state.rightTab,
    uiMode: state.uiMode,
    settingsTab: state.settingsTab,
  }),
  merge: (persisted, current) => {
    // Honour the dedicated ``kivski-ui-mode`` localStorage key if it
    // disagrees with the bundled persist payload — this lets us treat
    // that key as the source of truth (it's the one ``setUiMode``
    // writes, and the one a user is most likely to inspect / tweak).
    const override = readUiModeOverride();
    const merged = {
      ...current,
      ...(persisted as Partial<AppState> | undefined),
    } as AppState;
    if (override !== null) merged.uiMode = override;
    return merged;
  },
}));

// ---------- Selector helpers ----------

export const selectAttackers = (s: AppState): AgentSnapshot[] =>
  s.agents.filter((a) => a.side === "attacker");

export const selectDefenders = (s: AppState): AgentSnapshot[] =>
  s.agents.filter((a) => a.side === "defender");

/**
 * Sidebar grouping selectors. Agents are grouped by their persistent
 * `team` (yellow / blue) so the sidebar layout doesn't reshuffle when
 * sides switch at round 12 — only the role label / accent colour
 * changes inside each block. See `LeftSidebar` for the consuming view.
 */
export const selectYellowTeam = (s: AppState): AgentSnapshot[] =>
  s.agents.filter((a) => a.team === "yellow");

export const selectBlueTeam = (s: AppState): AgentSnapshot[] =>
  s.agents.filter((a) => a.team === "blue");

/** Current side role of a team, derived from any one of its agents. */
export const teamCurrentSide = (
  players: AgentSnapshot[],
): AgentSnapshot["side"] | null => {
  if (players.length === 0) return null;
  return players[0].side;
};

export const selectSelectedAgent = (s: AppState): AgentSnapshot | null => {
  if (!s.selectedAgentId) return null;
  return s.agents.find((a) => a.id === s.selectedAgentId) ?? null;
};

export const selectSelectedInspection = (s: AppState): AgentInspection | null => {
  if (!s.selectedAgentId) return null;
  return s.byAgent[s.selectedAgentId] ?? null;
};

/** Per-team economy snapshot (totals only). */
export const selectTeamEconomy = (
  s: AppState,
  side: AgentSnapshot["side"],
): { total: number; alive: number } => {
  let total = 0;
  let alive = 0;
  for (const a of s.agents) {
    if (a.side !== side) continue;
    total += a.money;
    if (a.isAlive) alive += 1;
  }
  return { total, alive };
};
