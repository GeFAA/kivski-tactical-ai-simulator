import { create } from "zustand";
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
}

export type RightTab = "events" | "inspector" | "comms" | "metrics" | "sys";

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

export const useStore = create<AppState>((set) => ({
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

  reset: () =>
    set({
      ...initialMatch,
      ...initialUI,
      ...initialInspection,
      ...initialFeed,
      ...initialMetrics,
    }),
}));

// ---------- Selector helpers ----------

export const selectAttackers = (s: AppState): AgentSnapshot[] =>
  s.agents.filter((a) => a.side === "attacker");

export const selectDefenders = (s: AppState): AgentSnapshot[] =>
  s.agents.filter((a) => a.side === "defender");

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
