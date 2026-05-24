/**
 * Tiny REST + WebSocket client for talking to the Kivski FastAPI backend.
 *
 * - REST: thin fetch wrappers under /api/*
 * - WS:   subscribeMatch() returns an unsubscribe fn and auto-reconnects
 *         with exponential backoff. JSON frames are typed as `WSFrame`.
 *
 * Wire protocol notes:
 *
 *   1. There is no "default" match endpoint. Every WebSocket subscription
 *      must target a concrete `match_id`, which is obtained by POSTing to
 *      `/api/match/new`. `subscribeMatch()` performs that handshake
 *      transparently and renews the match if the backend forgets it
 *      (e.g. after a restart) — see `_ensureMatch`.
 *
 *   2. The backend ships raw `Snapshot.to_json_dict()` payloads with
 *      snake_case keys and int-encoded enums. We translate those into the
 *      frontend's camelCase `MatchSnapshot` shape via `decodeMatchSnapshot`
 *      so React components see a fully-typed value.
 */

import type { MatchSnapshot, TrainingStatus, WSFrame } from "./types";
import {
  decodeEventItem,
  decodeMapInfo,
  decodeMatchSnapshot,
  decodeMessageItem,
} from "./wire";

// ---------- REST ----------

const API_BASE = "/api";

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status} ${res.statusText}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as T;
}

export interface CheckpointInfo {
  id: string;
  name: string;
  step: number;
  createdAt: string;
  notes?: string | null;
  /**
   * Source kind from the backend's :func:`_entry_for` sidecar parse —
   * ``"best"`` for the promoted ``best.pt`` entry, otherwise
   * ``"checkpoint"`` (per-run snapshot). Drives the icon + label
   * choice in the UI (crown for best, camera for snapshot).
   */
  kind?: "best" | "checkpoint" | string;
  /**
   * Raw epoch-seconds (or undefined) when the checkpoint was written.
   * Used to derive a human-friendly "Xm ago" / "yesterday" label so
   * the dropdown reads like "Best so far (1247 eps · 4m ago)" instead
   * of "best · 1247".
   */
  timestamp?: number;
  /**
   * Pre-rendered friendly label like
   * "⭐ Best so far" / "📸 Snapshot @ ep 500" / "Latest checkpoint".
   * The dropdown renders this directly so the same translation runs in
   * one place (api-client) regardless of which consumer (drawer card
   * list, footer dropdown) shows it.
   */
  prettyLabel?: string;
  /**
   * Pre-rendered "Xm ago" / "yesterday" / "" fragment so the card can
   * show the age as a small secondary line without re-computing it on
   * every render.
   */
  ageLabel?: string;
}

// ---------- Time-since helper ----------

/**
 * Translate either an ISO-8601 timestamp string or epoch-seconds float
 * into a compact "Xs ago / Xm ago / Xh ago / yesterday / 3d ago" label.
 * Returns an empty string when the input can't be parsed.
 */
export function formatTimeAgo(input: string | number | null | undefined): string {
  if (input === null || input === undefined || input === "") return "";
  let ms: number;
  if (typeof input === "number") {
    // Backend ships epoch-seconds (now_unix()) → multiply by 1000.
    ms = input < 1e12 ? input * 1000 : input;
  } else {
    const numeric = Number(input);
    if (Number.isFinite(numeric)) {
      ms = numeric < 1e12 ? numeric * 1000 : numeric;
    } else {
      const parsed = Date.parse(input);
      if (Number.isNaN(parsed)) return "";
      ms = parsed;
    }
  }
  const diffMs = Date.now() - ms;
  if (!Number.isFinite(diffMs)) return "";
  if (diffMs < 0) return "just now";
  const sec = Math.floor(diffMs / 1000);
  if (sec < 60) return sec <= 5 ? "just now" : `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day === 1) return "yesterday";
  if (day < 7) return `${day}d ago`;
  const week = Math.floor(day / 7);
  if (week < 5) return `${week}w ago`;
  return `${Math.floor(day / 30)}mo ago`;
}

/**
 * Build the human-friendly label shown in the checkpoint dropdown.
 * The rules (in order):
 *   - kind == "best" or name == "best"  → "⭐ Best so far"
 *   - name endsWith "_latest" / starts with "latest" → "🕐 Latest checkpoint"
 *   - name like "snapshot_ep_<N>" or "main_ep_<N>"   → "📸 Snapshot @ ep N"
 *   - fallback → "Checkpoint @ ep N" (or just the name when no episode)
 */
function buildPrettyCheckpointLabel(
  name: string,
  kind: string | undefined,
  step: number,
): string {
  const lower = name.toLowerCase();
  const isBest = kind === "best" || lower === "best";
  if (isBest) return "Best so far";
  if (lower === "latest" || lower.endsWith("_latest")) {
    return "Latest checkpoint";
  }
  // Common trainer naming: "snapshot_ep_500" / "main_ep_128" / "ep_42".
  const epMatch = name.match(/(?:^|_)ep[_-]?(\d+)/i);
  const ep = epMatch ? Number(epMatch[1]) : step;
  if (lower.startsWith("snapshot")) {
    return ep ? `Snapshot @ ep ${ep}` : "Snapshot";
  }
  if (lower.startsWith("main")) {
    return ep ? `Checkpoint @ ep ${ep}` : `Checkpoint (${name})`;
  }
  // Generic fallback so unknown checkpoint naming schemes still read
  // sensibly.
  return ep ? `Checkpoint @ ep ${ep}` : name;
}

/**
 * Single option in the "policy picker" dropdown for the comparison-match
 * setup modal. Backend (v0.3) exposes
 * ``GET /api/checkpoints/recommended`` returning ``{options: [...]}`` so
 * we can render a curated list (random / scripted variants + latest +
 * best + named ckpts) without leaking implementation details (paths,
 * heuristics) into the frontend.
 */
export interface PolicyOption {
  /** Stable identifier passed back to /api/match/new as `policy_yellow` / `policy_blue`. */
  id: string;
  /** Human-readable label shown in the dropdown. */
  name: string;
}

/**
 * Defense-in-depth fallback used when ``/api/checkpoints/recommended``
 * is not yet wired up on the backend. Keeps the comparison-match UI
 * functional with the always-available baselines.
 */
const FALLBACK_POLICY_OPTIONS: readonly PolicyOption[] = [
  { id: "random", name: "Random" },
  { id: "scripted_rush", name: "Scripted (Rush)" },
  { id: "scripted_hold", name: "Scripted (Hold)" },
  { id: "latest", name: "Latest Checkpoint" },
  { id: "best", name: "Best Checkpoint" },
] as const;

interface RawRecommendedResponse {
  options?: Array<{ id?: unknown; name?: unknown } | string>;
}

/**
 * Fetch the curated list of policy options for the comparison-match
 * picker. Falls back to a hardcoded baseline list when the endpoint is
 * unavailable so the modal always has something to render — the user
 * can still pick `random` / `scripted_rush` / `scripted_hold` against
 * an early backend.
 */
export async function getRecommendedPolicies(): Promise<PolicyOption[]> {
  try {
    const res = await fetch(`${API_BASE}/checkpoints/recommended`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!res.ok) return [...FALLBACK_POLICY_OPTIONS];
    const raw = (await res.json()) as RawRecommendedResponse | unknown;
    const list =
      raw && typeof raw === "object" && Array.isArray((raw as RawRecommendedResponse).options)
        ? (raw as RawRecommendedResponse).options!
        : Array.isArray(raw)
          ? (raw as RawRecommendedResponse["options"])!
          : [];
    const parsed: PolicyOption[] = [];
    for (const entry of list ?? []) {
      if (typeof entry === "string") {
        if (entry.length === 0) continue;
        parsed.push({ id: entry, name: entry });
      } else if (entry && typeof entry === "object") {
        const id = typeof entry.id === "string" ? entry.id : null;
        if (id === null || id.length === 0) continue;
        const name = typeof entry.name === "string" && entry.name.length > 0 ? entry.name : id;
        parsed.push({ id, name });
      }
    }
    if (parsed.length === 0) return [...FALLBACK_POLICY_OPTIONS];
    return parsed;
  } catch {
    return [...FALLBACK_POLICY_OPTIONS];
  }
}

/**
 * Backend wire shape for /api/checkpoints — list_checkpoints() returns
 * `{checkpoints: [...], loaded: name|null}` where each entry has
 * `{name, path, size_bytes, episodes, timestamp, metadata, loaded}`.
 * We unwrap + translate here so the rest of the app sees plain
 * `CheckpointInfo[]`.
 */
interface RawCheckpointsResponse {
  checkpoints?: Array<{
    name?: string;
    episodes?: number | null;
    timestamp?: string | number | null;
    metadata?: Record<string, unknown> | null;
    kind?: string | null;
  }>;
  loaded?: string | null;
}

export async function getCheckpoints(): Promise<CheckpointInfo[]> {
  try {
    const res = await fetch(`${API_BASE}/checkpoints`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!res.ok) return [];
    const raw = (await res.json()) as RawCheckpointsResponse | unknown;
    const list =
      raw && typeof raw === "object" && Array.isArray((raw as RawCheckpointsResponse).checkpoints)
        ? (raw as RawCheckpointsResponse).checkpoints!
        : Array.isArray(raw)
          ? (raw as RawCheckpointsResponse["checkpoints"])!
          : [];
    return list.map((c) => {
      const name = String(c?.name ?? "unnamed");
      const ts = c?.timestamp;
      const step = typeof c?.episodes === "number" ? c.episodes : 0;
      const kind =
        typeof c?.kind === "string" && c.kind.length > 0 ? c.kind : "checkpoint";
      const timestampNum =
        typeof ts === "number"
          ? ts
          : typeof ts === "string" && Number.isFinite(Number(ts))
            ? Number(ts)
            : typeof ts === "string"
              ? Date.parse(ts) / 1000
              : undefined;
      const ageLabel = formatTimeAgo(ts ?? null);
      const prettyLabel = buildPrettyCheckpointLabel(name, kind, step);
      return {
        id: name,
        name,
        step,
        createdAt: ts == null ? "" : String(ts),
        notes:
          c?.metadata && typeof c.metadata === "object" && Object.keys(c.metadata).length > 0
            ? JSON.stringify(c.metadata)
            : null,
        kind,
        timestamp:
          typeof timestampNum === "number" && Number.isFinite(timestampNum)
            ? timestampNum
            : undefined,
        ageLabel,
        prettyLabel,
      };
    });
  } catch {
    return [];
  }
}

/**
 * Available training configuration presets (rendered in the dropdown).
 * Always falls back to an empty list if the endpoint is unavailable.
 */
export interface TrainingConfigInfo {
  id: string;
  name: string;
  description?: string;
}

export async function getTrainingConfigs(): Promise<TrainingConfigInfo[]> {
  try {
    const res = await fetch(`${API_BASE}/training/configs`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!res.ok) return [];
    return (await res.json()) as TrainingConfigInfo[];
  } catch {
    return [];
  }
}

// ---------- Training goals ----------

/**
 * High-level training intent the user picks instead of a raw config
 * filename. Each goal maps to one of the YAML presets under
 * ``configs/`` — the choice is the user-facing UI surface, the raw
 * filename remains the wire/CLI surface.
 *
 * Mapping (chosen so the *biggest* speedup hits the "Watch progress
 * fast" button without sacrificing the viewer's 5v5 distribution):
 *   - ``watch``    → ``configs/turbo.yaml``   (5v5, frame_skip=6, 48 envs)
 *   - ``balanced`` → ``configs/viewer.yaml``  (5v5, frame_skip=4, 32 envs)
 *   - ``quality``  → ``configs/fast.yaml``    (curriculum 1v1→5v5)
 */
export type TrainingGoal = "watch" | "balanced" | "quality";

export interface TrainingGoalSpec {
  id: TrainingGoal;
  icon: string;
  title: string;
  blurb: string;
  /** Wire config path. Mirrors the ``id`` field in ``TrainingConfigInfo``. */
  configPath: string;
}

export const TRAINING_GOALS: readonly TrainingGoalSpec[] = [
  {
    id: "watch",
    icon: "👁",
    title: "Watch progress fast",
    blurb: "See agents stop wandering randomly within 1–3 hours.",
    configPath: "configs/turbo.yaml",
  },
  {
    id: "balanced",
    icon: "⚖",
    title: "Balanced (recommended)",
    blurb: "Good trade-off between learning speed and final quality.",
    configPath: "configs/viewer.yaml",
  },
  {
    id: "quality",
    icon: "🏆",
    title: "Max final skill",
    blurb: "Best end-game policy. Takes longer before the viewer pops.",
    configPath: "configs/fast.yaml",
  },
] as const;

/** Default goal for first-time visitors who have never picked one. */
export const DEFAULT_TRAINING_GOAL: TrainingGoal = "balanced";

/** Lookup a goal spec by id. Falls back to ``balanced`` for unknown ids. */
export function getTrainingGoalSpec(id: TrainingGoal | string | null | undefined): TrainingGoalSpec {
  const match = TRAINING_GOALS.find((g) => g.id === id);
  return match ?? TRAINING_GOALS[1]; // balanced is index 1
}

/**
 * Translate a config path (the backend's :class:`TrainingConfigInfo.id`)
 * back to a training-goal id. Returns ``null`` for paths that don't map
 * to one of the three curated goals — used to drive the "override"
 * visual state when the user picks a raw config in the Advanced reveal.
 */
export function inferGoalFromConfigPath(path: string | null | undefined): TrainingGoal | null {
  if (!path) return null;
  const lower = path.toLowerCase();
  for (const goal of TRAINING_GOALS) {
    if (goal.configPath.toLowerCase() === lower) return goal.id;
  }
  return null;
}

/**
 * Backend wire shape for /api/training/status:
 *   {running, job_id, pid, started_at, exit_code?, config_path?, episodes?,
 *    resume_from?, log_tail: string[], log_path?}
 * The frontend `TrainingStatus` uses camelCase + a subset of fields
 * (running/episode/totalEpisodes/policyLoss/valueLoss/entropy). Live
 * metric fields (policyLoss/...) are pushed via the WS `metrics_sample`
 * frame, so the /status endpoint only fills running + episode budgets.
 */
interface RawTrainingStatus {
  running?: boolean;
  episodes?: number | null;
  job_id?: string | null;
  pid?: number | null;
  started_at?: number;
  /** Cumulative training time across all runs (persistent across restarts). */
  training_clock_total_seconds?: number;
  /** Wall-clock since the running trainer started; 0 when idle. */
  current_session_seconds?: number;
}

/**
 * Wire shape for ``GET /api/training/resume-target``: tells the UI
 * whether the next ``POST /api/training/start`` would auto-resume from
 * a checkpoint on disk and, if so, which one. Used to render the
 * "Resumes from <X>" tooltip on the Start button so the user never
 * accidentally re-trains from scratch when they meant to continue.
 */
export interface ResumeTargetInfo {
  available: boolean;
  path: string | null;
  name: string | null;
}

export async function getResumeTarget(): Promise<ResumeTargetInfo> {
  try {
    const res = await fetch(`${API_BASE}/training/resume-target`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!res.ok) return { available: false, path: null, name: null };
    const raw = (await res.json()) as Partial<ResumeTargetInfo> | null;
    if (!raw || typeof raw !== "object") return { available: false, path: null, name: null };
    return {
      available: Boolean(raw.available),
      path: typeof raw.path === "string" ? raw.path : null,
      name: typeof raw.name === "string" ? raw.name : null,
    };
  } catch {
    return { available: false, path: null, name: null };
  }
}

/** Get a one-shot snapshot of the training loop state. */
export async function getTrainingStatus(): Promise<TrainingStatus | null> {
  try {
    const res = await fetch(`${API_BASE}/training/status`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!res.ok) return null;
    const raw = (await res.json()) as RawTrainingStatus | null;
    if (!raw || typeof raw !== "object") return null;
    const totalTrainedSeconds =
      typeof raw.training_clock_total_seconds === "number" &&
      Number.isFinite(raw.training_clock_total_seconds)
        ? raw.training_clock_total_seconds
        : undefined;
    const currentSessionSeconds =
      typeof raw.current_session_seconds === "number" &&
      Number.isFinite(raw.current_session_seconds)
        ? raw.current_session_seconds
        : undefined;
    return {
      running: Boolean(raw.running),
      episode: 0, // backend does not track in-flight episode for V1; comes via WS metrics_sample
      totalEpisodes: typeof raw.episodes === "number" ? raw.episodes : 0,
      totalTrainedSeconds,
      currentSessionSeconds,
    };
  } catch {
    return null;
  }
}

export type Command =
  | { type: "pause" }
  | { type: "resume" }
  | { type: "set_speed"; speed: number }
  | { type: "reset_match" }
  | { type: "start_training"; configId?: string }
  | { type: "stop_training" }
  | { type: "run_episodes"; n: number; configId?: string }
  | { type: "save_checkpoint"; name?: string }
  | { type: "load_checkpoint"; id: string };

/**
 * Module-level reference to the currently-active match id, set by
 * `subscribeMatch` after the create-match handshake. Match-scoped
 * commands (pause/resume/speed/reset) read this so they don't need
 * the caller to thread the id through every layer.
 *
 * Kept in sync with the store's `currentMatchId` field — both are
 * authoritative, but the module-level ref avoids a circular import
 * here (`store` imports types from this file).
 */
let _currentMatchId: string | null = null;

/** External setter used by `subscribeMatch` (or tests) to override the id. */
export function setCurrentMatchId(id: string | null): void {
  _currentMatchId = id;
}

export function getCurrentMatchId(): string | null {
  return _currentMatchId;
}

type CommandResult =
  | { ok: true; alreadyRunning?: boolean; detail?: string }
  | { ok: false; error: string };

/** Common error wrapper around `fetch` so each branch in `postCommand` is one line. */
async function postOrError(url: string, init?: RequestInit): Promise<CommandResult> {
  try {
    const res = await fetch(url, init);
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      return { ok: false, error: `${res.status} ${res.statusText}: ${body.slice(0, 200)}` };
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}

/**
 * Variant of `postOrError` that treats HTTP 409 ("already running") as a
 * graceful no-op rather than an error. Used by the training-start
 * branches so a double-click on the Start button doesn't surface a
 * scary error toast — the second POST simply reports
 * `{ok: true, alreadyRunning: true}` and the UI carries on.
 */
async function postOrError409Graceful(
  url: string,
  init?: RequestInit,
): Promise<CommandResult> {
  try {
    const res = await fetch(url, init);
    if (res.status === 409) {
      const body = await res.text().catch(() => "");
      return { ok: true, alreadyRunning: true, detail: body.slice(0, 200) };
    }
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      return { ok: false, error: `${res.status} ${res.statusText}: ${body.slice(0, 200)}` };
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}

/**
 * Dispatch a UI-level command to the correct backend endpoint.
 *
 * The backend deliberately exposes one endpoint per action (each with
 * its own validation + schema) rather than a single `/api/command`
 * collector — so this function fans the typed union out to:
 *
 *   pause/resume/speed/reset → /api/match/{currentMatchId}/...
 *   start_training/stop      → /api/training/start | /stop
 *   run_episodes             → /api/training/start with `episodes`
 *   save_checkpoint          → no-op in V1 (trainer auto-saves)
 *   load_checkpoint          → /api/checkpoints/{name}/load
 */
export async function postCommand(cmd: Command): Promise<CommandResult> {
  const matchId = _currentMatchId;
  const needsMatch = (): CommandResult =>
    matchId === null
      ? { ok: false, error: "no active match; reload the page" }
      : { ok: true };

  switch (cmd.type) {
    case "pause": {
      const guard = needsMatch();
      if (!guard.ok) return guard;
      return postOrError(`${API_BASE}/match/${encodeURIComponent(matchId!)}/pause`, {
        method: "POST",
      });
    }
    case "resume": {
      const guard = needsMatch();
      if (!guard.ok) return guard;
      return postOrError(`${API_BASE}/match/${encodeURIComponent(matchId!)}/resume`, {
        method: "POST",
      });
    }
    case "set_speed": {
      const guard = needsMatch();
      if (!guard.ok) return guard;
      const url =
        `${API_BASE}/match/${encodeURIComponent(matchId!)}/speed` +
        `?multiplier=${encodeURIComponent(String(cmd.speed))}`;
      return postOrError(url, { method: "POST" });
    }
    case "reset_match": {
      const guard = needsMatch();
      if (!guard.ok) return guard;
      return postOrError(`${API_BASE}/match/${encodeURIComponent(matchId!)}/reset`, {
        method: "POST",
      });
    }
    case "start_training": {
      const body = {
        config: cmd.configId || "configs/default.yaml",
        episodes: null,
        checkpoint: null,
      };
      return postOrError409Graceful(`${API_BASE}/training/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    }
    case "stop_training": {
      return postOrError(`${API_BASE}/training/stop`, { method: "POST" });
    }
    case "run_episodes": {
      const body = {
        config: cmd.configId || "configs/default.yaml",
        episodes: Math.max(1, Math.floor(cmd.n)),
        checkpoint: null,
      };
      return postOrError409Graceful(`${API_BASE}/training/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    }
    case "save_checkpoint": {
      // V1: the trainer subprocess writes checkpoints periodically
      // (controlled by configs/default.yaml `checkpoint_interval`).
      // There is no on-demand save endpoint -- surface that as a
      // "soft success" so the button gives feedback rather than 404.
      return { ok: true };
    }
    case "load_checkpoint": {
      const id = cmd.id.trim();
      if (id === "") return { ok: false, error: "no checkpoint selected" };
      return postOrError(`${API_BASE}/checkpoints/${encodeURIComponent(id)}/load`, {
        method: "POST",
      });
    }
    default: {
      // Exhaustiveness check -- a new Command variant will surface here as a TS error.
      const _exhaustive: never = cmd;
      return {
        ok: false,
        error: `unknown command: ${(_exhaustive as { type: string }).type}`,
      };
    }
  }
}

// ---------- Match lifecycle ----------

/** Body accepted by the backend's `/api/match/new` endpoint. */
export interface CreateMatchBody {
  map?: string;
  seed?: number;
  config?: string;
  policy_yellow?: string;
  policy_blue?: string;
  autostart?: boolean;
  /**
   * Per-side auto-reload of the trained-policy adapter. When the
   * backend completes a round and a newer checkpoint shows up under
   * ``models/checkpoints/*.pt``, it hot-swaps the policy and emits a
   * ``policy_reload`` WS event so the viewer can update its UI. Only
   * meaningful when the matching ``policy_*`` selects a checkpoint
   * ("latest" / "best" / explicit path); the server silently
   * normalises the flag to False for Random / Scripted policies.
   */
  auto_reload_yellow?: boolean;
  auto_reload_blue?: boolean;
}

export interface CreateMatchResult {
  match_id: string;
  map?: string;
  seed?: number | null;
  policy_yellow?: string | null;
  policy_blue?: string | null;
  /**
   * Human-readable label for the selected yellow policy (e.g. "Random",
   * "Best Checkpoint", "run-001-ep12000"). Backend v0.3 returns these so
   * the header can display the active policies without translating the
   * id locally. Optional for backwards compat with older backends.
   */
  policy_yellow_name?: string | null;
  policy_blue_name?: string | null;
  /**
   * Effective auto-reload state per side. Mirrors the request body but
   * normalised by the server: requesting auto-reload on a Random or
   * Scripted side returns False here even when the body asked for True.
   * Optional for backwards compat with older backends (treat undefined
   * as False).
   */
  auto_reload_yellow?: boolean;
  auto_reload_blue?: boolean;
  paused?: boolean;
}

/**
 * Create a new match session on the backend and return its descriptor.
 * Every WebSocket subscription targets the resulting `match_id`.
 *
 * The optional `signal` allows callers (notably `subscribeMatch` running
 * under React StrictMode's mount → cleanup → mount cycle) to cancel an
 * in-flight create handshake when the surrounding effect tears down.
 * Without this the StrictMode dance would silently create two matches.
 */
export async function createMatch(
  body: CreateMatchBody = {},
  signal?: AbortSignal,
): Promise<CreateMatchResult> {
  const payload: CreateMatchBody = { map: "dustline", ...body };
  const res = await fetch(`${API_BASE}/match/new`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  return jsonOrThrow<CreateMatchResult>(res);
}

/**
 * Fire-and-forget delete used to clean up a match that was created by
 * an aborted `subscribeMatch` handshake (StrictMode unmount before the
 * POST response arrived). Failures are intentionally swallowed — at
 * worst the match lingers until the backend's GC sweeps it.
 */
function fireAndForgetDeleteMatch(matchId: string): void {
  try {
    void fetch(`${API_BASE}/match/${encodeURIComponent(matchId)}`, {
      method: "DELETE",
      keepalive: true,
    }).catch(() => {
      /* ignore */
    });
  } catch {
    /* ignore */
  }
}

// ---------- WebSocket ----------

export interface WSHandle {
  /** Close permanently and stop reconnecting. */
  close: () => void;
}

export interface SubscribeOpts {
  /**
   * Optional override of the full WebSocket URL. When set, `subscribeMatch`
   * skips the match-create handshake and uses the URL as-is — useful for
   * tests that hit a fixture endpoint.
   */
  url?: string;
  /** Optional create-match body. Defaults to `{map: "dustline"}`. */
  createBody?: CreateMatchBody;
  onFrame: (frame: WSFrame) => void;
  onStatus?: (status: "connecting" | "open" | "closed" | "error") => void;
  /**
   * Notified whenever the backend match id changes (initial connect,
   * session-lost reconnect, or final close → null). The frontend uses
   * this to mirror the id into the global store so REST commands can
   * target /api/match/{id}/...
   */
  onMatchId?: (id: string | null) => void;
  /**
   * Notified whenever the create-match handshake completes with the
   * backend's authoritative policy descriptors
   * (`policy_yellow`/`policy_yellow_name` etc.). Used by the header to
   * display which policies are currently being compared.
   */
  onPolicies?: (yellow: PolicyAssignment | null, blue: PolicyAssignment | null) => void;
  /**
   * Notified whenever the create-match handshake completes with the
   * server's *effective* auto-reload flags. Mirrors what the user
   * requested via `createMatch(body)` but normalised by the backend
   * (a request to auto-reload a Random side comes back as False).
   */
  onAutoReload?: (yellow: boolean, blue: boolean) => void;
  /** Initial reconnect delay in ms. Doubles per attempt up to 15s cap. */
  baseDelayMs?: number;
}

/**
 * Same shape as `store.ts`'s `PolicyAssignment` — duplicated here to
 * avoid a cyclic import (the store consumes types from this file).
 */
export interface PolicyAssignment {
  id: string;
  name: string;
}

/** Build the WebSocket URL for a concrete `match_id` using the page origin. */
function wsUrlForMatch(matchId: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/match/${encodeURIComponent(matchId)}`;
}

/**
 * Build a `PolicyAssignment` from the create-match response fields.
 * Returns null when no policy id was reported (older backends, or
 * "auto" default not advertised on the wire).
 */
function packPolicy(
  id: string | null | undefined,
  name: string | null | undefined,
): PolicyAssignment | null {
  if (typeof id !== "string" || id.length === 0) return null;
  const display = typeof name === "string" && name.length > 0 ? name : id;
  return { id, name: display };
}

/**
 * Type guard: did the just-received frame indicate the backend has
 * forgotten our match session (so a fresh one must be created)?
 */
function isSessionLostFrame(frame: unknown): boolean {
  if (typeof frame !== "object" || frame === null) return false;
  const f = frame as { type?: unknown; detail?: unknown };
  if (f.type !== "error") return false;
  if (typeof f.detail !== "string") return false;
  return /match\b.*\bnot found/i.test(f.detail);
}

/**
 * Map a raw incoming JSON frame from the backend onto a list of typed
 * `WSFrame` objects the rest of the app consumes. Returns an array
 * because a single backend `snapshot` frame can fan out into one
 * snapshot frame plus N `event`/`message` frames (the backend keeps
 * per-tick comms and combat events nested inside the snapshot payload
 * rather than streaming them as their own frames).
 */
function adaptIncomingFrame(parsed: unknown, mapName: string): WSFrame[] {
  if (typeof parsed !== "object" || parsed === null) return [];
  const f = parsed as { type?: unknown };
  if (typeof f.type !== "string") return [];

  const kind = f.type;
  const obj = parsed as Record<string, unknown>;

  switch (kind) {
    case "snapshot": {
      const data = obj.data;
      const snap: MatchSnapshot = decodeMatchSnapshot(data, mapName);
      const out: WSFrame[] = [{ type: "snapshot", data: snap }];
      // Fan out per-tick comms and combat events that live inside the
      // raw snapshot payload. These are transient (cleared next tick),
      // so we forward each one to the App-level handler which routes
      // them into the comms/events feeds.
      const rawData = (typeof data === "object" && data !== null
        ? (data as Record<string, unknown>)
        : {}) as Record<string, unknown>;
      const rawMessages = Array.isArray(rawData.messages) ? rawData.messages : [];
      for (const rm of rawMessages) {
        out.push({ type: "message", data: decodeMessageItem(rm, snap.tick) });
      }
      const rawEvents = Array.isArray(rawData.events) ? rawData.events : [];
      for (const re of rawEvents) {
        out.push({ type: "event", data: decodeEventItem(re, snap.tick) });
      }
      return out;
    }
    case "map_info": {
      const info = decodeMapInfo(obj.data);
      return [{ type: "map_info", data: info }];
    }
    case "match_done": {
      const matchId = typeof obj.match_id === "string" ? obj.match_id : undefined;
      return [{ type: "match_done", matchId }];
    }
    case "pong": {
      const ts = typeof obj.ts === "number" ? obj.ts : Date.now() / 1000;
      return [{ type: "pong", ts }];
    }
    case "ack": {
      const forKind = typeof obj.for === "string" ? obj.for : "unknown";
      return [{ type: "ack", for: forKind, ...obj }];
    }
    case "error": {
      // Backend uses {type:"error", detail:"..."}; frontend WSFrame
      // historically used {data:{message:string}}. Adapt both shapes.
      const detail = typeof obj.detail === "string" ? obj.detail : undefined;
      const dataField = obj.data as { message?: unknown } | undefined;
      const message =
        detail ??
        (dataField && typeof dataField.message === "string" ? dataField.message : "unknown error");
      return [{ type: "error", data: { message } }];
    }
    case "metrics_sample": {
      // Normalise snake_case → camelCase so the rest of the app sees
      // the typed `MetricsSample` shape. The backend's
      // MetricsBroadcaster ships `{episode, winrate_vs_random,
      // winrate_vs_scripted, policy_loss, value_loss, entropy}`.
      const data = (obj.data ?? {}) as Record<string, unknown>;
      const pickNumber = (k: string): number | undefined => {
        const v = data[k];
        return typeof v === "number" && Number.isFinite(v) ? v : undefined;
      };
      const episode = pickNumber("episode") ?? 0;
      const sample = {
        episode,
        winrateVsRandom: pickNumber("winrate_vs_random") ?? pickNumber("winrateVsRandom"),
        winrateVsScripted:
          pickNumber("winrate_vs_scripted") ?? pickNumber("winrateVsScripted"),
        policyLoss: pickNumber("policy_loss") ?? pickNumber("policyLoss"),
        valueLoss: pickNumber("value_loss") ?? pickNumber("valueLoss"),
        entropy: pickNumber("entropy"),
      };
      return [{ type: "metrics_sample", data: sample }];
    }
    case "policy_reload": {
      // Hot-swap notification from the backend's per-round auto-reload.
      // Normalise the side field to the closed enum the frontend uses
      // and drop the frame if it's malformed (the rest of the pipeline
      // expects `side` to be 'yellow' | 'blue').
      const data = (obj.data ?? {}) as Record<string, unknown>;
      const side = data.side === "yellow" || data.side === "blue" ? data.side : null;
      const name = typeof data.name === "string" ? data.name : null;
      if (side === null || name === null) return [];
      const path = typeof data.path === "string" ? data.path : undefined;
      const previous =
        typeof data.previous === "string"
          ? data.previous
          : data.previous === null
            ? null
            : undefined;
      return [
        {
          type: "policy_reload",
          data: { side, name, path, previous },
        },
      ];
    }
    case "event":
    case "message":
    case "inspect":
    case "attention_update":
    case "training_status":
    case "round_result":
    case "hello":
      // These frame types pass through unchanged. The runtime shape
      // matches the WSFrame contract (still loosely-typed `as` cast —
      // upstream backend doesn't ship them yet in V1).
      return [parsed as WSFrame];
    default:
      console.warn("[kivski] unknown WS frame type:", kind);
      return [];
  }
}

/**
 * Subscribe to the live match WebSocket. Returns a handle whose `.close()`
 * stops the reconnect loop.
 *
 * Behaviour:
 *   - On first connect (and after the session goes missing) we POST to
 *     `/api/match/new` to obtain a fresh `match_id` and open the WebSocket
 *     against `/ws/match/{match_id}`.
 *   - On transient WebSocket failures we reconnect with exponential
 *     backoff, reusing the same `match_id` until the backend tells us
 *     the session is gone — at which point we create a new match.
 *   - If creating a new match fails repeatedly (e.g. backend offline)
 *     the backoff keeps growing, capped at 15s, but never gives up
 *     unless `.close()` is called.
 */
export function subscribeMatch(opts: SubscribeOpts): WSHandle {
  const baseDelay = opts.baseDelayMs ?? 500;
  const createBody = opts.createBody ?? { map: "dustline" };
  const forcedUrl = opts.url ?? null;

  let closed = false;
  let attempt = 0;
  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let matchId: string | null = null;
  let mapName = createBody.map ?? "dustline";
  let needsFreshMatch = forcedUrl === null;
  // AbortController for the in-flight create-match POST. close() aborts
  // it so React StrictMode's mount → cleanup → mount cycle can't leak a
  // second match.
  let pendingCreateAbort: AbortController | null = null;

  const status = (s: "connecting" | "open" | "closed" | "error") => {
    opts.onStatus?.(s);
  };

  /** Ensure we have a current `matchId`, creating one if necessary. */
  const ensureMatch = async (): Promise<string | null> => {
    if (forcedUrl) return null;
    if (matchId !== null && !needsFreshMatch) return matchId;
    pendingCreateAbort = new AbortController();
    const signal = pendingCreateAbort.signal;
    try {
      const res = await createMatch(createBody, signal);
      // If close() fired while the POST was in flight, the match was
      // still created server-side. Reap it before bailing out so we
      // don't leak orphaned sessions.
      if (closed) {
        fireAndForgetDeleteMatch(res.match_id);
        return null;
      }
      matchId = res.match_id;
      if (res.map) mapName = res.map;
      needsFreshMatch = false;

      // Make the new id discoverable by `postCommand` (module-level ref)
      // and the React store (via the consumer-provided callback).
      setCurrentMatchId(matchId);
      opts.onMatchId?.(matchId);

      // Forward policy descriptors (used by the header). Backend v0.3+
      // reports `policy_*` (id) + `policy_*_name` (display) on the
      // create response; older backends omit them and we forward `null`.
      const yellowPolicy = packPolicy(res.policy_yellow, res.policy_yellow_name);
      const bluePolicy = packPolicy(res.policy_blue, res.policy_blue_name);
      opts.onPolicies?.(yellowPolicy, bluePolicy);
      // v0.6+: echo the per-side auto-reload state. Older backends omit
      // the fields → treat as off.
      opts.onAutoReload?.(
        Boolean(res.auto_reload_yellow),
        Boolean(res.auto_reload_blue),
      );

      console.warn("[kivski] created match", matchId, "on", mapName);
      return matchId;
    } catch (err) {
      // AbortError is the expected outcome when close() ran while the
      // POST was pending (StrictMode cleanup) — stay quiet so the
      // console only carries genuine failures.
      if ((err as { name?: string })?.name === "AbortError") {
        return null;
      }

      console.warn("[kivski] match create failed:", err);
      status("error");
      return null;
    } finally {
      pendingCreateAbort = null;
    }
  };

  const connect = async () => {
    if (closed) return;
    status("connecting");

    let url: string;
    if (forcedUrl) {
      url = forcedUrl;
    } else {
      const id = await ensureMatch();
      if (id === null) {
        scheduleReconnect();
        return;
      }
      url = wsUrlForMatch(id);
    }

    try {
      ws = new WebSocket(url);
    } catch (err) {

      console.warn("[kivski] WS construct failed:", err);
      scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      attempt = 0;
      status("open");
    };

    ws.onmessage = (ev) => {
      try {
        const raw = typeof ev.data === "string" ? ev.data : "";
        if (!raw) return;
        const parsed = JSON.parse(raw) as unknown;

        // If the backend tells us the match is gone, force a new one
        // before the next reconnect.
        if (isSessionLostFrame(parsed)) {
          needsFreshMatch = true;
          matchId = null;
        }

        const adapted = adaptIncomingFrame(parsed, mapName);
        if (adapted.length === 0) return;

        for (const frame of adapted) {
          // Track the map name from the initial map_info frame so any
          // subsequent snapshots are tagged with the correct map.
          if (frame.type === "map_info") {
            mapName = frame.data.mapName;
          }
          opts.onFrame(frame);
        }
      } catch (err) {

        console.warn("[kivski] WS parse error:", err);
      }
    };

    ws.onerror = () => {
      status("error");
      // 'close' will fire right after, which schedules reconnect.
    };

    ws.onclose = (ev) => {
      status("closed");
      ws = null;
      // 4404 is the "match not found" close code used by the backend.
      if (ev && ev.code === 4404) {
        needsFreshMatch = true;
        matchId = null;
      }
      scheduleReconnect();
    };
  };

  const scheduleReconnect = () => {
    if (closed) return;
    attempt += 1;
    const delay = Math.min(15_000, baseDelay * 2 ** Math.min(attempt - 1, 6));
    // small jitter to avoid thundering herd if many clients reconnect at once
    const jitter = Math.floor(Math.random() * 250);
    reconnectTimer = setTimeout(() => {
      // Drop the awaited promise — the loop self-recovers on failure.
      void connect();
    }, delay + jitter);
  };

  void connect();

  return {
    close: () => {
      closed = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      // Cancel any in-flight create-match POST. If the server already
      // accepted the request before we abort, ensureMatch()'s
      // post-await `closed` check will fire-and-forget DELETE the
      // resulting match so we don't leak it.
      if (pendingCreateAbort) {
        try {
          pendingCreateAbort.abort();
        } catch {
          /* ignore */
        }
        pendingCreateAbort = null;
      }
      if (ws) {
        try {
          ws.close();
        } catch {
          /* ignore */
        }
        ws = null;
      }
      // If a match id was already negotiated (the POST resolved before
      // close() ran) but we haven't actually started consuming it,
      // delete it server-side. Best-effort: the backend has a TTL sweep
      // for orphaned matches so a transient failure here is harmless.
      if (matchId !== null) {
        fireAndForgetDeleteMatch(matchId);
        matchId = null;
      }
      // Clear the id so any post-close postCommand fails fast with
      // "no active match" instead of POST-ing to a stale URL.
      setCurrentMatchId(null);
      opts.onMatchId?.(null);
    },
  };
}
