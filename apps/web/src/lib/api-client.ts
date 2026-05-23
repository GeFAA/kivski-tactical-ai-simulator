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
import { decodeMapInfo, decodeMatchSnapshot } from "./wire";

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
}

export async function getCheckpoints(): Promise<CheckpointInfo[]> {
  const res = await fetch(`${API_BASE}/checkpoints`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  return jsonOrThrow<CheckpointInfo[]>(res);
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

/** Get a one-shot snapshot of the training loop state. */
export async function getTrainingStatus(): Promise<TrainingStatus | null> {
  try {
    const res = await fetch(`${API_BASE}/training/status`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!res.ok) return null;
    return (await res.json()) as TrainingStatus;
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
  | { type: "run_episodes"; n: number }
  | { type: "save_checkpoint"; name?: string }
  | { type: "load_checkpoint"; id: string };

export async function postCommand(cmd: Command): Promise<{ ok: true } | { ok: false; error: string }> {
  try {
    const res = await fetch(`${API_BASE}/command`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cmd),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      return { ok: false, error: `${res.status} ${res.statusText}: ${body.slice(0, 200)}` };
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
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
}

export interface CreateMatchResult {
  match_id: string;
  map?: string;
  seed?: number | null;
  policy_yellow?: string | null;
  policy_blue?: string | null;
  paused?: boolean;
}

/**
 * Create a new match session on the backend and return its descriptor.
 * Every WebSocket subscription targets the resulting `match_id`.
 */
export async function createMatch(body: CreateMatchBody = {}): Promise<CreateMatchResult> {
  const payload: CreateMatchBody = { map: "dustline", ...body };
  const res = await fetch(`${API_BASE}/match/new`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return jsonOrThrow<CreateMatchResult>(res);
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
  /** Initial reconnect delay in ms. Doubles per attempt up to 15s cap. */
  baseDelayMs?: number;
}

/** Build the WebSocket URL for a concrete `match_id` using the page origin. */
function wsUrlForMatch(matchId: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/match/${encodeURIComponent(matchId)}`;
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
 * Map a raw incoming JSON frame from the backend onto the typed
 * `WSFrame` union the rest of the app consumes. Performs the
 * snapshot decode in-line so consumers never see snake_case shapes.
 */
function adaptIncomingFrame(parsed: unknown, mapName: string): WSFrame | null {
  if (typeof parsed !== "object" || parsed === null) return null;
  const f = parsed as { type?: unknown };
  if (typeof f.type !== "string") return null;

  const kind = f.type;
  const obj = parsed as Record<string, unknown>;

  switch (kind) {
    case "snapshot": {
      const data = obj.data;
      const snap: MatchSnapshot = decodeMatchSnapshot(data, mapName);
      return { type: "snapshot", data: snap };
    }
    case "map_info": {
      const info = decodeMapInfo(obj.data);
      return { type: "map_info", data: info };
    }
    case "match_done": {
      const matchId = typeof obj.match_id === "string" ? obj.match_id : undefined;
      return { type: "match_done", matchId };
    }
    case "pong": {
      const ts = typeof obj.ts === "number" ? obj.ts : Date.now() / 1000;
      return { type: "pong", ts };
    }
    case "ack": {
      const forKind = typeof obj.for === "string" ? obj.for : "unknown";
      return { type: "ack", for: forKind, ...obj };
    }
    case "error": {
      // Backend uses {type:"error", detail:"..."}; frontend WSFrame
      // historically used {data:{message:string}}. Adapt both shapes.
      const detail = typeof obj.detail === "string" ? obj.detail : undefined;
      const dataField = obj.data as { message?: unknown } | undefined;
      const message =
        detail ??
        (dataField && typeof dataField.message === "string" ? dataField.message : "unknown error");
      return { type: "error", data: { message } };
    }
    case "event":
    case "message":
    case "inspect":
    case "attention_update":
    case "training_status":
    case "metrics_sample":
    case "round_result":
    case "hello":
      // These frame types pass through unchanged. The runtime shape
      // matches the WSFrame contract (still loosely-typed `as` cast —
      // upstream backend doesn't ship them yet in V1).
      return parsed as WSFrame;
    default:

      console.warn("[kivski] unknown WS frame type:", kind);
      return null;
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

  const status = (s: "connecting" | "open" | "closed" | "error") => {
    opts.onStatus?.(s);
  };

  /** Ensure we have a current `matchId`, creating one if necessary. */
  const ensureMatch = async (): Promise<string | null> => {
    if (forcedUrl) return null;
    if (matchId !== null && !needsFreshMatch) return matchId;
    try {
      const res = await createMatch(createBody);
      matchId = res.match_id;
      if (res.map) mapName = res.map;
      needsFreshMatch = false;

      console.warn("[kivski] created match", matchId, "on", mapName);
      return matchId;
    } catch (err) {

      console.warn("[kivski] match create failed:", err);
      status("error");
      return null;
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
        if (adapted === null) return;

        // Track the map name from the initial map_info frame so any
        // subsequent snapshots are tagged with the correct map.
        if (adapted.type === "map_info") {
          mapName = adapted.data.mapName;
        }

        opts.onFrame(adapted);
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
      if (ws) {
        try {
          ws.close();
        } catch {
          /* ignore */
        }
        ws = null;
      }
    },
  };
}
