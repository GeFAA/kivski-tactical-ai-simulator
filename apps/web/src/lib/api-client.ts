/**
 * Tiny REST + WebSocket client for talking to the Kivski FastAPI backend.
 *
 * - REST: thin fetch wrappers under /api/*
 * - WS:   subscribeMatch() returns an unsubscribe fn and auto-reconnects
 *         with exponential backoff. JSON frames are typed as `WSFrame`.
 */

import type { TrainingStatus, WSFrame } from "./types";

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

// ---------- WebSocket ----------

export interface WSHandle {
  /** Close permanently and stop reconnecting. */
  close: () => void;
}

export interface SubscribeOpts {
  url?: string;
  onFrame: (frame: WSFrame) => void;
  onStatus?: (status: "connecting" | "open" | "closed" | "error") => void;
  /** Initial reconnect delay in ms. Doubles per attempt up to 15s cap. */
  baseDelayMs?: number;
}

function defaultWsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/match`;
}

/**
 * Subscribe to the live match WebSocket. Returns a handle whose `.close()`
 * stops the reconnect loop. Each JSON frame is decoded and validated
 * loosely (must have a string `type`); malformed frames are dropped with
 * a console warning.
 */
export function subscribeMatch(opts: SubscribeOpts): WSHandle {
  const url = opts.url ?? defaultWsUrl();
  const baseDelay = opts.baseDelayMs ?? 500;

  let closed = false;
  let attempt = 0;
  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  const status = (s: "connecting" | "open" | "closed" | "error") => {
    opts.onStatus?.(s);
  };

  const connect = () => {
    if (closed) return;
    status("connecting");
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
        if (
          typeof parsed === "object" &&
          parsed !== null &&
          typeof (parsed as { type?: unknown }).type === "string"
        ) {
          opts.onFrame(parsed as WSFrame);
        } else {
           
          console.warn("[kivski] dropping malformed WS frame:", raw.slice(0, 120));
        }
      } catch (err) {
         
        console.warn("[kivski] WS parse error:", err);
      }
    };

    ws.onerror = () => {
      status("error");
      // 'close' will fire right after, which schedules reconnect.
    };

    ws.onclose = () => {
      status("closed");
      ws = null;
      scheduleReconnect();
    };
  };

  const scheduleReconnect = () => {
    if (closed) return;
    attempt += 1;
    const delay = Math.min(15_000, baseDelay * 2 ** Math.min(attempt - 1, 6));
    // small jitter to avoid thundering herd if many clients reconnect at once
    const jitter = Math.floor(Math.random() * 250);
    reconnectTimer = setTimeout(connect, delay + jitter);
  };

  connect();

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
