import { useEffect, useRef, useState, type ReactNode } from "react";
import { Application, Container, Graphics, Text, TextStyle } from "pixi.js";
import { useStore, selectSelectedAgent } from "@/lib/store";
import { loadMap } from "@/lib/map-loader";
import { PixiContext, type PixiContextValue } from "@/lib/pixi-context";
import type { AgentSnapshot, EventItem, MapData } from "@/lib/types";
import CommsOverlay from "@/components/CommsOverlay";
import InfluenceArrows from "@/components/InfluenceArrows";
import HeatmapOverlay from "@/components/HeatmapOverlay";

/**
 * Wall-clock interval (ms) over which we interpolate between two
 * snapshots. The backend broadcasts at ``server.tick_broadcast_hz``
 * (20 Hz default → 50 ms) and the engine ticks at 10 Hz (100 ms), so
 * 100 ms is a safe ceiling. We clamp ``alpha`` to ``[0, 1]`` so the dot
 * sits at the latest snapshot if a frame is late.
 */
const SNAPSHOT_INTERVAL_MS = 100;

// ---------- Color helpers ----------

const COLORS = {
  bgGrid: 0x131821,
  gridLine: 0x1a2030,
  wall: 0x2a2f3a,
  cover: 0x3b4252,
  siteA: 0xff6b6b,
  siteB: 0x6bcb77,
  spawnT: 0xffc833,
  spawnCT: 0x4da8ff,
  attacker: 0xffc833,
  defender: 0x4da8ff,
  bomb: 0xff8c42,
  selectionRing: 0xffffff,
  fovCone: 0xffd24a,
  soundRing: 0xa78bfa,
} as const;

/**
 * z-order constants for overlay layers. Higher = drawn on top.
 * Static map layers occupy 0..40 (background grid → walls).
 * Dynamic overlays sit on top so they never get clipped.
 */
export const Z = {
  background: 0,
  zones: 10,
  walls: 20,
  spawnLabels: 30,
  heatmap: 35,
  bomb: 40,
  players: 50,
  commsOverlay: 60,
  influence: 70,
  fov: 80,
  sound: 85,
} as const;

const sideColor = (s: AgentSnapshot["side"]): number =>
  s === "attacker" ? COLORS.attacker : COLORS.defender;

const zoneColor = (kind: MapData["zones"][number]["kind"]): number => {
  switch (kind) {
    case "site_a":
      return COLORS.siteA;
    case "site_b":
      return COLORS.siteB;
    case "spawn_attacker":
      return COLORS.spawnT;
    case "spawn_defender":
      return COLORS.spawnCT;
    case "buy":
      return 0x888888;
    default:
      return 0x444444;
  }
};

// ---------- Drawing helpers ----------

const drawBackgroundGrid = (g: Graphics, w: number, h: number, step = 4) => {
  g.clear();
  g.rect(0, 0, w, h).fill({ color: COLORS.bgGrid });
  for (let x = 0; x <= w; x += step) {
    g.moveTo(x, 0).lineTo(x, h);
  }
  for (let y = 0; y <= h; y += step) {
    g.moveTo(0, y).lineTo(w, y);
  }
  g.stroke({ color: COLORS.gridLine, width: 0.05, alpha: 0.8 });
};

const drawMap = (
  zones: Container,
  walls: Container,
  spawnLabels: Container,
  map: MapData,
) => {
  // Zones
  zones.removeChildren();
  spawnLabels.removeChildren();
  for (const z of map.zones) {
    const g = new Graphics();
    const pts = z.poly.flatMap((p) => [p.x, p.y]);
    g.poly(pts).fill({ color: zoneColor(z.kind), alpha: 0.18 });
    g.poly(pts).stroke({ color: zoneColor(z.kind), width: 0.2, alpha: 0.55 });
    zones.addChild(g);

    if (z.label) {
      const t = new Text({
        text: z.label,
        style: new TextStyle({
          fontFamily: "ui-monospace, monospace",
          fontSize: 2,
          fill: zoneColor(z.kind),
          fontWeight: "700",
          align: "center",
        }),
      });
      const cx = z.poly.reduce((s, p) => s + p.x, 0) / z.poly.length;
      const cy = z.poly.reduce((s, p) => s + p.y, 0) / z.poly.length;
      t.anchor.set(0.5);
      t.position.set(cx, cy);
      spawnLabels.addChild(t);
    }
  }

  // Walls / cover
  walls.removeChildren();
  for (const w of map.walls) {
    const g = new Graphics();
    const pts = w.poly.flatMap((p) => [p.x, p.y]);
    const color = w.kind === "wall" ? COLORS.wall : COLORS.cover;
    g.poly(pts).fill({ color });
    g.poly(pts).stroke({ color: 0x000000, width: 0.08, alpha: 0.4 });
    walls.addChild(g);
  }
};

// ----- Interpolated player rendering ------------------------------------
//
// We keep a stable per-agent dot in the players container and update its
// position every frame via Pixi's ticker. The dot's *visual* state (alive
// cross, selection ring, facing arrow) is rebuilt only when the snapshot
// changes — but its (x, y) is lerped between the previous and current
// snapshot so the user sees a smooth glide instead of a 0.45-tile jump
// every 100 ms. On a round reset (round id change) we skip interpolation
// and snap to the new spawn so players don't appear to slide across the
// map.
interface DotEntry {
  /** The Pixi node — kept across snapshots so position can be tweened. */
  graphic: Graphics;
  /** Hash of the last visual state we rendered into this Graphics. */
  visualKey: string;
  /** Last selection state we wired the click handler against. */
  selectedSnapshot: boolean;
}

interface AgentPosKeyframe {
  x: number;
  y: number;
  alive: boolean;
}

interface PlayerRenderState {
  /** Per-agent stable Pixi nodes. */
  dots: Map<string, DotEntry>;
  /**
   * Previous-snapshot positions, keyed by agent id. Filled at the moment
   * a fresh snapshot arrives. Missing entries fall through to "current".
   */
  prev: Map<string, AgentPosKeyframe>;
  /** Current-snapshot positions, keyed by agent id. */
  curr: Map<string, AgentPosKeyframe>;
  /** Wall-clock ms at which the current snapshot arrived. */
  receivedAt: number;
  /**
   * Round id of the latest snapshot. When the round flips we skip
   * interpolation for that one update so respawned agents pop into
   * place rather than slide across the entire map.
   */
  round: number;
}

const _visualKey = (
  a: AgentSnapshot,
  isSelected: boolean,
): string =>
  `${a.side}|${a.isAlive ? 1 : 0}|${isSelected ? 1 : 0}|${a.facing.toFixed(2)}`;

const _redrawDot = (
  dot: Graphics,
  a: AgentSnapshot,
  isSelected: boolean,
): void => {
  dot.clear();
  const r = 1.1;
  const color = sideColor(a.side);
  if (isSelected) {
    dot.circle(0, 0, r + 0.5).stroke({ color: COLORS.selectionRing, width: 0.25 });
  }
  if (a.isAlive) {
    dot.circle(0, 0, r).fill({ color, alpha: 0.95 });
    dot.circle(0, 0, r).stroke({ color: 0x000000, width: 0.15, alpha: 0.6 });
    const fx = Math.cos(a.facing) * (r + 0.9);
    const fy = Math.sin(a.facing) * (r + 0.9);
    dot.moveTo(0, 0).lineTo(fx, fy).stroke({ color, width: 0.3, alpha: 0.9 });
  } else {
    dot
      .moveTo(-r, -r)
      .lineTo(r, r)
      .moveTo(-r, r)
      .lineTo(r, -r)
      .stroke({ color, width: 0.3, alpha: 0.7 });
  }
};

/**
 * Merge a fresh snapshot into the persistent render state. Ensures one
 * dot per agent (creating / removing as needed), refreshes the dot's
 * visual state, and records the previous-vs-current keyframes used by
 * the per-frame interpolator.
 */
const ingestPlayersSnapshot = (
  layer: Container,
  state: PlayerRenderState,
  agents: AgentSnapshot[],
  selectedId: string | null,
  round: number,
  onSelect: (id: string | null) => void,
): void => {
  const seen = new Set<string>();
  // The new "previous" map is the old "current" — i.e. where the dots
  // are *right now* (or rather, where they were headed). If a round
  // change is detected we'll snap by overwriting prev with curr later.
  const newPrev = new Map<string, AgentPosKeyframe>();
  for (const [id, kf] of state.curr) {
    newPrev.set(id, kf);
  }
  // Also fold in any in-flight interpolated positions from the *old*
  // prev → curr so the new tween starts from the dot's current visible
  // position, not from the previous snapshot's keyframe. This avoids a
  // tiny visual rubber-band when snapshots arrive out of sync with the
  // 100 ms tween window.
  const tweenAlpha = state.curr.size === 0 ? 1 : Math.min(
    1,
    Math.max(0, (performance.now() - state.receivedAt) / SNAPSHOT_INTERVAL_MS),
  );
  if (tweenAlpha < 1) {
    for (const [id, prev] of state.prev) {
      const curr = state.curr.get(id);
      if (!curr) continue;
      newPrev.set(id, {
        x: prev.x + (curr.x - prev.x) * tweenAlpha,
        y: prev.y + (curr.y - prev.y) * tweenAlpha,
        alive: curr.alive,
      });
    }
  }

  const newCurr = new Map<string, AgentPosKeyframe>();
  for (const a of agents) {
    seen.add(a.id);
    newCurr.set(a.id, { x: a.pos.x, y: a.pos.y, alive: a.isAlive });

    const isSelected = selectedId === a.id;
    let entry = state.dots.get(a.id);
    if (!entry) {
      const g = new Graphics();
      g.eventMode = "static";
      g.cursor = "pointer";
      layer.addChild(g);
      entry = { graphic: g, visualKey: "", selectedSnapshot: false };
      state.dots.set(a.id, entry);
      // Brand-new dot — start interpolation at the same position as the
      // current snapshot to avoid a slide-in from (0, 0).
      newPrev.set(a.id, { x: a.pos.x, y: a.pos.y, alive: a.isAlive });
    }

    // Re-bind the click handler if the selection target flipped (cheap).
    if (entry.selectedSnapshot !== isSelected) {
      entry.graphic.removeAllListeners();
      entry.graphic.on("pointertap", () =>
        onSelect(isSelected ? null : a.id),
      );
      entry.selectedSnapshot = isSelected;
    } else {
      // The closure captures the agent id only — make sure it stays
      // current across snapshots even when selection didn't change.
      entry.graphic.removeAllListeners();
      entry.graphic.on("pointertap", () =>
        onSelect(isSelected ? null : a.id),
      );
    }

    const vk = _visualKey(a, isSelected);
    if (vk !== entry.visualKey) {
      _redrawDot(entry.graphic, a, isSelected);
      entry.visualKey = vk;
    }
  }

  // Remove dots for agents that disappeared (shouldn't happen mid-match
  // but is correct on match reset).
  for (const [id, entry] of state.dots) {
    if (!seen.has(id)) {
      if (!entry.graphic.destroyed) entry.graphic.destroy();
      state.dots.delete(id);
    }
  }

  // Round transition: snap. The respawn would otherwise look like every
  // agent telegraphing 40+ tiles across the map in 100 ms.
  const roundChanged = round !== state.round;
  if (roundChanged) {
    for (const [id, kf] of newCurr) {
      newPrev.set(id, { ...kf });
    }
  }

  state.prev = newPrev;
  state.curr = newCurr;
  state.receivedAt = performance.now();
  state.round = round;
};

/**
 * Per-frame position update. Called from Pixi's ticker. Lerps each
 * dot's (x, y) between its previous and current keyframe based on how
 * far along the 100 ms tween we are.
 */
const tickPlayerPositions = (state: PlayerRenderState): void => {
  if (state.dots.size === 0) return;
  const alpha = Math.min(
    1,
    Math.max(0, (performance.now() - state.receivedAt) / SNAPSHOT_INTERVAL_MS),
  );
  for (const [id, entry] of state.dots) {
    if (entry.graphic.destroyed) continue;
    const curr = state.curr.get(id);
    if (!curr) continue;
    const prev = state.prev.get(id) ?? curr;
    const x = prev.x + (curr.x - prev.x) * alpha;
    const y = prev.y + (curr.y - prev.y) * alpha;
    entry.graphic.position.set(x, y);
  }
};

const drawBomb = (
  layer: Container,
  bomb: { pos: { x: number; y: number } | null; phase: string },
) => {
  layer.removeChildren();
  if (!bomb.pos) return;
  const g = new Graphics();
  g.circle(0, 0, 0.9).fill({ color: COLORS.bomb });
  g.circle(0, 0, 1.4).stroke({
    color: COLORS.bomb,
    width: 0.2,
    alpha: bomb.phase === "planted" ? 0.9 : 0.5,
  });
  g.position.set(bomb.pos.x, bomb.pos.y);
  layer.addChild(g);
};

// ---------- FoV / Sound stubs ----------

const FOV_DEG = 144;
const FOV_RADIUS = 30;

const drawFovCone = (layer: Container, agent: AgentSnapshot | null) => {
  layer.removeChildren();
  if (!agent || !agent.isAlive) return;
  const g = new Graphics();
  const half = (FOV_DEG * Math.PI) / 180 / 2;
  const segments = 24;
  const start = agent.facing - half;
  const points: number[] = [agent.pos.x, agent.pos.y];
  for (let i = 0; i <= segments; i++) {
    const t = start + (i / segments) * 2 * half;
    points.push(agent.pos.x + Math.cos(t) * FOV_RADIUS);
    points.push(agent.pos.y + Math.sin(t) * FOV_RADIUS);
  }
  g.poly(points).fill({ color: COLORS.fovCone, alpha: 0.12 });
  g.poly(points).stroke({ color: COLORS.fovCone, width: 0.15, alpha: 0.5 });
  layer.addChild(g);
};

const drawSoundEvents = (
  layer: Container,
  events: EventItem[],
  currentTick: number,
) => {
  layer.removeChildren();
  // Window: last ~30 ticks of sound events with a position.
  const recent = events.filter(
    (e) => e.kind === "sound" && e.pos && currentTick - e.tick < 30,
  );
  for (const e of recent) {
    if (!e.pos) continue;
    const age = Math.max(0, currentTick - e.tick);
    const alpha = Math.max(0, 1 - age / 30);
    const radius = 2 + age * 0.4;
    const g = new Graphics();
    g.circle(0, 0, radius).stroke({
      color: COLORS.soundRing,
      width: 0.2,
      alpha: alpha * 0.7,
    });
    g.position.set(e.pos.x, e.pos.y);
    layer.addChild(g);
  }
};

// ---------- Component ----------

const MapViewer = () => {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const appRef = useRef<Application | null>(null);
  const worldRef = useRef<Container | null>(null);
  const mapDataRef = useRef<MapData | null>(null);
  /** Stable, key-addressable container registry so overlays can re-mount safely. */
  const layerRegistry = useRef<Map<string, Container>>(new Map());

  const selectAgent = useStore((s) => s.selectAgent);
  const mapName = useStore((s) => s.mapName);
  const showFov = useStore((s) => s.showFov);
  const showSound = useStore((s) => s.showSound);

  /** Forces a re-render once the async pixi init finishes — so PixiContext.Provider sees app+world. */
  const [pixiReady, setPixiReady] = useState(false);
  /** Non-fatal user-visible message when something pixi-side went wrong. */
  const [pixiError, setPixiError] = useState<string | null>(null);

  // Init PixiJS app once per host element.
  useEffect(() => {
    let disposed = false;
    const host = hostRef.current;
    if (!host) return;

    const app = new Application();
    appRef.current = app;

    (async () => {
      try {
        await app.init({
          background: 0x0a0e14,
          antialias: true,
          resolution: Math.min(window.devicePixelRatio || 1, 2),
          autoDensity: true,
          resizeTo: host,
        });
      } catch (err) {

        console.error("[kivski] PIXI app.init failed:", err);
        if (!disposed) {
          setPixiError(
            `PIXI init failed: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
        return;
      }
      if (disposed) {
        try {
          app.destroy(true, { children: true, texture: true });
        } catch {
          /* may throw if init never finished cleanly */
        }
        return;
      }
      host.appendChild(app.canvas);

      // Root world container that we scale to fit the host.
      const world = new Container();
      world.label = "world";
      world.sortableChildren = true;
      app.stage.addChild(world);
      worldRef.current = world;

      // Initial layers (static map). Overlays add themselves on mount.
      const background = ensureLayer(world, layerRegistry.current, "background", Z.background);
      const zones = ensureLayer(world, layerRegistry.current, "zones", Z.zones);
      const walls = ensureLayer(world, layerRegistry.current, "walls", Z.walls);
      const spawnLabels = ensureLayer(world, layerRegistry.current, "spawnLabels", Z.spawnLabels);
      const bomb = ensureLayer(world, layerRegistry.current, "bomb", Z.bomb);
      const players = ensureLayer(world, layerRegistry.current, "players", Z.players);

      const bgGfx = new Graphics();
      background.addChild(bgGfx);

      const fit = () => {
        const map = mapDataRef.current;
        if (!map) return;
        if (!app.renderer) return;
        const sx = app.renderer.width / map.width;
        const sy = app.renderer.height / map.height;
        const s = Math.min(sx, sy);
        world.scale.set(s);
        world.position.set(
          (app.renderer.width - map.width * s) / 2,
          (app.renderer.height - map.height * s) / 2,
        );
      };

      let map: MapData;
      try {
        map = await loadMap(mapName || "dustline");
      } catch (err) {

        console.error("[kivski] loadMap failed:", err);
        if (!disposed) {
          setPixiError(
            `Map load failed: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
        return;
      }
      if (disposed) return;
      mapDataRef.current = map;
      try {
        drawBackgroundGrid(bgGfx, map.width, map.height, 4);
        drawMap(zones, walls, spawnLabels, map);
        fit();
      } catch (err) {

        console.error("[kivski] map draw failed:", err);
        if (!disposed) {
          setPixiError(
            `Map draw failed: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
        return;
      }

      const ro = new ResizeObserver(fit);
      ro.observe(host);
      (app as Application & { __ro?: ResizeObserver }).__ro = ro;

      // Persistent render state for the players layer — see the docstring
      // on PlayerRenderState. Lives in this closure so cleanup can stop
      // the ticker and tear down the dots together with the Pixi app.
      const playerState: PlayerRenderState = {
        dots: new Map(),
        prev: new Map(),
        curr: new Map(),
        receivedAt: performance.now(),
        round: -1,
      };

      // Wire static-layer redraws on every store change. Players are
      // ingested into the persistent render state (positions are kept
      // animated by the ticker below); the bomb is small enough to
      // rebuild from scratch each snapshot. Guard against firing on
      // torn-down containers in case the unsub ordering ever races
      // with `destroy()`.
      const unsub = useStore.subscribe((state) => {
        if (disposed) return;
        if (players.destroyed || bomb.destroyed) return;
        try {
          ingestPlayersSnapshot(
            players,
            playerState,
            state.agents,
            state.selectedAgentId,
            state.round,
            selectAgent,
          );
          drawBomb(bomb, state.bomb);
        } catch (err) {

          console.error("[kivski] subscribe draw failed:", err);
        }
      });
      (app as Application & { __unsub?: () => void }).__unsub = unsub;

      // Per-frame interpolation hook — runs at the Pixi ticker's native
      // rate (capped at the browser's vsync). Cheap: at most O(N) where
      // N == active agents.
      const tickerFn = () => {
        if (disposed || players.destroyed) return;
        try {
          tickPlayerPositions(playerState);
        } catch (err) {
          console.error("[kivski] player ticker failed:", err);
        }
      };
      app.ticker.add(tickerFn);
      (app as Application & { __tickerFn?: () => void }).__tickerFn = tickerFn;

      // Mark context ready so overlays can mount.
      setPixiReady(true);
    })();

    // Copy ref values for cleanup (avoid stale-ref lint warning).
    const layers = layerRegistry.current;
    return () => {
      disposed = true;
      setPixiReady(false);
      const a = appRef.current;
      appRef.current = null;
      worldRef.current = null;
      mapDataRef.current = null;
      layers.clear();
      if (a) {
        const ro = (a as Application & { __ro?: ResizeObserver }).__ro;
        if (ro) ro.disconnect();
        const u = (a as Application & { __unsub?: () => void }).__unsub;
        if (u) {
          try { u(); } catch { /* ignore */ }
        }
        const tf = (a as Application & { __tickerFn?: () => void }).__tickerFn;
        if (tf && a.ticker) {
          try { a.ticker.remove(tf); } catch { /* ignore */ }
        }
        try {
          a.destroy(true, { children: true, texture: true });
        } catch {
          /* ignore — common during the StrictMode mount/unmount/mount cycle
             when destroy() is called before init() completes. */
        }
      }
    };
  }, [mapName, selectAgent]);

  // FoV overlay (selected agent only). Lightweight enough to handle here.
  useEffect(() => {
    const app = appRef.current;
    const world = worldRef.current;
    if (!app || !world) return;
    const layer = ensureLayer(world, layerRegistry.current, "fov", Z.fov);
    layer.visible = showFov;
    if (!showFov) {
      layer.removeChildren();
      return;
    }
    const unsub = useStore.subscribe((state) => {
      drawFovCone(layer, selectSelectedAgent(state));
    });
    drawFovCone(layer, selectSelectedAgent(useStore.getState()));
    return () => unsub();
  }, [showFov, pixiReady]);

  // Sound overlay — fades recent sound events.
  useEffect(() => {
    const app = appRef.current;
    const world = worldRef.current;
    if (!app || !world) return;
    const layer = ensureLayer(world, layerRegistry.current, "sound", Z.sound);
    layer.visible = showSound;
    if (!showSound) {
      layer.removeChildren();
      return;
    }
    const unsub = useStore.subscribe((state) => {
      drawSoundEvents(layer, state.eventFeed, state.tick);
    });
    drawSoundEvents(layer, useStore.getState().eventFeed, useStore.getState().tick);
    return () => unsub();
  }, [showSound, pixiReady]);

  // Provider value — only valid after pixi has initialised.
  const ctx: PixiContextValue | null =
    pixiReady && appRef.current && worldRef.current && mapDataRef.current
      ? {
          app: appRef.current,
          world: worldRef.current,
          mapWidth: mapDataRef.current.width,
          mapHeight: mapDataRef.current.height,
          addLayer: (key, zIndex) =>
            ensureLayer(worldRef.current!, layerRegistry.current, key, zIndex),
        }
      : null;

  return (
    <div
      ref={hostRef}
      className="relative h-full w-full overflow-hidden bg-kivski-bg"
    >
      {pixiError && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div
            className="pointer-events-auto max-w-md rounded border p-4 text-xs"
            style={{
              borderColor: "#5a2a2a",
              background: "rgba(20,10,10,0.85)",
              color: "#ff9d9d",
              fontFamily: "ui-monospace, monospace",
            }}
          >
            <div style={{ color: "#FFC833", fontWeight: 700, marginBottom: 6 }}>
              MapViewer error
            </div>
            <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{pixiError}</pre>
          </div>
        </div>
      )}
      {ctx && (
        <PixiContext.Provider value={ctx}>
          <OverlayMount />
        </PixiContext.Provider>
      )}
    </div>
  );
};

/**
 * Ensures a stable container exists at the given key + zIndex on the world.
 * Returns the same container on re-mount.
 */
const ensureLayer = (
  world: Container,
  registry: Map<string, Container>,
  key: string,
  zIndex: number,
): Container => {
  let layer = registry.get(key);
  if (!layer || layer.destroyed) {
    layer = new Container();
    layer.label = key;
    layer.zIndex = zIndex;
    world.addChild(layer);
    registry.set(key, layer);
  } else if (layer.parent !== world) {
    world.addChild(layer);
  }
  return layer;
};

/**
 * Wrapper that mounts every overlay component. Keeps MapViewer's JSX
 * focused on the host div, and gives overlays a single shared parent.
 */
const OverlayMount = (): ReactNode => (
  <>
    <HeatmapOverlay />
    <CommsOverlay />
    <InfluenceArrows />
  </>
);

export default MapViewer;
