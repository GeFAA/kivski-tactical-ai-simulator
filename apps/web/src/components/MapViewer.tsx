import { useEffect, useRef, useState, type ReactNode } from "react";
import { Application, Container, Graphics, Text, TextStyle } from "pixi.js";
import { useStore, selectSelectedAgent } from "@/lib/store";
import { loadMap } from "@/lib/map-loader";
import { PixiContext, type PixiContextValue } from "@/lib/pixi-context";
import type { AgentSnapshot, EventItem, MapData } from "@/lib/types";
import CommsOverlay from "@/components/CommsOverlay";
import InfluenceArrows from "@/components/InfluenceArrows";
import HeatmapOverlay from "@/components/HeatmapOverlay";

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

const drawPlayers = (
  layer: Container,
  agents: AgentSnapshot[],
  selectedId: string | null,
  onSelect: (id: string | null) => void,
) => {
  layer.removeChildren();
  for (const a of agents) {
    const dot = new Graphics();
    const r = 1.1;
    const color = sideColor(a.side);

    if (selectedId === a.id) {
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

    dot.position.set(a.pos.x, a.pos.y);
    dot.eventMode = "static";
    dot.cursor = "pointer";
    dot.on("pointertap", () => onSelect(selectedId === a.id ? null : a.id));
    layer.addChild(dot);
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

      // Wire static-layer redraws on every store change. Players + bomb only.
      // Guard against firing on torn-down containers in case the unsub
      // ordering ever races with `destroy()`.
      const unsub = useStore.subscribe((state) => {
        if (disposed) return;
        if (players.destroyed || bomb.destroyed) return;
        try {
          drawPlayers(players, state.agents, state.selectedAgentId, selectAgent);
          drawBomb(bomb, state.bomb);
        } catch (err) {

          console.error("[kivski] subscribe draw failed:", err);
        }
      });
      (app as Application & { __unsub?: () => void }).__unsub = unsub;

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
