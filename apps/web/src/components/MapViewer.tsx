import { useEffect, useRef, useState, type ReactNode } from "react";
import { Application, Container, Graphics, Text, TextStyle } from "pixi.js";
import { useStore, selectSelectedAgent } from "@/lib/store";
import { loadMap } from "@/lib/map-loader";
import { PixiContext, type PixiContextValue } from "@/lib/pixi-context";
import type {
  AgentSnapshot,
  BombSnapshot,
  EventItem,
  MapData,
  WeaponKind,
} from "@/lib/types";
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
  bgGrid: 0x0c1118,
  gridLine: 0x18202d,
  wall: 0x1a1f2a,
  wallEdge: 0x070a0f,
  cover: 0x252a35,
  coverEdge: 0x10141c,
  siteA: 0xff4d4d,
  siteB: 0x4ade80,
  spawnT: 0xffc833,
  spawnCT: 0x4da8ff,
  attacker: 0xffc833,
  defender: 0x4da8ff,
  bomb: 0xff8c42,
  bombAccent: 0xffb368,
  selectionRing: 0xffffff,
  fovCone: 0xffd24a,
  soundRing: 0xa78bfa,
  hpHigh: 0x4ade80,
  hpMed: 0xfacc15,
  hpLow: 0xf87171,
  weaponBody: 0xb6bcc5,
  weaponEdge: 0x2a2f3a,
  labelBg: 0x0a0e14,
  labelText: 0xf1f5f9,
  defuseKit: 0xe2e8f0,
} as const;

/**
 * z-order constants for overlay layers. Higher = drawn on top.
 * Static map layers occupy 0..40 (background grid → walls).
 * Dynamic overlays sit on top so they never get clipped.
 */
export const Z = {
  background: 0,
  zones: 10,
  siteLetters: 15,
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

const hpColor = (hpFrac: number): number => {
  if (hpFrac > 0.66) return COLORS.hpHigh;
  if (hpFrac > 0.33) return COLORS.hpMed;
  return COLORS.hpLow;
};

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
  g.stroke({ color: COLORS.gridLine, width: 0.05, alpha: 0.4 });
};

/**
 * Polygon centroid (good-enough average) for placing bombsite letters
 * and spawn labels. Falls back to (0,0) on an empty poly.
 */
const polyCentroid = (poly: { x: number; y: number }[]): { x: number; y: number } => {
  if (poly.length === 0) return { x: 0, y: 0 };
  let sx = 0;
  let sy = 0;
  for (const p of poly) {
    sx += p.x;
    sy += p.y;
  }
  return { x: sx / poly.length, y: sy / poly.length };
};

/**
 * Render the static map: zones (bombsites get a translucent fill +
 * bold A/B letter, spawns get a team-coloured outline), walls (full
 * blocks with a darker edge), cover (lighter gray + softer edge).
 */
const drawMap = (
  zones: Container,
  walls: Container,
  spawnLabels: Container,
  siteLetters: Container,
  map: MapData,
) => {
  zones.removeChildren();
  spawnLabels.removeChildren();
  siteLetters.removeChildren();

  for (const z of map.zones) {
    const g = new Graphics();
    const pts = z.poly.flatMap((p) => [p.x, p.y]);
    const color = zoneColor(z.kind);

    if (z.kind === "site_a" || z.kind === "site_b") {
      // Bombsite: translucent tint + bright thin outline + big letter.
      g.poly(pts).fill({ color, alpha: 0.12 });
      g.poly(pts).stroke({ color, width: 0.25, alpha: 0.55 });
      const c = polyCentroid(z.poly);
      const letter = new Text({
        text: z.kind === "site_a" ? "A" : "B",
        style: new TextStyle({
          fontFamily: "ui-monospace, monospace",
          fontSize: 6,
          fill: color,
          fontWeight: "900",
          align: "center",
        }),
      });
      letter.anchor.set(0.5);
      letter.position.set(c.x, c.y);
      letter.alpha = 0.32;
      siteLetters.addChild(letter);
    } else if (z.kind === "spawn_attacker" || z.kind === "spawn_defender") {
      // Spawn pad: no fill, just a 1.5px team-coloured outline.
      g.poly(pts).stroke({ color, width: 0.22, alpha: 0.55 });
    } else {
      // Neutral / buy / other: subtle translucent fill + faint outline.
      g.poly(pts).fill({ color, alpha: 0.08 });
      g.poly(pts).stroke({ color, width: 0.12, alpha: 0.35 });
    }
    zones.addChild(g);

    if (z.label) {
      const c = polyCentroid(z.poly);
      const t = new Text({
        text: z.label,
        style: new TextStyle({
          fontFamily: "ui-monospace, monospace",
          fontSize: 1.6,
          fill: zoneColor(z.kind),
          fontWeight: "700",
          align: "center",
        }),
      });
      t.anchor.set(0.5);
      t.position.set(c.x, c.y);
      t.alpha = 0.7;
      spawnLabels.addChild(t);
    }
  }

  walls.removeChildren();
  for (const w of map.walls) {
    const g = new Graphics();
    const pts = w.poly.flatMap((p) => [p.x, p.y]);
    if (w.kind === "wall") {
      g.poly(pts).fill({ color: COLORS.wall });
      g.poly(pts).stroke({ color: COLORS.wallEdge, width: 0.12, alpha: 0.95 });
    } else {
      g.poly(pts).fill({ color: COLORS.cover });
      g.poly(pts).stroke({ color: COLORS.coverEdge, width: 0.09, alpha: 0.85 });
    }
    walls.addChild(g);
  }
};

// ----- CS-observer-style player rendering --------------------------------
//
// Each agent gets a persistent `PlayerContainer` populated with stable
// child Graphics/Text nodes so per-frame updates only touch transforms
// and `clear()/redraw()` paths, never `new Text({...})`. The container
// position is lerped between two snapshots; alpha/rotation/visual state
// for sub-nodes is refreshed only when the agent's *visual key* changes.
//
// Visual layout (centred at container origin = agent.pos in world units):
//
//   weapon icon  ─── small gun shape, ~y = -2.6
//   facing arrow ─── triangle on body edge in facing direction
//   body circle  ─── 1.1 unit radius, team coloured
//   bomb / kit   ─── small icon overlay above-right of body
//   name label   ─── dark chip "A0"/"B5" at y = +2.0
//   hp bar       ─── thin coloured bar at y = +3.2
//   selection    ─── pulsing white ring at z = -1
//
// Sizes are in world units (the world container is scaled to fit the
// map's WxH); the Pixi resolution keeps text crisp.

const NAME_SCALE = 1; // world-unit sizing; Text fontSize is in world units

/** Short label like "A0" / "B5" derived from the agent id. */
const shortName = (a: AgentSnapshot): string => {
  // Backend ids look like "agent_0" .. "agent_9". Use the team
  // initial + idx so labels are 2 chars at most and packing stays nice.
  const idMatch = /(\d+)/.exec(a.id);
  const idx = idMatch ? idMatch[1] : "?";
  const prefix = a.team === "yellow" ? "Y" : "B";
  return `${prefix}${idx}`;
};

/**
 * Draw a small weapon silhouette into ``g`` centred at (0, 0). Sizes
 * are in *world units* — the parent container is scaled by the world
 * transform. The shapes are deliberately schematic (no photo-realism)
 * so they read at small sizes.
 */
const drawWeaponShape = (g: Graphics, kind: WeaponKind): void => {
  g.clear();
  const body = COLORS.weaponBody;
  const edge = COLORS.weaponEdge;
  const wStroke = 0.07;

  switch (kind) {
    case "knife": {
      // Triangular blade + small grip.
      g.poly([-0.8, 0.3, 0.6, -0.1, -0.4, -0.4])
        .fill({ color: body })
        .stroke({ color: edge, width: wStroke });
      g.rect(-1.0, 0.15, 0.3, 0.25).fill({ color: 0x4a3024 });
      break;
    }
    case "pistol": {
      // L-shape: slide on top + grip down-right.
      g.rect(-1.0, -0.35, 1.6, 0.45).fill({ color: body }).stroke({ color: edge, width: wStroke });
      g.rect(0.2, 0.1, 0.45, 0.7).fill({ color: body }).stroke({ color: edge, width: wStroke });
      break;
    }
    case "smg": {
      // Compact rectangle + stock fold.
      g.rect(-1.3, -0.35, 2.0, 0.55).fill({ color: body }).stroke({ color: edge, width: wStroke });
      g.rect(-1.7, -0.25, 0.45, 0.35).fill({ color: body }).stroke({ color: edge, width: wStroke });
      g.rect(-0.3, 0.2, 0.35, 0.55).fill({ color: body }).stroke({ color: edge, width: wStroke });
      break;
    }
    case "rifle":
    case "ar":
    case "lmg": {
      // Longer body + pistol grip + stock.
      g.rect(-1.6, -0.35, 2.6, 0.55).fill({ color: body }).stroke({ color: edge, width: wStroke });
      g.rect(-2.0, -0.2, 0.4, 0.3).fill({ color: body }).stroke({ color: edge, width: wStroke });
      g.rect(-0.2, 0.2, 0.4, 0.55).fill({ color: body }).stroke({ color: edge, width: wStroke });
      break;
    }
    case "sniper": {
      // Long body + scope circle on top + grip.
      g.rect(-1.8, -0.3, 3.0, 0.45).fill({ color: body }).stroke({ color: edge, width: wStroke });
      g.circle(-0.3, -0.6, 0.35).fill({ color: 0x2a2f3a }).stroke({ color: edge, width: wStroke });
      g.rect(0.2, 0.15, 0.4, 0.55).fill({ color: body }).stroke({ color: edge, width: wStroke });
      break;
    }
    case "shotgun": {
      // Body with a wider muzzle on the left.
      g.rect(-1.6, -0.3, 2.4, 0.5).fill({ color: body }).stroke({ color: edge, width: wStroke });
      g.rect(-1.9, -0.4, 0.4, 0.7).fill({ color: body }).stroke({ color: edge, width: wStroke });
      g.rect(0.0, 0.2, 0.4, 0.55).fill({ color: body }).stroke({ color: edge, width: wStroke });
      break;
    }
    case "grenade":
    case "flash":
    case "smoke":
    case "molotov":
    case "c4": {
      // Generic round/cylinder.
      g.circle(0, 0, 0.45).fill({ color: body }).stroke({ color: edge, width: wStroke });
      break;
    }
    default: {
      g.rect(-1.2, -0.3, 2.0, 0.5).fill({ color: body }).stroke({ color: edge, width: wStroke });
    }
  }
};

/**
 * Briefcase-style bomb icon. Drawn into ``g`` centred at (0, 0).
 * Reused for both the player overlay (when carrying) and the dropped/
 * planted bomb on the map.
 */
const drawBombShape = (g: Graphics, scale = 1, blinkPulse = 0): void => {
  g.clear();
  const s = scale;
  // Briefcase body.
  g.rect(-0.6 * s, -0.35 * s, 1.2 * s, 0.7 * s)
    .fill({ color: COLORS.bomb })
    .stroke({ color: 0x4a2a10, width: 0.07 * s });
  // Handle.
  g.rect(-0.25 * s, -0.55 * s, 0.5 * s, 0.18 * s).fill({ color: COLORS.bombAccent });
  // Blink dot (planted state pulses).
  const blinkA = 0.55 + 0.45 * blinkPulse;
  g.circle(0.32 * s, -0.05 * s, 0.13 * s).fill({ color: 0xff2a2a, alpha: blinkA });
};

/** Defuse-kit indicator: small wrench-on-pad square. */
const drawDefuseKitShape = (g: Graphics): void => {
  g.clear();
  g.rect(-0.4, -0.32, 0.8, 0.6)
    .fill({ color: COLORS.defuseKit })
    .stroke({ color: 0x2a3140, width: 0.07 });
  // Cross-detail inside the kit.
  g.rect(-0.18, -0.12, 0.36, 0.08).fill({ color: 0x2a3140 });
  g.rect(-0.06, -0.22, 0.12, 0.28).fill({ color: 0x2a3140 });
};

/**
 * Persistent Pixi nodes for a single agent. Keeping them stable across
 * snapshots means we never allocate new Text/Container objects in the
 * subscribe path, which used to be a memory leak in the previous
 * `_redrawDot` implementation that ``clear()``ed and re-stroked a flat
 * Graphics on every tick.
 */
interface PlayerNodes {
  container: Container;
  shadow: Graphics;
  selection: Graphics;
  body: Graphics;
  facing: Graphics;
  weapon: Graphics;
  bombIcon: Graphics;
  kitIcon: Graphics;
  nameBg: Graphics;
  nameText: Text;
  hpBar: Graphics;
  /** Most recently rendered visual key — skip redraws when unchanged. */
  visualKey: string;
  /** Current weapon kind for the cached weapon Graphics. */
  weaponKind: WeaponKind | null;
  /** True if the click handler is wired for the selected state. */
  selected: boolean;
  /** Currently-rendered hp fraction (lerped toward target each frame). */
  displayedHpFrac: number;
}

interface AgentPosKeyframe {
  x: number;
  y: number;
  alive: boolean;
}

interface PlayerRenderState {
  /** Per-agent stable Pixi node bundle. */
  nodes: Map<string, PlayerNodes>;
  /**
   * Previous-snapshot positions, keyed by agent id. Filled at the moment
   * a fresh snapshot arrives. Missing entries fall through to "current".
   */
  prev: Map<string, AgentPosKeyframe>;
  /** Current-snapshot positions, keyed by agent id. */
  curr: Map<string, AgentPosKeyframe>;
  /**
   * Snapshot of the per-agent target hp fraction, used by the per-frame
   * ticker to lerp the bar smoothly instead of popping.
   */
  hpTargets: Map<string, number>;
  /** Wall-clock ms at which the current snapshot arrived. */
  receivedAt: number;
  /**
   * Round id of the latest snapshot. When the round flips we skip
   * interpolation for that one update so respawned agents pop into
   * place rather than slide across the entire map.
   */
  round: number;
}

const _visualKey = (a: AgentSnapshot, isSelected: boolean): string =>
  [
    a.side,
    a.team,
    a.isAlive ? 1 : 0,
    isSelected ? 1 : 0,
    a.facing.toFixed(2),
    a.weapons[a.activeWeaponIdx]?.kind ?? "none",
    a.hasBomb ? 1 : 0,
    a.hasDefuseKit ? 1 : 0,
    a.isPlanting ? 1 : 0,
    a.isDefusing ? 1 : 0,
  ].join("|");

const buildPlayerNodes = (layer: Container, a: AgentSnapshot): PlayerNodes => {
  const container = new Container();
  container.label = `player_${a.id}`;
  container.sortableChildren = true;
  container.eventMode = "static";
  container.cursor = "pointer";

  const shadow = new Graphics();
  shadow.zIndex = -2;
  const selection = new Graphics();
  selection.zIndex = -1;
  const body = new Graphics();
  body.zIndex = 0;
  const facing = new Graphics();
  facing.zIndex = 1;
  const weapon = new Graphics();
  weapon.zIndex = 2;
  weapon.position.set(0, -2.6);
  const bombIcon = new Graphics();
  bombIcon.zIndex = 3;
  bombIcon.position.set(1.4, -1.6);
  bombIcon.visible = false;
  const kitIcon = new Graphics();
  kitIcon.zIndex = 3;
  kitIcon.position.set(-1.4, -1.6);
  kitIcon.visible = false;

  const nameBg = new Graphics();
  nameBg.zIndex = 4;
  // Render text at a "natural" pixel size (12px) and scale it down so
  // it occupies ~1.4 world units. This bypasses Pixi v8's tendency to
  // produce blurry tiny-font glyph atlases when fontSize <= 2.
  const nameText = new Text({
    text: shortName(a),
    style: new TextStyle({
      fontFamily: "ui-monospace, Menlo, Consolas, monospace",
      fontSize: 12 * NAME_SCALE,
      fill: COLORS.labelText,
      fontWeight: "700",
      align: "center",
      letterSpacing: 0.2,
    }),
    resolution: 2,
  });
  nameText.anchor.set(0.5, 0);
  nameText.scale.set(1 / 7); // 12px → ~1.7 world units total height
  nameText.position.set(0, 2.0);
  nameText.zIndex = 5;

  const hpBar = new Graphics();
  hpBar.zIndex = 5;
  hpBar.position.set(0, 3.55);

  container.addChild(shadow);
  container.addChild(selection);
  container.addChild(body);
  container.addChild(facing);
  container.addChild(weapon);
  container.addChild(bombIcon);
  container.addChild(kitIcon);
  container.addChild(nameBg);
  container.addChild(nameText);
  container.addChild(hpBar);

  layer.addChild(container);

  return {
    container,
    shadow,
    selection,
    body,
    facing,
    weapon,
    bombIcon,
    kitIcon,
    nameBg,
    nameText,
    hpBar,
    visualKey: "",
    weaponKind: null,
    selected: false,
    displayedHpFrac: 1,
  };
};

/**
 * Refresh per-tick visual state of a single player. Only touches the
 * Graphics/Text sub-nodes — never adds or removes children.
 */
const refreshPlayerVisuals = (
  nodes: PlayerNodes,
  a: AgentSnapshot,
  isSelected: boolean,
): void => {
  const color = sideColor(a.side);
  const r = 1.1;

  // Shadow ellipse — subtle drop, dimmer when dead.
  nodes.shadow.clear();
  nodes.shadow
    .ellipse(0, 0.35, r * 1.1, r * 0.45)
    .fill({ color: 0x000000, alpha: a.isAlive ? 0.45 : 0.18 });

  // Body.
  nodes.body.clear();
  if (a.isAlive) {
    nodes.body.circle(0, 0, r).fill({ color, alpha: 0.96 });
    nodes.body.circle(0, 0, r).stroke({ color: 0xffffff, width: 0.16, alpha: 0.85 });
  } else {
    // Dead: dimmer dot + X overlay.
    nodes.body.circle(0, 0, r * 0.85).fill({ color, alpha: 0.28 });
    nodes.body.circle(0, 0, r * 0.85).stroke({ color: 0x000000, width: 0.12, alpha: 0.4 });
    nodes.body
      .moveTo(-r * 0.55, -r * 0.55)
      .lineTo(r * 0.55, r * 0.55)
      .moveTo(-r * 0.55, r * 0.55)
      .lineTo(r * 0.55, -r * 0.55)
      .stroke({ color: 0xffffff, width: 0.18, alpha: 0.85 });
  }

  // Facing arrow (white triangle on body edge in facing direction).
  nodes.facing.clear();
  if (a.isAlive) {
    const cx = Math.cos(a.facing);
    const cy = Math.sin(a.facing);
    const tipX = cx * (r + 0.7);
    const tipY = cy * (r + 0.7);
    const baseX = cx * (r + 0.05);
    const baseY = cy * (r + 0.05);
    // Perpendicular to facing for the triangle base.
    const nx = -cy;
    const ny = cx;
    const halfW = 0.42;
    nodes.facing
      .poly([
        tipX,
        tipY,
        baseX + nx * halfW,
        baseY + ny * halfW,
        baseX - nx * halfW,
        baseY - ny * halfW,
      ])
      .fill({ color: 0xffffff, alpha: 0.95 });
  }

  // Weapon shape — only redraw on weapon-kind change.
  const activeWeapon = a.weapons[a.activeWeaponIdx]?.kind ?? null;
  if (activeWeapon && nodes.weaponKind !== activeWeapon) {
    drawWeaponShape(nodes.weapon, activeWeapon);
    nodes.weaponKind = activeWeapon;
  }
  nodes.weapon.visible = !!activeWeapon && a.isAlive;
  if (nodes.weapon.visible) {
    // Tilt slightly with facing so the weapon "follows" the player.
    nodes.weapon.rotation = Math.sin(a.facing) * 0.18;
  }

  // Bomb / defuse kit overlays.
  if (a.hasBomb) {
    drawBombShape(nodes.bombIcon, 0.85);
    nodes.bombIcon.visible = a.isAlive;
  } else {
    nodes.bombIcon.visible = false;
  }
  if (a.hasDefuseKit && a.side === "defender") {
    drawDefuseKitShape(nodes.kitIcon);
    nodes.kitIcon.visible = a.isAlive;
  } else {
    nodes.kitIcon.visible = false;
  }

  // Name label — short id, chip-style bg sized to text. `nameText.width`
  // reports raw glyph-atlas pixels, so we multiply by `scale.x` to get
  // its visible size in *world units*.
  const label = shortName(a);
  if (nodes.nameText.text !== label) nodes.nameText.text = label;
  nodes.nameText.alpha = a.isAlive ? 1 : 0.5;
  const textW = nodes.nameText.width * nodes.nameText.scale.x;
  const textH = nodes.nameText.height * nodes.nameText.scale.y;
  const padX = 0.32;
  const padY = 0.18;
  const w = Math.max(1.6, textW + padX * 2);
  const h = textH + padY * 2;
  nodes.nameBg.clear();
  nodes.nameBg
    .roundRect(-w / 2, 2.0 - padY, w, h, 0.3)
    .fill({ color: COLORS.labelBg, alpha: 0.82 })
    .stroke({ color, width: 0.1, alpha: 0.65 });

  // Selection ring — drawn at fixed radius, alpha pulse handled in ticker.
  nodes.selection.clear();
  if (isSelected) {
    nodes.selection.circle(0, 0, r + 0.7).stroke({
      color: COLORS.selectionRing,
      width: 0.22,
      alpha: 1,
    });
  }
  nodes.selection.visible = isSelected;

  // Hp bar — redraws are cheap so we run them in the ticker (smooth lerp).
};

/**
 * Per-frame hp bar redraw. Lerps the displayed fraction toward the
 * target (snapshot) value for a smooth shrink. Bar width is fixed at
 * 2.4 world units (~ the body diameter).
 */
const drawHpBar = (nodes: PlayerNodes, hpFrac: number, alive: boolean): void => {
  const W = 2.4;
  const H = 0.32;
  nodes.hpBar.clear();
  if (!alive) {
    nodes.hpBar.visible = false;
    return;
  }
  nodes.hpBar.visible = true;
  // Track.
  nodes.hpBar
    .roundRect(-W / 2, 0, W, H, 0.08)
    .fill({ color: 0x000000, alpha: 0.55 })
    .stroke({ color: 0x000000, width: 0.05, alpha: 0.8 });
  // Fill.
  const fillW = Math.max(0, Math.min(1, hpFrac)) * W;
  if (fillW > 0) {
    nodes.hpBar
      .roundRect(-W / 2, 0, fillW, H, 0.08)
      .fill({ color: hpColor(hpFrac), alpha: 0.95 });
  }
};

/**
 * Merge a fresh snapshot into the persistent render state. Creates /
 * removes per-agent node bundles as needed, refreshes visual state,
 * and records the previous-vs-current keyframes used by the per-frame
 * interpolator.
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
  const newPrev = new Map<string, AgentPosKeyframe>();
  for (const [id, kf] of state.curr) {
    newPrev.set(id, kf);
  }
  // Fold any in-flight interpolated positions so the next tween starts
  // from the dot's visible position (avoids rubber-band on lag).
  const tweenAlpha =
    state.curr.size === 0
      ? 1
      : Math.min(
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
    state.hpTargets.set(a.id, Math.max(0, Math.min(1, a.hp / 100)));

    const isSelected = selectedId === a.id;
    let nodes = state.nodes.get(a.id);
    if (!nodes) {
      nodes = buildPlayerNodes(layer, a);
      state.nodes.set(a.id, nodes);
      newPrev.set(a.id, { x: a.pos.x, y: a.pos.y, alive: a.isAlive });
    }

    // Click handler — always re-wire so the closure captures the latest
    // selection state without leaking a listener registry.
    nodes.container.removeAllListeners();
    const targetId = a.id;
    nodes.container.on("pointertap", () =>
      onSelect(isSelected ? null : targetId),
    );
    nodes.selected = isSelected;

    const vk = _visualKey(a, isSelected);
    if (vk !== nodes.visualKey) {
      refreshPlayerVisuals(nodes, a, isSelected);
      nodes.visualKey = vk;
    }
  }

  // Tear down any agents that vanished (e.g. match reset).
  for (const [id, nodes] of state.nodes) {
    if (!seen.has(id)) {
      if (!nodes.container.destroyed) {
        nodes.container.destroy({ children: true });
      }
      state.nodes.delete(id);
      state.hpTargets.delete(id);
    }
  }

  // Round transition: snap. Without this, every respawned agent would
  // appear to telegraph 40+ tiles across the map in 100 ms.
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
 * Per-frame container update. Lerps position, fades the selection ring
 * (sine pulse), and smooth-shrinks the hp bar toward its target value.
 */
const tickPlayerPositions = (state: PlayerRenderState): void => {
  if (state.nodes.size === 0) return;
  const alpha = Math.min(
    1,
    Math.max(0, (performance.now() - state.receivedAt) / SNAPSHOT_INTERVAL_MS),
  );
  // Selection ring pulse — 1.2 Hz sine.
  const t = performance.now() / 1000;
  const pulse = 0.55 + 0.45 * Math.sin(t * Math.PI * 2 * 1.2);

  for (const [id, nodes] of state.nodes) {
    if (nodes.container.destroyed) continue;
    const curr = state.curr.get(id);
    if (!curr) continue;
    const prev = state.prev.get(id) ?? curr;
    const x = prev.x + (curr.x - prev.x) * alpha;
    const y = prev.y + (curr.y - prev.y) * alpha;
    nodes.container.position.set(x, y);

    if (nodes.selection.visible) {
      nodes.selection.alpha = pulse;
    }

    // Smooth hp bar lerp toward the snapshot target (0..1 in ~250 ms).
    const target = state.hpTargets.get(id) ?? 0;
    const cur = nodes.displayedHpFrac;
    const blend = Math.min(1, 16 / 1000 * 6); // ~6 units per second
    const next = cur + (target - cur) * blend;
    if (Math.abs(next - cur) > 0.005 || nodes.displayedHpFrac !== target) {
      nodes.displayedHpFrac = next;
      drawHpBar(nodes, next, curr.alive);
    }
  }
};

// ---------- Bomb on the map (when not carried) ---------------------------

interface BombRenderNodes {
  container: Container;
  icon: Graphics;
  /** Currently rendered phase so we know when to re-draw the icon. */
  phase: string;
}

const buildBombNodes = (layer: Container): BombRenderNodes => {
  const container = new Container();
  container.label = "bomb-on-map";
  const icon = new Graphics();
  container.addChild(icon);
  layer.addChild(container);
  return { container, icon, phase: "" };
};

/**
 * Render the map-level bomb icon. The carried bomb is drawn as an
 * overlay on the carrier's container (see `bombIcon`), so we only
 * show this when the bomb is on the ground or planted.
 */
const updateBombOnMap = (
  state: { nodes: BombRenderNodes | null },
  layer: Container,
  bomb: BombSnapshot,
): void => {
  const showOnMap =
    bomb.pos !== null &&
    bomb.phase !== "carried" &&
    bomb.phase !== "none";

  if (!showOnMap) {
    if (state.nodes && !state.nodes.container.destroyed) {
      state.nodes.container.visible = false;
    }
    return;
  }

  if (!state.nodes) {
    state.nodes = buildBombNodes(layer);
  }
  const n = state.nodes;
  n.container.visible = true;
  n.container.position.set(bomb.pos!.x, bomb.pos!.y);

  // Blink the planted icon at 2 Hz so it reads as armed.
  const t = performance.now() / 1000;
  const blink = bomb.phase === "planted" ? 0.5 + 0.5 * Math.sin(t * Math.PI * 4) : 0;
  drawBombShape(n.icon, 1.25, blink);
  n.phase = bomb.phase;
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
      const siteLetters = ensureLayer(world, layerRegistry.current, "siteLetters", Z.siteLetters);
      const walls = ensureLayer(world, layerRegistry.current, "walls", Z.walls);
      const spawnLabels = ensureLayer(world, layerRegistry.current, "spawnLabels", Z.spawnLabels);
      const bombLayer = ensureLayer(world, layerRegistry.current, "bomb", Z.bomb);
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
        drawMap(zones, walls, spawnLabels, siteLetters, map);
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

      // Persistent render state for the players layer — see the
      // docstring on PlayerRenderState. Lives in this closure so cleanup
      // can stop the ticker and tear down the nodes together with the
      // Pixi app.
      const playerState: PlayerRenderState = {
        nodes: new Map(),
        prev: new Map(),
        curr: new Map(),
        hpTargets: new Map(),
        receivedAt: performance.now(),
        round: -1,
      };
      const bombState: { nodes: BombRenderNodes | null } = { nodes: null };

      // Wire static-layer redraws on every store change. Players are
      // ingested into the persistent render state (positions are kept
      // animated by the ticker below). Guard against firing on
      // torn-down containers in case the unsub ordering ever races
      // with `destroy()`.
      const unsub = useStore.subscribe((state) => {
        if (disposed) return;
        if (players.destroyed || bombLayer.destroyed) return;
        try {
          ingestPlayersSnapshot(
            players,
            playerState,
            state.agents,
            state.selectedAgentId,
            state.round,
            selectAgent,
          );
          updateBombOnMap(bombState, bombLayer, state.bomb);
        } catch (err) {

          console.error("[kivski] subscribe draw failed:", err);
        }
      });
      (app as Application & { __unsub?: () => void }).__unsub = unsub;

      // Per-frame interpolation / pulse / hp-lerp.
      const tickerFn = () => {
        if (disposed || players.destroyed) return;
        try {
          tickPlayerPositions(playerState);
          // Refresh the planted-bomb blink even without snapshot churn.
          if (bombState.nodes && bombState.nodes.container.visible) {
            const cur = useStore.getState().bomb;
            updateBombOnMap(bombState, bombLayer, cur);
          }
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
