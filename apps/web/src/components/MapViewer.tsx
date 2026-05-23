import { useEffect, useRef, useState, type ReactNode } from "react";
import { Application, Container, Graphics, Text, TextStyle } from "pixi.js";
import { useStore, selectSelectedAgent } from "@/lib/store";
import { loadMap } from "@/lib/map-loader";
import { clippedFovPolygon } from "@/lib/fov-clipping";
import { PixiContext, type PixiContextValue } from "@/lib/pixi-context";
import type {
  AgentSnapshot,
  BombSnapshot,
  EventItem,
  MapData,
  MessageItem,
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

// ----- Weapon silhouette palette (v2 polish) ----------------------------
//
// All firearm silhouettes share a four-tone palette:
//   - W_BODY     : dark mainframe / receiver
//   - W_HIGHLIGHT: lighter barrel / slide
//   - W_ACCENT   : cyan optics / electronic parts
//   - W_OUTLINE  : near-black stroke around the silhouette
//
// Sizes are in *world units*. The world container is fit to map size, so
// 1 unit ≈ 1 grid tile; weapons therefore measure ~2-3 units wide and
// render around 14-20 px at typical zoom levels.
const W_BODY = 0x3a4250;
const W_HIGHLIGHT = 0x6b7888;
const W_ACCENT = 0x4d9eff;
const W_OUTLINE = 0x1a1f2a;
const W_GRIP = 0x2a2e38;
const W_STROKE = 0.06;
const W_STROKE_FINE = 0.045;

/**
 * Draw a small weapon silhouette into ``g`` centred at (0, 0).
 *
 * The shapes are deliberately schematic — they read clearly even at
 * 14-20 px and stay legible when the world is zoomed out. Every kind
 * uses the shared palette (body / highlight / accent / outline) so the
 * set feels visually coherent.
 *
 * Default orientation: weapon points along **+x**. We offset the whole
 * silhouette by `OX` so it sits clearly *off* the body circle and reads
 * as a held weapon rather than overlapping with the player dot.
 */
const OX = 1.0; // offset from body origin so the weapon clears the dot
const drawWeaponShape = (g: Graphics, kind: WeaponKind): void => {
  g.clear();

  switch (kind) {
    case "knife": {
      // Narrow blade pointing +x, short dark grip behind it.
      // Blade: elongated triangle (length ~1.1, base 0.35).
      g.poly([OX + 0.7, 0, OX - 0.5, 0.18, OX - 0.5, -0.18])
        .fill({ color: W_HIGHLIGHT })
        .stroke({ color: W_OUTLINE, width: W_STROKE_FINE });
      // Grip.
      g.rect(OX - 0.95, -0.13, 0.45, 0.26)
        .fill({ color: W_GRIP })
        .stroke({ color: W_OUTLINE, width: W_STROKE_FINE });
      // Cross-guard.
      g.rect(OX - 0.52, -0.22, 0.08, 0.44).fill({ color: W_OUTLINE });
      break;
    }
    case "pistol": {
      // L-shape: slide (barrel) on top + grip below + trigger guard.
      g.rect(OX - 0.6, -0.22, 1.5, 0.34)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      g.rect(OX + 0.6, -0.18, 0.3, 0.26).fill({ color: W_HIGHLIGHT });
      // Grip below.
      g.rect(OX - 0.2, 0.06, 0.4, 0.62)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      g.rect(OX + 0.22, 0.06, 0.16, 0.2).fill({ color: W_OUTLINE });
      g.rect(OX - 0.1, -0.32, 0.14, 0.1).fill({ color: W_HIGHLIGHT });
      break;
    }
    case "smg": {
      // Compact body + grip + magazine drop + short stock fold behind.
      g.rect(OX - 0.9, -0.24, 1.95, 0.42)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      g.rect(OX + 0.75, -0.18, 0.3, 0.28).fill({ color: W_HIGHLIGHT });
      // Folding stock.
      g.rect(OX - 1.35, -0.14, 0.45, 0.22)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE_FINE });
      // Pistol grip.
      g.rect(OX - 0.3, 0.16, 0.32, 0.42)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      // Magazine.
      g.rect(OX + 0.05, 0.18, 0.3, 0.5)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      g.rect(OX + 0.05, 0.6, 0.3, 0.08).fill({ color: W_HIGHLIGHT });
      break;
    }
    case "rifle":
    case "ar":
    case "lmg": {
      // Longer body + buttstock + scope mount + magazine.
      // Buttstock.
      g.rect(OX - 1.55, -0.14, 0.55, 0.3)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      // Receiver.
      g.rect(OX - 1.0, -0.22, 1.85, 0.4)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      // Barrel extension.
      g.rect(OX + 0.85, -0.13, 0.55, 0.2)
        .fill({ color: W_HIGHLIGHT })
        .stroke({ color: W_OUTLINE, width: W_STROKE_FINE });
      // Muzzle tip.
      g.rect(OX + 1.35, -0.1, 0.12, 0.14).fill({ color: W_OUTLINE });
      // Scope mount on top.
      g.rect(OX - 0.2, -0.4, 0.5, 0.2)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE_FINE });
      g.rect(OX - 0.08, -0.36, 0.22, 0.1).fill({ color: W_ACCENT, alpha: 0.85 });
      // Pistol grip.
      g.rect(OX - 0.4, 0.18, 0.28, 0.4)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      // Magazine.
      g.rect(OX + 0.0, 0.18, 0.4, 0.55)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      g.rect(OX + 0.0, 0.64, 0.4, 0.09).fill({ color: W_HIGHLIGHT });
      break;
    }
    case "sniper": {
      // Very long barrel + big scope on top + bipod feet + buttstock.
      g.rect(OX - 1.8, -0.14, 0.6, 0.3)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      g.rect(OX - 1.2, -0.22, 1.1, 0.4)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      // Long barrel.
      g.rect(OX - 0.1, -0.13, 1.85, 0.22)
        .fill({ color: W_HIGHLIGHT })
        .stroke({ color: W_OUTLINE, width: W_STROKE_FINE });
      // Muzzle brake.
      g.rect(OX + 1.7, -0.17, 0.18, 0.3).fill({ color: W_OUTLINE });
      // Big scope.
      g.circle(OX - 0.4, -0.46, 0.32)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      g.circle(OX - 0.4, -0.46, 0.18).fill({ color: W_ACCENT, alpha: 0.9 });
      g.circle(OX - 0.4, -0.46, 0.07).fill({ color: 0xffffff, alpha: 0.6 });
      // Scope mount rails.
      g.rect(OX - 0.7, -0.28, 0.6, 0.08).fill({ color: W_OUTLINE });
      // Bipod feet (two small triangles).
      g.poly([OX + 0.65, 0.18, OX + 0.51, 0.5, OX + 0.79, 0.5])
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE_FINE });
      g.poly([OX + 1.1, 0.18, OX + 0.96, 0.5, OX + 1.24, 0.5])
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE_FINE });
      // Pistol grip.
      g.rect(OX - 0.8, 0.18, 0.28, 0.4)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      break;
    }
    case "shotgun": {
      // Thick barrel + pump action below + buttstock.
      g.rect(OX - 1.45, -0.18, 0.55, 0.36)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      g.rect(OX - 0.9, -0.26, 0.7, 0.48)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      // Pump action below barrel.
      g.rect(OX - 0.25, 0.16, 0.85, 0.18)
        .fill({ color: W_GRIP })
        .stroke({ color: W_OUTLINE, width: W_STROKE_FINE });
      // Thick barrel.
      g.rect(OX - 0.1, -0.2, 1.55, 0.32)
        .fill({ color: W_HIGHLIGHT })
        .stroke({ color: W_OUTLINE, width: W_STROKE_FINE });
      // Muzzle bore.
      g.circle(OX + 1.4, -0.04, 0.14)
        .fill({ color: W_OUTLINE })
        .stroke({ color: W_BODY, width: W_STROKE_FINE });
      // Pistol grip.
      g.rect(OX - 0.6, 0.22, 0.3, 0.4)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
      break;
    }
    case "grenade":
    case "flash":
    case "smoke":
    case "molotov":
    case "c4": {
      // Round grenade body + safety lever + pin ring.
      g.circle(OX, 0, 0.4).fill({ color: W_BODY }).stroke({ color: W_OUTLINE, width: W_STROKE });
      g.rect(OX - 0.34, -0.42, 0.68, 0.16)
        .fill({ color: W_HIGHLIGHT })
        .stroke({ color: W_OUTLINE, width: W_STROKE_FINE });
      g.circle(OX + 0.4, -0.42, 0.1).stroke({ color: W_HIGHLIGHT, width: 0.07 });
      break;
    }
    default: {
      g.rect(OX - 0.9, -0.22, 1.8, 0.42)
        .fill({ color: W_BODY })
        .stroke({ color: W_OUTLINE, width: W_STROKE });
    }
  }
};

/**
 * Weapon "tier" subtle backdrop glow. Drawn into ``g`` *behind* the
 * silhouette so we don't blow out the foreground colours. Returns true
 * if a glow was drawn, false otherwise.
 */
const drawWeaponTierGlow = (g: Graphics, kind: WeaponKind): boolean => {
  g.clear();
  // Knife / sidearm: no glow.
  if (kind === "knife" || kind === "pistol") return false;
  let color = 0xffffff;
  let alpha = 0.15;
  let radius = 1.4;
  switch (kind) {
    case "smg":
    case "shotgun":
      color = 0xffffff;
      alpha = 0.13;
      radius = 1.5;
      break;
    case "rifle":
    case "ar":
    case "lmg":
      color = 0x4d9eff;
      alpha = 0.16;
      radius = 1.8;
      break;
    case "sniper":
      color = 0xffd24a;
      alpha = 0.18;
      radius = 2.0;
      break;
    default:
      return false;
  }
  // Soft ellipse glow centred on the weapon midpoint and stretched
  // along its +x axis.
  g.ellipse(OX, 0, radius, radius * 0.55).fill({ color, alpha });
  return true;
};

/** Tip x-offset (in weapon-local units) used for muzzle-flash anchoring. */
const weaponMuzzleX = (kind: WeaponKind): number => {
  switch (kind) {
    case "knife":
      return OX + 0.7;
    case "pistol":
      return OX + 0.95;
    case "smg":
      return OX + 1.1;
    case "shotgun":
      return OX + 1.55;
    case "sniper":
      return OX + 1.9;
    case "rifle":
    case "ar":
    case "lmg":
      return OX + 1.5;
    default:
      return OX + 1.0;
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
 *
 * Layout note (v2 polish): the body, weapon and facing sub-nodes are now
 * parented to a `bobLayer` container so the walk-bob, hit-flash scale,
 * spawn-pulse scale, and death-fade alpha can all act on one transform
 * without disturbing the centred position labels / hp bar / progress ring.
 */
interface PlayerNodes {
  container: Container;
  /** Pulled-out container for body+weapon so we can scale/bob them. */
  bobLayer: Container;
  /** Container for the weapon group (glow + silhouette, with rotation). */
  weaponLayer: Container;
  shadow: Graphics;
  selection: Graphics;
  body: Graphics;
  facing: Graphics;
  weaponGlow: Graphics;
  weapon: Graphics;
  /** Muzzle-flash drawn relative to the weapon's tip, in weaponLayer. */
  muzzleFlash: Graphics;
  bombIcon: Graphics;
  kitIcon: Graphics;
  nameBg: Graphics;
  nameText: Text;
  hpBar: Graphics;
  /** Plant/defuse progress ring around the body. */
  progressRing: Graphics;
  /** Comm-pulse ring (transient outward fade). */
  commPulse: Graphics;
  /** Spawn-pulse ring (transient outward fade on respawn). */
  spawnPulse: Graphics;
  /** Floating damage number above head (transient). */
  damageText: Text;
  /** Most recently rendered visual key — skip redraws when unchanged. */
  visualKey: string;
  /** Current weapon kind for the cached weapon Graphics. */
  weaponKind: WeaponKind | null;
  /** True if the click handler is wired for the selected state. */
  selected: boolean;
  /** Currently-rendered hp fraction (lerped toward target each frame). */
  displayedHpFrac: number;
  /** Last snapshot hp value (for damage-detection diff). */
  lastHp: number;
  /** Last snapshot alive state (for spawn / death detection). */
  lastAlive: boolean;
  /** Walk-bob phase, randomised per agent so they don't bob in lockstep. */
  bobPhase: number;
  /** Last known position (used for velocity estimation between snapshots). */
  lastPos: { x: number; y: number };
  /** Smoothed velocity magnitude — drives walk-bob amplitude scaling. */
  speed: number;
  // ---- transient timestamps for one-shot animations (perf.now() ms) ----
  hitFlashStart: number;
  damageFloatStart: number;
  damageFloatAmount: number;
  deathFadeStart: number;
  /** Spawn-pulse start; 0 when not pulsing. */
  spawnPulseStart: number;
  /** Muzzle-flash start; 0 when not flashing. */
  muzzleFlashStart: number;
  /** Most-recent comm-pulse start; 0 when not pulsing. */
  commPulseStart: number;
  /** Color of the active comm pulse (matches comm action style). */
  commPulseColor: number;
  /** Last seen message-id from this agent so we trigger comm-pulse exactly once. */
  lastCommMsgId: string | null;
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
  /** Cached current bomb snapshot — drives the progress ring in the ticker. */
  bomb: BombSnapshot | null;
  /**
   * Per-agent planting/defusing flags from the latest snapshot. Used by
   * the ticker to compute the progress ring without re-reading the
   * store on every frame.
   */
  actState: Map<string, { planting: boolean; defusing: boolean }>;
  /** Most recently seen event id (avoid double-triggering muzzle flash). */
  lastEventId: string | null;
  /** Most recently seen message id (avoid double-triggering comm pulse). */
  lastMessageId: string | null;
}

// Visual key drives the "do we need to rebuild static visuals" check.
// We deliberately exclude:
//   - facing  (only updates the weaponLayer.rotation, applied per-tick)
//   - planting / defusing (handled via progress ring in the ticker)
const _visualKey = (a: AgentSnapshot, isSelected: boolean): string =>
  [
    a.side,
    a.team,
    a.isAlive ? 1 : 0,
    isSelected ? 1 : 0,
    a.weapons[a.activeWeaponIdx]?.kind ?? "none",
    a.hasBomb ? 1 : 0,
    a.hasDefuseKit ? 1 : 0,
  ].join("|");

const buildPlayerNodes = (layer: Container, a: AgentSnapshot): PlayerNodes => {
  const container = new Container();
  container.label = `player_${a.id}`;
  container.sortableChildren = true;
  container.eventMode = "static";
  container.cursor = "pointer";

  // Background-ish nodes hang off the root container so they are
  // unaffected by walk-bob / death-fade / spawn-pulse transforms.
  const shadow = new Graphics();
  shadow.zIndex = -2;
  const selection = new Graphics();
  selection.zIndex = -1;
  const progressRing = new Graphics();
  progressRing.zIndex = -1;
  const commPulse = new Graphics();
  commPulse.zIndex = -1;
  const spawnPulse = new Graphics();
  spawnPulse.zIndex = -1;

  // Body + facing + weapon live in a dedicated "bob" layer so we can
  // bob/scale/fade them as a unit.
  const bobLayer = new Container();
  bobLayer.label = "bob";
  bobLayer.zIndex = 0;
  bobLayer.sortableChildren = true;

  const body = new Graphics();
  body.zIndex = 0;
  const facing = new Graphics();
  facing.zIndex = 1;

  // Weapon group: glow (behind silhouette) + silhouette + muzzle flash.
  // The layer sits at the body's origin and *rotates* with `facing`;
  // individual weapon shapes are drawn shifted to +x so the weapon
  // sits roughly "in the player's hand" pointing outward. Much more
  // readable than the v1 "static icon above the head".
  const weaponLayer = new Container();
  weaponLayer.label = "weapon";
  weaponLayer.zIndex = 2;
  weaponLayer.position.set(0, 0);
  const weaponGlow = new Graphics();
  weaponGlow.zIndex = -1;
  const weapon = new Graphics();
  weapon.zIndex = 0;
  const muzzleFlash = new Graphics();
  muzzleFlash.zIndex = 1;
  muzzleFlash.visible = false;
  weaponLayer.addChild(weaponGlow);
  weaponLayer.addChild(weapon);
  weaponLayer.addChild(muzzleFlash);

  bobLayer.addChild(body);
  bobLayer.addChild(facing);
  bobLayer.addChild(weaponLayer);

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

  // Floating damage number — built once, hidden, animated when an event hits.
  const damageText = new Text({
    text: "",
    style: new TextStyle({
      fontFamily: "ui-monospace, Menlo, Consolas, monospace",
      fontSize: 14,
      fill: 0xff5d5d,
      fontWeight: "900",
      align: "center",
      stroke: { color: 0x0a0e14, width: 3 },
    }),
    resolution: 2,
  });
  damageText.anchor.set(0.5, 1);
  damageText.scale.set(1 / 7);
  damageText.zIndex = 6;
  damageText.visible = false;

  const hpBar = new Graphics();
  hpBar.zIndex = 5;
  hpBar.position.set(0, 3.55);

  container.addChild(shadow);
  container.addChild(selection);
  container.addChild(progressRing);
  container.addChild(commPulse);
  container.addChild(spawnPulse);
  container.addChild(bobLayer);
  container.addChild(bombIcon);
  container.addChild(kitIcon);
  container.addChild(nameBg);
  container.addChild(nameText);
  container.addChild(damageText);
  container.addChild(hpBar);

  layer.addChild(container);

  // Spawn-pulse on first appearance so the agent doesn't pop in cold.
  const now = performance.now();
  return {
    container,
    bobLayer,
    weaponLayer,
    shadow,
    selection,
    body,
    facing,
    weaponGlow,
    weapon,
    muzzleFlash,
    bombIcon,
    kitIcon,
    nameBg,
    nameText,
    hpBar,
    progressRing,
    commPulse,
    spawnPulse,
    damageText,
    visualKey: "",
    weaponKind: null,
    selected: false,
    displayedHpFrac: 1,
    lastHp: a.hp,
    lastAlive: a.isAlive,
    bobPhase: Math.random() * Math.PI * 2,
    lastPos: { x: a.pos.x, y: a.pos.y },
    speed: 0,
    hitFlashStart: 0,
    damageFloatStart: 0,
    damageFloatAmount: 0,
    deathFadeStart: 0,
    spawnPulseStart: a.isAlive ? now : 0,
    muzzleFlashStart: 0,
    commPulseStart: 0,
    commPulseColor: 0xffffff,
    lastCommMsgId: null,
  };
};

/**
 * Refresh per-tick visual state of a single player. Only touches the
 * Graphics/Text sub-nodes — never adds or removes children.
 *
 * The ticker handles all *transient* animation state (walk-bob, hit
 * flash, spawn pulse, death fade, muzzle flash, comm pulse, progress
 * ring). This function only rebuilds the *snapshot-keyed* visuals.
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

  // Weapon shape + tier glow — only redraw on weapon-kind change.
  const activeWeapon = a.weapons[a.activeWeaponIdx]?.kind ?? null;
  if (activeWeapon && nodes.weaponKind !== activeWeapon) {
    drawWeaponShape(nodes.weapon, activeWeapon);
    const drewGlow = drawWeaponTierGlow(nodes.weaponGlow, activeWeapon);
    nodes.weaponGlow.visible = drewGlow;
    nodes.weaponKind = activeWeapon;
  }
  nodes.weaponLayer.visible = !!activeWeapon && a.isAlive;
  if (nodes.weaponLayer.visible) {
    // Weapon shapes are drawn pointing +x; rotate the whole group to
    // match the agent's facing angle.
    nodes.weaponLayer.rotation = a.facing;
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
  // Name alpha is also touched by the death-fade ticker; this is the base.
  if (a.isAlive) nodes.nameText.alpha = 1;
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
  if (isSelected && a.isAlive) {
    nodes.selection.circle(0, 0, r + 0.7).stroke({
      color: COLORS.selectionRing,
      width: 0.22,
      alpha: 1,
    });
  }
  nodes.selection.visible = isSelected && a.isAlive;

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

// Animation timing constants (ms).
const HIT_FLASH_MS = 200;
const DAMAGE_FLOAT_MS = 800;
const DEATH_FADE_MS = 600;
const SPAWN_PULSE_MS = 400;
const MUZZLE_FLASH_MS = 180;
const COMM_PULSE_MS = 600;

const easeOutBack = (t: number): number => {
  // Classic easeOutBack with c1 = 1.70158, c3 = c1 + 1 = 2.70158.
  const c1 = 1.70158;
  const c3 = c1 + 1;
  const x = t - 1;
  return 1 + c3 * x * x * x + c1 * x * x;
};

/** Cyan/orange/etc colour lookup for comm pulses. Cheap default for `SILENT`. */
const commActionColor = (action: string | undefined): number => {
  switch (action) {
    case "PING_LOCATION":
      return 0x4d9eff;
    case "WARN_DANGER":
      return 0xff5d5d;
    case "REQUEST_SUPPORT":
      return 0xffd24a;
    case "SUGGEST_ROTATE":
      return 0xa78bfa;
    case "SUGGEST_ATTACK":
      return 0xff8c42;
    case "SUGGEST_FALLBACK":
      return 0x4ade80;
    case "CONTACT_ENEMY":
      return 0xff2a2a;
    case "BOMBSITE_CLEAR":
      return 0x4ade80;
    case "ACK":
      return 0xe2e8f0;
    default:
      return 0xffffff;
  }
};

/**
 * Merge a fresh snapshot into the persistent render state. Creates /
 * removes per-agent node bundles as needed, refreshes visual state,
 * and records the previous-vs-current keyframes used by the per-frame
 * interpolator.
 *
 * Also runs the *snapshot-driven* animation triggers (hit-flash on hp
 * drop, death-fade on alive→dead, spawn-pulse on dead→alive). Event /
 * message driven triggers (muzzle flash, comm pulse) come in via the
 * extra params and are applied to the matching agent's PlayerNodes.
 */
const ingestPlayersSnapshot = (
  layer: Container,
  state: PlayerRenderState,
  agents: AgentSnapshot[],
  selectedId: string | null,
  round: number,
  bomb: BombSnapshot,
  events: EventItem[],
  messages: MessageItem[],
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
  const now = performance.now();
  for (const a of agents) {
    seen.add(a.id);
    newCurr.set(a.id, { x: a.pos.x, y: a.pos.y, alive: a.isAlive });
    state.hpTargets.set(a.id, Math.max(0, Math.min(1, a.hp / 100)));
    state.actState.set(a.id, { planting: a.isPlanting, defusing: a.isDefusing });

    const isSelected = selectedId === a.id;
    let nodes = state.nodes.get(a.id);
    const isNew = !nodes;
    if (!nodes) {
      nodes = buildPlayerNodes(layer, a);
      state.nodes.set(a.id, nodes);
      newPrev.set(a.id, { x: a.pos.x, y: a.pos.y, alive: a.isAlive });
    }

    // ---- Animation trigger detection from snapshot diffs ---------------
    // Damage detection: hp dropped while still alive (or just died).
    const hpDelta = a.hp - nodes.lastHp;
    if (!isNew && hpDelta < -0.5 && nodes.lastAlive) {
      nodes.hitFlashStart = now;
      nodes.damageFloatStart = now;
      nodes.damageFloatAmount = Math.round(-hpDelta);
      nodes.damageText.text = `-${nodes.damageFloatAmount}`;
    }
    // Alive transitions.
    if (!isNew && !nodes.lastAlive && a.isAlive) {
      // Dead → alive: spawn pulse.
      nodes.spawnPulseStart = now;
      nodes.deathFadeStart = 0;
    }
    if (!isNew && nodes.lastAlive && !a.isAlive) {
      // Alive → dead: kick off death fade.
      nodes.deathFadeStart = now;
    }
    nodes.lastHp = a.hp;
    nodes.lastAlive = a.isAlive;

    // ---- Click handler — re-wire each tick so the closure captures the
    // latest selection state without leaking a listener registry.
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
    // Facing is *cheap* (just a transform), update each snapshot so the
    // weapon snaps to the new aim direction without re-running the
    // visual-key path.
    if (a.isAlive && nodes.weaponLayer.visible) {
      nodes.weaponLayer.rotation = a.facing;
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
      state.actState.delete(id);
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

  // ---- Event-driven triggers (muzzle flash) -------------------------------
  // Events arrive as newest-first list. Walk from newest to the last-seen
  // id and trigger muzzle flash on any kill/info combat event we haven't
  // surfaced yet. Cap iterations to avoid pathological backlog scans.
  if (events.length > 0) {
    const newestId = events[0].id;
    if (newestId !== state.lastEventId) {
      const maxScan = 20;
      for (let i = 0; i < Math.min(events.length, maxScan); i++) {
        const e = events[i];
        if (e.id === state.lastEventId) break;
        if ((e.kind === "kill" || e.kind === "info") && e.actorId) {
          const shooter = state.nodes.get(e.actorId);
          if (shooter) shooter.muzzleFlashStart = now;
        }
      }
      state.lastEventId = newestId;
    }
  }

  // ---- Message-driven triggers (comm pulse) -------------------------------
  if (messages.length > 0) {
    const newestId = messages[0].id;
    if (newestId !== state.lastMessageId) {
      const maxScan = 16;
      for (let i = 0; i < Math.min(messages.length, maxScan); i++) {
        const m = messages[i];
        if (m.id === state.lastMessageId) break;
        if (!m.action || m.action === "SILENT") continue;
        const sender = state.nodes.get(m.fromId);
        if (sender) {
          sender.commPulseStart = now;
          sender.commPulseColor = commActionColor(m.action);
        }
      }
      state.lastMessageId = newestId;
    }
  }

  state.prev = newPrev;
  state.curr = newCurr;
  state.receivedAt = now;
  state.round = round;
  state.bomb = bomb;
};

/**
 * Per-frame container update. Lerps position between snapshots, then
 * runs all transient animations (walk-bob, hit-flash, damage float,
 * death-fade, spawn-pulse, muzzle-flash, comm-pulse, progress ring,
 * selection-ring pulse, hp-bar lerp).
 *
 * Performance: no `new` allocations in this path. All sub-Graphics are
 * `clear()`-ed and re-drawn in place. The number of state lookups is
 * bounded by the number of agents (typ. 10).
 */
const tickPlayerPositions = (state: PlayerRenderState): void => {
  if (state.nodes.size === 0) return;
  const now = performance.now();
  const alpha = Math.min(
    1,
    Math.max(0, (now - state.receivedAt) / SNAPSHOT_INTERVAL_MS),
  );
  const tSec = now / 1000;
  // Selection ring pulse — 1.2 Hz sine.
  const selPulse = 0.55 + 0.45 * Math.sin(tSec * Math.PI * 2 * 1.2);

  for (const [id, nodes] of state.nodes) {
    if (nodes.container.destroyed) continue;
    const curr = state.curr.get(id);
    if (!curr) continue;
    const prev = state.prev.get(id) ?? curr;
    const baseX = prev.x + (curr.x - prev.x) * alpha;
    const baseY = prev.y + (curr.y - prev.y) * alpha;

    // ---- Velocity estimate (smoothed) for walk-bob amplitude --------
    const dxSnap = curr.x - prev.x;
    const dySnap = curr.y - prev.y;
    const distPerSnap = Math.hypot(dxSnap, dySnap);
    // distPerSnap is distance over SNAPSHOT_INTERVAL_MS (~100ms).
    // Smooth with a low-pass filter so brief pauses don't kill the bob.
    const targetSpeed = distPerSnap;
    nodes.speed += (targetSpeed - nodes.speed) * 0.15;

    nodes.container.position.set(baseX, baseY);

    // ---- Walk-bob (only when alive AND moving) ---------------------
    if (curr.alive && nodes.speed > 0.04) {
      nodes.bobPhase += (1 / 60) * 2 * Math.PI * 4; // ~4 Hz at 60fps
      // Sinus offset; amplitude scales gently with speed up to 0.3 tile.
      const amp = Math.min(0.3, nodes.speed * 1.2);
      const bob = -Math.abs(Math.sin(nodes.bobPhase)) * amp;
      nodes.bobLayer.position.set(0, bob);
    } else {
      nodes.bobLayer.position.set(0, 0);
    }

    // ---- Death fade -------------------------------------------------
    if (nodes.deathFadeStart > 0) {
      const dt = now - nodes.deathFadeStart;
      if (dt < DEATH_FADE_MS) {
        const k = dt / DEATH_FADE_MS;
        // Body alpha 1 → 0.25; name dims 1 → 0.5; weapon hides early.
        nodes.bobLayer.alpha = 1 - 0.75 * k;
        nodes.nameText.alpha = 1 - 0.5 * k;
        nodes.weaponLayer.visible = false;
        nodes.selection.visible = false;
      } else {
        nodes.bobLayer.alpha = 0.25;
        nodes.nameText.alpha = 0.5;
        nodes.deathFadeStart = 0;
      }
    } else if (curr.alive) {
      // Reset alpha after a respawn.
      nodes.bobLayer.alpha = 1;
    }

    // ---- Spawn pulse -----------------------------------------------
    if (nodes.spawnPulseStart > 0) {
      const dt = now - nodes.spawnPulseStart;
      if (dt < SPAWN_PULSE_MS) {
        const k = dt / SPAWN_PULSE_MS;
        // Scale body from 0.3 → 1.0 with easeOutBack.
        const eased = easeOutBack(k);
        const scale = 0.3 + 0.7 * Math.min(1, Math.max(0, eased));
        nodes.bobLayer.scale.set(scale);
        // Outward white ring fade.
        const ringR = 1.2 + k * 1.6;
        const ringA = (1 - k) * 0.8;
        nodes.spawnPulse.clear();
        nodes.spawnPulse
          .circle(0, 0, ringR)
          .stroke({ color: 0xffffff, width: 0.18, alpha: ringA });
        nodes.spawnPulse.visible = true;
      } else {
        nodes.bobLayer.scale.set(1);
        nodes.spawnPulse.visible = false;
        nodes.spawnPulseStart = 0;
      }
    }

    // ---- Hit flash (red body tint) ---------------------------------
    if (nodes.hitFlashStart > 0) {
      const dt = now - nodes.hitFlashStart;
      if (dt < HIT_FLASH_MS) {
        const k = 1 - dt / HIT_FLASH_MS;
        // Tint via colour matrix; cheaper than redrawing the graphic.
        nodes.body.tint = 0xff8888;
        nodes.body.alpha = 0.85 + 0.15 * k;
      } else {
        nodes.body.tint = 0xffffff;
        nodes.body.alpha = 1;
        nodes.hitFlashStart = 0;
      }
    }

    // ---- Damage number float ---------------------------------------
    if (nodes.damageFloatStart > 0) {
      const dt = now - nodes.damageFloatStart;
      if (dt < DAMAGE_FLOAT_MS) {
        const k = dt / DAMAGE_FLOAT_MS;
        nodes.damageText.visible = true;
        // Floats up from -1.6 to -3.4 over the lifetime.
        nodes.damageText.position.set(0, -1.6 - k * 1.8);
        nodes.damageText.alpha = 1 - k * k; // ease-out fade
      } else {
        nodes.damageText.visible = false;
        nodes.damageFloatStart = 0;
      }
    }

    // ---- Muzzle flash ----------------------------------------------
    if (nodes.muzzleFlashStart > 0 && nodes.weaponKind) {
      const dt = now - nodes.muzzleFlashStart;
      if (dt < MUZZLE_FLASH_MS) {
        const k = 1 - dt / MUZZLE_FLASH_MS;
        const tipX = weaponMuzzleX(nodes.weaponKind);
        const radius = 0.35 + 0.15 * (1 - k); // grows then fades
        nodes.muzzleFlash.clear();
        // Bright yellow-white half-disc.
        nodes.muzzleFlash
          .circle(tipX, 0, radius)
          .fill({ color: 0xfff6c0, alpha: k * 0.95 });
        // Inner hot core.
        nodes.muzzleFlash
          .circle(tipX, 0, radius * 0.5)
          .fill({ color: 0xffffff, alpha: k * 0.9 });
        // Forward spike (small triangle).
        nodes.muzzleFlash
          .poly([
            tipX + radius * 1.4, 0,
            tipX + radius * 0.3, radius * 0.6,
            tipX + radius * 0.3, -radius * 0.6,
          ])
          .fill({ color: 0xffe27a, alpha: k * 0.85 });
        nodes.muzzleFlash.visible = true;
      } else {
        nodes.muzzleFlash.visible = false;
        nodes.muzzleFlashStart = 0;
      }
    }

    // ---- Comm pulse ------------------------------------------------
    if (nodes.commPulseStart > 0) {
      const dt = now - nodes.commPulseStart;
      if (dt < COMM_PULSE_MS) {
        const k = dt / COMM_PULSE_MS;
        const radius = 1.2 + k * 1.0; // 12px → 22px feel
        const alphaR = (1 - k) * 0.8;
        nodes.commPulse.clear();
        nodes.commPulse
          .circle(0, 0, radius)
          .stroke({ color: nodes.commPulseColor, width: 0.18, alpha: alphaR });
        nodes.commPulse.visible = true;
      } else {
        nodes.commPulse.visible = false;
        nodes.commPulseStart = 0;
      }
    }

    // ---- Plant / defuse progress ring ------------------------------
    const act = state.actState.get(id);
    if (act && curr.alive && (act.planting || act.defusing) && state.bomb) {
      const total = act.planting ? 4.0 : 7.0; // engine defaults; ring is purely visual
      const elapsed = state.bomb.timer;
      const frac = Math.max(0, Math.min(1, elapsed / total));
      const ringColor = act.planting ? 0xff8c42 : 0x4d9eff;
      drawProgressRing(nodes.progressRing, frac, ringColor);
      nodes.progressRing.visible = true;
    } else if (nodes.progressRing.visible) {
      nodes.progressRing.clear();
      nodes.progressRing.visible = false;
    }

    // ---- Selection ring pulse --------------------------------------
    if (nodes.selection.visible) {
      nodes.selection.alpha = selPulse;
    }

    // ---- Smooth hp bar lerp -----------------------------------------
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

/**
 * Draw a 360° progress ring around the body. `frac` in [0,1]. The ring
 * starts at the top (12 o'clock) and fills clockwise.
 */
const drawProgressRing = (g: Graphics, frac: number, color: number): void => {
  g.clear();
  const r = 1.6;
  const startAngle = -Math.PI / 2;
  const endAngle = startAngle + Math.max(0, Math.min(1, frac)) * Math.PI * 2;
  // Dim base ring (unfilled portion).
  g.circle(0, 0, r).stroke({ color: 0x0a0e14, width: 0.32, alpha: 0.7 });
  // Filled arc.
  if (frac > 0.005) {
    g.arc(0, 0, r, startAngle, endAngle, false).stroke({
      color,
      width: 0.34,
      alpha: 0.95,
    });
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
/** Number of rays per cone — see `clippedFovPolygon` perf note. */
const FOV_RAY_COUNT = 64;

/**
 * Draw a wall-clipped FoV cone for the selected agent. The cone arc
 * is sampled by ``FOV_RAY_COUNT`` rays which are each clipped against
 * every sight-blocking wall / cover edge on the map, so the visual
 * stops at obstacles instead of bleeding through them.
 *
 * Pure visual — the real engine LoS lives in
 * ``packages/sim/kivski_sim/visibility.compute_fov`` and is unaffected.
 */
const drawFovCone = (
  layer: Container,
  agent: AgentSnapshot | null,
  mapData: MapData | null,
) => {
  layer.removeChildren();
  if (!agent || !agent.isAlive || !mapData) return;

  // Aggregate every sight-blocking obstacle. Both walls and cover
  // pieces stop visibility in the backend's compute_fov, so we treat
  // the union here. (If we ever introduce decorative walls without
  // `blocks_sight`, filter on a per-shape flag instead.)
  const obstacles = mapData.walls;

  const color = sideColor(agent.side);
  const poly = clippedFovPolygon(
    agent.pos,
    agent.facing,
    (FOV_DEG * Math.PI) / 180,
    FOV_RADIUS,
    obstacles,
    FOV_RAY_COUNT,
  );
  if (poly.length < 3) return;
  const flat = poly.flatMap((p) => [p.x, p.y]);

  const g = new Graphics();
  g.poly(flat).fill({ color, alpha: 0.12 });
  g.poly(flat).stroke({ color, width: 0.15, alpha: 0.5 });
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
        actState: new Map(),
        receivedAt: performance.now(),
        round: -1,
        bomb: null,
        lastEventId: null,
        lastMessageId: null,
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
            state.bomb,
            state.eventFeed,
            state.recentMessages,
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
  // The cone is wall-clipped against `mapDataRef.current`; we read it
  // fresh on every redraw so a late map load doesn't leave a stale
  // closure. `mapName` is a dep so this re-runs when the map changes
  // and the cone re-clips against the new walls.
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
      drawFovCone(layer, selectSelectedAgent(state), mapDataRef.current);
    });
    drawFovCone(
      layer,
      selectSelectedAgent(useStore.getState()),
      mapDataRef.current,
    );
    return () => unsub();
  }, [showFov, pixiReady, mapName]);

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
