/**
 * Map loading: try the backend first, normalise the wire schema to
 * the viewer's `MapData` shape, and fall back to a built-in placeholder
 * so the viewer always has something to render.
 *
 * Backend schema (see packages/sim/maps/loader.py) uses:
 *   { name, width, height, walls: [{polygon:[[x,y]...], blocks_*}],
 *     cover: [{polygon:[[x,y]...], blocks_*, low}],
 *     spawns: { attacker:[[x,y]...], defender:[[x,y]...] },
 *     bombsites: { A:{center,polygon}, B:{center,polygon} },
 *     named_areas: [{name, polygon}] }
 *
 * Viewer schema (`MapData`) uses:
 *   { name, width, height, pxPerUnit,
 *     walls: [{ kind:"wall"|"cover", poly: [{x,y}, ...] }],
 *     zones: [{ id, kind, label, poly: [{x,y}, ...] }] }
 *
 * `normaliseBackendMap` does the translation so the viewer renders the
 * real map shipped by the backend.
 */

import type { MapData, MapWall, MapZone, Vec2 } from "./types";

const MAP_API = "/api/maps";

/** Built-in 'dustline' placeholder map — small bombsite layout. */
export const DUSTLINE_FALLBACK: MapData = {
  name: "dustline",
  width: 64,
  height: 64,
  pxPerUnit: 12,
  walls: [
    // Outer wall ring
    {
      kind: "wall",
      poly: [
        { x: 0, y: 0 },
        { x: 64, y: 0 },
        { x: 64, y: 2 },
        { x: 0, y: 2 },
      ],
    },
    {
      kind: "wall",
      poly: [
        { x: 0, y: 62 },
        { x: 64, y: 62 },
        { x: 64, y: 64 },
        { x: 0, y: 64 },
      ],
    },
    {
      kind: "wall",
      poly: [
        { x: 0, y: 0 },
        { x: 2, y: 0 },
        { x: 2, y: 64 },
        { x: 0, y: 64 },
      ],
    },
    {
      kind: "wall",
      poly: [
        { x: 62, y: 0 },
        { x: 64, y: 0 },
        { x: 64, y: 64 },
        { x: 62, y: 64 },
      ],
    },
    // Mid divider
    {
      kind: "wall",
      poly: [
        { x: 22, y: 28 },
        { x: 42, y: 28 },
        { x: 42, y: 36 },
        { x: 22, y: 36 },
      ],
    },
    // Cover near A
    {
      kind: "cover",
      poly: [
        { x: 12, y: 14 },
        { x: 18, y: 14 },
        { x: 18, y: 18 },
        { x: 12, y: 18 },
      ],
    },
    // Cover near B
    {
      kind: "cover",
      poly: [
        { x: 46, y: 46 },
        { x: 52, y: 46 },
        { x: 52, y: 50 },
        { x: 46, y: 50 },
      ],
    },
  ],
  zones: [
    {
      id: "siteA",
      kind: "site_a",
      label: "A",
      poly: [
        { x: 6, y: 6 },
        { x: 22, y: 6 },
        { x: 22, y: 22 },
        { x: 6, y: 22 },
      ],
    },
    {
      id: "siteB",
      kind: "site_b",
      label: "B",
      poly: [
        { x: 42, y: 42 },
        { x: 58, y: 42 },
        { x: 58, y: 58 },
        { x: 42, y: 58 },
      ],
    },
    {
      id: "spawn_t",
      kind: "spawn_attacker",
      label: "T-Spawn",
      poly: [
        { x: 4, y: 54 },
        { x: 14, y: 54 },
        { x: 14, y: 60 },
        { x: 4, y: 60 },
      ],
    },
    {
      id: "spawn_ct",
      kind: "spawn_defender",
      label: "CT-Spawn",
      poly: [
        { x: 50, y: 4 },
        { x: 60, y: 4 },
        { x: 60, y: 10 },
        { x: 50, y: 10 },
      ],
    },
  ],
};

// ---------- Backend → viewer schema normalisation ----------

type PolyTuple = [number, number];

interface BackendWall {
  polygon?: PolyTuple[];
  poly?: { x: number; y: number }[];
}

interface BackendBombsite {
  center?: PolyTuple;
  polygon?: PolyTuple[];
}

interface BackendNamedArea {
  name?: string;
  polygon?: PolyTuple[];
}

interface BackendMap {
  name?: string;
  width?: number;
  height?: number;
  tile_size?: number;
  walls?: BackendWall[];
  cover?: BackendWall[];
  bombsites?: { A?: BackendBombsite; B?: BackendBombsite };
  spawns?: { attacker?: PolyTuple[]; defender?: PolyTuple[] };
  named_areas?: BackendNamedArea[];

  // Already-in-viewer-schema fields (some test maps may send these directly)
  walls_viewer?: MapWall[];
  zones?: MapZone[];
  pxPerUnit?: number;
}

const tupleToVec = (t: PolyTuple): Vec2 => ({ x: t[0], y: t[1] });

const polygonToPoly = (poly: PolyTuple[] | undefined): Vec2[] =>
  Array.isArray(poly) ? poly.map(tupleToVec) : [];

/** Build a small AABB polygon centred on `(x, y)` for spawn markers. */
const spawnPad = (
  cx: number,
  cy: number,
  half: number,
): Vec2[] => [
  { x: cx - half, y: cy - half },
  { x: cx + half, y: cy - half },
  { x: cx + half, y: cy + half },
  { x: cx - half, y: cy + half },
];

/**
 * Convert the backend's map JSON to the viewer's `MapData`. The wire
 * schema is more verbose (separate walls/cover/spawns/bombsites/named_areas)
 * — the viewer just wants two flat arrays: `walls` + `zones`.
 */
export const normaliseBackendMap = (raw: BackendMap): MapData | null => {
  if (
    typeof raw?.width !== "number" ||
    typeof raw?.height !== "number"
  ) {
    return null;
  }

  // If the payload already matches the viewer schema (zones present and walls
  // already have {poly}), trust it as-is.
  if (
    Array.isArray(raw.zones) &&
    Array.isArray(raw.walls) &&
    raw.walls.every((w) => Array.isArray(w.poly))
  ) {
    return {
      name: raw.name ?? "unknown",
      width: raw.width,
      height: raw.height,
      pxPerUnit: typeof raw.pxPerUnit === "number" ? raw.pxPerUnit : 12,
      walls: raw.walls as unknown as MapWall[],
      zones: raw.zones,
    };
  }

  // ---- Backend native shape: translate ----
  const walls: MapWall[] = [];
  for (const w of raw.walls ?? []) {
    const poly = polygonToPoly(w.polygon);
    if (poly.length >= 3) walls.push({ kind: "wall", poly });
  }
  for (const c of raw.cover ?? []) {
    const poly = polygonToPoly(c.polygon);
    if (poly.length >= 3) walls.push({ kind: "cover", poly });
  }

  const zones: MapZone[] = [];

  // Bombsites
  if (raw.bombsites?.A?.polygon) {
    zones.push({
      id: "siteA",
      kind: "site_a",
      label: "A",
      poly: polygonToPoly(raw.bombsites.A.polygon),
    });
  }
  if (raw.bombsites?.B?.polygon) {
    zones.push({
      id: "siteB",
      kind: "site_b",
      label: "B",
      poly: polygonToPoly(raw.bombsites.B.polygon),
    });
  }

  // Spawns — render as small pads around each spawn point.
  const spawnHalf = 0.6;
  (raw.spawns?.attacker ?? []).forEach((p, i) => {
    zones.push({
      id: `spawn_attacker_${i}`,
      kind: "spawn_attacker",
      label: i === 0 ? "Yellow" : undefined,
      poly: spawnPad(p[0], p[1], spawnHalf),
    });
  });
  (raw.spawns?.defender ?? []).forEach((p, i) => {
    zones.push({
      id: `spawn_defender_${i}`,
      kind: "spawn_defender",
      label: i === 0 ? "Blue" : undefined,
      poly: spawnPad(p[0], p[1], spawnHalf),
    });
  });

  // Named areas — render as neutral zones with their name as label.
  for (const a of raw.named_areas ?? []) {
    if (!a.name || !a.polygon) continue;
    // Skip the ones we already emitted via bombsites/spawns.
    const lower = a.name.toLowerCase();
    if (
      lower === "bombsitea" ||
      lower === "bombsiteb" ||
      lower === "yellowspawn" ||
      lower === "bluespawn"
    ) {
      continue;
    }
    zones.push({
      id: `area_${a.name}`,
      kind: "neutral",
      label: a.name,
      poly: polygonToPoly(a.polygon),
    });
  }

  return {
    name: raw.name ?? "unknown",
    width: raw.width,
    height: raw.height,
    pxPerUnit: typeof raw.tile_size === "number" ? 12 / raw.tile_size : 12,
    walls,
    zones,
  };
};

/**
 * Load a named map. Tries `/api/maps/<name>` first; if the request
 * fails, returns malformed data, or normalisation can't produce a
 * usable shape, returns the built-in fallback.
 */
export async function loadMap(name: string): Promise<MapData> {
  try {
    const res = await fetch(`${MAP_API}/${encodeURIComponent(name)}`, {
      headers: { Accept: "application/json" },
    });
    if (res.ok) {
      const raw = (await res.json()) as BackendMap;
      const normalised = normaliseBackendMap(raw);
      if (normalised && normalised.walls.length > 0) {
        return normalised;
      }

      console.warn(
        `[kivski] /api/maps/${name} returned data that could not be normalised — using fallback.`,
        { rawKeys: Object.keys(raw ?? {}) },
      );
    } else {

      console.warn(
        `[kivski] /api/maps/${name} HTTP ${res.status} — using fallback.`,
      );
    }
  } catch (err) {

    console.warn(`[kivski] /api/maps/${name} fetch failed — using fallback.`, err);
  }

  if (name === DUSTLINE_FALLBACK.name) return DUSTLINE_FALLBACK;
  // Unknown map and no backend — return fallback so the UI still renders.
  return { ...DUSTLINE_FALLBACK, name };
}

export type { MapData } from "./types";
