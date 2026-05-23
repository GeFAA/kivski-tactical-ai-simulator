/**
 * Map loading: try the backend first, fall back to a built-in
 * placeholder so the viewer always has something to render.
 */

import type { MapData } from "./types";

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

/**
 * Load a named map. Tries `/api/maps/<name>` first; if the request
 * fails or the server is offline, returns the built-in fallback (only
 * for 'dustline' — other names throw so the caller knows).
 */
export async function loadMap(name: string): Promise<MapData> {
  try {
    const res = await fetch(`${MAP_API}/${encodeURIComponent(name)}`, {
      headers: { Accept: "application/json" },
    });
    if (res.ok) {
      const data = (await res.json()) as MapData;
      // Trust-but-verify a couple of required fields.
      if (
        typeof data?.width === "number" &&
        typeof data?.height === "number" &&
        Array.isArray(data?.walls) &&
        Array.isArray(data?.zones)
      ) {
        return data;
      }
       
      console.warn(`[kivski] /api/maps/${name} returned malformed data — using fallback.`);
    }
  } catch (err) {
     
    console.warn(`[kivski] /api/maps/${name} fetch failed — using fallback.`, err);
  }

  if (name === DUSTLINE_FALLBACK.name) return DUSTLINE_FALLBACK;
  // Unknown map and no backend — return fallback so the UI still renders.
  return { ...DUSTLINE_FALLBACK, name };
}

export type { MapData } from "./types";
