/**
 * Shared PixiJS application context.
 *
 * The MapViewer creates a single `PIXI.Application` and a root world
 * container scaled to fit the host. Overlay components (CommsOverlay,
 * InfluenceArrows, HeatmapOverlay, FovOverlay, ...) consume this
 * context, hook a `Container` into the world at the right z-order,
 * and drive its draw calls via `useEffect` + a store subscription —
 * keeping React's render path completely off the per-tick hot path.
 *
 * Why a context instead of self-mounted React canvases?
 *   - One PixiJS app, one canvas, shared transform: panning / zoom
 *     applied to `world` automatically affects every overlay.
 *   - No double-buffer / no compositing weirdness.
 *   - Overlay z-order is explicit & declarative via `addLayer(zIndex)`.
 */

import { createContext, useContext } from "react";
import type { Application, Container } from "pixi.js";

export interface PixiContextValue {
  app: Application;
  /** Root container, scaled & centered to fit the map. */
  world: Container;
  /**
   * Register an overlay layer at a stable z-order. Returns the
   * container; the same `key` always returns the same container
   * (so re-mounts don't leak layers).
   */
  addLayer: (key: string, zIndex: number) => Container;
  /** Map dimensions in world units. */
  mapWidth: number;
  mapHeight: number;
}

export const PixiContext = createContext<PixiContextValue | null>(null);

/**
 * Read the Pixi app + helpers. Returns `null` while the MapViewer is
 * still booting (e.g. before the async `app.init()` resolves) so
 * overlay components can early-return safely.
 */
export const usePixi = (): PixiContextValue | null => useContext(PixiContext);
