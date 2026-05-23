import { useEffect } from "react";
import { Container, Graphics } from "pixi.js";
import { useStore } from "@/lib/store";
import { usePixi } from "@/lib/pixi-context";
import { Z } from "@/components/MapViewer";

/**
 * Aggregated position heatmap rendered as alpha-blended yellow/blue
 * rectangles. The store keeps a small ring buffer of sampled positions
 * (1-in-N ticks across all alive agents) that we bucket into a coarse
 * grid at draw time and shade by density.
 *
 * The overlay is only rendered when `showHeatmap` is true (toggled
 * via DebugToggles).
 */

const GRID_W = 30;
const GRID_H = 20;

const ATK_COLOR = 0xffc833;
const DEF_COLOR = 0x4da8ff;

const render = (
  layer: Container,
  positions: { side: "attacker" | "defender"; x: number; y: number }[],
  mapW: number,
  mapH: number,
) => {
  layer.removeChildren();
  if (positions.length === 0) return;

  const cellW = mapW / GRID_W;
  const cellH = mapH / GRID_H;
  // 2-D dense matrices for the two sides.
  const atk = new Float32Array(GRID_W * GRID_H);
  const def = new Float32Array(GRID_W * GRID_H);

  for (const p of positions) {
    const gx = Math.max(0, Math.min(GRID_W - 1, Math.floor(p.x / cellW)));
    const gy = Math.max(0, Math.min(GRID_H - 1, Math.floor(p.y / cellH)));
    const idx = gy * GRID_W + gx;
    if (p.side === "attacker") atk[idx] += 1;
    else def[idx] += 1;
  }
  let maxA = 1;
  let maxD = 1;
  for (let i = 0; i < atk.length; i++) {
    if (atk[i] > maxA) maxA = atk[i];
    if (def[i] > maxD) maxD = def[i];
  }

  const g = new Graphics();
  for (let gy = 0; gy < GRID_H; gy++) {
    for (let gx = 0; gx < GRID_W; gx++) {
      const idx = gy * GRID_W + gx;
      const a = atk[idx] / maxA;
      const d = def[idx] / maxD;
      if (a < 0.03 && d < 0.03) continue;
      const x = gx * cellW;
      const y = gy * cellH;
      if (a >= d) {
        g.rect(x, y, cellW, cellH).fill({ color: ATK_COLOR, alpha: Math.min(0.55, a * 0.55) });
      } else {
        g.rect(x, y, cellW, cellH).fill({ color: DEF_COLOR, alpha: Math.min(0.55, d * 0.55) });
      }
    }
  }
  layer.addChild(g);
};

const HeatmapOverlay = () => {
  const pixi = usePixi();
  const showHeatmap = useStore((s) => s.showHeatmap);

  useEffect(() => {
    if (!pixi) return;
    const layer = pixi.addLayer("heatmap", Z.heatmap);
    layer.visible = showHeatmap;
    if (!showHeatmap) {
      layer.removeChildren();
      return;
    }

    const drawNow = () => {
      const s = useStore.getState();
      render(layer, s.heatmapPositions, pixi.mapWidth, pixi.mapHeight);
    };
    drawNow();
    // Update at most ~2x/sec to keep cost negligible.
    const interval = setInterval(drawNow, 500);

    return () => {
      clearInterval(interval);
      layer.removeChildren();
    };
  }, [pixi, showHeatmap]);

  return null;
};

export default HeatmapOverlay;
