import { useEffect } from "react";
import { Container, Graphics } from "pixi.js";
import { useStore } from "@/lib/store";
import { usePixi } from "@/lib/pixi-context";
import { Z } from "@/components/MapViewer";
import type { AgentSnapshot } from "@/lib/types";

/**
 * Draws attention-influence arrows for the currently selected agent:
 *
 *   • Outgoing (blue): selected agent attends to teammate → arrow points
 *     to the teammate, line thickness scales with the weight in [0,1].
 *   • Incoming (yellow): another teammate is attending to the selected
 *     agent → arrow points from that teammate to the selected agent.
 *
 * Weights are read from `state.attentionWeights[observerId][targetId]`.
 * Only weights above THRESHOLD are drawn (keeps the overlay readable).
 */

const THRESHOLD = 0.05;
const OUTGOING_COLOR = 0x4da8ff; // blue (defender accent)
const INCOMING_COLOR = 0xffd24a; // yellow

const arrowHead = (
  g: Graphics,
  sx: number,
  sy: number,
  tx: number,
  ty: number,
  size: number,
  color: number,
  alpha: number,
) => {
  const ang = Math.atan2(ty - sy, tx - sx);
  const ax = tx - Math.cos(ang) * size;
  const ay = ty - Math.sin(ang) * size;
  const left = ang + Math.PI / 2;
  const right = ang - Math.PI / 2;
  const lx = ax + Math.cos(left) * (size * 0.45);
  const ly = ay + Math.sin(left) * (size * 0.45);
  const rx = ax + Math.cos(right) * (size * 0.45);
  const ry = ay + Math.sin(right) * (size * 0.45);
  g.poly([tx, ty, lx, ly, rx, ry]).fill({ color, alpha });
};

const drawWeightedArrow = (
  layer: Container,
  from: { x: number; y: number },
  to: { x: number; y: number },
  weight: number,
  color: number,
) => {
  const w = Math.max(0.1, Math.min(1, weight));
  const alpha = 0.4 + w * 0.5; // 0.4 .. 0.9
  const width = 0.15 + w * 0.6; // 0.15 .. 0.75
  // Shorten endpoints so they don't overlap the player dots.
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.hypot(dx, dy);
  if (len < 0.0001) return;
  const ux = dx / len;
  const uy = dy / len;
  const sx = from.x + ux * 1.3;
  const sy = from.y + uy * 1.3;
  const tx = to.x - ux * 1.6;
  const ty = to.y - uy * 1.6;

  const g = new Graphics();
  g.moveTo(sx, sy).lineTo(tx, ty).stroke({ color, width, alpha });
  arrowHead(g, sx, sy, tx, ty, 1.2, color, alpha);
  layer.addChild(g);
};

const render = (
  layer: Container,
  selectedId: string | null,
  agents: AgentSnapshot[],
  attention: Record<string, Record<string, number>>,
) => {
  layer.removeChildren();
  if (!selectedId) return;
  const sel = agents.find((a) => a.id === selectedId);
  if (!sel || !sel.isAlive) return;
  const byId = new Map<string, AgentSnapshot>();
  for (const a of agents) byId.set(a.id, a);

  // Outgoing: selected → teammates the selected agent attends to.
  const outgoing = attention[selectedId] ?? {};
  for (const [targetId, w] of Object.entries(outgoing)) {
    if (w < THRESHOLD || targetId === selectedId) continue;
    const tgt = byId.get(targetId);
    if (!tgt || !tgt.isAlive) continue;
    drawWeightedArrow(layer, sel.pos, tgt.pos, w, OUTGOING_COLOR);
  }

  // Incoming: any other observer attending to the selected agent.
  for (const [observerId, weights] of Object.entries(attention)) {
    if (observerId === selectedId) continue;
    const w = weights[selectedId];
    if (typeof w !== "number" || w < THRESHOLD) continue;
    const observer = byId.get(observerId);
    if (!observer || !observer.isAlive) continue;
    drawWeightedArrow(layer, observer.pos, sel.pos, w, INCOMING_COLOR);
  }
};

const InfluenceArrows = () => {
  const pixi = usePixi();

  useEffect(() => {
    if (!pixi) return;
    const layer = pixi.addLayer("influence", Z.influence);

    const drawNow = () => {
      const s = useStore.getState();
      render(layer, s.selectedAgentId, s.agents, s.attentionWeights);
    };
    drawNow();
    const unsub = useStore.subscribe(drawNow);

    return () => {
      unsub();
      layer.removeChildren();
    };
  }, [pixi]);

  return null;
};

export default InfluenceArrows;
