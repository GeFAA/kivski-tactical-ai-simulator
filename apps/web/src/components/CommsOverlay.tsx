import { useEffect } from "react";
import { Container, Graphics, Text, TextStyle } from "pixi.js";
import { useStore } from "@/lib/store";
import { usePixi } from "@/lib/pixi-context";
import { commActionStyle } from "@/lib/event-icons";
import { Z } from "@/components/MapViewer";
import type { AgentSnapshot, CommAction, MessageItem, Vec2 } from "@/lib/types";

/**
 * Renders communication markers (pings, callouts, suggestions) and
 * optional sender→receiver arrows on top of the map.
 *
 * Lifetime model: each `MessageItem` is rendered for `MAX_AGE_SEC`
 * seconds after `m.ts`, fading out linearly. We re-render on every
 * store update so player movements pull arrows along.
 */

const MAX_AGE_SEC = 3.0;

// Drive a soft animation pulse for ping markers.
const now = (): number => performance.now() / 1000;

interface DrawCtx {
  /** "now" in seconds, used for age + pulse. */
  t: number;
  /** Showing comms arrows from sender→receivers. */
  showArrows: boolean;
}

// ---------- Geometry primitives ----------

/** Draw an arrowhead at (tx, ty) pointing from (sx, sy). */
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
  const lx = ax + Math.cos(left) * (size * 0.5);
  const ly = ay + Math.sin(left) * (size * 0.5);
  const rx = ax + Math.cos(right) * (size * 0.5);
  const ry = ay + Math.sin(right) * (size * 0.5);
  g.poly([tx, ty, lx, ly, rx, ry]).fill({ color, alpha });
};

const dashedLine = (
  g: Graphics,
  sx: number,
  sy: number,
  tx: number,
  ty: number,
  dash: number,
  gap: number,
  color: number,
  width: number,
  alpha: number,
) => {
  const dx = tx - sx;
  const dy = ty - sy;
  const len = Math.hypot(dx, dy);
  if (len < 0.0001) return;
  const ux = dx / len;
  const uy = dy / len;
  let cursor = 0;
  while (cursor < len) {
    const segEnd = Math.min(cursor + dash, len);
    g.moveTo(sx + ux * cursor, sy + uy * cursor)
      .lineTo(sx + ux * segEnd, sy + uy * segEnd)
      .stroke({ color, width, alpha });
    cursor = segEnd + gap;
  }
};

// ---------- Marker styles ----------

const drawMarker = (
  layer: Container,
  _m: MessageItem,
  action: CommAction,
  pos: Vec2,
  age01: number,
) => {
  const style = commActionStyle(action);
  const alpha = 1 - age01;

  const g = new Graphics();
  g.position.set(pos.x, pos.y);

  switch (action) {
    case "PING_LOCATION": {
      // Two pulsing rings.
      const pulse = 1 + Math.sin(age01 * Math.PI * 3) * 0.4;
      g.circle(0, 0, 1.4 * pulse).stroke({ color: style.color, width: 0.3, alpha });
      g.circle(0, 0, 0.6).fill({ color: style.color, alpha: alpha * 0.9 });
      break;
    }
    case "WARN_DANGER": {
      // Red triangle.
      const s = 2.0;
      g.poly([0, -s, -s * 0.866, s * 0.5, s * 0.866, s * 0.5])
        .fill({ color: style.color, alpha: alpha * 0.65 })
        .stroke({ color: style.color, width: 0.25, alpha });
      break;
    }
    case "REQUEST_SUPPORT": {
      g.circle(0, 0, 1.5).fill({ color: style.color, alpha: alpha * 0.35 });
      const t = new Text({
        text: "?",
        style: new TextStyle({
          fontFamily: "ui-monospace, monospace",
          fontSize: 2.4,
          fill: style.color,
          fontWeight: "900",
        }),
      });
      t.anchor.set(0.5);
      t.alpha = alpha;
      g.addChild(t);
      break;
    }
    case "CONTACT_ENEMY": {
      g.circle(0, 0, 1.4).fill({ color: style.color, alpha: alpha * 0.4 });
      const t = new Text({
        text: "!",
        style: new TextStyle({
          fontFamily: "ui-monospace, monospace",
          fontSize: 2.6,
          fill: style.color,
          fontWeight: "900",
        }),
      });
      t.anchor.set(0.5);
      t.alpha = alpha;
      g.addChild(t);
      break;
    }
    case "BOMBSITE_CLEAR": {
      g.circle(0, 0, 1.4).fill({ color: style.color, alpha: alpha * 0.35 });
      // check mark.
      g.moveTo(-0.8, 0.0)
        .lineTo(-0.2, 0.7)
        .lineTo(1.0, -0.8)
        .stroke({ color: style.color, width: 0.4, alpha });
      break;
    }
    case "SUGGEST_FALLBACK": {
      g.circle(0, 0, 0.9).fill({ color: style.color, alpha: alpha * 0.3 });
      const t = new Text({
        text: "←",
        style: new TextStyle({
          fontFamily: "ui-monospace, monospace",
          fontSize: 2.4,
          fill: style.color,
          fontWeight: "900",
        }),
      });
      t.anchor.set(0.5);
      t.alpha = alpha;
      g.addChild(t);
      break;
    }
    default: {
      // Generic small dot for actions without a dedicated marker.
      g.circle(0, 0, 0.7).fill({ color: style.color, alpha });
    }
  }

  layer.addChild(g);
};

const drawSuggestionArrow = (
  layer: Container,
  from: Vec2,
  to: Vec2,
  color: number,
  alpha: number,
) => {
  const g = new Graphics();
  g.moveTo(from.x, from.y)
    .lineTo(to.x, to.y)
    .stroke({ color, width: 0.4, alpha });
  arrowHead(g, from.x, from.y, to.x, to.y, 1.6, color, alpha);
  layer.addChild(g);
};

const drawCommsArrow = (
  layer: Container,
  from: Vec2,
  to: Vec2,
  color: number,
  alpha: number,
) => {
  const g = new Graphics();
  dashedLine(g, from.x, from.y, to.x, to.y, 0.8, 0.6, color, 0.18, alpha * 0.85);
  arrowHead(g, from.x, from.y, to.x, to.y, 1.0, color, alpha * 0.9);
  layer.addChild(g);
};

// ---------- Top-level render ----------

const render = (
  layer: Container,
  messages: MessageItem[],
  agents: AgentSnapshot[],
  ctx: DrawCtx,
) => {
  layer.removeChildren();

  // Index agents by id for arrow endpoints. Skip dead agents (no markers).
  const byId = new Map<string, AgentSnapshot>();
  for (const a of agents) byId.set(a.id, a);

  for (const m of messages) {
    const ageSec = ctx.t - m.ts / 1000;
    if (ageSec < 0 || ageSec > MAX_AGE_SEC) continue;
    const age01 = ageSec / MAX_AGE_SEC;
    const alpha = Math.max(0, 1 - age01);

    const action = (m.action ?? "SILENT") as CommAction;
    const style = commActionStyle(action);
    const sender = byId.get(m.fromId);
    const targetPos = m.pos ?? sender?.pos ?? null;

    // Marker at message position.
    if (targetPos && action !== "SILENT" && action !== "ACK") {
      drawMarker(layer, m, action, targetPos, age01);
    }

    // Directional suggestion arrows (sender → target pos)
    if (
      sender &&
      targetPos &&
      (action === "SUGGEST_ROTATE" ||
        action === "SUGGEST_ATTACK" ||
        action === "SUGGEST_FALLBACK")
    ) {
      drawSuggestionArrow(layer, sender.pos, targetPos, style.color, alpha);
    }

    // Comms arrows (sender → each receiver) when the toggle is on.
    if (ctx.showArrows && sender) {
      for (const toId of m.toIds) {
        const recv = byId.get(toId);
        if (!recv || recv.id === sender.id) continue;
        drawCommsArrow(layer, sender.pos, recv.pos, style.color, alpha);
      }
    }
  }
};

// ---------- React shell ----------

const CommsOverlay = () => {
  const pixi = usePixi();

  useEffect(() => {
    if (!pixi) return;
    const layer = pixi.addLayer("commsOverlay", Z.commsOverlay);

    let raf = 0;
    const step = () => {
      const state = useStore.getState();
      const showComms = state.showComms;
      layer.visible = showComms || state.recentMessages.length > 0;
      render(layer, state.recentMessages, state.agents, {
        t: now(),
        showArrows: showComms,
      });
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);

    return () => {
      cancelAnimationFrame(raf);
      layer.removeChildren();
    };
  }, [pixi]);

  return null;
};

export default CommsOverlay;
