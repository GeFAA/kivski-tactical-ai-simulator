/**
 * Mapping tables for the Kivski viewer. Centralised so map overlays,
 * the comms tab, the inspector, and the metrics panel all stay in
 * visual sync (same colors, same emoji shorthand, same labels).
 *
 * All colors are returned as raw 0xRRGGBB integers so they're directly
 * usable as PixiJS `Graphics.fill({color})` arguments. CSS classes are
 * exposed separately for React components.
 */

import type { CommAction, EventKind, RoundOutcome } from "./types";

// ---------- Communications ----------

export interface CommActionStyle {
  /** Short label used in tabs / chips. */
  label: string;
  /** Single-glyph or emoji shorthand for tight overlays. */
  glyph: string;
  /** Pixi-friendly 0xRRGGBB color. */
  color: number;
  /** CSS color string (used for backgrounds / borders). */
  css: string;
}

/**
 * Style table for every CommAction the policy can emit.
 * The marker shapes themselves are picked by `CommsOverlay` based on
 * the action enum (circle / triangle / arrow / etc.).
 */
export const COMM_ACTION_STYLES: Record<CommAction, CommActionStyle> = {
  PING_LOCATION: { label: "Ping",        glyph: "•", color: 0x4da8ff, css: "#4DA8FF" },
  WARN_DANGER:   { label: "Danger",      glyph: "!", color: 0xff4d4d, css: "#FF4D4D" },
  REQUEST_SUPPORT:{ label: "Need Help",  glyph: "?", color: 0xff9933, css: "#FF9933" },
  SUGGEST_ROTATE:{ label: "Rotate",      glyph: "↻", color: 0x4da8ff, css: "#4DA8FF" },
  SUGGEST_ATTACK:{ label: "Push",        glyph: "→", color: 0xffd24a, css: "#FFD24A" },
  SUGGEST_FALLBACK:{ label: "Fall Back", glyph: "←", color: 0x9aa3b2, css: "#9AA3B2" },
  CONTACT_ENEMY: { label: "Enemy",       glyph: "!", color: 0xff4d4d, css: "#FF4D4D" },
  BOMBSITE_CLEAR:{ label: "Clear",       glyph: "✓", color: 0x4ade80, css: "#4ADE80" },
  ACK:           { label: "Ack",         glyph: "·", color: 0x6b7585, css: "#6B7585" },
  SILENT:        { label: "Silent",      glyph: " ", color: 0x444a55, css: "#444A55" },
};

/** Safe lookup that always returns a style (falls back to SILENT). */
export const commActionStyle = (a: CommAction | undefined | null): CommActionStyle => {
  if (!a) return COMM_ACTION_STYLES.SILENT;
  return COMM_ACTION_STYLES[a] ?? COMM_ACTION_STYLES.SILENT;
};

// ---------- Round outcomes ----------

export interface OutcomeStyle {
  label: string;
  /** Pixi color. */
  color: number;
  css: string;
  /** "yellow" | "blue" | "draw" — used for team-tinted UI. */
  team: "attacker" | "defender" | "draw";
}

export const ROUND_OUTCOME_STYLES: Record<RoundOutcome, OutcomeStyle> = {
  attacker_elim:  { label: "Atk Elim",     color: 0xffc833, css: "#FFC833", team: "attacker" },
  defender_elim:  { label: "Def Elim",     color: 0x4da8ff, css: "#4DA8FF", team: "defender" },
  bomb_explode:   { label: "Bomb Boom",    color: 0xffc833, css: "#FFC833", team: "attacker" },
  bomb_defused:   { label: "Defused",      color: 0x4da8ff, css: "#4DA8FF", team: "defender" },
  time_out:       { label: "Time Out",     color: 0x9aa3b2, css: "#9AA3B2", team: "draw" },
  draw:           { label: "Draw",         color: 0x9aa3b2, css: "#9AA3B2", team: "draw" },
};

export const outcomeStyle = (o: RoundOutcome | undefined | null): OutcomeStyle => {
  if (!o) return ROUND_OUTCOME_STYLES.draw;
  return ROUND_OUTCOME_STYLES[o] ?? ROUND_OUTCOME_STYLES.draw;
};

// ---------- Event kinds ----------

export const EVENT_KIND_LABEL: Record<EventKind, string> = {
  kill: "kill",
  death: "death",
  plant: "plant",
  defuse: "defuse",
  round_start: "round start",
  round_end: "round end",
  bomb_explode: "bomb",
  purchase: "buy",
  comm: "comm",
  info: "info",
  sound: "sound",
};

// ---------- Observation feature groups ----------
//
// Used by AgentInspector to render the per-group observation bar chart.
// Mirrors `packages/marl/obs/groups.py` group names on the backend.

export const OBSERVATION_SCHEMA: { id: string; label: string; color: string }[] = [
  { id: "self",     label: "Self state",     color: "#4DA8FF" },
  { id: "team",     label: "Teammates",      color: "#6BCB77" },
  { id: "enemies",  label: "Visible enemies", color: "#FF6B6B" },
  { id: "vision",   label: "Vision rays",    color: "#FACC15" },
  { id: "comm",     label: "Comm channels",  color: "#A78BFA" },
  { id: "memory",   label: "Memory / last-known", color: "#9AA3B2" },
  { id: "phase",    label: "Phase / timer",  color: "#FFC833" },
  { id: "economy",  label: "Economy",        color: "#FACC15" },
];
