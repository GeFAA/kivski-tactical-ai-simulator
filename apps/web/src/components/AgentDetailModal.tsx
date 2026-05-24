/**
 * Modal sheet that opens when the user clicks an agent dot on the map
 * or an agent card in the left sidebar. It shows the agent's:
 *  - Identity header (team-coloured avatar + editable display name)
 *  - Live status (HP / armor / money / weapon)
 *  - Round + match K/D/A stats
 *  - Interpreted "what they're doing" line (planting / defusing /
 *    carrying bomb / dead / combat / patrolling)
 *  - Last 3 comm messages addressed *to* this agent
 *
 * The custom display name is persisted to localStorage via the
 * ``setCustomAgentName`` store action, so the chosen handle survives
 * across reloads, match resets, and backend restarts. The modal lazily
 * mounts (the parent only renders it when ``agentDetailOpen`` is true),
 * so an off-screen modal costs nothing.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  resolveAgentName,
  selectSelectedAgent,
  useStore,
} from "@/lib/store";
import { commActionStyle } from "@/lib/event-icons";
import type { AgentSnapshot, EventItem, MessageItem } from "@/lib/types";

// ---------- Small presentational bits ----------

const ProgressBar = ({
  pct,
  colorClass,
  height = "h-2",
}: {
  pct: number;
  colorClass: string;
  height?: string;
}) => {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div className={`relative ${height} w-full overflow-hidden rounded bg-[#0f131a]`}>
      <div
        className={`absolute inset-y-0 left-0 ${colorClass} transition-all`}
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
};

const StatTile = ({
  label,
  value,
  valueClass = "text-kivski-text",
}: {
  label: string;
  value: string | number;
  valueClass?: string;
}) => (
  <div className="rounded bg-kivski-bg p-2 text-center">
    <div className={`stat text-lg font-semibold ${valueClass}`}>{value}</div>
    <div className="mt-0.5 text-[10px] uppercase tracking-widest text-kivski-muted">
      {label}
    </div>
  </div>
);

const SectionTitle = ({ children }: { children: React.ReactNode }) => (
  <div className="mb-1.5 text-[10px] uppercase tracking-widest text-kivski-muted">
    {children}
  </div>
);

// ---------- Domain helpers ----------

const weaponKindToLabel: Record<AgentSnapshot["weapons"][number]["kind"], string> = {
  knife: "Blade",
  pistol: "Pistol",
  smg: "SMG",
  rifle: "Rifle",
  ar: "Rifle",
  sniper: "Marksman",
  shotgun: "Shotgun",
  lmg: "LMG",
  grenade: "Grenade",
  flash: "Flash",
  smoke: "Smoke",
  molotov: "Molotov",
  c4: "C4",
};

const weaponKindToGlyph: Partial<Record<AgentSnapshot["weapons"][number]["kind"], string>> = {
  knife: "⚔",
  pistol: "🔫",
  smg: "▮",
  rifle: "▰",
  ar: "▰",
  sniper: "⌖",
  shotgun: "▤",
};

/**
 * "What are they doing right now" — a one-line interpretation of the
 * agent's current state. Priority-ordered so a planting agent reads as
 * "planting" instead of "in combat" if they happen to also be taking
 * fire. Falls back to a passive descriptor when nothing notable is
 * happening so the line is never empty.
 */
const describeAction = (
  agent: AgentSnapshot,
  recentEvents: EventItem[],
): { glyph: string; text: string; tone: string } => {
  if (!agent.isAlive) {
    return { glyph: "💀", text: "Down — waiting for respawn", tone: "text-kivski-hp-low" };
  }
  if (agent.isPlanting) {
    return { glyph: "💣", text: "Planting the bomb", tone: "text-kivski-bomb" };
  }
  if (agent.isDefusing) {
    return { glyph: "🛠️", text: "Defusing the bomb", tone: "text-kivski-defender" };
  }
  if (agent.hasBomb) {
    return { glyph: "🎒", text: "Carrying the bomb to site", tone: "text-kivski-bomb" };
  }
  // Recent damage scan — if this agent was involved in a kill / death /
  // info event in the last ~5 seconds, treat them as "in combat".
  const now = Date.now();
  const inCombat = recentEvents.some(
    (e) =>
      now - e.ts < 5_000 &&
      (e.kind === "kill" || e.kind === "death" || e.kind === "info") &&
      (e.actorId === agent.id || e.targetId === agent.id),
  );
  if (inCombat) {
    return { glyph: "⚔️", text: "Engaged in combat", tone: "text-kivski-hp-low" };
  }
  if (agent.intent) {
    return { glyph: "🎯", text: agent.intent, tone: "text-kivski-text" };
  }
  if (agent.side === "defender") {
    return { glyph: "👁️", text: "Holding position / watching", tone: "text-kivski-muted" };
  }
  return { glyph: "🚶", text: "Moving up the map", tone: "text-kivski-muted" };
};

// ---------- Recent comm row ----------

const MessageRow = ({
  m,
  nameOf,
}: {
  m: MessageItem;
  nameOf: (id: string) => string;
}) => {
  const style = commActionStyle(m.action);
  return (
    <li className="flex items-start gap-2 border-b border-kivski-border/40 py-1.5 last:border-b-0">
      <span
        className="pill shrink-0"
        style={{ background: `${style.css}22`, color: style.css }}
      >
        <span className="mr-1 font-bold">{style.glyph}</span>
        {m.actionLabel ?? style.label}
      </span>
      <div className="min-w-0 flex-1 leading-tight">
        <div className="truncate text-[11px] text-kivski-text">
          from <span className="font-medium">{nameOf(m.fromId)}</span>
        </div>
        <div className="stat text-[10px] text-kivski-muted">tick {m.tick}</div>
      </div>
    </li>
  );
};

// ---------- Main modal ----------

const AgentDetailModal = () => {
  const isOpen = useStore((s) => s.agentDetailOpen);
  const closeAgentDetail = useStore((s) => s.closeAgentDetail);
  const agent = useStore(selectSelectedAgent);
  const customAgentNames = useStore((s) => s.customAgentNames);
  const setCustomAgentName = useStore((s) => s.setCustomAgentName);
  const allMessages = useStore((s) => s.recentMessages);
  const allEvents = useStore((s) => s.eventFeed);
  const agents = useStore((s) => s.agents);

  const [draftName, setDraftName] = useState("");
  // Track the last agent id we synced the draft for, so switching from
  // one agent to another (modal stays mounted) resets the input box
  // instead of carrying over the previous name.
  const lastSyncedId = useRef<string | null>(null);

  // Sync draft input → resolved display name whenever the selected
  // agent changes (or when the modal re-opens for a fresh selection).
  useEffect(() => {
    if (!isOpen || !agent) return;
    if (lastSyncedId.current === agent.id) return;
    const resolved = resolveAgentName(agent, customAgentNames);
    setDraftName(resolved);
    lastSyncedId.current = agent.id;
  }, [isOpen, agent, customAgentNames]);

  // Close on ESC.
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeAgentDetail();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isOpen, closeAgentDetail]);

  // Reset the tracked id when the modal closes so re-opening for the
  // same agent still pulls in any backend updates.
  useEffect(() => {
    if (!isOpen) lastSyncedId.current = null;
  }, [isOpen]);

  const receivedMessages = useMemo(() => {
    if (!agent) return [];
    return allMessages.filter((m) => m.toIds.includes(agent.id)).slice(0, 3);
  }, [agent, allMessages]);

  /** Translate raw agent ids to friendly display names for comm authors. */
  const nameOf = useMemo(() => {
    const lookup = new Map<string, string>();
    for (const a of agents) {
      lookup.set(a.id, resolveAgentName(a, customAgentNames));
    }
    return (id: string): string => lookup.get(id) ?? id;
  }, [agents, customAgentNames]);

  if (!isOpen || !agent) return null;

  const isAttacker = agent.side === "attacker";
  const teamColor = agent.team === "yellow" ? "text-kivski-attacker" : "text-kivski-defender";
  const teamBg = agent.team === "yellow" ? "bg-kivski-attacker" : "bg-kivski-defender";
  const sideBadgeClass = isAttacker
    ? "bg-kivski-attacker/20 text-kivski-attacker"
    : "bg-kivski-defender/20 text-kivski-defender";

  const hpPct = Math.max(0, Math.min(100, agent.hp));
  const hpBarColor = hpPct < 33 ? "bg-kivski-hp-low" : "bg-kivski-hp";
  const armorPct = Math.max(0, Math.min(100, agent.armor));
  const primary = agent.weapons[agent.activeWeaponIdx] ?? agent.weapons[0];
  const weaponLabel = primary ? weaponKindToLabel[primary.kind] : "—";
  const weaponGlyph = primary ? weaponKindToGlyph[primary.kind] ?? "·" : "·";

  const resolvedName = resolveAgentName(agent, customAgentNames);
  const action = describeAction(agent, allEvents);

  const handleSave = () => {
    setCustomAgentName(agent.id, draftName);
  };
  const handleReset = () => {
    setCustomAgentName(agent.id, "");
    setDraftName(agent.name);
  };

  return (
    <div
      className="fixed inset-0 z-40 flex items-stretch justify-end"
      role="dialog"
      aria-label={`Agent details — ${resolvedName}`}
      aria-modal="true"
    >
      {/* Backdrop — clicking outside the sheet closes it. */}
      <button
        type="button"
        aria-label="close-backdrop"
        onClick={closeAgentDetail}
        className="absolute inset-0 bg-black/55 backdrop-blur-sm"
      />

      {/* Right-side sheet */}
      <div className="relative flex h-full w-full max-w-md flex-col overflow-hidden border-l border-kivski-border bg-kivski-panel shadow-2xl">
        {/* Header */}
        <div className="flex items-start gap-3 border-b border-kivski-border px-4 py-3">
          <div
            className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-full ${teamBg}/30 ring-2 ring-inset ${
              isAttacker ? "ring-kivski-attacker/60" : "ring-kivski-defender/60"
            }`}
          >
            <span className={`text-lg font-bold ${teamColor}`}>
              {agent.team === "yellow" ? "Y" : "B"}
            </span>
          </div>
          <div className="min-w-0 flex-1">
            <div className="mb-1 flex items-center gap-2">
              <span className={`text-[10px] uppercase tracking-widest ${teamColor}`}>
                {agent.team} team
              </span>
              <span className={`pill ${sideBadgeClass}`}>{agent.side}</span>
              {!agent.isAlive && (
                <span className="pill bg-kivski-hp-low/15 text-kivski-hp-low">dead</span>
              )}
            </div>
            <div className="text-[10px] uppercase tracking-wider text-kivski-muted">
              backend id: {agent.id}
            </div>
          </div>
          <button
            type="button"
            onClick={closeAgentDetail}
            aria-label="close-modal"
            className="rounded p-1 text-kivski-muted hover:bg-kivski-panel-2 hover:text-kivski-text"
          >
            {"✕"}
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 space-y-3 overflow-y-auto px-4 py-3">
          {/* Editable name input */}
          <section>
            <SectionTitle>Display name</SectionTitle>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={draftName}
                onChange={(e) => setDraftName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave();
                }}
                placeholder="Agent name"
                aria-label="agent-display-name"
                maxLength={32}
                className="flex-1 rounded border border-kivski-border bg-kivski-bg px-2.5 py-1.5 text-sm text-kivski-text focus:border-kivski-defender focus:outline-none focus:ring-1 focus:ring-kivski-defender/40"
              />
              <button
                type="button"
                onClick={handleSave}
                className="btn btn-primary"
                disabled={draftName.trim() === resolvedName}
              >
                Save
              </button>
            </div>
            <div className="mt-1 flex items-center justify-between text-[10px] text-kivski-muted">
              <span>
                Currently shown as{" "}
                <span className="font-medium text-kivski-text">{resolvedName}</span>
              </span>
              {customAgentNames[agent.id] && (
                <button
                  type="button"
                  onClick={handleReset}
                  className="text-kivski-muted underline hover:text-kivski-text"
                >
                  Reset
                </button>
              )}
            </div>
          </section>

          {/* What are they doing */}
          <section className="rounded border border-kivski-border bg-kivski-panel-2 px-3 py-2">
            <SectionTitle>Right now</SectionTitle>
            <div className={`flex items-center gap-2 text-sm font-medium ${action.tone}`}>
              <span className="text-lg leading-none">{action.glyph}</span>
              <span>{action.text}</span>
            </div>
          </section>

          {/* Live status */}
          <section className="space-y-2">
            <SectionTitle>Live status</SectionTitle>
            <div className="space-y-1.5">
              <div>
                <div className="mb-0.5 flex items-baseline justify-between text-[11px]">
                  <span className="text-kivski-muted">HP</span>
                  <span className="stat">
                    <span className="text-kivski-text">{agent.hp}</span>
                    <span className="text-kivski-muted">/100</span>
                  </span>
                </div>
                <ProgressBar pct={hpPct} colorClass={hpBarColor} />
              </div>
              <div>
                <div className="mb-0.5 flex items-baseline justify-between text-[11px]">
                  <span className="text-kivski-muted">Armor</span>
                  <span className="stat">
                    <span className="text-kivski-armor">{agent.armor}</span>
                    <span className="text-kivski-muted">/100</span>
                  </span>
                </div>
                <ProgressBar pct={armorPct} colorClass="bg-kivski-armor" />
              </div>
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-kivski-muted">Money</span>
                <span className="stat text-kivski-money">${agent.money}</span>
              </div>
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-kivski-muted">Weapon</span>
                <span className="stat text-kivski-text">
                  {weaponGlyph} {weaponLabel}
                </span>
              </div>
              {(agent.hasBomb || agent.hasDefuseKit) && (
                <div className="flex flex-wrap gap-1 pt-1">
                  {agent.hasBomb && (
                    <span className="pill bg-kivski-bomb/15 text-kivski-bomb">
                      Carrying C4
                    </span>
                  )}
                  {agent.hasDefuseKit && (
                    <span className="pill bg-kivski-defender/15 text-kivski-defender">
                      Defuse Kit
                    </span>
                  )}
                </div>
              )}
            </div>
          </section>

          {/* Round stats */}
          <section>
            <SectionTitle>Match stats</SectionTitle>
            <div className="grid grid-cols-3 gap-2">
              <StatTile label="Kills" value={agent.kills} />
              <StatTile
                label="Deaths"
                value={agent.deaths}
                valueClass="text-kivski-hp-low"
              />
              <StatTile label="Assists" value={agent.assists} />
            </div>
          </section>

          {/* Recent comms received */}
          <section>
            <SectionTitle>
              Recent messages received{" "}
              <span className="normal-case text-kivski-muted">
                · {receivedMessages.length} of last 3
              </span>
            </SectionTitle>
            {receivedMessages.length === 0 ? (
              <div className="rounded border border-dashed border-kivski-border/60 px-3 py-4 text-center text-[11px] text-kivski-muted">
                No incoming messages yet.
              </div>
            ) : (
              <ul className="rounded border border-kivski-border bg-kivski-panel-2 px-3 py-1">
                {receivedMessages.map((m) => (
                  <MessageRow key={m.id} m={m} nameOf={nameOf} />
                ))}
              </ul>
            )}
          </section>
        </div>
      </div>
    </div>
  );
};

export default AgentDetailModal;
