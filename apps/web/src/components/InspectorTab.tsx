/**
 * Right-sidebar Inspector tab. Renders an at-a-glance dossier for the
 * currently-selected agent: vitals, position, equipment, K/D, and the
 * most recent comm messages they received.
 *
 * When no agent is selected we render a hint instead of an empty panel
 * so the tab never looks broken on a fresh page-load.
 */

import { useMemo } from "react";
import {
  resolveAgentName,
  selectSelectedAgent,
  selectSelectedInspection,
  useStore,
} from "@/lib/store";
import { commActionStyle, OBSERVATION_SCHEMA } from "@/lib/event-icons";
import type { AgentSnapshot, MessageItem } from "@/lib/types";

// ---------- Tiny shared bits ----------

const StatRow = ({
  label,
  children,
  hint,
}: {
  label: string;
  children: React.ReactNode;
  hint?: string;
}) => (
  <div className="flex items-baseline justify-between gap-2 text-[11px]">
    <span className="text-kivski-muted">{label}</span>
    <span className="stat text-kivski-text" title={hint}>
      {children}
    </span>
  </div>
);

const ProgressBar = ({
  pct,
  colorClass,
}: {
  pct: number;
  colorClass: string;
}) => {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div className="relative h-1.5 w-full overflow-hidden rounded bg-[#0f131a]">
      <div
        className={`absolute inset-y-0 left-0 ${colorClass} transition-all`}
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
};

const Section = ({
  title,
  children,
  hint,
}: {
  title: string;
  children: React.ReactNode;
  hint?: string;
}) => (
  <section className="panel p-2">
    <div className="mb-1.5 flex items-baseline justify-between">
      <span className="text-[10px] uppercase tracking-widest text-kivski-muted">
        {title}
      </span>
      {hint && <span className="stat text-[10px] text-kivski-muted">{hint}</span>}
    </div>
    {children}
  </section>
);

// ---------- Helpers ----------

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

const formatFacing = (radians: number): string => {
  // Backend yaw 0 = +x axis, CCW positive; normalise to 0..360°.
  const deg = (radians * 180) / Math.PI;
  const normalized = ((deg % 360) + 360) % 360;
  return `${normalized.toFixed(0)}°`;
};

const formatTimeAgo = (ts: number): string => {
  const sec = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (sec < 60) return `${sec}s ago`;
  return `${Math.floor(sec / 60)}m ago`;
};

// ---------- Message preview (received only) ----------

const ReceivedMessageRow = ({
  m,
  nameOf,
}: {
  m: MessageItem;
  nameOf: (id: string) => string;
}) => {
  const style = commActionStyle(m.action);
  return (
    <li className="flex items-start gap-2 border-b border-kivski-border/60 px-1 py-1 text-[11px] last:border-b-0">
      <span
        className="pill shrink-0"
        style={{ background: `${style.css}22`, color: style.css }}
      >
        <span className="mr-1 font-bold">{style.glyph}</span>
        {m.actionLabel ?? style.label}
      </span>
      <div className="min-w-0 flex-1 leading-tight">
        <div className="truncate text-kivski-text">from {nameOf(m.fromId)}</div>
        <div className="stat text-[10px] text-kivski-muted">
          tick {m.tick} · {formatTimeAgo(m.ts)}
        </div>
      </div>
    </li>
  );
};

// ---------- Observation feature groups (compact stats) ----------

const FeatureGroupBars = ({ groups }: { groups: Record<string, number[]> }) => (
  <ul className="space-y-1">
    {OBSERVATION_SCHEMA.map((g) => {
      const v = groups[g.id];
      if (!v || v.length === 0) {
        return (
          <li
            key={g.id}
            className="flex items-center gap-2 text-[10px] opacity-50"
          >
            <span className="w-24 truncate text-kivski-text">{g.label}</span>
            <span className="text-kivski-muted">empty</span>
          </li>
        );
      }
      const max = Math.max(1, ...v.map((x) => Math.abs(x)));
      return (
        <li key={g.id} className="text-[10px]">
          <div className="mb-0.5 flex items-baseline justify-between">
            <span className="text-kivski-text">{g.label}</span>
            <span className="stat text-kivski-muted">{v.length} dims</span>
          </div>
          <div className="flex h-3 items-end gap-px">
            {v.slice(0, 24).map((x, i) => {
              const h = Math.max(6, Math.round((Math.abs(x) / max) * 100));
              return (
                <div
                  key={i}
                  style={{ height: `${h}%`, background: g.color, opacity: 0.7 }}
                  className="w-1"
                  title={x.toFixed(3)}
                />
              );
            })}
          </div>
        </li>
      );
    })}
  </ul>
);

// ---------- Container ----------

const InspectorTab = () => {
  const agent = useStore(selectSelectedAgent);
  const inspection = useStore(selectSelectedInspection);
  const allMessages = useStore((s) => s.recentMessages);
  const agents = useStore((s) => s.agents);
  const customAgentNames = useStore((s) => s.customAgentNames);

  const receivedMessages = useMemo(() => {
    if (!agent) return [];
    return allMessages.filter((m) => m.toIds.includes(agent.id)).slice(0, 6);
  }, [agent, allMessages]);

  /** Translate raw agent ids to friendly "Y-3" / "B-7" (or custom) names. */
  const nameOf = useMemo(() => {
    const lookup = new Map<string, string>();
    for (const a of agents) {
      lookup.set(a.id, resolveAgentName(a, customAgentNames));
    }
    return (id: string): string => lookup.get(id) ?? id;
  }, [agents, customAgentNames]);

  const displayName = agent
    ? resolveAgentName(agent, customAgentNames)
    : "";

  if (!agent) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-4 text-center text-xs text-kivski-muted">
        <div className="text-2xl opacity-50">·</div>
        <div className="text-kivski-text">Click an agent dot on the map</div>
        <div className="text-kivski-muted">
          (or use the team sidebar on the left) to see their HP, weapon,
          position, and recent comms.
        </div>
      </div>
    );
  }

  const isAttacker = agent.side === "attacker";
  const teamColor = agent.team === "yellow" ? "text-kivski-attacker" : "text-kivski-defender";
  const sideBadgeClass = isAttacker
    ? "bg-kivski-attacker/20 text-kivski-attacker"
    : "bg-kivski-defender/20 text-kivski-defender";

  const hpPct = Math.max(0, Math.min(100, agent.hp));
  const hpColor = hpPct < 33 ? "bg-kivski-hp-low" : "bg-kivski-hp";
  const armorPct = Math.max(0, Math.min(100, agent.armor));

  const primary = agent.weapons[agent.activeWeaponIdx] ?? agent.weapons[0];
  const weaponLabel = primary ? weaponKindToLabel[primary.kind] : "—";
  const weaponGlyph = primary ? weaponKindToGlyph[primary.kind] ?? "·" : "·";

  return (
    <div className="flex flex-col gap-2 p-2 text-xs">
      {/* Header */}
      <section className="panel p-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 min-w-0">
            <span
              className={`inline-block h-2.5 w-2.5 rounded-full ${
                isAttacker ? "bg-kivski-attacker" : "bg-kivski-defender"
              }`}
            />
            <span className={`truncate font-semibold ${teamColor}`}>
              {displayName}
            </span>
            {!agent.isAlive && (
              <span className="pill bg-kivski-hp-low/15 text-kivski-hp-low">
                dead
              </span>
            )}
          </div>
          <span className={`pill ${sideBadgeClass}`}>{agent.side}</span>
        </div>
        <div className="mt-1 text-[10px] uppercase tracking-wider text-kivski-muted">
          {agent.id} · {agent.team} team
        </div>
      </section>

      {/* Vitals */}
      <Section title="Vitals">
        <div className="flex flex-col gap-1.5">
          <div>
            <StatRow label="HP" hint="hit points 0-100">
              <span className="text-kivski-text">{agent.hp}</span>
              <span className="text-kivski-muted">/100</span>
            </StatRow>
            <div className="mt-0.5">
              <ProgressBar pct={hpPct} colorClass={hpColor} />
            </div>
          </div>
          <div>
            <StatRow label="Armor">
              <span className="text-kivski-armor">{agent.armor}</span>
              <span className="text-kivski-muted">/100</span>
            </StatRow>
            <div className="mt-0.5">
              <ProgressBar pct={armorPct} colorClass="bg-kivski-armor" />
            </div>
          </div>
          <StatRow label="Money">
            <span className="text-kivski-money">${agent.money}</span>
          </StatRow>
        </div>
      </Section>

      {/* Position */}
      <Section title="Position">
        <div className="flex flex-col gap-0.5">
          <StatRow label="Tile">
            ({agent.pos.x.toFixed(1)}, {agent.pos.y.toFixed(1)})
          </StatRow>
          <StatRow label="Facing">{formatFacing(agent.facing)}</StatRow>
          {(agent.isPlanting || agent.isDefusing) && (
            <StatRow label="State">
              <span className="text-kivski-bomb">
                {agent.isPlanting ? "Planting bomb" : "Defusing bomb"}
              </span>
            </StatRow>
          )}
        </div>
      </Section>

      {/* Equipment */}
      <Section title="Equipment">
        <div className="flex flex-col gap-1">
          <StatRow label="Weapon">
            <span className="text-kivski-text">
              {weaponGlyph} {weaponLabel}
            </span>
          </StatRow>
          {agent.hasBomb && (
            <span className="pill bg-kivski-bomb/15 text-kivski-bomb">
              Carrying Bomb
            </span>
          )}
          {agent.hasDefuseKit && (
            <span className="pill bg-kivski-defender/15 text-kivski-defender">
              Defuse Kit
            </span>
          )}
          {!agent.hasBomb && !agent.hasDefuseKit && (
            <div className="text-[10px] text-kivski-muted">no special items</div>
          )}
        </div>
      </Section>

      {/* Round stats */}
      <Section title="Round stats">
        <div className="grid grid-cols-3 gap-2">
          <div className="rounded bg-kivski-bg p-1.5 text-center">
            <div className="stat text-sm text-kivski-text">{agent.kills}</div>
            <div className="text-[9px] uppercase text-kivski-muted">Kills</div>
          </div>
          <div className="rounded bg-kivski-bg p-1.5 text-center">
            <div className="stat text-sm text-kivski-hp-low">{agent.deaths}</div>
            <div className="text-[9px] uppercase text-kivski-muted">Deaths</div>
          </div>
          <div className="rounded bg-kivski-bg p-1.5 text-center">
            <div className="stat text-sm text-kivski-text">{agent.assists}</div>
            <div className="text-[9px] uppercase text-kivski-muted">Assists</div>
          </div>
        </div>
      </Section>

      {/* Recent comms received */}
      <Section title="Recent comms received" hint={`${receivedMessages.length} msg`}>
        {receivedMessages.length === 0 ? (
          <div className="text-[11px] text-kivski-muted">
            No incoming messages yet.
          </div>
        ) : (
          <ul className="flex flex-col">
            {receivedMessages.map((m) => (
              <ReceivedMessageRow key={m.id} m={m} nameOf={nameOf} />
            ))}
          </ul>
        )}
      </Section>

      {/* Optional inspection blob — only shown when present so the panel
          stays compact for randomly-driven matches without inspection. */}
      {inspection?.observationGroups && (
        <Section title="Observation groups" hint="policy view">
          <FeatureGroupBars groups={inspection.observationGroups} />
        </Section>
      )}
      {typeof inspection?.valueEstimate === "number" && (
        <Section title="Value estimate" hint="critic head">
          <div className="stat text-sm text-kivski-text">
            {inspection.valueEstimate.toFixed(3)}
          </div>
        </Section>
      )}
    </div>
  );
};

export default InspectorTab;
