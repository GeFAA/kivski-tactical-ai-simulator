import { selectSelectedAgent, selectSelectedInspection, useStore } from "@/lib/store";
import type { RightTab } from "@/lib/store";
import type { EventItem } from "@/lib/types";
import { OBSERVATION_SCHEMA } from "@/lib/event-icons";
import CommsTab from "@/components/CommsTab";
import MetricsPanel from "@/components/MetricsPanel";
import SystemInfo from "@/components/SystemInfo";

const tabs: { id: RightTab; label: string }[] = [
  { id: "events", label: "Events" },
  { id: "inspector", label: "Inspector" },
  { id: "comms", label: "Comms" },
  { id: "metrics", label: "Metrics" },
  { id: "sys", label: "Sys" },
];

// ---------- Event Feed ----------

const eventChipClass = (kind: EventItem["kind"]): string => {
  switch (kind) {
    case "kill":
    case "death":
      return "bg-kivski-hp-low/15 text-kivski-hp-low";
    case "plant":
    case "bomb_explode":
      return "bg-kivski-bomb/15 text-kivski-bomb";
    case "defuse":
      return "bg-kivski-hp/15 text-kivski-hp";
    case "round_start":
    case "round_end":
      return "bg-kivski-defender/15 text-kivski-defender";
    case "purchase":
      return "bg-kivski-money/15 text-kivski-money";
    case "sound":
      return "bg-[#A78BFA]/15 text-[#A78BFA]";
    default:
      return "bg-kivski-panel-2 text-kivski-muted";
  }
};

const EventRow = ({ e }: { e: EventItem }) => (
  <li className="flex items-start gap-2 border-b border-kivski-border/60 px-2 py-1.5 text-xs last:border-b-0">
    <span className={`pill ${eventChipClass(e.kind)}`}>{e.kind.replace("_", " ")}</span>
    <div className="min-w-0 flex-1">
      <div className="truncate text-kivski-text">{e.text}</div>
      <div className="stat text-[10px] text-kivski-muted">tick {e.tick}</div>
    </div>
  </li>
);

const EventFeed = () => {
  const events = useStore((s) => s.eventFeed);
  if (events.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-4 text-center text-xs text-kivski-muted">
        No events yet — waiting for live snapshots.
      </div>
    );
  }
  return (
    <ul className="flex flex-col">
      {events.map((e) => (
        <EventRow key={e.id} e={e} />
      ))}
    </ul>
  );
};

// ---------- Inspector helpers ----------

const Gauge = ({ value, range = 1 }: { value: number; range?: number }) => {
  // Map [-range, +range] → [0, 100] for an offset bar centered at 50.
  const clamped = Math.max(-range, Math.min(range, value));
  const pct = ((clamped + range) / (range * 2)) * 100;
  const positive = clamped >= 0;
  return (
    <div className="relative h-2 w-full overflow-hidden rounded bg-kivski-bg">
      <div className="absolute inset-y-0 left-1/2 w-px bg-kivski-border" />
      <div
        className={`absolute inset-y-0 ${
          positive ? "bg-kivski-hp" : "bg-kivski-hp-low"
        }`}
        style={
          positive
            ? { left: "50%", width: `${pct - 50}%` }
            : { right: "50%", width: `${50 - pct}%` }
        }
      />
    </div>
  );
};

const FeatureGroupBars = ({
  groups,
}: {
  groups: Record<string, number[]>;
}) => {
  return (
    <ul className="space-y-1">
      {OBSERVATION_SCHEMA.map((g) => {
        const v = groups[g.id];
        if (!v || v.length === 0) {
          return (
            <li key={g.id} className="flex items-center gap-2 text-[10px] opacity-50">
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
};

// ---------- Agent Inspector ----------

const AgentInspector = () => {
  const agent = useStore(selectSelectedAgent);
  const inspection = useStore(selectSelectedInspection);
  const events = useStore((s) => s.eventFeed);

  if (!agent) {
    return (
      <div className="flex h-full items-center justify-center px-4 text-center text-xs text-kivski-muted">
        Select an agent on the map or sidebar to inspect.
      </div>
    );
  }

  const attEntries = inspection?.attention?.byAgent
    ? Object.entries(inspection.attention.byAgent).sort((a, b) => b[1] - a[1])
    : [];

  const traceEvents = events
    .filter((e) => e.actorId === agent.id)
    .slice(0, 5);

  const heads = inspection?.lastHeads ?? {};

  return (
    <div className="flex flex-col gap-2 p-2 text-xs">
      <div className="panel p-2">
        <div className="flex items-center justify-between">
          <div className="font-semibold text-kivski-text">{agent.name}</div>
          <div
            className={`pill ${
              agent.side === "attacker"
                ? "bg-kivski-attacker/20 text-kivski-attacker"
                : "bg-kivski-defender/20 text-kivski-defender"
            }`}
          >
            {agent.side}
          </div>
        </div>
        <div className="mt-1 grid grid-cols-2 gap-x-2 gap-y-1 text-[11px] text-kivski-muted">
          <span>
            HP <span className="stat text-kivski-text">{agent.hp}</span>
          </span>
          <span>
            Armor <span className="stat text-kivski-text">{agent.armor}</span>
          </span>
          <span>
            Money <span className="stat text-kivski-money">${agent.money}</span>
          </span>
          <span>
            K/D/A{" "}
            <span className="stat text-kivski-text">
              {agent.kills}/{agent.deaths}/{agent.assists}
            </span>
          </span>
        </div>
      </div>

      {/* Multi-head action */}
      <div className="panel p-2">
        <div className="mb-1 text-[10px] uppercase tracking-widest text-kivski-muted">
          Last action (per head)
        </div>
        <div className="grid grid-cols-2 gap-1 text-[11px]">
          {(["move", "micro", "comm", "buy", "aimTarget"] as const).map((k) => (
            <div key={k} className="flex items-center justify-between rounded bg-kivski-bg px-1.5 py-1">
              <span className="text-kivski-muted">{k}</span>
              <span className="stat text-kivski-text">{heads[k] ?? "—"}</span>
            </div>
          ))}
        </div>
        {!inspection?.lastHeads && inspection?.lastAction && (
          <pre className="mt-1 max-h-24 overflow-auto rounded bg-kivski-bg p-1.5 text-[10px] leading-snug text-kivski-muted">
            {JSON.stringify(inspection.lastAction.params, null, 2)}
          </pre>
        )}
      </div>

      {/* Value estimate */}
      <div className="panel p-2">
        <div className="mb-1 flex items-baseline justify-between">
          <span className="text-[10px] uppercase tracking-widest text-kivski-muted">
            Value estimate
          </span>
          <span className="stat text-kivski-text">
            {inspection?.valueEstimate?.toFixed(3) ?? "—"}
          </span>
        </div>
        <Gauge value={inspection?.valueEstimate ?? 0} range={1} />
      </div>

      {/* Observation feature groups */}
      <div className="panel p-2">
        <div className="mb-1 text-[10px] uppercase tracking-widest text-kivski-muted">
          Observation feature groups
        </div>
        {inspection?.observationGroups ? (
          <FeatureGroupBars groups={inspection.observationGroups} />
        ) : (
          <div className="text-kivski-muted">no group-level observation logged</div>
        )}
      </div>

      {/* Attention */}
      <div className="panel p-2">
        <div className="mb-1 text-[10px] uppercase tracking-widest text-kivski-muted">
          Attention (top sources)
        </div>
        {attEntries.length === 0 ? (
          <div className="text-kivski-muted">no attention data</div>
        ) : (
          <ul className="space-y-1">
            {attEntries.slice(0, 6).map(([id, w]) => (
              <li key={id} className="flex items-center gap-2">
                <span className="w-20 truncate text-kivski-text">{id}</span>
                <div className="relative h-1.5 flex-1 overflow-hidden rounded bg-kivski-bg">
                  <div
                    className="absolute inset-y-0 left-0 bg-kivski-defender"
                    style={{ width: `${Math.round(w * 100)}%` }}
                  />
                </div>
                <span className="stat w-10 text-right text-kivski-muted">{w.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Hidden state stub */}
      <div className="panel p-2">
        <div className="mb-1 text-[10px] uppercase tracking-widest text-kivski-muted">
          Hidden state (preview)
        </div>
        {inspection?.hiddenStatePreview && inspection.hiddenStatePreview.length > 0 ? (
          <div className="flex h-4 items-end gap-px">
            {inspection.hiddenStatePreview.slice(0, 32).map((v, i) => {
              const max = Math.max(
                1,
                ...inspection.hiddenStatePreview!.map((x) => Math.abs(x)),
              );
              const h = Math.max(6, Math.round((Math.abs(v) / max) * 100));
              return (
                <div
                  key={i}
                  className={v >= 0 ? "bg-kivski-defender" : "bg-kivski-hp-low"}
                  style={{ height: `${h}%`, opacity: 0.6 }}
                />
              );
            })}
          </div>
        ) : (
          <div className="text-kivski-muted">no hidden state captured (stub)</div>
        )}
      </div>

      {/* Decision trace */}
      <div className="panel p-2">
        <div className="mb-1 text-[10px] uppercase tracking-widest text-kivski-muted">
          Decision trace (recent)
        </div>
        {traceEvents.length === 0 ? (
          <div className="text-kivski-muted">no actions yet</div>
        ) : (
          <ul className="space-y-0.5 text-[11px]">
            {traceEvents.map((e) => (
              <li key={e.id} className="flex items-start gap-2">
                <span className="stat shrink-0 text-kivski-muted">t{e.tick}</span>
                <span className={`pill shrink-0 ${eventChipClass(e.kind)}`}>{e.kind}</span>
                <span className="truncate text-kivski-text">{e.text}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

// ---------- Container ----------

const RightSidebar = () => {
  const tab = useStore((s) => s.rightTab);
  const setTab = useStore((s) => s.setRightTab);

  return (
    <aside className="panel flex min-h-0 flex-col">
      <div className="flex border-b border-kivski-border">
        {tabs.map((t) => {
          const active = tab === t.id;
          return (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`flex-1 px-2 py-2 text-xs font-medium transition-colors ${
                active
                  ? "border-b-2 border-kivski-defender text-kivski-text"
                  : "text-kivski-muted hover:text-kivski-text"
              }`}
            >
              {t.label}
            </button>
          );
        })}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {tab === "events" && <EventFeed />}
        {tab === "inspector" && <AgentInspector />}
        {tab === "comms" && <CommsTab />}
        {tab === "metrics" && <MetricsPanel />}
        {tab === "sys" && <SystemInfo />}
      </div>
    </aside>
  );
};

export default RightSidebar;
