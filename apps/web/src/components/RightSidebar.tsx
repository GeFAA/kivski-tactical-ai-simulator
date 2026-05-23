import { useStore } from "@/lib/store";
import type { RightTab } from "@/lib/store";
import type { EventItem } from "@/lib/types";
import CommsTab from "@/components/CommsTab";
import InspectorTab from "@/components/InspectorTab";
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
      <div className="flex h-full flex-col items-center justify-center gap-2 px-4 text-center text-xs text-kivski-muted">
        <div className="text-2xl opacity-50">·</div>
        <div className="text-kivski-text">No combat events yet</div>
        <div className="text-kivski-muted">
          Random matches produce few kills early on. Events appear here as
          agents shoot, plant, and die.
        </div>
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
        {tab === "inspector" && <InspectorTab />}
        {tab === "comms" && <CommsTab />}
        {tab === "metrics" && <MetricsPanel />}
        {tab === "sys" && <SystemInfo />}
      </div>
    </aside>
  );
};

export default RightSidebar;
