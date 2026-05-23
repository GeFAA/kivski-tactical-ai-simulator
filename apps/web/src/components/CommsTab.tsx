import { useMemo } from "react";
import { useStore, selectSelectedAgent } from "@/lib/store";
import { commActionStyle } from "@/lib/event-icons";
import type { MessageItem } from "@/lib/types";

/**
 * Right-sidebar Comms tab. Shows per-agent message history (Received,
 * Sent) and a small bar chart of the latest attention weights toward
 * teammates.
 *
 * When no agent is selected we render a hint to nudge the user to pick
 * one — without a focus we'd just be dumping the whole stream which is
 * already in the Event Feed tab.
 */

// ---------- Payload preview ----------

const PayloadBars = ({ values }: { values: number[] }) => {
  if (!values || values.length === 0) {
    return <span className="stat text-[10px] text-kivski-muted">no payload</span>;
  }
  // Normalize to [-1, 1] range for visual scale.
  const max = Math.max(1, ...values.map((v) => Math.abs(v)));
  return (
    <div className="flex h-5 items-end gap-px">
      {values.slice(0, 16).map((v, i) => {
        const h = Math.max(2, Math.round((Math.abs(v) / max) * 100));
        const positive = v >= 0;
        return (
          <div
            key={i}
            className={`w-1 ${positive ? "bg-kivski-defender" : "bg-kivski-hp-low"}`}
            style={{ height: `${h}%`, opacity: 0.65 + (Math.abs(v) / max) * 0.35 }}
            title={v.toFixed(3)}
          />
        );
      })}
    </div>
  );
};

// ---------- Message row ----------

const MessageRow = ({
  m,
  selfId,
  direction,
}: {
  m: MessageItem;
  selfId: string;
  direction: "in" | "out";
}) => {
  const style = commActionStyle(m.action);
  const other = direction === "in" ? m.fromId : m.toIds.filter((id) => id !== selfId).join(", ") || "broadcast";

  return (
    <li className="border-b border-kivski-border/60 px-2 py-1.5 last:border-b-0">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <span
            className="pill"
            style={{ background: `${style.css}22`, color: style.css }}
            title={style.label}
          >
            <span className="mr-1 font-bold">{style.glyph}</span>
            {m.actionLabel ?? style.label}
          </span>
          <span className="truncate text-[11px] text-kivski-text">
            {direction === "in" ? "from " : "to "}
            <span className="font-medium">{other}</span>
          </span>
        </div>
        <span className="stat shrink-0 text-[10px] text-kivski-muted">tick {m.tick}</span>
      </div>
      {m.text && <div className="mt-0.5 text-[11px] text-kivski-muted">{m.text}</div>}
      <div className="mt-1">
        <PayloadBars values={m.payload ?? []} />
      </div>
    </li>
  );
};

// ---------- Attention chart ----------

const AttentionPanel = ({ selectedId }: { selectedId: string }) => {
  const agents = useStore((s) => s.agents);
  const weights = useStore((s) => s.attentionWeights[selectedId]);

  const sortedEntries = useMemo(() => {
    if (!weights) return [];
    return Object.entries(weights)
      .filter(([id]) => id !== selectedId)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
  }, [weights, selectedId]);

  if (sortedEntries.length === 0) {
    return (
      <div className="text-[11px] text-kivski-muted">
        No attention data for this agent yet.
      </div>
    );
  }

  const nameOf = (id: string) => agents.find((a) => a.id === id)?.name ?? id;
  const sideOf = (id: string) => agents.find((a) => a.id === id)?.side;

  return (
    <ul className="space-y-1">
      {sortedEntries.map(([id, w]) => {
        const side = sideOf(id);
        const bar = side === "attacker" ? "bg-kivski-attacker" : "bg-kivski-defender";
        return (
          <li key={id} className="flex items-center gap-2 text-[11px]">
            <span className="w-20 truncate text-kivski-text">{nameOf(id)}</span>
            <div className="relative h-1.5 flex-1 overflow-hidden rounded bg-kivski-bg">
              <div
                className={`absolute inset-y-0 left-0 ${bar}`}
                style={{ width: `${Math.round(Math.min(1, Math.max(0, w)) * 100)}%` }}
              />
            </div>
            <span className="stat w-10 text-right text-kivski-muted">{w.toFixed(2)}</span>
          </li>
        );
      })}
    </ul>
  );
};

// ---------- Container ----------

const CommsTab = () => {
  const agent = useStore(selectSelectedAgent);
  const messages = useStore((s) => s.recentMessages);

  if (!agent) {
    return (
      <div className="flex h-full items-center justify-center px-4 text-center text-xs text-kivski-muted">
        Select an agent on the map or in the sidebar to inspect their comms.
      </div>
    );
  }

  const received = messages.filter((m) => m.toIds.includes(agent.id)).slice(0, 10);
  const sent = messages.filter((m) => m.fromId === agent.id).slice(0, 10);

  return (
    <div className="flex flex-col gap-2 p-2">
      <section className="panel p-2">
        <div className="mb-1 flex items-center justify-between">
          <span className="text-[10px] uppercase tracking-widest text-kivski-muted">
            Received
          </span>
          <span className="stat text-[10px] text-kivski-muted">{received.length}</span>
        </div>
        {received.length === 0 ? (
          <div className="text-[11px] text-kivski-muted">No incoming messages.</div>
        ) : (
          <ul className="flex flex-col">
            {received.map((m) => (
              <MessageRow key={m.id} m={m} selfId={agent.id} direction="in" />
            ))}
          </ul>
        )}
      </section>

      <section className="panel p-2">
        <div className="mb-1 flex items-center justify-between">
          <span className="text-[10px] uppercase tracking-widest text-kivski-muted">
            Sent
          </span>
          <span className="stat text-[10px] text-kivski-muted">{sent.length}</span>
        </div>
        {sent.length === 0 ? (
          <div className="text-[11px] text-kivski-muted">No outgoing messages.</div>
        ) : (
          <ul className="flex flex-col">
            {sent.map((m) => (
              <MessageRow key={m.id} m={m} selfId={agent.id} direction="out" />
            ))}
          </ul>
        )}
      </section>

      <section className="panel p-2">
        <div className="mb-1.5 text-[10px] uppercase tracking-widest text-kivski-muted">
          Attention weights
        </div>
        <AttentionPanel selectedId={agent.id} />
      </section>
    </div>
  );
};

export default CommsTab;
