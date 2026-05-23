import { useMemo } from "react";
import { useStore, selectSelectedAgent } from "@/lib/store";
import { commActionStyle } from "@/lib/event-icons";
import type { CommAction, MessageItem } from "@/lib/types";

/**
 * Right-sidebar Comms tab. Shows the global stream of inter-agent comm
 * messages as a single timeline ("newest on top"), with a header that
 * summarises how many of each comm-action have been seen recently.
 *
 * If an agent is selected we render a "Focus" filter chip at the top so
 * the user can flip between "all comms" and "only what this agent
 * sent/received" — without that flip we'd hide the global feed any
 * time someone clicks a dot on the map, which is more annoying than
 * helpful.
 */

// ---------- Payload preview ----------

const PayloadBars = ({ values }: { values: number[] }) => {
  if (!values || values.length === 0) return null;
  // Normalize to [-1, 1] range for visual scale.
  const max = Math.max(1, ...values.map((v) => Math.abs(v)));
  return (
    <div className="flex h-5 items-end gap-px" title="payload vector">
      {values.slice(0, 16).map((v, i) => {
        const h = Math.max(2, Math.round((Math.abs(v) / max) * 100));
        const positive = v >= 0;
        return (
          <div
            key={i}
            className={`w-1 ${positive ? "bg-kivski-defender" : "bg-kivski-hp-low"}`}
            style={{
              height: `${h}%`,
              opacity: 0.65 + (Math.abs(v) / max) * 0.35,
            }}
            title={v.toFixed(3)}
          />
        );
      })}
    </div>
  );
};

// ---------- Header summary ----------

const ACTION_ORDER: CommAction[] = [
  "PING_LOCATION",
  "WARN_DANGER",
  "REQUEST_SUPPORT",
  "SUGGEST_ROTATE",
  "SUGGEST_ATTACK",
  "SUGGEST_FALLBACK",
  "CONTACT_ENEMY",
  "BOMBSITE_CLEAR",
  "ACK",
  "SILENT",
];

const ActionCountChips = ({ messages }: { messages: MessageItem[] }) => {
  const counts = useMemo(() => {
    const c: Partial<Record<CommAction, number>> = {};
    for (const m of messages) {
      const a = m.action ?? "SILENT";
      c[a] = (c[a] ?? 0) + 1;
    }
    return c;
  }, [messages]);

  const visible = ACTION_ORDER.filter((a) => (counts[a] ?? 0) > 0);
  if (visible.length === 0) {
    return (
      <span className="stat text-[10px] text-kivski-muted">no comms yet</span>
    );
  }

  return (
    <div className="flex flex-wrap gap-1">
      {visible.map((a) => {
        const style = commActionStyle(a);
        return (
          <span
            key={a}
            className="pill"
            style={{ background: `${style.css}22`, color: style.css }}
            title={style.label}
          >
            <span className="mr-1 font-bold">{style.glyph}</span>
            {counts[a]}
          </span>
        );
      })}
    </div>
  );
};

// ---------- One row in the feed ----------

const formatTimeAgo = (ts: number): string => {
  const sec = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (sec < 60) return `${sec}s ago`;
  return `${Math.floor(sec / 60)}m ago`;
};

const MessageRow = ({
  m,
  nameOf,
}: {
  m: MessageItem;
  nameOf: (id: string) => string;
}) => {
  const style = commActionStyle(m.action);
  const receiverNames = m.toIds.map(nameOf);
  const receiverPreview = receiverNames.length === 0 ? "—" : receiverNames.join(", ");
  const trimmedReceivers =
    receiverNames.length > 3
      ? `${receiverNames.slice(0, 3).join(", ")} +${receiverNames.length - 3}`
      : receiverPreview;
  return (
    <li className="flex items-stretch gap-2 border-b border-kivski-border/60 px-2 py-1.5 last:border-b-0">
      <div
        className="w-0.5 shrink-0 rounded"
        style={{ background: style.css, opacity: 0.7 }}
        title={style.label}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 min-w-0">
            <span className="stat text-[11px] font-medium text-kivski-text">
              {nameOf(m.fromId)}
            </span>
            <span
              className="pill shrink-0"
              style={{ background: `${style.css}22`, color: style.css }}
              title={style.label}
            >
              <span className="mr-1 font-bold">{style.glyph}</span>
              {m.actionLabel ?? style.label}
            </span>
          </div>
          <span className="stat shrink-0 text-[10px] text-kivski-muted">
            tick {m.tick}
          </span>
        </div>
        <div className="mt-0.5 truncate text-[11px] text-kivski-muted">
          → {trimmedReceivers}
        </div>
        <div className="mt-0.5 flex items-end justify-between gap-2">
          <div className="min-w-0 flex-1">
            {m.pos && (
              <span className="stat text-[10px] text-kivski-muted">
                @({m.pos.x.toFixed(1)}, {m.pos.y.toFixed(1)})
              </span>
            )}
          </div>
          <div className="shrink-0">
            <PayloadBars values={m.payload ?? []} />
          </div>
          <span className="stat shrink-0 text-[10px] text-kivski-muted">
            {formatTimeAgo(m.ts)}
          </span>
        </div>
      </div>
    </li>
  );
};

// ---------- Container ----------

const CommsTab = () => {
  const allMessages = useStore((s) => s.recentMessages);
  const agents = useStore((s) => s.agents);
  const selected = useStore(selectSelectedAgent);
  const selectAgent = useStore((s) => s.selectAgent);

  const filtered = useMemo(() => {
    if (!selected) return allMessages;
    return allMessages.filter(
      (m) => m.fromId === selected.id || m.toIds.includes(selected.id),
    );
  }, [allMessages, selected]);

  /**
   * Translate a raw `agent_<n>` id into the friendly "Y-3"/"B-7" display
   * name set by the wire decoder. Falls back to the raw id if the agent
   * has been removed from the snapshot (e.g. mid-respawn).
   */
  const nameOf = useMemo(() => {
    const lookup = new Map<string, string>();
    for (const a of agents) lookup.set(a.id, a.name);
    return (id: string): string => lookup.get(id) ?? id;
  }, [agents]);

  return (
    <div className="flex flex-col gap-2 p-2">
      <section className="panel p-2">
        <div className="mb-1.5 flex items-baseline justify-between gap-2">
          <span className="text-[10px] uppercase tracking-widest text-kivski-muted">
            Comm stream
          </span>
          <span className="stat text-[10px] text-kivski-muted">
            {allMessages.length} total · showing {filtered.length}
          </span>
        </div>
        <ActionCountChips messages={allMessages} />
        {selected && (
          <div className="mt-2 flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider text-kivski-muted">
              Focus
            </span>
            <span className="pill bg-kivski-defender/15 text-kivski-defender">
              {selected.name}
            </span>
            <button
              type="button"
              onClick={() => selectAgent(null)}
              className="ml-auto text-[10px] text-kivski-muted underline hover:text-kivski-text"
            >
              clear
            </button>
          </div>
        )}
      </section>

      <section className="panel min-h-0">
        {filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-1 px-4 py-6 text-center text-[11px] text-kivski-muted">
            <div className="text-2xl opacity-50">·</div>
            <div className="text-kivski-text">
              {selected ? `No comms involving ${selected.name} yet.` : "No agent comms yet."}
            </div>
            <div className="text-kivski-muted">
              Random policy is mostly silent. Comms appear when an agent
              fires a CommAction (ping, warn, request, rotate, etc.).
              Training-driven policies emit far more frequent comms.
            </div>
          </div>
        ) : (
          <ul className="flex flex-col">
            {filtered.map((m) => (
              <MessageRow key={m.id} m={m} nameOf={nameOf} />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
};

export default CommsTab;
