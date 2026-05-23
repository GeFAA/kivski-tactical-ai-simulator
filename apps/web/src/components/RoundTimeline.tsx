import { useState } from "react";
import { useStore } from "@/lib/store";
import { outcomeStyle } from "@/lib/event-icons";

/**
 * Horizontal strip below the map, above BottomControls. One coloured
 * box per finished round (yellow = attackers won, blue = defenders,
 * grey "X" = draw). Hover for a tooltip; clicking visually highlights
 * the round (V1: no time-travel yet).
 */

const RoundTimeline = () => {
  const results = useStore((s) => s.roundResults);
  const currentRound = useStore((s) => s.round);
  const [highlight, setHighlight] = useState<number | null>(null);

  if (results.length === 0) {
    return (
      <div className="panel flex items-center gap-2 px-3 py-1.5 text-[10px]">
        <span className="uppercase tracking-widest text-kivski-muted">Rounds</span>
        <span className="stat text-kivski-muted">no completed rounds yet</span>
      </div>
    );
  }

  return (
    <div className="panel flex items-center gap-2 px-3 py-1.5">
      <span className="text-[10px] uppercase tracking-widest text-kivski-muted">
        Rounds
      </span>
      <div className="flex flex-1 gap-0.5 overflow-x-auto">
        {results.map((r) => {
          const s = outcomeStyle(r.outcome);
          const isCurrent = highlight === r.round;
          return (
            <button
              key={`${r.round}-${r.outcome}`}
              type="button"
              onClick={() => setHighlight((h) => (h === r.round ? null : r.round))}
              title={`R${r.round} · ${s.label} (${r.winner})`}
              className={`group relative flex h-5 min-w-[1.25rem] items-center justify-center rounded-sm border text-[9px] font-bold leading-none transition-transform ${
                isCurrent ? "scale-110 ring-1 ring-white/50" : ""
              }`}
              style={{
                background: `${s.css}33`,
                color: s.css,
                borderColor: `${s.css}88`,
              }}
            >
              {r.winner === "draw" ? "X" : r.round}
            </button>
          );
        })}
      </div>
      <span className="stat shrink-0 text-[10px] text-kivski-muted">
        live: <span className="text-kivski-text">R{currentRound || "—"}</span>
      </span>
    </div>
  );
};

export default RoundTimeline;
