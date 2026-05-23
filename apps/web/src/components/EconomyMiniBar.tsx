import { useMemo } from "react";
import { useStore, selectTeamEconomy } from "@/lib/store";
import type { Side } from "@/lib/types";

/**
 * Small in-sidebar widget that summarises a team's economy:
 *  - Total cash on the books
 *  - Last-round delta (income from the most recent round_end event)
 *  - Up/down arrow indicating whether this team is gaining or losing
 *    cash relative to its own previous snapshot.
 */

const TREND_LOOKBACK = 30; // ticks

const EconomyMiniBar = ({ side }: { side: Side }) => {
  const econ = useStore((s) => selectTeamEconomy(s, side));
  const history = useStore((s) => s.economyHistory);

  const trend = useMemo(() => {
    if (history.length < 2) return 0;
    const latest = history[history.length - 1];
    // Find the sample roughly TREND_LOOKBACK ticks before the latest.
    const target = latest.tick - TREND_LOOKBACK;
    let baseline = history[0];
    for (let i = history.length - 1; i >= 0; i--) {
      if (history[i].tick <= target) {
        baseline = history[i];
        break;
      }
    }
    const cur = side === "attacker" ? latest.attackerTotal : latest.defenderTotal;
    const base = side === "attacker" ? baseline.attackerTotal : baseline.defenderTotal;
    return cur - base;
  }, [history, side]);

  const accent = side === "attacker" ? "text-kivski-attacker" : "text-kivski-defender";
  const bar = side === "attacker" ? "bg-kivski-attacker/70" : "bg-kivski-defender/70";
  // Cap the bar at $32k (typical buy-round budget for the whole team).
  const fill = Math.min(1, econ.total / 32000);

  const trendArrow = trend > 0 ? "▲" : trend < 0 ? "▼" : "•";
  const trendColor =
    trend > 0
      ? "text-kivski-hp"
      : trend < 0
        ? "text-kivski-hp-low"
        : "text-kivski-muted";

  return (
    <div className="flex items-center gap-2 px-2 pb-1.5 text-[10px]">
      <span className={`stat shrink-0 font-semibold ${accent}`}>
        ${econ.total.toLocaleString()}
      </span>
      <div className="relative h-1.5 flex-1 overflow-hidden rounded bg-kivski-bg">
        <div
          className={`absolute inset-y-0 left-0 ${bar} transition-all`}
          style={{ width: `${Math.round(fill * 100)}%` }}
        />
      </div>
      <span className={`stat shrink-0 ${trendColor}`} title="last 30-tick delta">
        {trendArrow} {Math.abs(trend).toLocaleString()}
      </span>
    </div>
  );
};

export default EconomyMiniBar;
