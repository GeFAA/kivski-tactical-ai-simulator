import { useStore } from "@/lib/store";

const formatMmSs = (seconds: number): string => {
  const s = Math.max(0, Math.floor(seconds));
  const mm = Math.floor(s / 60)
    .toString()
    .padStart(2, "0");
  const ss = (s % 60).toString().padStart(2, "0");
  return `${mm}:${ss}`;
};

const phaseLabel: Record<string, string> = {
  warmup: "Warm-up",
  buy: "Buy Phase",
  live: "Live",
  post_round: "Post-Round",
  halftime: "Half-Time",
  match_over: "Match Over",
};

const MatchHeader = () => {
  const round = useStore((s) => s.round);
  const phase = useStore((s) => s.phase);
  const secondsLeft = useStore((s) => s.secondsLeft);
  const score = useStore((s) => s.score);
  const mapName = useStore((s) => s.mapName);
  const connected = useStore((s) => s.connected);
  const tick = useStore((s) => s.tick);

  return (
    <header className="flex h-14 items-center gap-4 border-b border-kivski-border bg-kivski-panel px-4">
      {/* Left: branding */}
      <div className="flex items-center gap-2">
        <div className="h-6 w-6 rounded bg-gradient-to-br from-kivski-attacker to-kivski-defender" />
        <div className="leading-tight">
          <div className="text-sm font-semibold tracking-wide">Kivski</div>
          <div className="text-[10px] uppercase tracking-widest text-kivski-muted">
            Tactical AI Sim
          </div>
        </div>
      </div>

      <div className="h-8 w-px bg-kivski-border" />

      {/* Center: round + timer + score */}
      <div className="flex flex-1 items-center justify-center gap-6">
        <div className="text-center">
          <div className="text-[10px] uppercase tracking-widest text-kivski-muted">Round</div>
          <div className="stat text-lg font-semibold">{round || "-"}</div>
        </div>

        <div className="text-center">
          <div className="text-[10px] uppercase tracking-widest text-kivski-muted">Timer</div>
          <div className="stat text-2xl font-semibold tabular-nums text-white">
            {formatMmSs(secondsLeft)}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="stat text-2xl font-bold text-kivski-attacker">
            {score.attacker.toString().padStart(2, "0")}
          </span>
          <span className="text-sm text-kivski-muted">:</span>
          <span className="stat text-2xl font-bold text-kivski-defender">
            {score.defender.toString().padStart(2, "0")}
          </span>
        </div>

        <div className="text-center">
          <div className="text-[10px] uppercase tracking-widest text-kivski-muted">Phase</div>
          <div className="text-sm font-medium">{phaseLabel[phase] ?? phase}</div>
        </div>
      </div>

      {/* Right: connection + map + tick */}
      <div className="flex items-center gap-3 text-xs text-kivski-muted">
        <div>
          map <span className="text-kivski-text">{mapName}</span>
        </div>
        <div>
          tick <span className="stat text-kivski-text">{tick}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              connected ? "bg-kivski-hp animate-pulse-slow" : "bg-kivski-hp-low"
            }`}
          />
          <span>{connected ? "live" : "offline"}</span>
        </div>
      </div>
    </header>
  );
};

export default MatchHeader;
