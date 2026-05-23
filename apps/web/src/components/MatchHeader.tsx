import { useStore } from "@/lib/store";
import type { PolicyAssignment } from "@/lib/store";

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

/**
 * Round display. The backend uses zero-based `round_id` (0, 1, 2 ...). For
 * human readers we show the 1-based round number ("Round 1", "Round 2",
 * ...) — and during warmup / match over we replace the number with a
 * descriptive label so the user is never staring at a meaningless "-" or
 * a wrong-by-one figure. Total rounds are not yet shipped over the wire,
 * so we render the bare ordinal for now.
 */
const formatRound = (round: number, phase: string): string => {
  if (phase === "warmup") return "Warmup";
  if (phase === "match_over") return "Final";
  const n = Math.max(0, Math.floor(round)) + 1;
  return String(n);
};

/**
 * Categorize a policy by its id for the checkpoint indicator badge.
 *
 * Recognised prefixes / aliases:
 *   - "random"           → random baseline (gray)
 *   - "scripted*"        → scripted baseline (blue, brain emoji)
 *   - "checkpoint:..."   → trained policy (yellow)
 *   - "latest" / "best"  → trained policy (yellow)
 *   - anything else      → trained (assume neural)
 *
 * Returns a tuple of [icon, label, className].
 */
type PolicyKind = "trained" | "random" | "scripted" | "unknown";

const classifyPolicy = (p: PolicyAssignment | null): PolicyKind => {
  if (!p) return "unknown";
  const id = p.id.toLowerCase();
  if (id === "random") return "random";
  if (id.startsWith("scripted")) return "scripted";
  if (id.startsWith("checkpoint:") || id === "latest" || id === "best") {
    return "trained";
  }
  // Default: anything else (path, hash) is assumed trained.
  return "trained";
};

const PolicyBadge = ({
  policy,
  teamLabel,
  teamColor,
}: {
  policy: PolicyAssignment | null;
  teamLabel: string;
  teamColor: string;
}) => {
  const kind = classifyPolicy(policy);
  const tooltip = policy?.name ?? policy?.id ?? "policy unknown";

  let icon = "?";
  let kindLabel = "Unknown";
  let kindClass = "bg-kivski-panel-2 text-kivski-muted";
  if (kind === "trained") {
    icon = "[N]";
    kindLabel = "Trained";
    kindClass = "bg-kivski-attacker/15 text-kivski-attacker";
  } else if (kind === "random") {
    icon = "[R]";
    kindLabel = "Random";
    kindClass = "bg-kivski-panel-2 text-kivski-muted";
  } else if (kind === "scripted") {
    icon = "[S]";
    kindLabel = "Scripted";
    kindClass = "bg-kivski-defender/15 text-kivski-defender";
  }

  return (
    <span
      className="inline-flex items-center gap-1 text-[10px]"
      title={`${teamLabel}: ${tooltip} (${kindLabel})`}
    >
      <span
        className="inline-block h-1.5 w-1.5 rounded-full"
        style={{ background: teamColor }}
      />
      <span className="text-kivski-muted">{teamLabel}:</span>
      <span className={`stat rounded px-1 py-px text-[9px] ${kindClass}`}>
        {icon}
      </span>
      <span className="stat max-w-[10rem] truncate text-kivski-text">
        {policy?.name ?? "auto"}
      </span>
    </span>
  );
};

/**
 * Live winrate display vs the random and scripted baselines, pulled from
 * the most-recent `metrics_sample` WS frame. Only shown when training is
 * actively running and at least one sample has been received.
 *
 * Each metric is rendered with a delta arrow indicating change vs the
 * second-most-recent value:
 *   ▲ improvement, ▼ regression, · no change / first sample.
 */
const WinrateStrip = () => {
  const trainingRunning = useStore((s) => s.trainingStatus.running);
  const history = useStore((s) => s.metricsHistory);

  if (!trainingRunning || history.length === 0) return null;

  const latest = history[history.length - 1];
  const prev = history.length > 1 ? history[history.length - 2] : null;

  // Helper: render one metric with a delta arrow.
  const renderOne = (
    label: string,
    valueFn: (m: typeof latest) => number | undefined,
  ): JSX.Element | null => {
    const v = valueFn(latest);
    if (typeof v !== "number") return null;
    const pv = prev ? valueFn(prev) : undefined;
    let arrow = "·";
    let arrowColor = "text-kivski-muted";
    if (typeof pv === "number") {
      const d = v - pv;
      if (d > 1e-4) {
        arrow = "↑";
        arrowColor = "text-kivski-hp";
      } else if (d < -1e-4) {
        arrow = "↓";
        arrowColor = "text-kivski-hp-low";
      }
    }
    return (
      <span
        className="inline-flex items-center gap-1 text-[10px]"
        title={`${label}: ${v.toFixed(3)}${
          typeof pv === "number" ? ` (was ${pv.toFixed(3)})` : ""
        }`}
      >
        <span className="text-kivski-muted">{label}</span>
        <span className="stat text-kivski-text">{v.toFixed(2)}</span>
        <span className={`stat ${arrowColor}`}>{arrow}</span>
      </span>
    );
  };

  return (
    <div className="flex items-center gap-3">
      {renderOne("WR vs Random", (m) => m.winrateVsRandom)}
      {renderOne("WR vs Scripted", (m) => m.winrateVsScripted)}
    </div>
  );
};

const MatchHeader = () => {
  const round = useStore((s) => s.round);
  const phase = useStore((s) => s.phase);
  const secondsLeft = useStore((s) => s.secondsLeft);
  const score = useStore((s) => s.score);
  const mapName = useStore((s) => s.mapName);
  const connected = useStore((s) => s.connected);
  const tick = useStore((s) => s.tick);
  const policies = useStore((s) => s.currentPolicies);

  return (
    <header className="flex flex-col border-b border-kivski-border bg-kivski-panel">
      {/* Main row */}
      <div className="flex h-14 items-center gap-4 px-4">
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
            <div className="stat text-lg font-semibold">{formatRound(round, phase)}</div>
          </div>

          <div className="text-center">
            <div className="text-[10px] uppercase tracking-widest text-kivski-muted">Timer</div>
            <div className="stat text-2xl font-semibold tabular-nums text-white">
              {phase === "match_over" ? "--:--" : formatMmSs(secondsLeft)}
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

        {/* Right: connection + map + tick + winrate */}
        <div className="flex items-center gap-3 text-xs text-kivski-muted">
          <WinrateStrip />
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
      </div>

      {/* Policy strip (always rendered so the header height stays stable). */}
      <div className="flex items-center gap-4 border-t border-kivski-border/60 bg-kivski-bg/40 px-4 py-1">
        <PolicyBadge
          policy={policies.yellow}
          teamLabel="Yellow"
          teamColor="#FFC833"
        />
        <PolicyBadge
          policy={policies.blue}
          teamLabel="Blue"
          teamColor="#4DA8FF"
        />
      </div>
    </header>
  );
};

export default MatchHeader;
