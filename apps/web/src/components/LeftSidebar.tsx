import type { AgentSnapshot, Side, Team } from "@/lib/types";
import {
  selectBlueTeam,
  selectYellowTeam,
  teamCurrentSide,
  useStore,
} from "@/lib/store";
import EconomyMiniBar from "@/components/EconomyMiniBar";

const weaponShort = (a: AgentSnapshot): string => {
  const w = a.weapons[a.activeWeaponIdx];
  if (!w) return "—";
  switch (w.kind) {
    case "knife":
      return "Knife";
    case "pistol":
      return "Pistol";
    case "smg":
      return "SMG";
    case "rifle":
    case "ar":
      return "Rifle";
    case "sniper":
      return "AWP";
    case "shotgun":
      return "Sht";
    case "lmg":
      return "LMG";
    case "grenade":
      return "HE";
    case "flash":
      return "Flash";
    case "smoke":
      return "Smoke";
    case "molotov":
      return "Molly";
    case "c4":
      return "C4";
    default:
      return "?";
  }
};

const PlayerRow = ({ a }: { a: AgentSnapshot }) => {
  const selectedId = useStore((s) => s.selectedAgentId);
  const selectAgent = useStore((s) => s.selectAgent);
  const isSelected = selectedId === a.id;
  const isAttacker = a.side === "attacker";

  const hpPct = Math.max(0, Math.min(100, a.hp));
  const hpBarColor = hpPct < 33 ? "bg-kivski-hp-low" : "bg-kivski-hp";
  const accent = isAttacker
    ? "border-l-kivski-attacker"
    : "border-l-kivski-defender";

  return (
    <button
      type="button"
      onClick={() => selectAgent(isSelected ? null : a.id)}
      className={`group w-full rounded-sm border border-kivski-border border-l-2 ${accent} bg-kivski-panel-2 px-2 py-1.5 text-left transition-colors ${
        isSelected ? "ring-1 ring-inset ring-kivski-defender/60 bg-[#1d2738]" : "hover:bg-[#1c2535]"
      } ${a.isAlive ? "" : "opacity-50"}`}
    >
      <div className="mb-1 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              isAttacker ? "bg-kivski-attacker" : "bg-kivski-defender"
            }`}
          />
          <span className="truncate text-xs font-medium">{a.name}</span>
          {a.hasBomb && (
            <span className="pill bg-kivski-bomb/20 text-kivski-bomb">C4</span>
          )}
          {a.isTalking && (
            <span className="pill bg-kivski-defender/20 text-kivski-defender">say</span>
          )}
        </div>
        <span className="stat text-[10px] text-kivski-muted">
          {a.kills}/{a.deaths}/{a.assists}
        </span>
      </div>

      {/* HP bar */}
      <div className="mb-1 h-1.5 w-full overflow-hidden rounded-sm bg-[#0f131a]">
        <div
          className={`h-full ${hpBarColor} transition-all`}
          style={{ width: `${hpPct}%` }}
        />
      </div>

      <div className="flex items-center justify-between text-[10px] text-kivski-muted">
        <span className="stat">
          <span className="text-kivski-text">{a.hp}</span> hp
          {a.armor > 0 && <span className="ml-1 text-kivski-armor">+{a.armor}</span>}
        </span>
        <span className="stat text-kivski-money">${a.money}</span>
        <span className="stat">{weaponShort(a)}</span>
      </div>
    </button>
  );
};

/**
 * Sidebar grouping is by **team** (yellow / blue) — that identity is
 * stable across the side-switch round, so the panels don't reshuffle
 * visually mid-match. The subheader shows the current side role
 * ("playing as attackers" / "playing as defenders") which DOES flip at
 * the switch, plus the alive count. The bullet/accent colour follows the
 * current side so the colour-coding inside the map viewer (yellow dots
 * for attackers, blue dots for defenders) stays in sync.
 */
const TEAM_LABEL: Record<Team, string> = {
  yellow: "Yellow Team",
  blue: "Blue Team",
};

const TEAM_COLOR_CLASS: Record<Team, string> = {
  // Yellow team is rendered in the attacker accent at the start of the
  // match; we keep the brand colour stable so the header is recognisable.
  yellow: "text-kivski-attacker",
  blue: "text-kivski-defender",
};

const TEAM_DOT_CLASS: Record<Team, string> = {
  yellow: "bg-kivski-attacker",
  blue: "bg-kivski-defender",
};

const sideRoleLabel = (side: Side | null): string => {
  if (side === "attacker") return "playing as attackers";
  if (side === "defender") return "playing as defenders";
  return "side pending";
};

const TeamBlock = ({
  team,
  players,
}: {
  team: Team;
  players: AgentSnapshot[];
}) => {
  const aliveCount = players.filter((p) => p.isAlive).length;
  const currentSide = teamCurrentSide(players);
  // The economy mini-bar is keyed by side (attacker / defender), so we
  // hand it the team's current role. The bar will swap whenever the
  // side flips, but the panel header stays put.
  const econSide: Side = currentSide ?? "attacker";

  return (
    <section className="panel flex min-h-0 flex-1 flex-col">
      <header className="panel-header flex-col items-start gap-0.5">
        <div className="flex w-full items-center justify-between">
          <div className="flex items-center gap-2">
            <span className={`h-2.5 w-2.5 rounded-full ${TEAM_DOT_CLASS[team]}`} />
            <span className={TEAM_COLOR_CLASS[team]}>{TEAM_LABEL[team]}</span>
          </div>
          <span className="stat normal-case text-kivski-muted">
            {aliveCount}/{players.length} alive
          </span>
        </div>
        <span className="text-[10px] uppercase tracking-wider text-kivski-muted">
          {sideRoleLabel(currentSide)}
        </span>
      </header>
      <EconomyMiniBar side={econSide} />
      <div className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto p-2">
        {players.length === 0 ? (
          <div className="px-1 py-6 text-center text-xs text-kivski-muted">No agents yet.</div>
        ) : (
          players.map((p) => <PlayerRow key={p.id} a={p} />)
        )}
      </div>
    </section>
  );
};

const LeftSidebar = () => {
  const yellow = useStore(selectYellowTeam);
  const blue = useStore(selectBlueTeam);

  return (
    <aside className="flex min-h-0 flex-col gap-2">
      <TeamBlock team="yellow" players={yellow} />
      <TeamBlock team="blue" players={blue} />
    </aside>
  );
};

export default LeftSidebar;
