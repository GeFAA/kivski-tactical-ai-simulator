import type { AgentSnapshot, Side } from "@/lib/types";
import { selectAttackers, selectDefenders, useStore } from "@/lib/store";
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

const TeamBlock = ({
  side,
  label,
  players,
}: {
  side: Side;
  label: string;
  players: AgentSnapshot[];
}) => {
  const aliveCount = players.filter((p) => p.isAlive).length;
  const accent = side === "attacker" ? "text-kivski-attacker" : "text-kivski-defender";
  return (
    <section className="panel flex min-h-0 flex-1 flex-col">
      <header className="panel-header">
        <div className="flex items-center gap-2">
          <span className={`h-2.5 w-2.5 rounded-full ${side === "attacker" ? "bg-kivski-attacker" : "bg-kivski-defender"}`} />
          <span className={accent}>{label}</span>
        </div>
        <span className="stat normal-case text-kivski-muted">
          {aliveCount}/{players.length} alive
        </span>
      </header>
      <EconomyMiniBar side={side} />
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
  const attackers = useStore(selectAttackers);
  const defenders = useStore(selectDefenders);

  return (
    <aside className="flex min-h-0 flex-col gap-2">
      <TeamBlock side="attacker" label="Attackers" players={attackers} />
      <TeamBlock side="defender" label="Defenders" players={defenders} />
    </aside>
  );
};

export default LeftSidebar;
