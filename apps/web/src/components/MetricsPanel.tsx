import { useMemo } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useStore, selectSelectedInspection } from "@/lib/store";
import { outcomeStyle } from "@/lib/event-icons";
import type { RoundOutcome } from "@/lib/types";

/**
 * "Metrics" tab in the right sidebar. Aggregates the four headline
 * charts the user wants while training / evaluating:
 *
 *   1. WinrateChart   - episodic winrate vs. random and scripted baselines
 *   2. EconomyChart   - per-tick total money per side for the current match
 *   3. ActionDistChart- last-100-ticks action distribution for the selected agent
 *   4. RoundOutcomePie- distribution of round outcomes over the last 20 rounds
 *
 * Every chart degrades to a small "no data yet" message so the panel
 * never just shows an empty white block during cold boot.
 */

// ---------- Shared chart shell ----------

const ChartCard = ({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
}) => (
  <section className="panel p-2">
    <div className="mb-1 flex items-baseline justify-between">
      <span className="text-[10px] uppercase tracking-widest text-kivski-muted">
        {title}
      </span>
      {hint && (
        <span className="stat text-[10px] text-kivski-muted">{hint}</span>
      )}
    </div>
    <div className="h-40">{children}</div>
  </section>
);

const NoData = ({ msg }: { msg: string }) => (
  <div className="flex h-full items-center justify-center text-[11px] text-kivski-muted">
    {msg}
  </div>
);

// ---------- 1. Winrate ----------

const WinrateChart = () => {
  const history = useStore((s) => s.metricsHistory);
  const data = useMemo(
    () =>
      history.map((s) => ({
        episode: s.episode,
        vsRandom:
          typeof s.winrateVsRandom === "number"
            ? Number((s.winrateVsRandom * 100).toFixed(1))
            : null,
        vsScripted:
          typeof s.winrateVsScripted === "number"
            ? Number((s.winrateVsScripted * 100).toFixed(1))
            : null,
      })),
    [history],
  );

  if (data.length === 0) {
    return <NoData msg="No metrics samples yet — start training or run eval episodes." />;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid stroke="#222B3A" strokeDasharray="2 2" />
        <XAxis dataKey="episode" stroke="#6B7585" tick={{ fontSize: 10 }} />
        <YAxis
          domain={[0, 100]}
          stroke="#6B7585"
          tick={{ fontSize: 10 }}
          unit="%"
        />
        <Tooltip
          contentStyle={{
            background: "#131821",
            border: "1px solid #222B3A",
            fontSize: 11,
            color: "#E5E7EB",
          }}
        />
        <Legend wrapperStyle={{ fontSize: 10, color: "#6B7585" }} />
        <Line
          type="monotone"
          dataKey="vsRandom"
          name="vs random"
          stroke="#4ADE80"
          strokeWidth={1.5}
          dot={false}
          connectNulls
        />
        <Line
          type="monotone"
          dataKey="vsScripted"
          name="vs scripted"
          stroke="#FACC15"
          strokeWidth={1.5}
          dot={false}
          connectNulls
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

// ---------- 2. Economy ----------

const EconomyChart = () => {
  const history = useStore((s) => s.economyHistory);
  if (history.length === 0) {
    return <NoData msg="No economy data yet — waiting for live match." />;
  }
  const data = history.map((s) => ({
    tick: s.tick,
    Attackers: s.attackerTotal,
    Defenders: s.defenderTotal,
  }));
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid stroke="#222B3A" strokeDasharray="2 2" />
        <XAxis dataKey="tick" stroke="#6B7585" tick={{ fontSize: 10 }} />
        <YAxis stroke="#6B7585" tick={{ fontSize: 10 }} />
        <Tooltip
          contentStyle={{
            background: "#131821",
            border: "1px solid #222B3A",
            fontSize: 11,
            color: "#E5E7EB",
          }}
        />
        <Legend wrapperStyle={{ fontSize: 10, color: "#6B7585" }} />
        <Area
          type="monotone"
          dataKey="Attackers"
          stackId="1"
          stroke="#FFC833"
          fill="#FFC833"
          fillOpacity={0.35}
        />
        <Area
          type="monotone"
          dataKey="Defenders"
          stackId="1"
          stroke="#4DA8FF"
          fill="#4DA8FF"
          fillOpacity={0.35}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};

// ---------- 3. Action Distribution ----------

const ActionDistChart = () => {
  const selectedId = useStore((s) => s.selectedAgentId);
  const events = useStore((s) => s.eventFeed);
  const inspection = useStore(selectSelectedInspection);

  const data = useMemo(() => {
    if (!selectedId) return [];
    // Prefer the event-feed (comm events tagged to the actor) — falls back
    // to inspection.lastHeads if no events are available.
    const counts = new Map<string, number>();
    const recent = events
      .filter((e) => e.actorId === selectedId && e.kind === "comm")
      .slice(0, 100);
    for (const e of recent) {
      const lbl = e.text.split(":")[0] || e.text;
      counts.set(lbl, (counts.get(lbl) ?? 0) + 1);
    }
    if (counts.size === 0 && inspection?.lastHeads) {
      const { lastHeads } = inspection;
      for (const v of Object.values(lastHeads)) {
        if (!v) continue;
        counts.set(v, (counts.get(v) ?? 0) + 1);
      }
    }
    return Array.from(counts.entries()).map(([action, count]) => ({ action, count }));
  }, [events, inspection, selectedId]);

  if (!selectedId) {
    return <NoData msg="Select an agent to see its recent action distribution." />;
  }
  if (data.length === 0) {
    return <NoData msg="No actions captured for this agent yet." />;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 16 }}>
        <CartesianGrid stroke="#222B3A" strokeDasharray="2 2" />
        <XAxis
          dataKey="action"
          stroke="#6B7585"
          tick={{ fontSize: 9 }}
          angle={-30}
          textAnchor="end"
          height={40}
          interval={0}
        />
        <YAxis stroke="#6B7585" tick={{ fontSize: 10 }} allowDecimals={false} />
        <Tooltip
          contentStyle={{
            background: "#131821",
            border: "1px solid #222B3A",
            fontSize: 11,
            color: "#E5E7EB",
          }}
        />
        <Bar dataKey="count" fill="#4DA8FF" />
      </BarChart>
    </ResponsiveContainer>
  );
};

// ---------- 4. Round Outcome Pie ----------

const RoundOutcomePie = () => {
  const results = useStore((s) => s.roundResults);
  const recent = results.slice(-20);
  const data = useMemo(() => {
    const counts = new Map<RoundOutcome, number>();
    for (const r of recent) counts.set(r.outcome, (counts.get(r.outcome) ?? 0) + 1);
    return Array.from(counts.entries()).map(([outcome, value]) => ({
      outcome,
      value,
      label: outcomeStyle(outcome).label,
      color: outcomeStyle(outcome).css,
    }));
  }, [recent]);

  if (data.length === 0) {
    return <NoData msg="No completed rounds yet." />;
  }
  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="label"
          cx="50%"
          cy="50%"
          outerRadius="80%"
          stroke="#0A0E14"
          strokeWidth={1}
          label={(d) => `${d.label} ${d.value}`}
          labelLine={false}
        >
          {data.map((d, i) => (
            <Cell key={i} fill={d.color} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            background: "#131821",
            border: "1px solid #222B3A",
            fontSize: 11,
            color: "#E5E7EB",
          }}
        />
      </PieChart>
    </ResponsiveContainer>
  );
};

// ---------- Container ----------

const MetricsPanel = () => {
  return (
    <div className="flex flex-col gap-2 p-2">
      <ChartCard title="Winrate vs baselines" hint="last samples · %">
        <WinrateChart />
      </ChartCard>
      <ChartCard title="Economy" hint="$ stacked per side">
        <EconomyChart />
      </ChartCard>
      <ChartCard title="Action distribution" hint="last 100 ticks · selected agent">
        <ActionDistChart />
      </ChartCard>
      <ChartCard title="Round outcomes" hint="last 20 rounds">
        <RoundOutcomePie />
      </ChartCard>
    </div>
  );
};

export default MetricsPanel;
