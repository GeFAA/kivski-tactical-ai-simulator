import { useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "@/lib/store";
import {
  getCheckpoints,
  getTrainingConfigs,
  getTrainingStatus,
  postCommand,
  type CheckpointInfo,
  type TrainingConfigInfo,
} from "@/lib/api-client";

/**
 * Training-specific control deck. Lives below BottomControls (or
 * inlined into it). Polls `/api/training/status` every 2s while the
 * trainer is running so the agent worker doesn't have to push a
 * dedicated WS frame on every change.
 *
 * Visualises live policy/value/entropy with tiny inline sparklines.
 */

const POLL_INTERVAL_MS = 2000;
const SPARK_WINDOW = 60; // last N samples

// ---------- Sparkline ----------

const Sparkline = ({
  values,
  color,
  width = 80,
  height = 18,
}: {
  values: number[];
  color: string;
  width?: number;
  height?: number;
}) => {
  if (values.length < 2) {
    return (
      <svg width={width} height={height} className="opacity-40">
        <line x1={0} y1={height - 1} x2={width} y2={height - 1} stroke={color} strokeWidth={1} />
      </svg>
    );
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const stepX = width / (values.length - 1);
  const points = values
    .map((v, i) => {
      const x = i * stepX;
      const y = height - ((v - min) / range) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width={width} height={height}>
      <polyline points={points} fill="none" stroke={color} strokeWidth={1.25} />
    </svg>
  );
};

// ---------- Stat row ----------

const StatRow = ({
  label,
  value,
  spark,
  color,
}: {
  label: string;
  value: number | undefined;
  spark: number[];
  color: string;
}) => (
  <div className="flex items-center justify-between gap-2 text-[10px]">
    <span className="uppercase tracking-widest text-kivski-muted">{label}</span>
    <div className="flex items-center gap-2">
      <Sparkline values={spark} color={color} />
      <span className="stat w-14 text-right text-kivski-text">
        {typeof value === "number" ? value.toFixed(3) : "—"}
      </span>
    </div>
  </div>
);

// ---------- Component ----------

const TrainingPanel = () => {
  const trainingStatus = useStore((s) => s.trainingStatus);
  const setTrainingStatus = useStore((s) => s.setTrainingStatus);

  const [configs, setConfigs] = useState<TrainingConfigInfo[]>([]);
  const [selectedConfig, setSelectedConfig] = useState<string>("");
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [selectedCkpt, setSelectedCkpt] = useState<string>("");
  const [episodeCount, setEpisodeCount] = useState(10);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Sparkline buffers. Plain refs so we don't trigger re-renders on push.
  const policyBuf = useRef<number[]>([]);
  const valueBuf = useRef<number[]>([]);
  const entropyBuf = useRef<number[]>([]);

  // Initial config + checkpoint fetch.
  useEffect(() => {
    let alive = true;
    (async () => {
      const [cfgs, ckpts] = await Promise.all([
        getTrainingConfigs(),
        getCheckpoints().catch(() => [] as CheckpointInfo[]),
      ]);
      if (!alive) return;
      // Defense-in-depth: enforce array shape so a wire-protocol regression
      // (object wrapper, null, error response) cannot crash the .map() below.
      const safeCfgs = Array.isArray(cfgs) ? cfgs : [];
      const safeCkpts = Array.isArray(ckpts) ? ckpts : [];
      setConfigs(safeCfgs);
      if (safeCfgs[0]) setSelectedConfig(safeCfgs[0].id);
      setCheckpoints(safeCkpts);
      if (safeCkpts[0]) setSelectedCkpt(safeCkpts[0].id);
    })();
    return () => {
      alive = false;
    };
  }, []);

  // Poll training status while the loop is running.
  useEffect(() => {
    if (!trainingStatus.running) return;
    let alive = true;
    const tick = async () => {
      const s = await getTrainingStatus();
      if (!alive || !s) return;
      setTrainingStatus(s);
      if (typeof s.policyLoss === "number") {
        policyBuf.current = [...policyBuf.current.slice(-SPARK_WINDOW + 1), s.policyLoss];
      }
      if (typeof s.valueLoss === "number") {
        valueBuf.current = [...valueBuf.current.slice(-SPARK_WINDOW + 1), s.valueLoss];
      }
      if (typeof s.entropy === "number") {
        entropyBuf.current = [...entropyBuf.current.slice(-SPARK_WINDOW + 1), s.entropy];
      }
    };
    tick();
    const id = setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [trainingStatus.running, setTrainingStatus]);

  const send = async (label: string, body: Parameters<typeof postCommand>[0]) => {
    setBusy(label);
    setError(null);
    const r = await postCommand(body);
    setBusy(null);
    if (!r.ok) setError(`${label}: ${r.error}`);
  };

  const statusLabel = useMemo(() => {
    if (!trainingStatus.running) return "Idle";
    return `Running ep ${trainingStatus.episode} / ${
      trainingStatus.totalEpisodes || "∞"
    }`;
  }, [trainingStatus]);

  return (
    <section className="panel flex flex-col gap-2 p-2">
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-widest text-kivski-muted">
            Training
          </span>
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              trainingStatus.running ? "bg-kivski-hp animate-pulse-slow" : "bg-kivski-muted"
            }`}
          />
          <span className="stat text-[11px] text-kivski-text">{statusLabel}</span>
        </div>
        {error && (
          <span className="truncate text-[10px] text-kivski-hp-low" title={error}>
            {error}
          </span>
        )}
      </header>

      {/* Live metrics */}
      <div className="flex flex-col gap-1 rounded bg-kivski-bg/60 p-1.5">
        <StatRow
          label="policy loss"
          value={trainingStatus.policyLoss}
          spark={policyBuf.current}
          color="#FF6B6B"
        />
        <StatRow
          label="value loss"
          value={trainingStatus.valueLoss}
          spark={valueBuf.current}
          color="#FFC833"
        />
        <StatRow
          label="entropy"
          value={trainingStatus.entropy}
          spark={entropyBuf.current}
          color="#4DA8FF"
        />
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <select
          value={selectedConfig}
          onChange={(e) => setSelectedConfig(e.target.value)}
          className="stat rounded border border-kivski-border bg-kivski-bg px-1.5 py-1 text-[11px] text-kivski-text outline-none focus:border-kivski-defender"
          disabled={configs.length === 0}
        >
          {configs.length === 0 && <option value="">(default)</option>}
          {configs.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="btn btn-primary"
          onClick={() =>
            send("start training", {
              type: "start_training",
              configId: selectedConfig || undefined,
            })
          }
          disabled={busy !== null || trainingStatus.running}
        >
          Start
        </button>
        <button
          type="button"
          className="btn btn-danger"
          onClick={() => send("stop training", { type: "stop_training" })}
          disabled={busy !== null || !trainingStatus.running}
        >
          Stop
        </button>

        <label className="flex items-center gap-1 text-[10px] uppercase tracking-widest text-kivski-muted">
          N
          <input
            type="number"
            min={1}
            max={10000}
            value={episodeCount}
            onChange={(e) =>
              setEpisodeCount(Math.max(1, Math.min(10000, Number(e.target.value) || 1)))
            }
            className="stat w-16 rounded border border-kivski-border bg-kivski-bg px-1 py-0.5 text-[11px] text-kivski-text outline-none focus:border-kivski-defender"
          />
        </label>
        <button
          type="button"
          className="btn"
          onClick={() => send(`run ${episodeCount} eps`, { type: "run_episodes", n: episodeCount })}
          disabled={busy !== null}
        >
          Run Eps
        </button>

        <span className="ml-2 h-4 w-px bg-kivski-border" />

        <button
          type="button"
          className="btn"
          onClick={() => send("save checkpoint", { type: "save_checkpoint" })}
          disabled={busy !== null}
        >
          Save
        </button>
        <select
          value={selectedCkpt}
          onChange={(e) => setSelectedCkpt(e.target.value)}
          className="stat rounded border border-kivski-border bg-kivski-bg px-1.5 py-1 text-[11px] text-kivski-text outline-none focus:border-kivski-defender"
        >
          {checkpoints.length === 0 && <option value="">(none)</option>}
          {checkpoints.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name} · {c.step}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="btn"
          disabled={!selectedCkpt || busy !== null}
          onClick={() => send("load checkpoint", { type: "load_checkpoint", id: selectedCkpt })}
        >
          Load
        </button>
      </div>
    </section>
  );
};

export default TrainingPanel;
