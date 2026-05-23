import { useEffect, useMemo, useState } from "react";
import { useStore } from "@/lib/store";
import {
  getCheckpoints,
  getResumeTarget,
  getTrainingConfigs,
  getTrainingStatus,
  postCommand,
  type CheckpointInfo,
  type ResumeTargetInfo,
  type TrainingConfigInfo,
} from "@/lib/api-client";

/**
 * Training-specific control deck. Lives below BottomControls (or
 * inlined into it). Reacts to the WebSocket-pushed `training_status` /
 * `metrics_sample` frames maintained by the backend's
 * :class:`MetricsBroadcaster`. The /api/training/status REST endpoint is
 * still polled at low cadence as a fallback for the `running` flag in
 * case the broadcaster's stream is interrupted.
 *
 * Visualises live policy/value/entropy with tiny inline sparklines.
 */

const POLL_INTERVAL_MS = 5000;
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
  // Sparklines now derived from metricsHistory, which gets a fresh
  // point on every WS push (App.tsx mirrors training_status frames into
  // pushMetricsSample). This bypasses React's useEffect-dependency dedup
  // that otherwise pinned the sparkline at one sample whenever the
  // trainer re-emitted an identical metric.
  const metricsHistory = useStore((s) => s.metricsHistory);

  const [configs, setConfigs] = useState<TrainingConfigInfo[]>([]);
  const [selectedConfig, setSelectedConfig] = useState<string>("");
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [selectedCkpt, setSelectedCkpt] = useState<string>("");
  const [episodeCount, setEpisodeCount] = useState(10);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Mirrors GET /api/training/resume-target so the Start button can
  // advertise "Resumes from <name>" instead of silently re-using the
  // latest checkpoint without any UI signal.
  const [resumeTarget, setResumeTarget] = useState<ResumeTargetInfo | null>(null);

  // Derived sparkline buffers — tail of metricsHistory.
  const policyBuf = useMemo(
    () =>
      metricsHistory
        .slice(-SPARK_WINDOW)
        .map((s) => s.policyLoss)
        .filter((v): v is number => typeof v === "number"),
    [metricsHistory],
  );
  const valueBuf = useMemo(
    () =>
      metricsHistory
        .slice(-SPARK_WINDOW)
        .map((s) => s.valueLoss)
        .filter((v): v is number => typeof v === "number"),
    [metricsHistory],
  );
  const entropyBuf = useMemo(
    () =>
      metricsHistory
        .slice(-SPARK_WINDOW)
        .map((s) => s.entropy)
        .filter((v): v is number => typeof v === "number"),
    [metricsHistory],
  );

  // Initial config + checkpoint + resume-target fetch.
  useEffect(() => {
    let alive = true;
    (async () => {
      const [cfgs, ckpts, resume] = await Promise.all([
        getTrainingConfigs(),
        getCheckpoints().catch(() => [] as CheckpointInfo[]),
        getResumeTarget().catch(
          (): ResumeTargetInfo => ({ available: false, path: null, name: null }),
        ),
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
      setResumeTarget(resume);
    })();
    return () => {
      alive = false;
    };
  }, []);

  // Low-frequency fallback poll for the running flag. The bulk of live
  // updates (loss / entropy / episode counters) arrives via the WS
  // `training_status` frame published by the MetricsBroadcaster.
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      const s = await getTrainingStatus();
      if (!alive || !s) return;
      // Only merge the boolean + totalEpisodes from REST so we don't stomp
      // the live numeric fields supplied by the WS push.
      setTrainingStatus({
        running: s.running,
        episode: s.episode,
        totalEpisodes: s.totalEpisodes,
      });
    };
    tick();
    const id = setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [setTrainingStatus]);

  const send = async (label: string, body: Parameters<typeof postCommand>[0]) => {
    setBusy(label);
    setError(null);
    const r = await postCommand(body);
    setBusy(null);
    if (!r.ok) {
      setError(`${label}: ${r.error}`);
    } else if (r.alreadyRunning) {
      // 409 was gracefully absorbed; show a non-error hint that the
      // job was already running so the user understands the no-op.
      setError(`${label}: already running (no-op)`);
      window.setTimeout(() => setError(null), 1_500);
    }
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
          spark={policyBuf}
          color="#FF6B6B"
        />
        <StatRow
          label="value loss"
          value={trainingStatus.valueLoss}
          spark={valueBuf}
          color="#FFC833"
        />
        <StatRow
          label="entropy"
          value={trainingStatus.entropy}
          spark={entropyBuf}
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
          title={
            resumeTarget?.available
              ? `Resumes from ${resumeTarget.name ?? resumeTarget.path}`
              : "Starts a fresh training run (no checkpoint to resume)"
          }
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
