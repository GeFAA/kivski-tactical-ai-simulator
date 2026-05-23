import { useEffect, useState } from "react";
import { getCheckpoints, type CheckpointInfo } from "@/lib/api-client";

/**
 * System Info panel: live host CPU / RAM / platform / torch readout.
 *
 * Polls `/api/system/info` every 5 s while mounted. The endpoint is cheap
 * (a single ``psutil.cpu_percent(interval=0.1)`` sample plus a torch
 * version probe) so polling is acceptable -- there is no need for the
 * heavier WebSocket plumbing the snapshot stream uses.
 *
 * All fields degrade gracefully when missing so a partially-installed
 * backend (e.g. no torch) renders an em-dash instead of crashing.
 */

const POLL_INTERVAL_MS = 5000;
const CKPT_POLL_INTERVAL_MS = 10_000;

interface SystemInfoPayload {
  cpu_count?: number | null;
  cpu_percent?: number | null;
  memory_total_gb?: number | null;
  memory_used_gb?: number | null;
  memory_percent?: number | null;
  load_avg?: number[] | null;
  platform?: string | null;
  python?: string | null;
  kivski_api_version?: string | null;
  kivski_sim_version?: string | null;
  torch_version?: string | null;
  cuda_available?: boolean | null;
  cuda_device?: string | null;
  cuda_compute_capability?: string | null;
  gpu_total_memory_gb?: number | null;
  gpu_used_memory_gb?: number | null;
  gpu_reserved_memory_gb?: number | null;
  uptime_s?: number | null;
  pid?: number | null;
}

const fmtNumber = (
  value: number | null | undefined,
  digits = 1,
  suffix = "",
): string => {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `${value.toFixed(digits)}${suffix}`;
};

const fmtUptime = (s: number | null | undefined): string => {
  if (typeof s !== "number" || s < 0) return "—";
  if (s < 60) return `${s.toFixed(0)}s`;
  const mins = Math.floor(s / 60);
  const secs = Math.floor(s % 60);
  if (mins < 60) return `${mins}m ${secs}s`;
  const hrs = Math.floor(mins / 60);
  const rmins = mins % 60;
  return `${hrs}h ${rmins}m`;
};

/** A simple horizontal bar -- value is a fraction in [0, 1]. */
const Bar = ({
  fraction,
  color = "#4DA8FF",
}: {
  fraction: number;
  color?: string;
}) => {
  const pct = Math.max(0, Math.min(1, fraction)) * 100;
  return (
    <div className="relative h-1.5 w-full overflow-hidden rounded bg-kivski-bg">
      <div
        className="absolute inset-y-0 left-0"
        style={{ width: `${pct}%`, background: color }}
      />
    </div>
  );
};

const Row = ({
  label,
  value,
  title,
}: {
  label: string;
  value: string;
  title?: string;
}) => (
  <div
    className="flex items-baseline justify-between gap-2 text-[11px]"
    title={title}
  >
    <span className="text-kivski-muted">{label}</span>
    <span className="stat truncate text-right text-kivski-text">{value}</span>
  </div>
);

const SystemInfo = () => {
  const [info, setInfo] = useState<SystemInfoPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastFetchAt, setLastFetchAt] = useState<number | null>(null);
  const [latestCkpt, setLatestCkpt] = useState<CheckpointInfo | null>(null);

  useEffect(() => {
    let alive = true;
    const fetchOnce = async () => {
      try {
        const res = await fetch("/api/system/info", {
          headers: { Accept: "application/json" },
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as SystemInfoPayload;
        if (!alive) return;
        setInfo(data);
        setError(null);
        setLastFetchAt(Date.now());
      } catch (err) {
        if (!alive) return;
        setError(err instanceof Error ? err.message : String(err));
      }
    };
    void fetchOnce();
    const id = window.setInterval(fetchOnce, POLL_INTERVAL_MS);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  // Lower-cadence poll for the most-recent checkpoint. We sort by step
  // (descending) and show the top entry so the user can see how far the
  // last training session got. Failures degrade silently — the section
  // simply renders an em-dash.
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      const list = await getCheckpoints();
      if (!alive) return;
      if (!Array.isArray(list) || list.length === 0) {
        setLatestCkpt(null);
        return;
      }
      const sorted = [...list].sort((a, b) => (b.step ?? 0) - (a.step ?? 0));
      setLatestCkpt(sorted[0] ?? null);
    };
    void tick();
    const id = window.setInterval(tick, CKPT_POLL_INTERVAL_MS);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  if (!info && !error) {
    return (
      <div className="flex h-full items-center justify-center px-4 text-center text-xs text-kivski-muted">
        Loading system info…
      </div>
    );
  }

  if (error && !info) {
    return (
      <div className="flex h-full items-center justify-center px-4 text-center text-xs text-kivski-hp-low">
        Could not load system info: {error}
      </div>
    );
  }

  const cpuPct =
    typeof info?.cpu_percent === "number" ? info.cpu_percent : null;
  const memPct =
    typeof info?.memory_percent === "number" ? info.memory_percent : null;
  const memUsed =
    typeof info?.memory_used_gb === "number" ? info.memory_used_gb : null;
  const memTotal =
    typeof info?.memory_total_gb === "number" ? info.memory_total_gb : null;

  const cpuColor =
    cpuPct === null
      ? "#4DA8FF"
      : cpuPct >= 85
        ? "#FF6B6B"
        : cpuPct >= 60
          ? "#FFC833"
          : "#4DA8FF";

  const memColor =
    memPct === null
      ? "#4DA8FF"
      : memPct >= 85
        ? "#FF6B6B"
        : memPct >= 70
          ? "#FFC833"
          : "#4DA8FF";

  return (
    <div className="flex flex-col gap-2 p-2 text-xs">
      {/* CPU */}
      <section className="panel p-2">
        <div className="mb-1 flex items-baseline justify-between">
          <span className="text-[10px] uppercase tracking-widest text-kivski-muted">
            CPU
          </span>
          <span className="stat text-kivski-text">
            {fmtNumber(cpuPct, 1, "%")}
          </span>
        </div>
        <Bar fraction={(cpuPct ?? 0) / 100} color={cpuColor} />
        <div className="mt-1 grid grid-cols-2 gap-1 text-[10px] text-kivski-muted">
          <span>
            Cores{" "}
            <span className="stat text-kivski-text">
              {info?.cpu_count ?? "—"}
            </span>
          </span>
          {Array.isArray(info?.load_avg) && info.load_avg.length >= 3 && (
            <span title="1m / 5m / 15m load average">
              Load{" "}
              <span className="stat text-kivski-text">
                {info.load_avg
                  .slice(0, 3)
                  .map((v) => v.toFixed(2))
                  .join(" / ")}
              </span>
            </span>
          )}
        </div>
      </section>

      {/* Memory */}
      <section className="panel p-2">
        <div className="mb-1 flex items-baseline justify-between">
          <span className="text-[10px] uppercase tracking-widest text-kivski-muted">
            Memory
          </span>
          <span className="stat text-kivski-text">
            {fmtNumber(memPct, 1, "%")}
          </span>
        </div>
        <Bar fraction={(memPct ?? 0) / 100} color={memColor} />
        <div className="mt-1 text-[10px] text-kivski-muted">
          {memUsed !== null && memTotal !== null ? (
            <span>
              <span className="stat text-kivski-text">
                {memUsed.toFixed(2)}
              </span>{" "}
              /{" "}
              <span className="stat text-kivski-text">
                {memTotal.toFixed(2)}
              </span>{" "}
              GB
            </span>
          ) : (
            <span>—</span>
          )}
        </div>
      </section>

      {/* Latest checkpoint */}
      <section className="panel flex flex-col gap-0.5 p-2">
        <div className="mb-1 text-[10px] uppercase tracking-widest text-kivski-muted">
          Latest Checkpoint
        </div>
        {latestCkpt ? (
          <>
            <Row
              label="Name"
              value={latestCkpt.name}
              title={latestCkpt.name}
            />
            <Row
              label="Episodes"
              value={String(latestCkpt.step ?? 0)}
              title="trainer-reported episode count"
            />
            <Row
              label="Created"
              value={latestCkpt.createdAt || "—"}
              title={latestCkpt.createdAt}
            />
          </>
        ) : (
          <div className="text-[11px] text-kivski-muted">
            no checkpoint saved yet
          </div>
        )}
      </section>

      {/* Runtime / platform */}
      <section className="panel flex flex-col gap-0.5 p-2">
        <div className="mb-1 text-[10px] uppercase tracking-widest text-kivski-muted">
          Runtime
        </div>
        <Row label="Platform" value={info?.platform ?? "—"} title={info?.platform ?? ""} />
        <Row
          label="Python"
          value={
            info?.python
              ? info.python.split(" ", 1)[0] ?? "—"
              : "—"
          }
          title={info?.python ?? ""}
        />
        <Row
          label="Torch"
          value={info?.torch_version ?? "—"}
          title={info?.torch_version ?? ""}
        />
        <Row
          label="CUDA"
          value={
            info?.cuda_available
              ? `yes${info.cuda_device ? ` (${info.cuda_device})` : ""}`
              : "no"
          }
          title={info?.cuda_device ?? ""}
        />
        {info?.cuda_available && info?.cuda_compute_capability && (
          <Row
            label="Compute"
            value={`sm_${info.cuda_compute_capability.replace(".", "")}`}
            title={`CUDA compute capability ${info.cuda_compute_capability}`}
          />
        )}
        {info?.cuda_available &&
          typeof info?.gpu_total_memory_gb === "number" && (
            <Row
              label="GPU Mem"
              value={`${fmtNumber(info?.gpu_used_memory_gb ?? 0, 2)} / ${fmtNumber(
                info.gpu_total_memory_gb,
                1,
              )} GB`}
              title={
                typeof info?.gpu_reserved_memory_gb === "number"
                  ? `allocated / total — reserved by torch: ${info.gpu_reserved_memory_gb.toFixed(
                      2,
                    )} GB`
                  : "torch.cuda.memory_allocated / device total"
              }
            />
          )}
        <Row
          label="Kivski API"
          value={info?.kivski_api_version ?? "—"}
        />
        <Row
          label="Kivski Sim"
          value={info?.kivski_sim_version ?? "—"}
        />
        <Row label="Uptime" value={fmtUptime(info?.uptime_s)} />
        <Row
          label="PID"
          value={typeof info?.pid === "number" ? String(info.pid) : "—"}
        />
      </section>

      {error && (
        <div className="text-[10px] text-kivski-hp-low">
          last refresh failed: {error}
        </div>
      )}
      {lastFetchAt && !error && (
        <div className="text-[10px] text-kivski-muted">
          refreshed {new Date(lastFetchAt).toLocaleTimeString()} (auto every{" "}
          {POLL_INTERVAL_MS / 1000}s)
        </div>
      )}
    </div>
  );
};

export default SystemInfo;
