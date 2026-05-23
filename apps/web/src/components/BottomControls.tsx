import { useEffect, useState } from "react";
import { getCheckpoints, postCommand, type CheckpointInfo } from "@/lib/api-client";
import { useStore } from "@/lib/store";
import MatchSetupModal from "@/components/MatchSetupModal";

const SPEEDS = [0.5, 1, 2, 4, 16];

const BottomControls = () => {
  const paused = useStore((s) => s.paused);
  const togglePaused = useStore((s) => s.togglePaused);
  const speed = useStore((s) => s.speed);
  const setSpeed = useStore((s) => s.setSpeed);
  const round = useStore((s) => s.round);
  // Mirrors /api/training/status — used to disable the Start button
  // while a job is in flight so a double-click can't even attempt the
  // POST. Belt-and-braces alongside the api-client's 409-graceful
  // handling.
  const trainingRunning = useStore((s) => s.trainingStatus.running);

  const [episodeCount, setEpisodeCount] = useState(10);
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [selectedCkpt, setSelectedCkpt] = useState<string>("");
  const [busy, setBusy] = useState<string | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const [matchModalOpen, setMatchModalOpen] = useState(false);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const list = await getCheckpoints();
        if (!alive) return;
        setCheckpoints(list);
        if (list[0]) setSelectedCkpt(list[0].id);
      } catch (err) {
         
        console.warn("[kivski] checkpoint list unavailable:", err);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const send = async (label: string, body: Parameters<typeof postCommand>[0]) => {
    setBusy(label);
    setLastError(null);
    const r = await postCommand(body);
    setBusy(null);
    if (!r.ok) {
      setLastError(`${label}: ${r.error}`);
    } else if (r.alreadyRunning) {
      // 409 was gracefully absorbed by the api-client; flash an
      // informational note rather than a scary error toast.
      setLastError(`${label}: already running (no-op)`);
      window.setTimeout(() => setLastError(null), 1_500);
    }
  };

  const onPlayPause = async () => {
    togglePaused();
    await send(paused ? "resume" : "pause", { type: paused ? "resume" : "pause" });
  };

  const onSpeed = async (s: number) => {
    setSpeed(s);
    await send(`speed ${s}x`, { type: "set_speed", speed: s });
  };

  return (
    <footer className="flex h-20 items-stretch gap-2 border-t border-kivski-border bg-kivski-panel px-3 py-2">
      {/* Playback */}
      <div className="panel flex items-center gap-2 px-3">
        <button
          type="button"
          className="btn btn-primary min-w-[5rem]"
          onClick={onPlayPause}
          disabled={busy !== null}
        >
          {paused ? "Play" : "Pause"}
        </button>

        <div className="flex items-center gap-1">
          <span className="text-[10px] uppercase tracking-widest text-kivski-muted">Speed</span>
          <div className="flex gap-0.5">
            {SPEEDS.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => onSpeed(s)}
                className={`btn px-1.5 text-[10px] ${
                  speed === s ? "border-kivski-defender text-kivski-defender" : ""
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        <button
          type="button"
          className="btn"
          onClick={() => setMatchModalOpen(true)}
          disabled={busy !== null}
          title="Pick policies for both teams and start a fresh comparison match"
        >
          New Match
        </button>

        <button
          type="button"
          className="btn"
          onClick={() => send("reset", { type: "reset_match" })}
          disabled={busy !== null}
        >
          Reset Match
        </button>
      </div>

      {/* Timeline strip */}
      <div className="panel flex min-w-0 flex-1 flex-col justify-center px-3">
        <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-widest text-kivski-muted">
          <span>Round Timeline</span>
          <span className="stat normal-case text-kivski-text">round {round || "—"}</span>
        </div>
        <div className="relative h-3 w-full overflow-hidden rounded bg-kivski-bg">
          {/* Placeholder ticks per round, up to 30 */}
          <div className="absolute inset-0 flex">
            {Array.from({ length: 30 }, (_, i) => (
              <div
                key={i}
                className={`flex-1 border-r border-kivski-border/60 ${
                  i < round ? "bg-kivski-defender/40" : ""
                } ${i === round - 1 ? "bg-kivski-defender" : ""}`}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Training controls */}
      <div className="panel flex items-center gap-2 px-3">
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => send("start training", { type: "start_training" })}
          disabled={busy !== null || trainingRunning}
          title={trainingRunning ? "A training job is already running" : undefined}
        >
          Start Training
        </button>
        <button
          type="button"
          className="btn btn-danger"
          onClick={() => send("stop training", { type: "stop_training" })}
          disabled={busy !== null || !trainingRunning}
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
              setEpisodeCount(
                Math.max(1, Math.min(10000, Number(e.target.value) || 1)),
              )
            }
            className="stat w-16 rounded border border-kivski-border bg-kivski-bg px-1 py-0.5 text-xs text-kivski-text outline-none focus:border-kivski-defender"
          />
        </label>
        <button
          type="button"
          className="btn"
          onClick={() =>
            send(`run ${episodeCount} eps`, {
              type: "run_episodes",
              n: episodeCount,
            })
          }
          disabled={busy !== null}
        >
          Run Eps
        </button>

        <div className="ml-2 flex items-center gap-1">
          <button
            type="button"
            className="btn"
            onClick={() => send("save checkpoint (auto)", { type: "save_checkpoint" })}
            disabled={busy !== null}
            title="Trainer auto-saves periodically (see configs/default.yaml checkpoint_interval)"
          >
            Save Ckpt
          </button>
          <select
            value={selectedCkpt}
            onChange={(e) => setSelectedCkpt(e.target.value)}
            className="stat rounded border border-kivski-border bg-kivski-bg px-1.5 py-1 text-xs text-kivski-text outline-none focus:border-kivski-defender"
          >
            {checkpoints.length === 0 && <option value="">(no checkpoints)</option>}
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
            onClick={() =>
              send("load checkpoint", { type: "load_checkpoint", id: selectedCkpt })
            }
          >
            Load
          </button>
        </div>
      </div>

      {/* Status line */}
      {(busy || lastError) && (
        <div className="panel flex items-center px-3 text-xs">
          {busy && <span className="text-kivski-muted">… {busy}</span>}
          {lastError && (
            <span className="ml-2 truncate text-kivski-hp-low" title={lastError}>
              {lastError}
            </span>
          )}
        </div>
      )}

      <MatchSetupModal
        open={matchModalOpen}
        onClose={() => setMatchModalOpen(false)}
      />
    </footer>
  );
};

export default BottomControls;
