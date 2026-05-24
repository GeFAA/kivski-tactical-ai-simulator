import { useEffect, useState } from "react";
import {
  getCheckpoints,
  getResumeTarget,
  postCommand,
  type CheckpointInfo,
  type ResumeTargetInfo,
} from "@/lib/api-client";
import { useStore } from "@/lib/store";
import MatchSetupModal from "@/components/MatchSetupModal";

const SPEEDS = [0.5, 1, 2, 4, 16];

const SETTINGS_ICON = (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);

/**
 * Simple-mode footer: just play/pause, a speed picker, and a settings
 * shortcut. Everything else (training, checkpoints, match setup) lives
 * in the settings drawer. Keeps the default view from being "voll
 * geschissen" with debug controls a non-technical user can't parse.
 */
const SimpleBottomControls = () => {
  const paused = useStore((s) => s.paused);
  const togglePaused = useStore((s) => s.togglePaused);
  const speed = useStore((s) => s.speed);
  const setSpeed = useStore((s) => s.setSpeed);
  const setSettingsOpen = useStore((s) => s.setSettingsOpen);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const send = async (
    label: string,
    body: Parameters<typeof postCommand>[0],
  ) => {
    setBusy(true);
    setError(null);
    const r = await postCommand(body);
    setBusy(false);
    if (!r.ok) setError(`${label}: ${r.error}`);
  };

  const onPlayPause = async () => {
    togglePaused();
    await send(paused ? "resume" : "pause", {
      type: paused ? "resume" : "pause",
    });
  };

  const onSpeed = async (s: number) => {
    setSpeed(s);
    await send(`speed ${s}x`, { type: "set_speed", speed: s });
  };

  return (
    <footer className="flex h-14 items-center justify-between gap-3 border-t border-kivski-border bg-kivski-panel px-4">
      <div className="flex items-center gap-2">
        <button
          type="button"
          className="btn btn-primary min-w-[6rem]"
          onClick={onPlayPause}
          disabled={busy}
        >
          {paused ? "Play" : "Pause"}
        </button>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-[10px] uppercase tracking-widest text-kivski-muted">
          Speed
        </span>
        <div className="flex gap-0.5">
          {SPEEDS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => onSpeed(s)}
              className={`btn px-2 py-1 text-xs ${
                speed === s
                  ? "border-kivski-defender text-kivski-defender"
                  : ""
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-2">
        {error && (
          <span
            className="truncate text-[10px] text-kivski-hp-low"
            title={error}
          >
            {error}
          </span>
        )}
        <button
          type="button"
          aria-label="Settings"
          onClick={() => setSettingsOpen(true)}
          title="Open settings (Match · Training · View · About)"
          className="flex h-9 items-center gap-1.5 rounded border border-kivski-border bg-kivski-panel-2 px-3 text-xs text-kivski-text transition-colors hover:border-kivski-defender hover:text-kivski-defender"
        >
          {SETTINGS_ICON}
          <span>Settings</span>
        </button>
      </div>
    </footer>
  );
};

const BottomControls = () => {
  const uiMode = useStore((s) => s.uiMode);
  if (uiMode === "simple") return <SimpleBottomControls />;
  return <AdvancedBottomControls />;
};

const AdvancedBottomControls = () => {
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
  // Auto-resume target: populated from GET /api/training/resume-target so
  // the Start Training button can show "Resumes from <name>" instead of
  // a generic tooltip. Refreshes on mount; backend is the source of truth.
  const [resumeTarget, setResumeTarget] = useState<ResumeTargetInfo | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [list, resume] = await Promise.all([
          getCheckpoints(),
          getResumeTarget(),
        ]);
        if (!alive) return;
        setCheckpoints(list);
        if (list[0]) setSelectedCkpt(list[0].id);
        setResumeTarget(resume);
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
          title={
            trainingRunning
              ? "A training job is already running"
              : resumeTarget?.available
                ? `Resumes from ${resumeTarget.name ?? resumeTarget.path}`
                : "Starts a fresh training run (no checkpoint to resume)"
          }
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
