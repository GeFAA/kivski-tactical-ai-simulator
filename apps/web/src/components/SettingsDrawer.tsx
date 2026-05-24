import { useEffect, useMemo, useState } from "react";
import {
  getCheckpoints,
  getResumeTarget,
  getTrainingConfigs,
  getTrainingGoalSpec,
  inferGoalFromConfigPath,
  postCommand,
  TRAINING_GOALS,
  type CheckpointInfo,
  type ResumeTargetInfo,
  type TrainingConfigInfo,
  type TrainingGoal,
} from "@/lib/api-client";
import { useStore } from "@/lib/store";
import type { SettingsTab, UiMode } from "@/lib/store";
import MatchSetupModal from "@/components/MatchSetupModal";

/**
 * Sliding settings drawer. The single user-facing access point for
 * power-user functionality when the viewer is in Simple mode (and a
 * convenient consolidation in Advanced mode too).
 *
 * Layout: fixed 380px-wide panel sliding in from the right with a
 * translucent backdrop. The drawer overlays the map without resizing
 * it. Closes on backdrop click, ESC, or the close button.
 *
 * Tabs:
 *   - Match    — pause/resume, speed, reset, new match
 *   - Training — start/stop, run N, config picker, save/load ckpt
 *   - View     — simple/advanced mode toggle + debug overlay toggles
 *   - About    — project pitch + GPU info
 */

const SPEEDS = [0.5, 1, 2, 4, 8, 16] as const;

const SPEED_LABELS: Record<number, string> = {
  0.5: "0.5x · slow-mo",
  1: "1x · normal",
  2: "2x",
  4: "4x · brisk",
  8: "8x",
  16: "16x · turbo",
};

interface SystemInfoPayload {
  cpu_count?: number | null;
  platform?: string | null;
  python?: string | null;
  kivski_api_version?: string | null;
  kivski_sim_version?: string | null;
  torch_version?: string | null;
  cuda_available?: boolean | null;
  cuda_device?: string | null;
}

// ---------- Small UI primitives ----------

const TabButton = ({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) => (
  <button
    type="button"
    onClick={onClick}
    className={`flex-1 px-2 py-2 text-xs font-medium transition-colors ${
      active
        ? "border-b-2 border-kivski-defender text-kivski-text"
        : "border-b-2 border-transparent text-kivski-muted hover:text-kivski-text"
    }`}
  >
    {label}
  </button>
);

const Toggle = ({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description?: string;
  checked: boolean;
  onChange: () => void;
}) => (
  <label className="flex cursor-pointer items-start gap-3 rounded border border-transparent px-2 py-2 text-xs hover:border-kivski-border hover:bg-kivski-panel-2/60">
    <input
      type="checkbox"
      checked={checked}
      onChange={onChange}
      className="mt-0.5 h-4 w-4 accent-kivski-defender"
    />
    <span className="min-w-0 flex-1">
      <span className="block font-medium text-kivski-text">{label}</span>
      {description && (
        <span className="mt-0.5 block text-[10px] leading-tight text-kivski-muted">
          {description}
        </span>
      )}
    </span>
  </label>
);

const ModeCard = ({
  active,
  title,
  blurb,
  onClick,
}: {
  active: boolean;
  title: string;
  blurb: string;
  onClick: () => void;
}) => (
  <button
    type="button"
    onClick={onClick}
    className={`flex-1 rounded border px-3 py-3 text-left transition-colors ${
      active
        ? "border-kivski-defender bg-kivski-defender/10 ring-1 ring-kivski-defender/40"
        : "border-kivski-border bg-kivski-panel-2 hover:border-kivski-defender/60"
    }`}
  >
    <div className="flex items-center justify-between">
      <span className="text-sm font-semibold text-kivski-text">{title}</span>
      {active && (
        <span className="rounded bg-kivski-defender/20 px-1.5 py-0.5 text-[9px] uppercase tracking-widest text-kivski-defender">
          active
        </span>
      )}
    </div>
    <p className="mt-1 text-[11px] leading-tight text-kivski-muted">{blurb}</p>
  </button>
);

const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <div className="mb-2 text-[10px] uppercase tracking-widest text-kivski-muted">
    {children}
  </div>
);

// ---------- Match tab ----------

const MatchTab = ({ onClose }: { onClose: () => void }) => {
  const paused = useStore((s) => s.paused);
  const togglePaused = useStore((s) => s.togglePaused);
  const speed = useStore((s) => s.speed);
  const setSpeed = useStore((s) => s.setSpeed);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [matchModalOpen, setMatchModalOpen] = useState(false);

  const send = async (
    label: string,
    body: Parameters<typeof postCommand>[0],
  ) => {
    setBusy(label);
    setError(null);
    const r = await postCommand(body);
    setBusy(null);
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
    <div className="flex flex-col gap-4 p-4">
      <section>
        <SectionLabel>Playback</SectionLabel>
        <button
          type="button"
          onClick={onPlayPause}
          disabled={busy !== null}
          className="btn btn-primary w-full py-2 text-sm"
        >
          {paused ? "Resume Match" : "Pause Match"}
        </button>
      </section>

      <section>
        <SectionLabel>Speed</SectionLabel>
        <div className="grid grid-cols-3 gap-1.5">
          {SPEEDS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => onSpeed(s)}
              className={`btn px-2 py-2 text-xs ${
                speed === s
                  ? "border-kivski-defender bg-kivski-defender/15 text-kivski-defender"
                  : ""
              }`}
              title={SPEED_LABELS[s]}
            >
              {s}x
            </button>
          ))}
        </div>
        <p className="mt-1 text-[10px] text-kivski-muted">
          {SPEED_LABELS[speed] ?? `${speed}x`}
        </p>
      </section>

      <section>
        <SectionLabel>Match</SectionLabel>
        <div className="flex flex-col gap-1.5">
          <button
            type="button"
            className="btn"
            onClick={() => setMatchModalOpen(true)}
            disabled={busy !== null}
            title="Pick policies for both teams and start a fresh comparison match"
          >
            New Match…
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => send("reset", { type: "reset_match" })}
            disabled={busy !== null}
            title="Restart the current match from round 1"
          >
            Reset Match
          </button>
        </div>
      </section>

      {(busy || error) && (
        <div className="rounded border border-kivski-border bg-kivski-panel-2 px-2 py-1.5 text-[10px]">
          {busy && <div className="text-kivski-muted">… {busy}</div>}
          {error && (
            <div className="truncate text-kivski-hp-low" title={error}>
              {error}
            </div>
          )}
        </div>
      )}

      <MatchSetupModal
        open={matchModalOpen}
        onClose={() => {
          setMatchModalOpen(false);
          // Close the drawer too so the freshly-started match is
          // immediately visible — clicking "Start" feels like a
          // commit action.
          onClose();
        }}
      />
    </div>
  );
};

// ---------- Training tab ----------

/**
 * Big-button picker for one of the three training goals. Replaces the
 * raw "Config" dropdown with intent-based wording so non-technical users
 * don't have to map ``fast.yaml`` / ``turbo.yaml`` to their actual
 * behaviour. The selected button gets a defender-coloured ring, and the
 * card title carries an `aria-pressed` toggle for screen readers.
 *
 * When ``advancedOverride`` is true the goal cards are rendered greyed
 * out and non-interactive — the user has overridden the goal with a
 * raw config file in the Advanced reveal below.
 */
const GoalCard = ({
  spec,
  active,
  overridden,
  onClick,
}: {
  spec: (typeof TRAINING_GOALS)[number];
  active: boolean;
  overridden: boolean;
  onClick: () => void;
}) => (
  <button
    type="button"
    onClick={onClick}
    aria-pressed={active}
    aria-label={`Pick training goal: ${spec.title}`}
    className={`flex w-full items-start gap-3 rounded border px-3 py-2.5 text-left transition-colors ${
      overridden
        ? "border-kivski-border bg-kivski-panel-2/50 text-kivski-muted opacity-60"
        : active
          ? "border-kivski-defender bg-kivski-defender/10 ring-1 ring-kivski-defender/40"
          : "border-kivski-border bg-kivski-panel-2 hover:border-kivski-defender/60"
    }`}
  >
    <span aria-hidden className="mt-0.5 text-xl leading-none">
      {spec.icon}
    </span>
    <span className="min-w-0 flex-1">
      <span className="flex items-baseline justify-between gap-2">
        <span className="text-[12px] font-semibold text-kivski-text">
          {spec.title}
        </span>
        {active && !overridden && (
          <span className="rounded bg-kivski-defender/20 px-1.5 py-0.5 text-[9px] uppercase tracking-widest text-kivski-defender">
            picked
          </span>
        )}
      </span>
      <span className="mt-0.5 block text-[10px] leading-tight text-kivski-muted">
        {spec.blurb}
      </span>
    </span>
  </button>
);

/**
 * Single-line card representing a checkpoint in the load-checkpoint
 * picker. Clicking it sets it as the load target; the active card
 * carries a coloured border so it's obviously selected.
 *
 * The label / icon / age are pre-computed in ``api-client.getCheckpoints``
 * so this component is purely presentational — keeps the translation
 * layer in one place.
 */
const CheckpointCard = ({
  ckpt,
  active,
  onClick,
}: {
  ckpt: CheckpointInfo;
  active: boolean;
  onClick: () => void;
}) => {
  const isBest = ckpt.kind === "best";
  const isSnapshot = (ckpt.name ?? "").toLowerCase().startsWith("snapshot");
  const icon = isBest ? "★" : isSnapshot ? "📸" : "💾";
  const epsLine = ckpt.step ? `ep ${ckpt.step}` : null;
  const ageLine = ckpt.ageLabel ?? "";
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      aria-label={`Select checkpoint ${ckpt.prettyLabel ?? ckpt.name}`}
      title={ckpt.name}
      className={`flex w-full items-center gap-2.5 rounded border px-2.5 py-2 text-left transition-colors ${
        active
          ? "border-kivski-defender bg-kivski-defender/10 ring-1 ring-kivski-defender/40"
          : "border-kivski-border bg-kivski-panel-2 hover:border-kivski-defender/60"
      }`}
    >
      <span aria-hidden className={`text-base leading-none ${isBest ? "text-yellow-400" : ""}`}>
        {icon}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[12px] font-medium text-kivski-text">
          {ckpt.prettyLabel ?? ckpt.name}
        </span>
        <span className="mt-0.5 flex items-center gap-2 text-[10px] text-kivski-muted">
          {epsLine && <span className="stat">{epsLine}</span>}
          {ageLine && <span className="truncate">{ageLine}</span>}
        </span>
      </span>
      {active && (
        <span className="rounded bg-kivski-defender/20 px-1.5 py-0.5 text-[9px] uppercase tracking-widest text-kivski-defender">
          load
        </span>
      )}
    </button>
  );
};

const TrainingTab = () => {
  const trainingStatus = useStore((s) => s.trainingStatus);
  const trainingGoal = useStore((s) => s.trainingGoal);
  const setTrainingGoal = useStore((s) => s.setTrainingGoal);
  const metricsHistory = useStore((s) => s.metricsHistory);

  const [configs, setConfigs] = useState<TrainingConfigInfo[]>([]);
  const [rawConfigOverride, setRawConfigOverride] = useState<string>("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [selectedCkpt, setSelectedCkpt] = useState<string>("");
  const [episodeCount, setEpisodeCount] = useState(10);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [resumeTarget, setResumeTarget] = useState<ResumeTargetInfo | null>(
    null,
  );

  useEffect(() => {
    let alive = true;
    (async () => {
      const [cfgs, ckpts, resume] = await Promise.all([
        getTrainingConfigs(),
        getCheckpoints().catch(() => [] as CheckpointInfo[]),
        getResumeTarget().catch(
          (): ResumeTargetInfo => ({
            available: false,
            path: null,
            name: null,
          }),
        ),
      ]);
      if (!alive) return;
      const safeCfgs = Array.isArray(cfgs) ? cfgs : [];
      const safeCkpts = Array.isArray(ckpts) ? ckpts : [];
      setConfigs(safeCfgs);
      setCheckpoints(safeCkpts);
      if (safeCkpts[0]) setSelectedCkpt(safeCkpts[0].id);
      setResumeTarget(resume);
    })();
    return () => {
      alive = false;
    };
  }, []);

  const send = async (
    label: string,
    body: Parameters<typeof postCommand>[0],
  ) => {
    setBusy(label);
    setError(null);
    const r = await postCommand(body);
    setBusy(null);
    if (!r.ok) {
      setError(`${label}: ${r.error}`);
    } else if (r.alreadyRunning) {
      setError(`${label}: already running (no-op)`);
      window.setTimeout(() => setError(null), 1_500);
    }
  };

  // Effective config that will be POSTed: explicit raw override beats
  // the goal-mapped path. The override is cleared when the user
  // re-picks a goal so the cards don't stay greyed out forever.
  const effectiveConfigId = useMemo(() => {
    if (rawConfigOverride) return rawConfigOverride;
    return getTrainingGoalSpec(trainingGoal).configPath;
  }, [rawConfigOverride, trainingGoal]);

  const inferredFromOverride = useMemo(
    () => inferGoalFromConfigPath(rawConfigOverride || null),
    [rawConfigOverride],
  );

  const goalIsOverridden = rawConfigOverride !== "" && inferredFromOverride === null;

  // Latest win-rate vs random — surfaced in the status header so a user
  // who picked "Watch progress fast" can actually see the trained
  // policy beating the baseline within the first 1-3 hours.
  const latestWinrate = useMemo(() => {
    for (let i = metricsHistory.length - 1; i >= 0; i--) {
      const v = metricsHistory[i].winrateVsRandom;
      if (typeof v === "number" && Number.isFinite(v)) return v;
    }
    return undefined;
  }, [metricsHistory]);

  const activeGoalSpec = getTrainingGoalSpec(trainingGoal);

  const statusLabel = useMemo(() => {
    if (!trainingStatus.running) return "Idle";
    return `Running · ep ${trainingStatus.episode}${
      trainingStatus.totalEpisodes
        ? ` / ${trainingStatus.totalEpisodes}`
        : ""
    }`;
  }, [trainingStatus]);

  const pickGoal = (id: TrainingGoal) => {
    setTrainingGoal(id);
    // Clearing the raw override re-enables the cards visually.
    setRawConfigOverride("");
  };

  return (
    <div className="flex flex-col gap-4 p-4">
      <section>
        <SectionLabel>Status</SectionLabel>
        <div className="flex items-center gap-2 rounded border border-kivski-border bg-kivski-panel-2 px-3 py-2 text-xs">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              trainingStatus.running
                ? "bg-kivski-hp animate-pulse-slow"
                : "bg-kivski-muted"
            }`}
          />
          <span className="text-kivski-text">{statusLabel}</span>
        </div>
        {trainingStatus.running && (
          <p className="mt-1 text-[10px] leading-tight text-kivski-muted">
            Goal:{" "}
            <span className="text-kivski-text">{activeGoalSpec.title}</span>
            {typeof latestWinrate === "number" && (
              <>
                {" · "}
                WR vs Random:{" "}
                <span className="text-kivski-text">{latestWinrate.toFixed(2)}</span>
              </>
            )}
          </p>
        )}
        {resumeTarget?.available && (
          <p
            className="mt-1 text-[10px] leading-tight text-kivski-muted"
            title={resumeTarget.path ?? undefined}
          >
            Next start resumes from{" "}
            <span className="text-kivski-text">
              {resumeTarget.name ?? resumeTarget.path}
            </span>
          </p>
        )}
      </section>

      <section>
        <SectionLabel>Training goal</SectionLabel>
        <div className="flex flex-col gap-1.5">
          {TRAINING_GOALS.map((spec) => (
            <GoalCard
              key={spec.id}
              spec={spec}
              active={trainingGoal === spec.id}
              overridden={goalIsOverridden}
              onClick={() => pickGoal(spec.id)}
            />
          ))}
        </div>
        <button
          type="button"
          onClick={() => setShowAdvanced((v) => !v)}
          className="mt-2 text-[10px] text-kivski-muted hover:text-kivski-defender"
          aria-expanded={showAdvanced}
        >
          {showAdvanced ? "Hide advanced" : "Advanced: pick raw config"}
        </button>
        {showAdvanced && (
          <div className="mt-2 rounded border border-kivski-border bg-kivski-bg/40 p-2">
            <p className="mb-1.5 text-[10px] leading-tight text-kivski-muted">
              Override goal mapping by selecting any YAML preset directly.
              Picking one greys out the goal cards.
            </p>
            <select
              value={rawConfigOverride}
              onChange={(e) => setRawConfigOverride(e.target.value)}
              disabled={configs.length === 0}
              className="stat w-full rounded border border-kivski-border bg-kivski-bg px-2 py-1.5 text-xs text-kivski-text outline-none focus:border-kivski-defender"
              aria-label="Raw training config override"
            >
              <option value="">(use goal above)</option>
              {configs.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} — {c.id}
                </option>
              ))}
            </select>
            {goalIsOverridden && (
              <p className="mt-1 text-[10px] leading-tight text-kivski-defender">
                Goal override active — using raw config {rawConfigOverride}.
              </p>
            )}
          </div>
        )}
      </section>

      <section>
        <SectionLabel>Start / Stop</SectionLabel>
        <div className="grid grid-cols-2 gap-1.5">
          <button
            type="button"
            className="btn btn-primary"
            onClick={() =>
              send("start training", {
                type: "start_training",
                configId: effectiveConfigId || undefined,
              })
            }
            disabled={busy !== null || trainingStatus.running}
            title={
              resumeTarget?.available
                ? `Resumes from ${resumeTarget.name ?? resumeTarget.path}`
                : `Starts a fresh training run (${effectiveConfigId})`
            }
          >
            Start
          </button>
          <button
            type="button"
            className="btn btn-danger"
            onClick={() =>
              send("stop training", { type: "stop_training" })
            }
            disabled={busy !== null || !trainingStatus.running}
          >
            Stop
          </button>
        </div>
      </section>

      <section>
        <SectionLabel>Run a fixed number of episodes</SectionLabel>
        <div className="flex items-center gap-2">
          <input
            type="number"
            min={1}
            max={10000}
            value={episodeCount}
            onChange={(e) =>
              setEpisodeCount(
                Math.max(
                  1,
                  Math.min(10000, Number(e.target.value) || 1),
                ),
              )
            }
            className="stat w-20 rounded border border-kivski-border bg-kivski-bg px-2 py-2 text-xs text-kivski-text outline-none focus:border-kivski-defender"
            aria-label="Episode count"
          />
          <button
            type="button"
            className="btn flex-1"
            onClick={() =>
              send(`run ${episodeCount} eps`, {
                type: "run_episodes",
                n: episodeCount,
                configId: effectiveConfigId || undefined,
              })
            }
            disabled={busy !== null}
          >
            Run {episodeCount} eps
          </button>
        </div>
      </section>

      <section>
        <SectionLabel>Load checkpoint</SectionLabel>
        <div className="flex flex-col gap-1.5">
          {checkpoints.length === 0 && (
            <div className="rounded border border-dashed border-kivski-border px-2 py-3 text-center text-[11px] text-kivski-muted">
              No checkpoints on disk yet. Start training to create one.
            </div>
          )}
          <div className="flex max-h-72 flex-col gap-1 overflow-y-auto pr-0.5">
            {checkpoints.map((c) => (
              <CheckpointCard
                key={c.id}
                ckpt={c}
                active={selectedCkpt === c.id}
                onClick={() => setSelectedCkpt(c.id)}
              />
            ))}
          </div>
          <button
            type="button"
            className="btn"
            disabled={!selectedCkpt || busy !== null}
            onClick={() =>
              send("load checkpoint", {
                type: "load_checkpoint",
                id: selectedCkpt,
              })
            }
            aria-label="Load selected checkpoint"
          >
            Load selected
          </button>
          <button
            type="button"
            className="btn"
            onClick={() =>
              send("save checkpoint (auto)", { type: "save_checkpoint" })
            }
            disabled={busy !== null}
            title="Trainer auto-saves periodically — this is a soft hint, not a forced save."
          >
            Save Checkpoint
          </button>
        </div>
      </section>

      {(busy || error) && (
        <div className="rounded border border-kivski-border bg-kivski-panel-2 px-2 py-1.5 text-[10px]">
          {busy && <div className="text-kivski-muted">… {busy}</div>}
          {error && (
            <div className="truncate text-kivski-hp-low" title={error}>
              {error}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ---------- View tab ----------

const ViewTab = () => {
  const uiMode = useStore((s) => s.uiMode);
  const setUiMode = useStore((s) => s.setUiMode);
  const showFov = useStore((s) => s.showFov);
  const showSound = useStore((s) => s.showSound);
  const showComms = useStore((s) => s.showComms);
  const showLastKnown = useStore((s) => s.showLastKnown);
  const showHeatmap = useStore((s) => s.showHeatmap);
  const toggleFov = useStore((s) => s.toggleFov);
  const toggleSound = useStore((s) => s.toggleSound);
  const toggleComms = useStore((s) => s.toggleComms);
  const toggleLastKnown = useStore((s) => s.toggleLastKnown);
  const toggleHeatmap = useStore((s) => s.toggleHeatmap);

  const pick = (mode: UiMode) => () => setUiMode(mode);

  return (
    <div className="flex flex-col gap-4 p-4">
      <section>
        <SectionLabel>Interface density</SectionLabel>
        <div className="flex gap-2">
          <ModeCard
            active={uiMode === "simple"}
            title="Simple"
            blurb="Just the match. Map, score, timer."
            onClick={pick("simple")}
          />
          <ModeCard
            active={uiMode === "advanced"}
            title="Advanced"
            blurb="Inspector, comms, metrics, training panel."
            onClick={pick("advanced")}
          />
        </div>
      </section>

      <section>
        <SectionLabel>Map overlays</SectionLabel>
        <div className="flex flex-col gap-0.5">
          <Toggle
            label="Field of view"
            description="Tint visible area for each agent."
            checked={showFov}
            onChange={toggleFov}
          />
          <Toggle
            label="Sound radii"
            description="Concentric circles for footstep / gunfire noise."
            checked={showSound}
            onChange={toggleSound}
          />
          <Toggle
            label="Comms arrows"
            description="Arrows for speech-bubble events between teammates."
            checked={showComms}
            onChange={toggleComms}
          />
          <Toggle
            label="Last-known positions"
            description="Ghost markers for the last seen position of each enemy."
            checked={showLastKnown}
            onChange={toggleLastKnown}
          />
          <Toggle
            label="Heatmap"
            description="Hot-spots where agents have spent time."
            checked={showHeatmap}
            onChange={toggleHeatmap}
          />
        </div>
        {uiMode === "simple" && (
          <p className="mt-2 text-[10px] leading-tight text-kivski-muted">
            Overlays still render on top of the map even in Simple mode —
            they only stay hidden if their toggle here is off.
          </p>
        )}
      </section>
    </div>
  );
};

// ---------- About tab ----------

const AboutTab = () => {
  const [info, setInfo] = useState<SystemInfoPayload | null>(null);
  const trainingRunning = useStore((s) => s.trainingStatus.running);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await fetch("/api/system/info", {
          headers: { Accept: "application/json" },
        });
        if (!res.ok) return;
        const data = (await res.json()) as SystemInfoPayload;
        if (alive) setInfo(data);
      } catch {
        /* ignore */
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="flex flex-col gap-4 p-4">
      <section>
        <div className="flex items-center gap-2">
          <div className="h-7 w-7 rounded bg-gradient-to-br from-kivski-attacker to-kivski-defender" />
          <div className="leading-tight">
            <div className="text-sm font-semibold text-kivski-text">
              Kivski Tactical AI Simulator
            </div>
            <div className="text-[10px] uppercase tracking-widest text-kivski-muted">
              v{info?.kivski_api_version ?? "0.3.0"}
            </div>
          </div>
        </div>
        <p className="mt-3 text-xs leading-relaxed text-kivski-text">
          A multi-agent reinforcement learning bomb-defuse simulation.
          Watch 5v5 AI agents learn tactics from scratch — taking cover,
          coordinating pushes, planting and defusing the bomb — using a
          PPO trainer and self-play.
        </p>
      </section>

      <section>
        <SectionLabel>Runtime</SectionLabel>
        <div className="flex flex-col gap-1 text-[11px]">
          <Row label="Backend" value={info?.kivski_api_version ?? "—"} />
          <Row label="Simulator" value={info?.kivski_sim_version ?? "—"} />
          <Row label="Torch" value={info?.torch_version ?? "—"} />
          <Row
            label="GPU"
            value={
              info?.cuda_available
                ? info.cuda_device
                  ? `CUDA · ${info.cuda_device}`
                  : "CUDA"
                : "CPU only"
            }
          />
          <Row label="Platform" value={info?.platform ?? "—"} />
          <Row
            label="Training"
            value={trainingRunning ? "running" : "idle"}
          />
        </div>
      </section>

      <section>
        <p className="text-[10px] leading-relaxed text-kivski-muted">
          Tip: switch to <span className="text-kivski-text">Advanced</span>{" "}
          in the View tab to see inspector, comms log, and live training
          metrics. Switch back any time — your debug toggles are kept.
        </p>
      </section>
    </div>
  );
};

const Row = ({ label, value }: { label: string; value: string }) => (
  <div className="flex items-baseline justify-between gap-2">
    <span className="text-kivski-muted">{label}</span>
    <span
      className="stat truncate text-right text-kivski-text"
      title={value}
    >
      {value}
    </span>
  </div>
);

// ---------- Drawer shell ----------

const SettingsDrawer = () => {
  const open = useStore((s) => s.settingsOpen);
  const setOpen = useStore((s) => s.setSettingsOpen);
  const tab = useStore((s) => s.settingsTab);
  const setTab = useStore((s) => s.setSettingsTab);

  // ESC closes the drawer.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  const onClose = () => setOpen(false);

  // We render the drawer container unconditionally and animate `open`
  // via CSS so the slide-in transition is smooth and the DOM doesn't
  // flicker. `pointer-events-none` keeps the inert side from blocking
  // the map.
  return (
    <>
      {/* Backdrop */}
      <div
        aria-hidden={!open}
        onClick={onClose}
        className={`fixed inset-0 z-40 bg-black/40 backdrop-blur-[2px] transition-opacity duration-200 ${
          open
            ? "opacity-100 pointer-events-auto"
            : "opacity-0 pointer-events-none"
        }`}
      />

      {/* Drawer panel */}
      <aside
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
        className={`fixed right-0 top-0 z-50 flex h-full w-[380px] max-w-[92vw] flex-col border-l border-kivski-border bg-kivski-panel shadow-2xl transition-transform duration-200 ease-out ${
          open ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <header className="flex items-center justify-between border-b border-kivski-border px-4 py-3">
          <div className="leading-tight">
            <div className="text-sm font-semibold text-kivski-text">
              Kivski Settings
            </div>
            <div className="text-[10px] uppercase tracking-widest text-kivski-muted">
              Match · Training · View · About
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="close-settings"
            className="btn px-2.5 py-1 text-sm"
            title="Close (Esc)"
          >
            {"✕"}
          </button>
        </header>

        {/* Tabs */}
        <div className="flex border-b border-kivski-border">
          {(
            [
              { id: "match", label: "Match" },
              { id: "training", label: "Training" },
              { id: "view", label: "View" },
              { id: "about", label: "About" },
            ] as { id: SettingsTab; label: string }[]
          ).map((t) => (
            <TabButton
              key={t.id}
              label={t.label}
              active={tab === t.id}
              onClick={() => setTab(t.id)}
            />
          ))}
        </div>

        {/* Body */}
        <div className="min-h-0 flex-1 overflow-y-auto">
          {tab === "match" && <MatchTab onClose={onClose} />}
          {tab === "training" && <TrainingTab />}
          {tab === "view" && <ViewTab />}
          {tab === "about" && <AboutTab />}
        </div>
      </aside>
    </>
  );
};

export default SettingsDrawer;
