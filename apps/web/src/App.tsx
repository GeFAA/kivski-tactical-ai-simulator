import { useEffect, useMemo, useState } from "react";
import MatchHeader from "@/components/MatchHeader";
import LeftSidebar from "@/components/LeftSidebar";
import RightSidebar from "@/components/RightSidebar";
import MapViewer from "@/components/MapViewer";
import BottomControls from "@/components/BottomControls";
import DebugToggles from "@/components/DebugToggles";
import RoundTimeline from "@/components/RoundTimeline";
import TrainingPanel from "@/components/TrainingPanel";
import SettingsDrawer from "@/components/SettingsDrawer";
import AgentDetailModal from "@/components/AgentDetailModal";
import {
  getTrainingGoalSpec,
  getTrainingStatus,
  subscribeMatch,
} from "@/lib/api-client";
import { formatDuration } from "@/lib/format";
import { useStore } from "@/lib/store";

/**
 * Subtle bottom-right pill rendered in Simple mode while training is
 * running. Clicking opens the settings drawer on the Training tab so a
 * curious user can see what's going on without learning the rest of the
 * Advanced UI. Stays hidden when training is idle (no clutter).
 *
 * Surfaces the active training goal title + the latest win-rate vs the
 * random baseline so a glance at the pill tells the user both *what*
 * the trainer is optimising for and *how it's going* — without having
 * to open the drawer.
 */
const TrainingPill = () => {
  const uiMode = useStore((s) => s.uiMode);
  const running = useStore((s) => s.trainingStatus.running);
  const episode = useStore((s) => s.trainingStatus.episode);
  const currentSessionSeconds = useStore(
    (s) => s.trainingStatus.currentSessionSeconds,
  );
  const totalTrainedSeconds = useStore(
    (s) => s.trainingStatus.totalTrainedSeconds,
  );
  const trainingGoal = useStore((s) => s.trainingGoal);
  const metricsHistory = useStore((s) => s.metricsHistory);
  const setSettingsOpen = useStore((s) => s.setSettingsOpen);
  const setSettingsTab = useStore((s) => s.setSettingsTab);

  // Latest non-null WR vs random for the trailing chip.
  const latestWr = useMemo(() => {
    for (let i = metricsHistory.length - 1; i >= 0; i--) {
      const v = metricsHistory[i].winrateVsRandom;
      if (typeof v === "number" && Number.isFinite(v)) return v;
    }
    return undefined;
  }, [metricsHistory]);

  if (uiMode !== "simple" || !running) return null;

  const spec = getTrainingGoalSpec(trainingGoal);
  // Prefer the running-session time; fall back to the cumulative
  // counter for the rare moment between trainer-start and the first
  // clock tick when only the total is known.
  const sessionLabel =
    typeof currentSessionSeconds === "number" && currentSessionSeconds > 0
      ? formatDuration(currentSessionSeconds)
      : null;
  const totalLabel =
    typeof totalTrainedSeconds === "number"
      ? formatDuration(totalTrainedSeconds)
      : null;
  const tooltip = [
    `Goal: ${spec.title}`,
    totalLabel ? `Total trained: ${totalLabel}` : null,
    sessionLabel ? `Current session: ${sessionLabel}` : null,
    "Click to open the Training panel.",
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <button
      type="button"
      onClick={() => {
        setSettingsTab("training");
        setSettingsOpen(true);
      }}
      className="fixed bottom-20 right-4 z-30 inline-flex items-center gap-2 rounded-full border border-kivski-defender/40 bg-kivski-panel/95 px-3 py-1.5 text-[11px] text-kivski-text shadow-lg backdrop-blur transition-colors hover:border-kivski-defender hover:bg-kivski-panel"
      title={tooltip}
      aria-label={`Training running: ${spec.title}, episode ${episode}${
        sessionLabel ? `, ${sessionLabel}` : ""
      }`}
    >
      <span className="inline-block h-1.5 w-1.5 rounded-full bg-kivski-hp animate-pulse-slow" />
      <span className="font-medium text-kivski-defender">Training</span>
      {sessionLabel && (
        <>
          <span className="stat text-kivski-text">{sessionLabel}</span>
          <span className="text-kivski-muted">·</span>
        </>
      )}
      <span className="stat text-kivski-muted">ep {episode}</span>
      {typeof latestWr === "number" && (
        <>
          <span className="text-kivski-muted">·</span>
          <span className="stat text-kivski-text" aria-label={`Win rate vs random: ${latestWr.toFixed(2)}`}>
            WR {latestWr.toFixed(2)}
          </span>
        </>
      )}
    </button>
  );
};

const ONBOARDED_KEY = "kivski-onboarded";

/**
 * First-visit teaching moment: a subtle balloon pointing the user at
 * the settings gear so they discover Advanced mode + training control.
 * Auto-dismisses after the user clicks "Got it", or whenever they open
 * the drawer themselves. localStorage-flagged so it never repeats.
 */
const OnboardingTooltip = () => {
  const uiMode = useStore((s) => s.uiMode);
  const settingsOpen = useStore((s) => s.settingsOpen);
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (uiMode !== "simple") return;
    try {
      const seen = window.localStorage.getItem(ONBOARDED_KEY);
      if (seen === "true") return;
    } catch {
      return;
    }
    // Slight delay so the bubble doesn't pop in at the same instant the
    // header finishes its first paint.
    const t = window.setTimeout(() => setShow(true), 800);
    return () => window.clearTimeout(t);
  }, [uiMode]);

  // Auto-dismiss the moment the user opens the drawer on their own.
  useEffect(() => {
    if (!settingsOpen || !show) return;
    setShow(false);
    try {
      window.localStorage.setItem(ONBOARDED_KEY, "true");
    } catch {
      /* ignore */
    }
  }, [settingsOpen, show]);

  if (!show) return null;

  const dismiss = () => {
    setShow(false);
    try {
      window.localStorage.setItem(ONBOARDED_KEY, "true");
    } catch {
      /* ignore */
    }
  };

  return (
    <div
      className="fixed right-3 top-16 z-30 w-72 rounded-lg border border-kivski-defender/40 bg-kivski-panel/95 p-3 text-xs shadow-2xl backdrop-blur"
      role="dialog"
      aria-label="Welcome tip"
    >
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="text-[10px] uppercase tracking-widest text-kivski-defender">
          Welcome to Kivski
        </span>
        <button
          type="button"
          onClick={dismiss}
          className="text-kivski-muted hover:text-kivski-text"
          aria-label="dismiss-onboarding"
        >
          {"✕"}
        </button>
      </div>
      <p className="leading-relaxed text-kivski-text">
        Click the gear icon on the top-right for{" "}
        <span className="font-semibold text-kivski-defender">Advanced</span>{" "}
        mode, training controls, and view options.
      </p>
      <div className="mt-2 flex justify-end">
        <button
          type="button"
          onClick={dismiss}
          className="btn px-3 py-1 text-[11px]"
        >
          Got it
        </button>
      </div>
    </div>
  );
};

const App = () => {
  const setConnected = useStore((s) => s.setConnected);
  const setCurrentMatchId = useStore((s) => s.setCurrentMatchId);
  const setMatchSnapshot = useStore((s) => s.setMatchSnapshot);
  const pushEvent = useStore((s) => s.pushEvent);
  const pushMessage = useStore((s) => s.pushMessage);
  const setInspection = useStore((s) => s.setInspection);
  const setMapName = useStore((s) => s.setMapName);
  const setAttentionWeights = useStore((s) => s.setAttentionWeights);
  const setTrainingStatus = useStore((s) => s.setTrainingStatus);
  const pushMetricsSample = useStore((s) => s.pushMetricsSample);
  const pushRoundResult = useStore((s) => s.pushRoundResult);
  const setCurrentPolicies = useStore((s) => s.setCurrentPolicies);
  const setAutoReload = useStore((s) => s.setAutoReload);
  const pushPolicyReload = useStore((s) => s.pushPolicyReload);
  // `matchToken` is incremented by `MatchSetupModal` after POSTing a new
  // comparison match — it forces this effect to re-run, which tears down
  // the current WebSocket and opens a fresh one against the new match.
  const matchToken = useStore((s) => s.matchToken);
  // Top-level UI density: drives whether the right sidebar, training
  // panel, round timeline, and debug-toggles overlay are shown.
  const uiMode = useStore((s) => s.uiMode);

  // Low-frequency poll of /api/training/status so the simple-mode
  // training pill knows when a job is running. TrainingPanel does its
  // own poll when mounted, but it lives only in Advanced mode — without
  // this fallback poll, a user in Simple mode would never see the pill
  // (the running flag stays false until the first WS metrics_sample
  // frame, which can lag by tens of seconds in turbo configs).
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      const s = await getTrainingStatus();
      if (!alive || !s) return;
      setTrainingStatus({
        running: s.running,
        episode: s.episode,
        totalEpisodes: s.totalEpisodes,
        totalTrainedSeconds: s.totalTrainedSeconds,
        currentSessionSeconds: s.currentSessionSeconds,
      });
    };
    void tick();
    const id = window.setInterval(tick, 5_000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [setTrainingStatus]);

  // Wire up the live match WebSocket once at mount. The handle's `.close()`
  // tears down the reconnect loop on hot-reload / unmount.
  useEffect(() => {
    const handle = subscribeMatch({
      onStatus: (status) => setConnected(status === "open"),
      onMatchId: (id) => setCurrentMatchId(id),
      onPolicies: (yellow, blue) => setCurrentPolicies({ yellow, blue }),
      onAutoReload: (yellow, blue) => setAutoReload({ yellow, blue }),
      onFrame: (frame) => {
        switch (frame.type) {
          case "hello":
            setMapName(frame.data.mapName);
            break;
          case "map_info":
            setMapName(frame.data.mapName);
            break;
          case "snapshot":
            setMatchSnapshot(frame.data);
            break;
          case "event":
            pushEvent(frame.data);
            break;
          case "message":
            pushMessage(frame.data);
            break;
          case "inspect":
            setInspection(frame.data);
            break;
          case "attention_update":
            setAttentionWeights(frame.data);
            break;
          case "training_status":
            setTrainingStatus(frame.data);
            // Mirror the metric fields into the metrics stream so the
            // sparklines get a fresh data point on every WS push, even
            // when the trainer is between PPO updates and re-emits the
            // same numeric values. Otherwise React's dependency array
            // dedups identical values and the sparkline sits at 1 point.
            if (
              typeof frame.data.policyLoss === "number" ||
              typeof frame.data.valueLoss === "number" ||
              typeof frame.data.entropy === "number"
            ) {
              pushMetricsSample({
                episode: frame.data.episode ?? 0,
                policyLoss: frame.data.policyLoss,
                valueLoss: frame.data.valueLoss,
                entropy: frame.data.entropy,
              });
              pushEvent({
                id: `train-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
                ts: Date.now(),
                tick: 0,
                kind: "info",
                text: `Training update — ep ${frame.data.episode ?? 0}, loss=${(frame.data.policyLoss ?? 0).toFixed(4)}, entropy=${(frame.data.entropy ?? 0).toFixed(2)}`,
              });
            }
            break;
          case "metrics_sample":
            pushMetricsSample(frame.data);
            break;
          case "round_result":
            pushRoundResult(frame.data);
            break;
          case "match_done":
            // Engine signals the match is finished; the store keeps the
            // last snapshot, the api-client auto-reconnects with a new
            // match on the next loop iteration.

            console.warn("[kivski] match_done:", frame.matchId ?? "(unknown id)");
            break;
          case "policy_reload":
            // Per-round auto-reload swapped one side's checkpoint;
            // mirror the new label into `currentPolicies` and seed
            // the transient toast.
            pushPolicyReload({
              side: frame.data.side,
              name: frame.data.name,
              previous: frame.data.previous ?? null,
            });
            // Surface as a generic info event too so it appears in the
            // event feed alongside kills / plants / defuses.
            pushEvent({
              id: `policy-reload-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
              ts: Date.now(),
              tick: 0,
              kind: "info",
              text: `${frame.data.side === "yellow" ? "Yellow" : "Blue"} hot-swapped to ${frame.data.name}`,
            });
            break;
          case "pong":
          case "ack":
            // Control acks aren't user-visible; nothing to do.
            break;
          case "error":

            console.warn("[kivski] server error frame:", frame.data.message);
            break;
        }
      },
    });
    return () => handle.close();
  }, [
    matchToken,
    setConnected,
    setCurrentMatchId,
    setCurrentPolicies,
    setMatchSnapshot,
    pushEvent,
    pushMessage,
    setInspection,
    setMapName,
    setAttentionWeights,
    setTrainingStatus,
    pushMetricsSample,
    pushRoundResult,
    pushPolicyReload,
    setAutoReload,
  ]);

  const isAdvanced = uiMode === "advanced";

  // Grid template:
  //   Advanced: 3 columns (sidebar | map | right tabs) — power layout
  //   Simple  : 2 columns (slim sidebar | map) — clean, map-first
  const gridCols = isAdvanced
    ? "grid-cols-[18rem_minmax(0,1fr)_22rem]"
    : "grid-cols-[14rem_minmax(0,1fr)]";

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-kivski-bg text-kivski-text">
      {/* Header: round/score/phase (compact in Simple, full in Advanced) */}
      <MatchHeader />

      {/* Body */}
      <div className={`grid min-h-0 flex-1 gap-2 p-2 ${gridCols}`}>
        <LeftSidebar />

        <div className="flex min-h-0 min-w-0 flex-col gap-2">
          <div className="panel relative flex min-h-0 flex-1 overflow-hidden">
            <MapViewer />
            {isAdvanced && (
              <div className="pointer-events-none absolute right-2 top-2">
                <div className="pointer-events-auto">
                  <DebugToggles />
                </div>
              </div>
            )}
          </div>
          {isAdvanced && <RoundTimeline />}
          {isAdvanced && <TrainingPanel />}
        </div>

        {isAdvanced && <RightSidebar />}
      </div>

      {/* Footer: playback + training controls (slim in Simple mode) */}
      <BottomControls />

      {/* Global overlays — always mounted */}
      <SettingsDrawer />
      <AgentDetailModal />
      <TrainingPill />
      <OnboardingTooltip />
    </div>
  );
};

export default App;
