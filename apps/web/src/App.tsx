import { useEffect } from "react";
import MatchHeader from "@/components/MatchHeader";
import LeftSidebar from "@/components/LeftSidebar";
import RightSidebar from "@/components/RightSidebar";
import MapViewer from "@/components/MapViewer";
import BottomControls from "@/components/BottomControls";
import DebugToggles from "@/components/DebugToggles";
import RoundTimeline from "@/components/RoundTimeline";
import TrainingPanel from "@/components/TrainingPanel";
import { subscribeMatch } from "@/lib/api-client";
import { useStore } from "@/lib/store";

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
  // `matchToken` is incremented by `MatchSetupModal` after POSTing a new
  // comparison match — it forces this effect to re-run, which tears down
  // the current WebSocket and opens a fresh one against the new match.
  const matchToken = useStore((s) => s.matchToken);

  // Wire up the live match WebSocket once at mount. The handle's `.close()`
  // tears down the reconnect loop on hot-reload / unmount.
  useEffect(() => {
    const handle = subscribeMatch({
      onStatus: (status) => setConnected(status === "open"),
      onMatchId: (id) => setCurrentMatchId(id),
      onPolicies: (yellow, blue) => setCurrentPolicies({ yellow, blue }),
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
  ]);

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-kivski-bg text-kivski-text">
      {/* Header: round/score/phase */}
      <MatchHeader />

      {/* Body: 3-column layout */}
      <div className="grid min-h-0 flex-1 grid-cols-[18rem_minmax(0,1fr)_22rem] gap-2 p-2">
        <LeftSidebar />

        <div className="flex min-h-0 min-w-0 flex-col gap-2">
          <div className="panel relative flex min-h-0 flex-1 overflow-hidden">
            <MapViewer />
            <div className="pointer-events-none absolute right-2 top-2">
              <div className="pointer-events-auto">
                <DebugToggles />
              </div>
            </div>
          </div>
          <RoundTimeline />
          <TrainingPanel />
        </div>

        <RightSidebar />
      </div>

      {/* Footer: playback + training controls */}
      <BottomControls />
    </div>
  );
};

export default App;
