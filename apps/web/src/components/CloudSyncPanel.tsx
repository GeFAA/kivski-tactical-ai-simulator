import { useCallback, useEffect, useRef, useState } from "react";
import {
  formatTimeAgo,
  getCloudStatus,
  pullAndLoadCloudCheckpoint,
  pullCloudCheckpoint,
  type CloudStatusInfo,
} from "@/lib/api-client";

/**
 * Cloud Sync panel mounted inside the Settings drawer's Training tab.
 *
 * Talks to ``/api/cloud/*`` to:
 *   - poll the latest remote checkpoint (HF Hub repo set via env vars),
 *   - one-click pull the bytes into ``models/checkpoints/cloud/``,
 *   - optionally load the pulled file as the active viewer policy.
 *
 * Auto-refreshes the status every 60 s while mounted. Visually mirrors
 * the SettingsDrawer's panel-2 / rounded-border idiom.
 */

const REFRESH_INTERVAL_MS = 60_000;

const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <div className="mb-2 text-[10px] uppercase tracking-widest text-kivski-muted">
    {children}
  </div>
);

const Row = ({ label, value }: { label: string; value: React.ReactNode }) => (
  <div className="flex items-baseline justify-between gap-2 text-[11px]">
    <span className="text-kivski-muted">{label}</span>
    <span className="stat truncate text-right text-kivski-text">{value}</span>
  </div>
);

const CloudSyncPanel = () => {
  const [status, setStatus] = useState<CloudStatusInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<string | null>(null);
  const aliveRef = useRef(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await getCloudStatus();
      if (aliveRef.current) setStatus(s);
    } catch (err) {
      if (aliveRef.current) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (aliveRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    aliveRef.current = true;
    void refresh();
    const id = window.setInterval(() => {
      void refresh();
    }, REFRESH_INTERVAL_MS);
    return () => {
      aliveRef.current = false;
      window.clearInterval(id);
    };
  }, [refresh]);

  const onPull = async () => {
    setBusy("pull");
    setError(null);
    setConfirm(null);
    try {
      const r = await pullCloudCheckpoint();
      setConfirm(`Pulled ${r.name}`);
      window.setTimeout(() => setConfirm(null), 2_500);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const onPullAndLoad = async () => {
    setBusy("pull-and-load");
    setError(null);
    setConfirm(null);
    try {
      const r = await pullAndLoadCloudCheckpoint();
      setConfirm(`Loaded ${r.name}`);
      window.setTimeout(() => setConfirm(null), 2_500);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="rounded border border-kivski-border bg-kivski-bg/40 p-3">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-[12px] font-semibold text-kivski-text">
          {"☁️ Cloud Sync"}
        </div>
        <span className="text-[9px] uppercase tracking-widest text-kivski-muted">
          HF Hub
        </span>
      </div>

      {status === null && loading && (
        <div className="text-[11px] text-kivski-muted">Loading…</div>
      )}

      {status !== null && !status.configured && (
        <div className="rounded border border-dashed border-kivski-border bg-kivski-panel-2/50 px-2 py-2 text-[11px] leading-snug text-kivski-muted">
          <p className="text-kivski-text">Not configured.</p>
          <p className="mt-1">
            Set <code className="stat text-kivski-defender">HF_TOKEN</code> and{" "}
            <code className="stat text-kivski-defender">KIVSKI_HF_REPO</code> on
            the backend, then restart the API.
          </p>
          <p className="mt-1">
            See{" "}
            <a
              href="./docker/README.md"
              className="text-kivski-defender underline hover:text-kivski-text"
              target="_blank"
              rel="noreferrer"
            >
              docker/README.md
            </a>{" "}
            for details.
          </p>
        </div>
      )}

      {status !== null && status.configured && (
        <div className="flex flex-col gap-2">
          <SectionLabel>Remote</SectionLabel>
          <div className="rounded border border-kivski-border bg-kivski-panel-2 px-2 py-1.5">
            <Row label="Repo" value={status.repo_id ?? "—"} />
            <Row
              label="Last local pull"
              value={
                status.last_pull
                  ? formatTimeAgo(status.last_pull) || "just now"
                  : "never"
              }
            />
          </div>

          <SectionLabel>Latest checkpoint</SectionLabel>
          {status.latest_checkpoint ? (
            <div className="rounded border border-kivski-border bg-kivski-panel-2 px-2 py-1.5">
              <Row
                label="Name"
                value={
                  <span title={status.latest_checkpoint.name}>
                    {status.latest_checkpoint.name}
                  </span>
                }
              />
              <Row
                label="Uploaded"
                value={
                  status.latest_checkpoint.uploaded_at
                    ? formatTimeAgo(status.latest_checkpoint.uploaded_at) ||
                      "just now"
                    : "—"
                }
              />
              {status.metrics_summary && (
                <>
                  <Row
                    label="Episode"
                    value={String(status.metrics_summary.episode)}
                  />
                  <Row
                    label="Score"
                    value={status.metrics_summary.score.toFixed(3)}
                  />
                </>
              )}
            </div>
          ) : (
            <div className="rounded border border-dashed border-kivski-border px-2 py-3 text-center text-[11px] text-kivski-muted">
              {status.error
                ? `HF API error: ${status.error}`
                : "No checkpoints found in remote repo."}
            </div>
          )}

          <div className="grid grid-cols-3 gap-1.5">
            <button
              type="button"
              className="btn px-2 py-1.5 text-[11px]"
              onClick={() => void refresh()}
              disabled={loading || busy !== null}
              title="Re-fetch cloud status"
            >
              Refresh
            </button>
            <button
              type="button"
              className="btn px-2 py-1.5 text-[11px]"
              onClick={() => void onPull()}
              disabled={busy !== null || !status.latest_checkpoint}
              title="Download the latest checkpoint into models/checkpoints/cloud/"
            >
              Pull latest
            </button>
            <button
              type="button"
              className="btn btn-primary px-2 py-1.5 text-[11px]"
              onClick={() => void onPullAndLoad()}
              disabled={busy !== null || !status.latest_checkpoint}
              title="Pull and immediately load for the active viewer match"
            >
              Pull &amp; Load
            </button>
          </div>
        </div>
      )}

      {(busy || error || confirm) && (
        <div className="mt-2 rounded border border-kivski-border bg-kivski-panel-2 px-2 py-1.5 text-[10px]">
          {busy && <div className="text-kivski-muted">… {busy}</div>}
          {confirm && <div className="text-kivski-defender">{confirm}</div>}
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

export default CloudSyncPanel;
