import { useEffect, useState } from "react";
import {
  createMatch,
  getRecommendedPolicies,
  type PolicyOption,
} from "@/lib/api-client";
import { useStore } from "@/lib/store";

/**
 * Comparison-match setup modal.
 *
 * Picks two policies (one per team) from the curated list returned by
 * ``GET /api/checkpoints/recommended`` and starts a fresh match via
 * ``POST /api/match/new`` with `policy_yellow` / `policy_blue` set.
 *
 * The frontend then reconnects to the new match id via `setMatchToken`,
 * which forces `App.tsx`'s WebSocket effect to tear down the existing
 * subscription and re-handshake. Snapshot decode, training metrics, etc.
 * resume seamlessly — only the match id (and thus the active policies)
 * change.
 *
 * Defense-in-depth: if `/api/checkpoints/recommended` is unavailable
 * (e.g. backend not yet on v0.3) the api-client falls back to a
 * hardcoded baseline list, so the modal is always usable.
 */

interface MatchSetupModalProps {
  open: boolean;
  onClose: () => void;
}

/**
 * Sentinel used when no policy is selected. Mapped to `null` before
 * sending to the backend so the server can apply its default
 * (auto-pick latest checkpoint).
 */
const AUTO_VALUE = "__auto__";

/**
 * Decide whether the auto-reload toggle is meaningful for a given
 * policy selection. We allow it when:
 *   - the user picked AUTO (server resolves to "latest" when a ckpt
 *     exists), or
 *   - the user explicitly picked the "latest" or "best" id, or
 *   - the option's kind is "checkpoint" (named snapshot).
 * Random / scripted baselines have nothing to reload, so the checkbox
 * stays disabled and unchecked.
 */
const supportsAutoReload = (
  selectedId: string,
  options: readonly PolicyOption[],
): boolean => {
  if (selectedId === AUTO_VALUE) return true;
  const lowered = selectedId.toLowerCase();
  if (lowered === "latest" || lowered === "best") return true;
  const opt = options.find((o) => o.id === selectedId) as
    | (PolicyOption & { kind?: string })
    | undefined;
  if (opt && typeof opt.kind === "string" && opt.kind === "checkpoint") {
    return true;
  }
  return false;
};

const MatchSetupModal = ({ open, onClose }: MatchSetupModalProps) => {
  const setMatchToken = useStore((s) => s.setMatchToken);
  const setCurrentPolicies = useStore((s) => s.setCurrentPolicies);
  const setAutoReload = useStore((s) => s.setAutoReload);

  const [options, setOptions] = useState<PolicyOption[]>([]);
  const [yellow, setYellow] = useState<string>(AUTO_VALUE);
  const [blue, setBlue] = useState<string>(AUTO_VALUE);
  const [autoReloadYellow, setAutoReloadYellow] = useState<boolean>(true);
  const [autoReloadBlue, setAutoReloadBlue] = useState<boolean>(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Fetch options whenever the modal becomes visible. Cached implicitly
  // by browser HTTP cache headers; keeping the fetch on-open avoids
  // a startup hit when the modal is never used.
  useEffect(() => {
    if (!open) return;
    let alive = true;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const opts = await getRecommendedPolicies();
        if (!alive) return;
        setOptions(opts);
      } catch (err) {
        if (!alive) return;
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [open]);

  // ESC closes the modal.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose, busy]);

  if (!open) return null;

  const resolveOption = (id: string): PolicyOption | null => {
    if (id === AUTO_VALUE) return null;
    return options.find((o) => o.id === id) ?? { id, name: id };
  };

  const yellowSupportsReload = supportsAutoReload(yellow, options);
  const blueSupportsReload = supportsAutoReload(blue, options);

  const onStart = async () => {
    setBusy(true);
    setError(null);
    const body: {
      policy_yellow?: string;
      policy_blue?: string;
      auto_reload_yellow?: boolean;
      auto_reload_blue?: boolean;
    } = {};
    if (yellow !== AUTO_VALUE) body.policy_yellow = yellow;
    if (blue !== AUTO_VALUE) body.policy_blue = blue;
    // Only forward auto_reload=true when the selected policy can
    // actually use it; otherwise the backend would normalise it back
    // to false and the request would be misleading.
    if (yellowSupportsReload && autoReloadYellow) body.auto_reload_yellow = true;
    if (blueSupportsReload && autoReloadBlue) body.auto_reload_blue = true;
    try {
      const result = await createMatch(body);
      // Optimistically mirror what the user selected; the WS handshake
      // will overwrite these with the authoritative
      // `policy_*_name` from the backend response when it arrives.
      const yOpt = resolveOption(yellow);
      const bOpt = resolveOption(blue);
      setCurrentPolicies({
        yellow:
          result.policy_yellow_name && result.policy_yellow
            ? { id: result.policy_yellow, name: result.policy_yellow_name }
            : yOpt,
        blue:
          result.policy_blue_name && result.policy_blue
            ? { id: result.policy_blue, name: result.policy_blue_name }
            : bOpt,
      });
      // Mirror the *effective* auto-reload flags from the server so
      // the header badge knows whether to render the "auto" suffix.
      setAutoReload({
        yellow: Boolean(result.auto_reload_yellow),
        blue: Boolean(result.auto_reload_blue),
      });
      // Bump the match token to trigger App.tsx's WS effect to
      // re-subscribe against the new match id.
      setMatchToken();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 backdrop-blur-sm"
      onClick={() => {
        if (!busy) onClose();
      }}
    >
      <div
        className="panel mt-20 w-full max-w-lg p-4"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between border-b border-kivski-border pb-2">
          <div>
            <div className="text-sm font-semibold text-kivski-text">
              Comparison Match
            </div>
            <div className="text-[10px] uppercase tracking-widest text-kivski-muted">
              Pick a policy for each team
            </div>
          </div>
          <button
            type="button"
            className="btn"
            onClick={onClose}
            disabled={busy}
            aria-label="Close"
          >
            ✕
          </button>
        </header>

        <div className="mt-3 grid grid-cols-2 gap-3">
          {/* Yellow */}
          <div className="flex flex-col gap-1">
            <label className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-kivski-muted">
              <span className="inline-block h-2 w-2 rounded-full bg-kivski-attacker" />
              Yellow Team
            </label>
            <select
              value={yellow}
              onChange={(e) => setYellow(e.target.value)}
              className="stat rounded border border-kivski-border bg-kivski-bg px-2 py-1.5 text-xs text-kivski-text outline-none focus:border-kivski-attacker"
              disabled={loading || busy}
            >
              <option value={AUTO_VALUE}>Auto (latest checkpoint)</option>
              {options.map((o) => (
                <option key={`y-${o.id}`} value={o.id}>
                  {o.name}
                </option>
              ))}
            </select>
            <label
              className={`flex items-center gap-1.5 pt-1 text-[10px] ${
                yellowSupportsReload
                  ? "cursor-pointer text-kivski-text"
                  : "cursor-not-allowed text-kivski-muted/60"
              }`}
              title={
                yellowSupportsReload
                  ? "Hot-swap the policy to the newest checkpoint at the end of every round"
                  : "Only available for trained-checkpoint policies"
              }
            >
              <input
                type="checkbox"
                checked={yellowSupportsReload && autoReloadYellow}
                onChange={(e) => setAutoReloadYellow(e.target.checked)}
                disabled={!yellowSupportsReload || loading || busy}
                className="accent-kivski-attacker"
              />
              <span>Auto-reload latest each round</span>
            </label>
          </div>

          {/* Blue */}
          <div className="flex flex-col gap-1">
            <label className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-kivski-muted">
              <span className="inline-block h-2 w-2 rounded-full bg-kivski-defender" />
              Blue Team
            </label>
            <select
              value={blue}
              onChange={(e) => setBlue(e.target.value)}
              className="stat rounded border border-kivski-border bg-kivski-bg px-2 py-1.5 text-xs text-kivski-text outline-none focus:border-kivski-defender"
              disabled={loading || busy}
            >
              <option value={AUTO_VALUE}>Auto (latest checkpoint)</option>
              {options.map((o) => (
                <option key={`b-${o.id}`} value={o.id}>
                  {o.name}
                </option>
              ))}
            </select>
            <label
              className={`flex items-center gap-1.5 pt-1 text-[10px] ${
                blueSupportsReload
                  ? "cursor-pointer text-kivski-text"
                  : "cursor-not-allowed text-kivski-muted/60"
              }`}
              title={
                blueSupportsReload
                  ? "Hot-swap the policy to the newest checkpoint at the end of every round"
                  : "Only available for trained-checkpoint policies"
              }
            >
              <input
                type="checkbox"
                checked={blueSupportsReload && autoReloadBlue}
                onChange={(e) => setAutoReloadBlue(e.target.checked)}
                disabled={!blueSupportsReload || loading || busy}
                className="accent-kivski-defender"
              />
              <span>Auto-reload latest each round</span>
            </label>
          </div>
        </div>

        {loading && (
          <div className="mt-3 text-[10px] text-kivski-muted">
            Loading policy options…
          </div>
        )}

        {error && (
          <div className="mt-3 rounded border border-kivski-hp-low/40 bg-kivski-hp-low/10 px-2 py-1 text-[11px] text-kivski-hp-low">
            {error}
          </div>
        )}

        <footer className="mt-4 flex items-center justify-between gap-2">
          <span className="text-[10px] text-kivski-muted">
            Starts a fresh match and reconnects the viewer.
          </span>
          <div className="flex gap-2">
            <button
              type="button"
              className="btn"
              onClick={onClose}
              disabled={busy}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={onStart}
              disabled={loading || busy}
            >
              {busy ? "Starting…" : "Start Comparison Match"}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default MatchSetupModal;
