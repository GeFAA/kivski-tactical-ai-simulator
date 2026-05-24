/**
 * Shared string-formatting helpers used by the viewer UI.
 *
 * Keep these pure (no React, no DOM, no I/O) so they can be unit-tested
 * in isolation and reused freely between components.
 */

/**
 * Format a duration (in seconds) as a compact human-readable string:
 *
 *   < 60s             → "45s"
 *   < 1h              → "12m 30s" / "12m" (drops 0s tail)
 *   < 1d              → "2h 15m"  / "2h"  (drops 0m tail)
 *   >= 1d             → "1d 4h"   / "3d"
 *
 * Always rounds *down* — the goal is to avoid surfacing a "1h 0m"
 * label as "1h 1m" purely because of sub-second drift on a poll cycle.
 *
 * NaN / negative / non-finite inputs return ``"—"`` so the caller can
 * safely render the result without guarding every call site.
 */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const total = Math.floor(seconds);
  if (total < 60) return `${total}s`;
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (days > 0) {
    return hours > 0 ? `${days}d ${hours}h` : `${days}d`;
  }
  if (hours > 0) {
    return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
  }
  // < 1h: include seconds tail when minutes are small so a fresh
  // session reads "12m 30s" rather than just "12m".
  return secs > 0 ? `${minutes}m ${secs}s` : `${minutes}m`;
}
