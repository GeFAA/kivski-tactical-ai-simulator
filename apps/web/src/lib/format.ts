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
  const WEEK = 7 * 86400;
  const weeks = Math.floor(total / WEEK);
  const days = Math.floor((total % WEEK) / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (weeks > 0) {
    return days > 0 ? `${weeks}w ${days}d` : `${weeks}w`;
  }
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

/**
 * Format a large count as a compact human-readable string with SI-ish
 * suffixes that match how engineers eyeball big numbers:
 *
 *   < 1 000          → "342"           (no suffix)
 *   < 1 000 000      → "12.4k"         (kilo)
 *   < 1 000 000 000  → "3.7M"          (mega)
 *   else             → "2.1B" / "5.6T" (giga / tera)
 *
 * Always keeps one decimal for the leading magnitude so a count like
 * 12 345 reads as "12.3k" rather than "12k" — except for tiny values
 * where the integer reads better ("342" not "342.0").
 *
 * Returns ``"—"`` for non-finite / negative input so callers don't
 * have to guard every site.
 */
export function formatCompactNumber(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "—";
  if (value < 1_000) return Math.round(value).toString();
  const abs = Math.abs(value);
  const tiers: Array<[number, string]> = [
    [1_000_000_000_000, "T"],
    [1_000_000_000, "B"],
    [1_000_000, "M"],
    [1_000, "k"],
  ];
  for (const [magnitude, suffix] of tiers) {
    if (abs >= magnitude) {
      const scaled = value / magnitude;
      // 1 decimal for the leading magnitude (e.g. 12.4k), but drop the
      // decimal once the integer part hits triple digits so the label
      // stays short (e.g. 248k, not 248.3k).
      const decimals = scaled >= 100 ? 0 : 1;
      return `${scaled.toFixed(decimals)}${suffix}`;
    }
  }
  return Math.round(value).toString();
}
