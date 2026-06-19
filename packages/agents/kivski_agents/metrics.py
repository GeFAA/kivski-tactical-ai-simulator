"""Typed aggregate metrics that the trainer hands off to telemetry.

The goal of this module is to provide a stable schema for every metric we
emit. Sinks (CSV, TensorBoard, W&B) all consume flat ``dict[str, float]``
payloads, so the dataclasses defined here are paired with
``*_to_dict`` helpers that produce a sink-friendly representation.

Keeping these structures separate from the telemetry sinks lets us:

    * Unit-test metric construction independently from I/O.
    * Re-use the same struct for in-process consumers (e.g. the FastAPI
      live viewer) that do not want a CSV row.
    * Document the expected key set so dashboards can be built against a
      known surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "EpisodeStats",
    "TrainStepMetrics",
    "CommUsageStats",
    "episode_stats_to_dict",
    "train_metrics_to_dict",
    "comm_usage_to_dict",
]


# ---------------------------------------------------------------------------
# Episode-level outcome and aggregate stats
# ---------------------------------------------------------------------------


@dataclass
class EpisodeStats:
    """Per-episode (a full match) aggregate stats."""

    episode: int
    match_done: bool
    yellow_score: int
    blue_score: int
    winner: str  # "yellow" | "blue" | "draw"
    total_rounds: int
    avg_round_duration_ticks: float
    # Sum of survivors across all round summaries (a per-round "alive count"
    # aggregate). Historically named ``total_kills`` but the formula is
    # ``sum(survivors_yellow + survivors_blue)`` — never tracked kills.
    total_survivors: int
    total_deaths: int
    bombs_planted: int
    bombs_defused: int
    bombs_detonated: int
    total_rewards_yellow: float
    total_rewards_blue: float
    timestamp: float


def episode_stats_to_dict(s: EpisodeStats) -> dict[str, float]:
    """Flatten :class:`EpisodeStats` to a sink-friendly ``dict[str, float]``.

    ``winner`` is encoded as a numeric label (yellow=1, blue=-1, draw=0)
    so it can be plotted on a scalar axis. The string winner is preserved
    on the dataclass for human-readable logs.
    """
    return {
        "episode/episode": float(s.episode),
        "episode/match_done": float(bool(s.match_done)),
        "episode/yellow_score": float(s.yellow_score),
        "episode/blue_score": float(s.blue_score),
        "episode/winner_code": float(_winner_code(s.winner)),
        "episode/total_rounds": float(s.total_rounds),
        "episode/avg_round_duration_ticks": float(s.avg_round_duration_ticks),
        # Emit both names so existing CSV / TensorBoard / W&B dashboards keep
        # working while consumers migrate. TODO: drop ``episode/total_kills``
        # alias once downstream dashboards reference ``episode/total_survivors``.
        "episode/total_survivors": float(s.total_survivors),
        "episode/total_kills": float(s.total_survivors),
        "episode/total_deaths": float(s.total_deaths),
        "episode/bombs_planted": float(s.bombs_planted),
        "episode/bombs_defused": float(s.bombs_defused),
        "episode/bombs_detonated": float(s.bombs_detonated),
        "episode/total_rewards_yellow": float(s.total_rewards_yellow),
        "episode/total_rewards_blue": float(s.total_rewards_blue),
        "episode/timestamp": float(s.timestamp),
    }


def _winner_code(winner: str) -> int:
    """Map the winner string to a stable numeric code for plotting."""
    w = (winner or "").strip().lower()
    if w == "yellow":
        return 1
    if w == "blue":
        return -1
    return 0


# ---------------------------------------------------------------------------
# Per-train-step optimizer / RL diagnostics
# ---------------------------------------------------------------------------


@dataclass
class TrainStepMetrics:
    """Diagnostics emitted once per PPO / MAPPO optimization step."""

    step: int
    episode: int
    policy_loss: float
    value_loss: float
    entropy: float
    kl_divergence: float
    explained_variance: float
    grad_norm: float
    learning_rate: float
    advantage_mean: float
    advantage_std: float
    fps: float


def train_metrics_to_dict(m: TrainStepMetrics) -> dict[str, float]:
    """Flatten :class:`TrainStepMetrics` to a sink-friendly dict."""
    return {
        "train/step": float(m.step),
        "train/episode": float(m.episode),
        "train/policy_loss": float(m.policy_loss),
        "train/value_loss": float(m.value_loss),
        "train/entropy": float(m.entropy),
        "train/kl_divergence": float(m.kl_divergence),
        "train/explained_variance": float(m.explained_variance),
        "train/grad_norm": float(m.grad_norm),
        "train/learning_rate": float(m.learning_rate),
        "train/advantage_mean": float(m.advantage_mean),
        "train/advantage_std": float(m.advantage_std),
        "train/fps": float(m.fps),
    }


# ---------------------------------------------------------------------------
# Communication usage (TarMAC-style comms)
# ---------------------------------------------------------------------------


@dataclass
class CommUsageStats:
    """How often each comm action was emitted across the rollout.

    Attributes:
        counts: Mapping from discrete comm action id -> emission count.
        entropy: Empirical entropy (nats) of the comm action distribution.
        mean_payload_norm: Mean L2 norm of emitted comm payload vectors.
    """

    counts: dict[int, int] = field(default_factory=dict)
    entropy: float = 0.0
    mean_payload_norm: float = 0.0


def comm_usage_to_dict(stats: CommUsageStats) -> dict[str, float]:
    """Flatten :class:`CommUsageStats` to a sink-friendly dict.

    The per-action counts are emitted as ``comm/count/<id>`` so dashboards
    can chart individual channels.
    """
    out: dict[str, float] = {
        "comm/entropy": float(stats.entropy),
        "comm/mean_payload_norm": float(stats.mean_payload_norm),
        "comm/total_messages": float(sum(stats.counts.values())),
    }
    for action_id, count in sorted(stats.counts.items()):
        out[f"comm/count/{int(action_id)}"] = float(count)
    return out


# ---------------------------------------------------------------------------
# Lightweight validation helpers
# ---------------------------------------------------------------------------


def _is_finite_number(v: Any) -> bool:
    """Return True if ``v`` is a finite int/float (excludes NaN/inf)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
