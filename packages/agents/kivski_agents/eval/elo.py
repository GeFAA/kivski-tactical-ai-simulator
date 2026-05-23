"""Elo and TrueSkill rating tracking for evaluation tournaments.

We provide two trackers: a simple :class:`EloTracker` (default, no extra
deps) and a :class:`TrueSkillTracker` (opt-in, uses the ``trueskill``
package listed in ``pyproject.toml``). Both expose the same minimal API:

    tracker.add_policy(name)
    tracker.update(a, b, outcome)
    tracker.expected_score(a, b)
    tracker.to_dict() / .from_dict(...)
    tracker.to_json(path) / .from_json(path)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = ["EloRating", "EloTracker", "TrueSkillTracker"]


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


@dataclass
class EloRating:
    """One policy's rolling Elo record."""

    policy_name: str
    rating: float = 1000.0
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0


class EloTracker:
    """Standard Elo rating book with a configurable K-factor.

    The default K of 32 is the classic chess constant and works reasonably
    well for short evaluation series (~20-100 matches per matchup). Lower it
    for stability when running long head-to-head suites.
    """

    def __init__(self, k_factor: float = 32.0) -> None:
        self.k_factor: float = float(k_factor)
        self.ratings: dict[str, EloRating] = {}

    # ------------------------------------------------------------------

    def add_policy(self, name: str, initial_rating: float = 1000.0) -> None:
        """Register ``name`` if not already present. Idempotent."""
        if name in self.ratings:
            return
        self.ratings[name] = EloRating(policy_name=str(name), rating=float(initial_rating))

    # ------------------------------------------------------------------

    def expected_score(self, a: str, b: str) -> float:
        """Return the Elo-implied probability of ``a`` beating ``b``.

        Uses the standard logistic formula::

            P(a wins) = 1 / (1 + 10^((R_b - R_a) / 400))
        """
        self.add_policy(a)
        self.add_policy(b)
        ra = float(self.ratings[a].rating)
        rb = float(self.ratings[b].rating)
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    # ------------------------------------------------------------------

    def update(self, a: str, b: str, outcome: float) -> None:
        """Record one match outcome and update both ratings.

        Args:
            a: First policy name.
            b: Second policy name.
            outcome: ``1.0`` if ``a`` won, ``0.0`` if ``b`` won, ``0.5`` draw.
                Values outside [0, 1] are clipped.
        """
        self.add_policy(a)
        self.add_policy(b)
        score = float(min(1.0, max(0.0, float(outcome))))
        e_a = self.expected_score(a, b)
        e_b = 1.0 - e_a
        delta_a = self.k_factor * (score - e_a)
        delta_b = self.k_factor * ((1.0 - score) - e_b)
        ra = self.ratings[a]
        rb = self.ratings[b]
        ra.rating = float(ra.rating + delta_a)
        rb.rating = float(rb.rating + delta_b)
        ra.matches += 1
        rb.matches += 1
        if score > 0.5:
            ra.wins += 1
            rb.losses += 1
        elif score < 0.5:
            ra.losses += 1
            rb.wins += 1
        else:
            ra.draws += 1
            rb.draws += 1

    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the tracker to a plain dict suitable for JSON."""
        return {
            "k_factor": float(self.k_factor),
            "ratings": {name: asdict(r) for name, r in sorted(self.ratings.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EloTracker:
        """Reconstruct an :class:`EloTracker` from :meth:`to_dict` output."""
        tracker = cls(k_factor=float(data.get("k_factor", 32.0)))
        for name, raw in data.get("ratings", {}).items():
            tracker.ratings[str(name)] = EloRating(
                policy_name=str(raw.get("policy_name", name)),
                rating=float(raw.get("rating", 1000.0)),
                matches=int(raw.get("matches", 0)),
                wins=int(raw.get("wins", 0)),
                draws=int(raw.get("draws", 0)),
                losses=int(raw.get("losses", 0)),
            )
        return tracker

    def to_json(self, path: str | Path) -> None:
        """Write :meth:`to_dict` output to ``path`` as pretty-printed JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> EloTracker:
        """Load an :class:`EloTracker` previously written by :meth:`to_json`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# TrueSkill (optional)
# ---------------------------------------------------------------------------


@dataclass
class _TSRating:
    """Plain serialisation record for a TrueSkill rating."""

    policy_name: str
    mu: float = 25.0
    sigma: float = 25.0 / 3.0
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0


class TrueSkillTracker:
    """TrueSkill rating book backed by the ``trueskill`` package.

    Multi-policy comparisons (e.g. round-robin tournaments) benefit from
    TrueSkill's uncertainty bookkeeping more than plain Elo does. Falls back
    to a clear :class:`ImportError` at construction time if the optional
    dependency is missing.
    """

    def __init__(
        self,
        mu: float = 25.0,
        sigma: float = 25.0 / 3.0,
    ) -> None:
        try:
            import trueskill  # noqa: PLC0415 - keep optional
        except ImportError as exc:
            raise ImportError(
                "TrueSkillTracker requires the 'trueskill' package. Install with "
                "`pip install trueskill` or use EloTracker."
            ) from exc

        self._ts = trueskill
        self._env = trueskill.TrueSkill(mu=float(mu), sigma=float(sigma))
        self._default_mu = float(mu)
        self._default_sigma = float(sigma)
        self.ratings: dict[str, _TSRating] = {}

    # ------------------------------------------------------------------

    def add_policy(self, name: str) -> None:
        if name in self.ratings:
            return
        self.ratings[name] = _TSRating(policy_name=str(name), mu=self._default_mu, sigma=self._default_sigma)

    # ------------------------------------------------------------------

    def _make_rating(self, name: str) -> Any:
        rec = self.ratings[name]
        return self._env.create_rating(mu=rec.mu, sigma=rec.sigma)

    def expected_score(self, a: str, b: str) -> float:
        """Return the TrueSkill win-probability of ``a`` against ``b``."""
        self.add_policy(a)
        self.add_policy(b)
        # ``quality`` is the symmetric match-quality, not the win probability.
        # We approximate the win probability via the normal CDF formula used
        # by trueskill's standard examples.
        import math  # noqa: PLC0415

        ra = self._make_rating(a)
        rb = self._make_rating(b)
        beta = self._env.beta
        delta_mu = ra.mu - rb.mu
        denom = math.sqrt(2.0 * (beta**2) + ra.sigma**2 + rb.sigma**2)
        return 0.5 * (1.0 + math.erf(delta_mu / (denom * math.sqrt(2.0))))

    def update(self, a: str, b: str, outcome: float) -> None:
        """Apply one match outcome and update both ratings."""
        self.add_policy(a)
        self.add_policy(b)
        score = float(min(1.0, max(0.0, float(outcome))))
        ra = self._make_rating(a)
        rb = self._make_rating(b)
        if score == 0.5:
            new_ra, new_rb = self._env.rate_1vs1(ra, rb, drawn=True)
        elif score > 0.5:
            new_ra, new_rb = self._env.rate_1vs1(ra, rb)
        else:
            new_rb, new_ra = self._env.rate_1vs1(rb, ra)
        rec_a = self.ratings[a]
        rec_b = self.ratings[b]
        rec_a.mu = float(new_ra.mu)
        rec_a.sigma = float(new_ra.sigma)
        rec_b.mu = float(new_rb.mu)
        rec_b.sigma = float(new_rb.sigma)
        rec_a.matches += 1
        rec_b.matches += 1
        if score > 0.5:
            rec_a.wins += 1
            rec_b.losses += 1
        elif score < 0.5:
            rec_a.losses += 1
            rec_b.wins += 1
        else:
            rec_a.draws += 1
            rec_b.draws += 1

    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_mu": float(self._default_mu),
            "default_sigma": float(self._default_sigma),
            "ratings": {name: asdict(r) for name, r in sorted(self.ratings.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrueSkillTracker:
        tracker = cls(
            mu=float(data.get("default_mu", 25.0)),
            sigma=float(data.get("default_sigma", 25.0 / 3.0)),
        )
        for name, raw in data.get("ratings", {}).items():
            tracker.ratings[str(name)] = _TSRating(
                policy_name=str(raw.get("policy_name", name)),
                mu=float(raw.get("mu", tracker._default_mu)),
                sigma=float(raw.get("sigma", tracker._default_sigma)),
                matches=int(raw.get("matches", 0)),
                wins=int(raw.get("wins", 0)),
                draws=int(raw.get("draws", 0)),
                losses=int(raw.get("losses", 0)),
            )
        return tracker

    def to_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> TrueSkillTracker:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
