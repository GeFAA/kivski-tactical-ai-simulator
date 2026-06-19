"""Smoke tests for v0.5+v0.6 reward + league fixes baked into production.yaml.

These guard against silent regressions of the win-rate (WR) fix:
  * defenders_elim_bonus > 0  -- terminal reward when attackers wipe defenders
  * plant_progress_per_second > 0  -- shaped reward for plant progress
  * league fractions sum to 1.0  -- no silent self-play vs the live policy
  * dense plant aggregate < terminal bonus -- prevents per-tick reward from
    dominating the terminal win signal at frame_skip=6.
"""

from __future__ import annotations

from pathlib import Path

from kivski_sim.config import load_config

# Resolve the production.yaml path relative to the repo root so the test works
# regardless of the working directory pytest is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROD_YAML = _REPO_ROOT / "configs" / "production.yaml"


def test_production_yaml_loads():
    assert _PROD_YAML.is_file(), f"missing {_PROD_YAML}"
    cfg = load_config(_PROD_YAML)
    assert cfg is not None


def test_league_fractions_sum_to_one():
    """league.py falls through to live self-play if fractions don't sum to 1.0,
    which silently undermines the v0.5 WR-fix (exploit_fraction = 0)."""
    cfg = load_config(_PROD_YAML)
    total = (
        cfg.league.exploit_fraction
        + cfg.league.random_fraction
        + cfg.league.scripted_fraction
    )
    assert abs(total - 1.0) < 1e-6, (
        f"league fractions must sum to 1.0 to avoid silent self-play, got {total}"
    )


def test_attacker_win_path_rewards_active():
    """defenders_elim_bonus + plant_progress_per_second must be > 0 in
    production.yaml -- they are the WR-fix that lets YELLOW (attacker) win."""
    cfg = load_config(_PROD_YAML)
    rs = cfg.reward_shaping
    assert rs.defenders_elim_bonus > 0, (
        "WR-fix regression: attacker has no terminal win reward"
    )
    assert rs.plant_progress_per_second > 0, (
        "WR-fix regression: no progress reward for planting"
    )


def test_dense_plant_aggregate_below_terminal():
    """successful_plant fires PER inner-tick at frame_skip=6 + tick_rate=10
    + ~40s planted phase = up to ~400 inner ticks per planted round.
    The aggregate must NOT dwarf defenders_elim_bonus, otherwise the dense
    bonus wins by accident (regression of the v0.6.1 per-tick fix)."""
    cfg = load_config(_PROD_YAML)
    rs = cfg.reward_shaping
    plant_aggregate = rs.successful_plant * 400
    assert plant_aggregate < rs.defenders_elim_bonus, (
        f"plant dense aggregate {plant_aggregate} >= defenders_elim_bonus "
        f"{rs.defenders_elim_bonus} -- dense reward dominates terminal, "
        "regression of v0.6.1 fix"
    )


def test_no_self_play_exploits():
    """Until WR > 0, league.exploit_fraction must be 0.0 -- otherwise the
    live policy plays against snapshots of itself before it can win."""
    cfg = load_config(_PROD_YAML)
    assert cfg.league.exploit_fraction == 0.0, (
        "exploit_fraction must be 0 until WR > 0"
    )


def test_curriculum_enabled():
    """The killshoot curriculum stage must gate rewards on the early policy
    -- without it, the agent gets a confusing mixed signal from day one."""
    cfg = load_config(_PROD_YAML)
    assert cfg.reward_curriculum.enabled, (
        "curriculum must be enabled for killshoot stage to gate rewards"
    )
