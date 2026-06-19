"""Single-match smoke test: trained policy vs random on full_pistol.

Designed to finish in well under a minute on CPU so we get fast feedback.
Reduces max_rounds to 4 to shorten the match further.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import torch
from kivski_agents.baselines import get_baseline
from kivski_agents.eval.runner import EvalRunner
from kivski_agents.eval.scenarios import ALL_SCENARIOS
from kivski_agents.policy_runner import PolicyBundle
from kivski_sim.config import KivskiConfig, load_config


def _shorten(cfg: KivskiConfig, max_rounds: int = 4) -> KivskiConfig:
    raw = cfg.model_dump()
    raw.setdefault("simulation", {})["max_rounds"] = int(max_rounds)
    raw["simulation"]["round_time_seconds"] = 30  # 30s instead of 90s
    return KivskiConfig.model_validate(raw)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    cfg = _shorten(load_config("configs/turbo.yaml"))
    device = torch.device("cpu")
    ckpt = Path("models/checkpoints/best.pt")
    print(f"[setup] device={device} ckpt={ckpt} max_rounds={cfg.simulation.max_rounds}", flush=True)

    bundle = PolicyBundle.from_checkpoint(ckpt)
    print(
        f"[setup] bundle ep={bundle.metadata.get('episode')} score={bundle.metadata.get('score')}", flush=True
    )

    for scenario_name in ("full_pistol", "default_5v5", "retake_2v3"):
        spec = next((s for s in ALL_SCENARIOS if s.name == scenario_name), None)
        if spec is None:
            continue
        for baseline_name in ("random", "scripted_rush", "scripted_hold"):
            t0 = time.perf_counter()
            runner = EvalRunner(spec, cfg)
            opp = get_baseline(baseline_name, runner.env, runner.map_data, seed=42)
            trained = bundle.to_runner(device=device, deterministic=False)
            result = runner.run(trained, opp, num_matches=2, seed=99)
            dt = time.perf_counter() - t0
            outcomes = Counter(r.outcome for r in result.rounds)
            outcome_str = " ".join(f"{k}={v}" for k, v in sorted(outcomes.items(), key=lambda kv: -kv[1]))
            print(
                f"[{spec.name:14s} vs {baseline_name:13s}] "
                f"y_wr={result.yellow_winrate:.2f} b_wr={result.blue_winrate:.2f} "
                f"matches={result.num_matches} rounds/m={result.avg_rounds_per_match:.1f} "
                f"plants={result.bomb_plant_rate:.2f} defuses={result.bomb_defuse_rate:.2f} "
                f"| outcomes: {outcome_str or '(none)'} | took {dt:.1f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
