"""Manual eval debug: trained policy vs random/scripted, per-scenario WR + outcome dist.

Diagnoses Problem 1 (WR=0 stuck) by running the latest best.pt against the
canonical baselines across every ALL_SCENARIOS spec and dumping:

    - yellow_winrate / blue_winrate / draws per scenario
    - matches played, avg rounds per match
    - bomb plant + defuse rates
    - round-outcome distribution (ATTACKERS_ELIM / DEFENDERS_ELIM / TIMEOUT /
      BOMB_DETONATED / BOMB_DEFUSED) so we can see whether agents are even
      shooting or just sitting around until timeout.

Usage:
    .venv\\Scripts\\python.exe scripts\\manual_eval_debug.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import torch
from kivski_agents.baselines import get_baseline
from kivski_agents.eval.runner import EvalRunner
from kivski_agents.eval.scenarios import ALL_SCENARIOS
from kivski_agents.policy_runner import PolicyBundle
from kivski_sim.config import load_config


def main(ckpt_path: str = "models/checkpoints/best.pt") -> int:
    # Unbuffered output so we see progress live even when stdout is piped.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    cfg = load_config("configs/turbo.yaml")
    # CPU eval: keeps the training GPU run undisturbed and is fast enough for
    # the tiny match counts we use here.
    device = torch.device("cpu")
    print(f"[setup] device={device} cfg=configs/turbo.yaml ckpt={ckpt_path}", flush=True)

    ckpt_file = Path(ckpt_path)
    if not ckpt_file.is_file():
        print(f"[err] checkpoint not found: {ckpt_file}")
        return 1

    bundle = PolicyBundle.from_checkpoint(ckpt_file)
    print(
        f"[setup] bundle loaded: ep={bundle.metadata.get('episode')} score={bundle.metadata.get('score')}",
        flush=True,
    )

    num_matches = int(__import__("os").environ.get("EVAL_MATCHES", "4"))
    print(f"[setup] num_matches={num_matches} per (scenario, baseline)", flush=True)

    for spec in ALL_SCENARIOS:
        print(f"\n=== scenario={spec.name} (team_size={spec.team_size}) ===", flush=True)
        runner = EvalRunner(spec, cfg)
        for baseline_name in ("random", "scripted_rush", "scripted_hold"):
            try:
                opp = get_baseline(baseline_name, runner.env, runner.map_data, seed=42)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"  {baseline_name:20s}: failed to build baseline: {exc!r}")
                continue
            # Rebuild trained runner for each pairing so its hidden state is fresh.
            trained = bundle.to_runner(device=device, deterministic=False)
            try:
                result = runner.run(trained, opp, num_matches=num_matches, seed=99)
            except Exception as exc:
                print(f"  {baseline_name:20s}: runner.run failed: {exc!r}", flush=True)
                continue

            outcome_dist: Counter[str] = Counter(r.outcome for r in result.rounds)
            outcome_summary = ", ".join(
                f"{k}={v}" for k, v in sorted(outcome_dist.items(), key=lambda kv: -kv[1])
            )
            print(
                f"  {baseline_name:20s}: y_wr={result.yellow_winrate:.2f} "
                f"b_wr={result.blue_winrate:.2f} "
                f"draws={result.draws} "
                f"matches={result.num_matches} "
                f"rounds={result.avg_rounds_per_match:.1f} "
                f"plants={result.bomb_plant_rate:.2f} "
                f"defuses={result.bomb_defuse_rate:.2f}",
                flush=True,
            )
            print(f"      outcomes: {outcome_summary or '(no rounds)'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "models/checkpoints/best.pt"))
