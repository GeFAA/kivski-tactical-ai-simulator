"""Eval the BC-pretrained policy: does it plant?

Success criterion: BC YELLOW plant_rate >= 0.10 vs random BLUE.
- 0.00 => BC failed; policy still random-equivalent
- > scripted_rush plant_rate (0.32) => BC matched the teacher closely
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
from kivski_sim.config import KivskiConfig, load_config


def _shorten(cfg: KivskiConfig, max_rounds: int = 4) -> KivskiConfig:
    raw = cfg.model_dump()
    raw.setdefault("simulation", {})["max_rounds"] = int(max_rounds)
    raw["simulation"]["round_time_seconds"] = 45
    return KivskiConfig.model_validate(raw)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    cfg = _shorten(load_config("configs/production.yaml"))
    ckpt = Path("models/checkpoints/bc_pretrained.pt")
    device = torch.device("cpu")

    bundle = PolicyBundle.from_checkpoint(ckpt)
    meta = bundle.metadata
    print(f"[setup] ckpt={ckpt}", flush=True)
    final_loss = meta.get("bc_final_loss", meta.get("bc_final_nll", float("nan")))
    print(
        f"[setup] meta: episode={meta.get('episode')} source={meta.get('source')} "
        f"bc_epochs={meta.get('bc_epochs')} demo_steps={meta.get('demo_steps')} "
        f"bc_final_loss={final_loss:.4f}",
        flush=True,
    )
    print(f"[setup] arch={meta.get('model_arch')} env={meta.get('env_shape')}", flush=True)

    spec = next((s for s in ALL_SCENARIOS if s.name == "full_pistol"), None)
    NUM = 16

    runs: list[tuple[str, str, str]] = [
        ("BC YELLOW (deterministic)", "deterministic", "random"),
        ("BC YELLOW (stochastic)", "stochastic", "random"),
        ("BC YELLOW (stochastic)", "stochastic", "scripted_rush"),
        ("BC YELLOW (stochastic)", "stochastic", "scripted_hold"),
        ("RANDOM YELLOW (baseline)", None, "random"),
        ("SCRIPTED_RUSH YELLOW (baseline)", None, "random"),
    ]

    for label, mode, opponent_name in runs:
        runner = EvalRunner(spec, cfg)
        opp = get_baseline(opponent_name, runner.env, runner.map_data, seed=99)
        if mode is None:
            # YELLOW is a baseline (random or scripted_rush)
            yellow = get_baseline(label.split(" ")[0].lower(), runner.env, runner.map_data, seed=11)
        else:
            yellow = bundle.to_runner(device=device, deterministic=(mode == "deterministic"))
        r = runner.run(yellow, opp, num_matches=NUM, seed=1000 + len(label))
        outcomes = Counter(rd.outcome for rd in r.rounds)
        print(
            f"\n=== {label}  vs  BLUE={opponent_name}  ({NUM} matches) ===\n"
            f"  yellow_wr={r.yellow_winrate:.3f}  plant_rate={r.bomb_plant_rate:.2f}  defuse_rate={r.bomb_defuse_rate:.2f}\n"
            f"  outcomes: {dict(outcomes)}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
