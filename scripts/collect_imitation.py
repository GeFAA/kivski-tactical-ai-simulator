"""Collect behavior-cloning demos: ScriptedRush YELLOW vs Random BLUE.

Runs the eval-style env loop (build_scenario from full_pistol) for a
configurable number of matches. Logs every YELLOW-side (obs, action) tuple
at the env's outer step rate (post-frame_skip) into a flat dict that
ImitationDataset expects.

joint_obs is computed as the concatenation of all YELLOW per-agent obs --
this is the centralized-critic input shape the trainer feeds at the same
moment. received_comm is zero-padded to comm_value_dim because the
scripted teacher doesn't produce comm payloads; the BC student will see
the same zero placeholder at inference time on its first action.

hidden_state is logged as zeros because we BC-train one step at a time
(treating each demo as independent). The MAPPO resume path re-initialises
hidden state at episode boundaries anyway.

Usage:
    python -m scripts.collect_imitation \
        --config configs/production.yaml \
        --matches 500 \
        --out data/imitation_demos.pt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import typer

from kivski_agents.baselines import get_baseline
from kivski_agents.eval.scenarios import ALL_SCENARIOS, build_scenario
from kivski_sim.config import load_config
from kivski_sim.env import agent_name

app = typer.Typer(add_completion=False)


def _comm_value_dim(cfg) -> int:
    """Mirror factory.build_model's comm-dim derivation."""
    heads = max(1, int(cfg.ml.comm_attention_heads))
    total = max(2 * heads, int(cfg.ml.comm_embedding_dim))
    half = total // 2
    sig_dim = max(heads, ((half + heads - 1) // heads) * heads)
    return int(sig_dim)


@app.command()
def collect(
    config: str = typer.Option("configs/production.yaml", "--config", "-c"),
    matches: int = typer.Option(500, "--matches"),
    out: str = typer.Option("data/imitation_demos.pt", "--out"),
    seed: int = typer.Option(42, "--seed"),
    scenario: str = typer.Option("full_pistol", "--scenario"),
    max_rounds: int = typer.Option(4, "--max-rounds", help="Shorter rounds = faster collect, still teaches navigate+plant."),
    round_time_seconds: int = typer.Option(45, "--round-time", help="Shorter rounds time limit."),
) -> None:
    """Collect scripted_rush demos as a torch tensor dict."""
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    cfg = load_config(config)
    # Override simulation params to keep collection fast: 4 short rounds per
    # match (vs production's 12 x 90s) still includes buy-phase + rush +
    # plant attempts but finishes in ~30s wall-clock per match instead of
    # ~10 minutes. Same skill demonstration, 20x faster.
    raw = cfg.model_dump()
    raw.setdefault("simulation", {})["max_rounds"] = int(max_rounds)
    raw["simulation"]["round_time_seconds"] = int(round_time_seconds)
    from kivski_sim.config import KivskiConfig as _Cfg
    cfg = _Cfg.model_validate(raw)
    team_size = int(cfg.simulation.team_size)
    comm_value_dim = _comm_value_dim(cfg)
    hidden_size = int(cfg.ml.hidden_size)

    spec = next((s for s in ALL_SCENARIOS if s.name == scenario), None)
    if spec is None:
        typer.echo(f"[collect] unknown scenario {scenario!r}; available: {[s.name for s in ALL_SCENARIOS]}")
        raise typer.Exit(1)

    yellow_names = [agent_name(i) for i in range(team_size)]
    blue_names = [agent_name(i + team_size) for i in range(team_size)]

    typer.echo(
        f"[collect] cfg={config} matches={matches} scenario={scenario} "
        f"team_size={team_size} comm_value_dim={comm_value_dim} hidden_size={hidden_size}"
    )

    obs_buf: list[np.ndarray] = []
    joint_buf: list[np.ndarray] = []
    move_buf: list[np.ndarray] = []
    discrete_buf: list[np.ndarray] = []

    t0 = time.perf_counter()
    total_steps = 0

    for match_idx in range(matches):
        match_seed = int(seed) + int(match_idx) * 17
        env = build_scenario(spec, cfg, seed=match_seed)
        teacher = get_baseline("scripted_rush", env, env._map, seed=match_seed)
        opponent = get_baseline("random", env, env._map, seed=match_seed + 1)

        # Per-match policy reset (scripted_rush picks per-agent target sites).
        try:
            teacher.reset(yellow_names)
        except Exception:
            pass
        try:
            opponent.reset(blue_names)
        except Exception:
            pass

        observations, _infos = env.reset(seed=match_seed)

        done = False
        match_steps = 0
        safety_cap = int(cfg.simulation.max_rounds) * int(cfg.simulation.max_ticks_per_round)
        while not done and match_steps < safety_cap:
            obs_yellow = {n: observations[n] for n in yellow_names if n in observations}
            obs_blue = {n: observations[n] for n in blue_names if n in observations}

            try:
                y_result = teacher.act(obs_yellow)
            except Exception:
                # Fallback: skip this match if teacher errors
                break
            try:
                b_result = opponent.act(obs_blue)
            except Exception:
                break

            y_actions = y_result[0] if isinstance(y_result, tuple) else y_result
            b_actions = b_result[0] if isinstance(b_result, tuple) else b_result

            # Build joint_obs from YELLOW agents only (centralized critic input).
            # If any agent missing from obs (e.g. terminated), use zeros for that slot.
            obs_dim = int(observations[yellow_names[0]].shape[0]) if yellow_names[0] in observations else 0
            if obs_dim == 0:
                break
            joint_parts = []
            for n in yellow_names:
                if n in observations:
                    joint_parts.append(observations[n].astype(np.float32))
                else:
                    joint_parts.append(np.zeros(obs_dim, dtype=np.float32))
            joint_obs = np.concatenate(joint_parts, axis=0)

            # Log per-agent YELLOW demos BEFORE stepping.
            for name in yellow_names:
                if name not in y_actions or name not in observations:
                    continue
                act = y_actions[name]
                if not isinstance(act, dict):
                    # Older scripted path may emit a flat numpy array; skip
                    # those frames -- BC needs the v0.4 mixed dict format.
                    continue
                obs_buf.append(observations[name].astype(np.float32))
                joint_buf.append(joint_obs)
                move_buf.append(np.asarray(act["move"], dtype=np.float32))
                discrete_buf.append(np.asarray(act["discrete"], dtype=np.int64))
                total_steps += 1

            merged = {**y_actions, **b_actions}
            try:
                observations, _rewards, terminations, truncations, _infos = env.step(merged)
            except Exception:
                break
            done = all(terminations.values()) or all(truncations.values())
            match_steps += 1

        if (match_idx + 1) % 25 == 0:
            dt = time.perf_counter() - t0
            rate = total_steps / max(dt, 1e-9)
            typer.echo(
                f"  match {match_idx + 1:4d}/{matches} demos={total_steps:_} "
                f"({rate:.0f} demos/s, {dt:.0f}s elapsed)"
            )

    typer.echo(f"\n[collect] total demos: {total_steps:_}")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obs_t = torch.from_numpy(np.stack(obs_buf))
    joint_t = torch.from_numpy(np.stack(joint_buf))
    move_t = torch.from_numpy(np.stack(move_buf))
    discrete_t = torch.from_numpy(np.stack(discrete_buf))
    n = obs_t.shape[0]
    demos = {
        "obs": obs_t,
        "joint_obs": joint_t,
        "received_comm": torch.zeros(n, comm_value_dim, dtype=torch.float32),
        "move": move_t,
        "discrete": discrete_t,
        "hidden_state": torch.zeros(n, hidden_size, dtype=torch.float32),
    }
    torch.save(demos, out_path)
    size_mb = out_path.stat().st_size / 1e6
    typer.echo(
        f"[collect] saved -> {out_path}  ({size_mb:.1f} MB)\n"
        f"  obs shape:      {tuple(obs_t.shape)}\n"
        f"  joint_obs shape:{tuple(joint_t.shape)}\n"
        f"  move shape:     {tuple(move_t.shape)}\n"
        f"  discrete shape: {tuple(discrete_t.shape)}"
    )


if __name__ == "__main__":
    app()
