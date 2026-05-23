"""CLI entry-point for the Kivski MAPPO trainer.

Exposed via the ``kivski-train`` console script declared in ``pyproject.toml``.
The same module is runnable directly with ``python scripts/train.py train ...``
(or with ``python scripts/train.py smoke ...`` for a tiny CI / verification run).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from kivski_agents.run_naming import generate_run_name
from kivski_agents.telemetry import make_sink
from kivski_agents.training.trainer import Trainer, TrainerConfig
from kivski_sim.config import KivskiConfig, load_config


app = typer.Typer(add_completion=False, help="Run the Kivski MAPPO training loop.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_device(name: str):  # -> torch.device  (lazy import for tests)
    """Resolve ``"auto" / "cuda" / "cpu" / "mps"`` to a ``torch.device``."""
    import torch  # local import keeps the CLI importable without torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(str(name))


def _override_seed(cfg: KivskiConfig, seed: int) -> KivskiConfig:
    """Return a copy of ``cfg`` with ``seed`` overridden."""
    raw = cfg.model_dump()
    raw["seed"] = int(seed)
    return KivskiConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def train(
    config: str = typer.Option(
        "configs/default.yaml", "--config", "-c", help="Path to KivskiConfig YAML."
    ),
    episodes: Optional[int] = typer.Option(
        None,
        "--episodes",
        help="Override training.total_episodes (number of full matches to train on).",
    ),
    num_envs: Optional[int] = typer.Option(
        None, "--num-envs", help="Override training.num_envs (parallel envs)."
    ),
    rollout_steps: Optional[int] = typer.Option(
        None, "--rollout-steps", help="Override training.rollout_steps."
    ),
    map_name: str = typer.Option(
        "dustline", "--map", help="Map name to train on."
    ),
    resume: Optional[str] = typer.Option(
        None, "--resume", help="Path to a checkpoint .pt to resume from."
    ),
    run_name: Optional[str] = typer.Option(
        None, "--run-name", help="Override the auto-generated run name."
    ),
    device: str = typer.Option(
        "auto", "--device", help='Device to train on: "auto", "cpu", "cuda", "mps".'
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Override the deterministic seed."
    ),
    backend: Optional[str] = typer.Option(
        None,
        "--telemetry",
        help='Telemetry backend: "csv" | "tensorboard" | "wandb" | "all" | "none".',
    ),
) -> None:
    """Launch a full MAPPO training run."""
    cfg = load_config(config)
    if seed is not None:
        cfg = _override_seed(cfg, int(seed))

    torch_device = _pick_device(device)
    rn = run_name or generate_run_name("kivski")
    log_dir = Path("models/logs") / rn
    ckpt_dir = Path("models/checkpoints") / rn
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    sink_backend = backend if backend is not None else cfg.telemetry.backend
    sink = make_sink(sink_backend, Path("models/logs"), rn)

    tcfg = TrainerConfig(
        total_episodes=int(episodes) if episodes is not None else int(cfg.training.total_episodes),
        rollout_steps=int(rollout_steps)
        if rollout_steps is not None
        else int(cfg.training.rollout_steps),
        num_envs=int(num_envs) if num_envs is not None else int(cfg.training.num_envs),
        checkpoint_every=int(cfg.training.checkpoint_every_episodes),
        eval_every=int(cfg.training.eval_every_episodes),
        snapshot_every=int(cfg.league.snapshot_every_episodes),
        log_dir=log_dir,
        checkpoint_dir=ckpt_dir,
        device=torch_device,
        map_name=map_name,
        resume_from=Path(resume) if resume else None,
        run_name=rn,
    )

    typer.echo(
        f"[kivski-train] run={rn} device={torch_device} num_envs={tcfg.num_envs} "
        f"rollout_steps={tcfg.rollout_steps} total_episodes={tcfg.total_episodes}"
    )
    trainer = Trainer(cfg, tcfg, telemetry=sink)
    try:
        trainer.train()
    finally:
        try:
            sink.close()
        except Exception:
            pass


@app.command()
def smoke(
    config: str = typer.Option(
        "configs/default.yaml", "--config", "-c", help="Path to KivskiConfig YAML."
    ),
    num_envs: int = typer.Option(2, "--num-envs", help="Tiny env count."),
    rollout_steps: int = typer.Option(32, "--rollout-steps", help="Few rollout steps."),
    episodes: int = typer.Option(
        4, "--episodes", help="Run until this many episodes finish (or buffer fills)."
    ),
    map_name: str = typer.Option("dustline", "--map", help="Map name."),
    device: str = typer.Option("cpu", "--device", help='Smoke runs on CPU by default.'),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Override run name."),
) -> None:
    """Tiny end-to-end verification run (~1 minute on CPU)."""
    cfg = load_config(config)
    # Smoke-specific overrides: tiny team / short matches / minimal PPO epochs
    # so the whole loop completes in a few seconds.
    raw = cfg.model_dump()
    sim = raw.setdefault("simulation", {})
    sim["team_size"] = 2
    sim["max_rounds"] = 2
    sim["side_switch_round"] = 999
    sim["round_time_seconds"] = 6
    sim["bomb_timer_seconds"] = 3
    sim["plant_time_seconds"] = 1.0
    sim["defuse_time_seconds"] = 1.0
    sim["defuse_time_with_kit_seconds"] = 0.5
    sim["buy_time_seconds"] = 1
    sim["max_ticks_per_round"] = 100
    sim["starting_money"] = 800
    ml = raw.setdefault("ml", {})
    ml["ppo_epochs"] = 1
    ml["minibatch_size"] = 64
    ml["hidden_size"] = 64
    league = raw.setdefault("league", {})
    league["snapshot_every_episodes"] = max(int(episodes), 1) + 1
    raw.setdefault("training", {})["num_envs"] = int(num_envs)
    raw["training"]["rollout_steps"] = int(rollout_steps)
    raw["training"]["total_episodes"] = int(episodes)
    raw["training"]["checkpoint_every_episodes"] = max(int(episodes), 1) + 1
    raw["training"]["eval_every_episodes"] = max(int(episodes), 1) + 1
    raw.setdefault("telemetry", {})["backend"] = "none"
    cfg = KivskiConfig.model_validate(raw)

    torch_device = _pick_device(device)
    rn = run_name or generate_run_name("smoke")
    log_dir = Path("models/logs") / rn
    ckpt_dir = Path("models/checkpoints") / rn
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sink = make_sink("none", Path("models/logs"), rn)

    tcfg = TrainerConfig(
        total_episodes=int(episodes),
        rollout_steps=int(rollout_steps),
        num_envs=int(num_envs),
        checkpoint_every=int(episodes) + 1,
        eval_every=int(episodes) + 1,
        snapshot_every=int(episodes) + 1,
        log_dir=log_dir,
        checkpoint_dir=ckpt_dir,
        device=torch_device,
        map_name=map_name,
        resume_from=None,
        run_name=rn,
        eval_matches=1,
        print_every=1,
    )

    typer.echo(
        f"[kivski-smoke] run={rn} device={torch_device} num_envs={tcfg.num_envs} "
        f"rollout_steps={tcfg.rollout_steps} target_episodes={tcfg.total_episodes}"
    )
    trainer = Trainer(cfg, tcfg, telemetry=sink)
    try:
        trainer.train()
    finally:
        try:
            sink.close()
        except Exception:
            pass
    typer.echo(
        f"[kivski-smoke] DONE episodes={trainer.episode_count} updates={trainer.update_step} "
        f"env_steps={trainer.env_steps}"
    )


if __name__ == "__main__":
    app()
