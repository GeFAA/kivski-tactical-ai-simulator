"""CLI entry-point for the Kivski MAPPO trainer.

Exposed via the ``kivski-train`` console script declared in ``pyproject.toml``.
The same module is runnable directly with ``python scripts/train.py train ...``
(or with ``python scripts/train.py smoke ...`` for a tiny CI / verification run).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import typer
from kivski_agents.run_naming import generate_run_name
from kivski_agents.telemetry import make_sink
from kivski_agents.training.auto_tune import detect_optimal_num_envs, detect_optimal_workers
from kivski_agents.training.trainer import Trainer, TrainerConfig
from kivski_sim.config import KivskiConfig, load_config

app = typer.Typer(add_completion=False, help="Run the Kivski MAPPO training loop.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_device(name: str):  # -> torch.device  (lazy import for tests)
    """Resolve ``"auto" / "cuda" / "cpu" / "mps"`` to a ``torch.device``.

    ``"auto"`` prefers CUDA, then MPS, falling back to CPU. Explicit
    ``"cuda"`` raises if no CUDA device is visible -- this is intentional
    so a misconfigured launcher surfaces immediately instead of silently
    falling back to a 10x slower CPU run.
    """
    import torch  # local import keeps the CLI importable without torch

    name_l = str(name).strip().lower()
    if name_l == "cpu":
        return torch.device("cpu")
    if name_l == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "[kivski-train] --device cuda requested but torch.cuda.is_available() is False. "
                "Install a CUDA build: pip install --index-url "
                "https://download.pytorch.org/whl/cu126 torch torchvision"
            )
        return torch.device("cuda")
    if name_l == "mps":
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _describe_device(device) -> str:  # -> str
    """Render a short human-readable device tag for the startup log line."""
    import torch  # local

    try:
        if device.type == "cuda":
            idx = device.index if device.index is not None else 0
            name = torch.cuda.get_device_name(idx)
            try:
                props = torch.cuda.get_device_properties(idx)
                total_gb = float(props.total_memory) / 1e9
                major, minor = torch.cuda.get_device_capability(idx)
                return f"cuda:{idx} ({name}, {total_gb:.1f} GB, sm_{major}{minor})"
            except Exception:
                return f"cuda:{idx} ({name})"
        return str(device)
    except Exception:
        return str(device)


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
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Path to KivskiConfig YAML."),
    episodes: int | None = typer.Option(
        None,
        "--episodes",
        help="Override training.total_episodes (number of full matches to train on).",
    ),
    num_envs: int | None = typer.Option(
        None, "--num-envs", help="Override training.num_envs (parallel envs)."
    ),
    rollout_steps: int | None = typer.Option(
        None, "--rollout-steps", help="Override training.rollout_steps."
    ),
    map_name: str = typer.Option("dustline", "--map", help="Map name to train on."),
    resume: str | None = typer.Option(None, "--resume", help="Path to a checkpoint .pt to resume from."),
    run_name: str | None = typer.Option(None, "--run-name", help="Override the auto-generated run name."),
    device: str = typer.Option("auto", "--device", help='Device to train on: "auto", "cpu", "cuda", "mps".'),
    seed: int | None = typer.Option(None, "--seed", help="Override the deterministic seed."),
    backend: str | None = typer.Option(
        None,
        "--telemetry",
        help='Telemetry backend: "csv" | "tensorboard" | "wandb" | "all" | "none".',
    ),
    vec_kind: str = typer.Option(
        "subproc",
        "--vec-kind",
        help='Vectorised env backend: "subproc" (multi-process, default), "thread", or "sync".',
    ),
    num_workers: int | None = typer.Option(
        None,
        "--num-workers",
        help="Worker count for parallel vec env. Auto-detected from CPU count when omitted.",
    ),
    auto_envs: bool = typer.Option(
        False,
        "--auto-envs/--no-auto-envs",
        help="Auto-detect num_envs from CPU count (default uses --num-envs or config).",
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

    # Resolve num_envs:
    # 1) explicit --num-envs always wins
    # 2) --auto-envs forces CPU-based detection
    # 3) otherwise fall back to config value
    if num_envs is not None:
        resolved_num_envs = int(num_envs)
    elif auto_envs:
        resolved_num_envs = detect_optimal_num_envs(None)
    else:
        resolved_num_envs = int(cfg.training.num_envs)

    # Resolve num_workers for parallel backends.
    if num_workers is None and vec_kind in ("subproc", "thread"):
        resolved_num_workers: int | None = detect_optimal_workers(resolved_num_envs)
    else:
        resolved_num_workers = num_workers

    tcfg = TrainerConfig(
        total_episodes=int(episodes) if episodes is not None else int(cfg.training.total_episodes),
        rollout_steps=int(rollout_steps) if rollout_steps is not None else int(cfg.training.rollout_steps),
        num_envs=int(resolved_num_envs),
        checkpoint_every=int(cfg.training.checkpoint_every_episodes),
        eval_every=int(cfg.training.eval_every_episodes),
        snapshot_every=int(cfg.league.snapshot_every_episodes),
        log_dir=log_dir,
        checkpoint_dir=ckpt_dir,
        device=torch_device,
        map_name=map_name,
        resume_from=Path(resume) if resume else None,
        run_name=rn,
        vec_kind=str(vec_kind),
        num_workers=resolved_num_workers,
    )

    typer.echo(
        f"[kivski-train] run={rn} device={_describe_device(torch_device)} "
        f"vec_kind={tcfg.vec_kind} "
        f"num_envs={tcfg.num_envs} num_workers={tcfg.num_workers} "
        f"rollout_steps={tcfg.rollout_steps} total_episodes={tcfg.total_episodes}"
    )
    trainer = Trainer(cfg, tcfg, telemetry=sink)
    if torch_device.type == "cuda":
        typer.echo(
            f"[kivski-train] mixed_precision=bf16:{trainer.mappo.amp_enabled} "
            f"amp_dtype={trainer.mappo.amp_dtype}"
        )
    try:
        trainer.train()
    finally:
        with contextlib.suppress(Exception):
            sink.close()


@app.command()
def smoke(
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Path to KivskiConfig YAML."),
    num_envs: int = typer.Option(2, "--num-envs", help="Tiny env count."),
    rollout_steps: int = typer.Option(32, "--rollout-steps", help="Few rollout steps."),
    episodes: int = typer.Option(
        4, "--episodes", help="Run until this many episodes finish (or buffer fills)."
    ),
    map_name: str = typer.Option("dustline", "--map", help="Map name."),
    device: str = typer.Option("cpu", "--device", help="Smoke runs on CPU by default."),
    run_name: str | None = typer.Option(None, "--run-name", help="Override run name."),
    vec_kind: str = typer.Option(
        "sync",
        "--vec-kind",
        help='Vectorised env backend: "sync" (default for smoke), "thread", or "subproc".',
    ),
    num_workers: int | None = typer.Option(
        None,
        "--num-workers",
        help="Worker count for parallel vec env. Auto-detected from CPU count when omitted.",
    ),
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

    resolved_workers = num_workers
    if num_workers is None and vec_kind in ("subproc", "thread"):
        resolved_workers = detect_optimal_workers(int(num_envs))

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
        vec_kind=str(vec_kind),
        num_workers=resolved_workers,
    )

    import time as _time

    t0 = _time.perf_counter()
    typer.echo(
        f"[kivski-smoke] run={rn} device={_describe_device(torch_device)} "
        f"vec_kind={tcfg.vec_kind} "
        f"num_envs={tcfg.num_envs} num_workers={tcfg.num_workers} "
        f"rollout_steps={tcfg.rollout_steps} target_episodes={tcfg.total_episodes}"
    )
    trainer = Trainer(cfg, tcfg, telemetry=sink)
    try:
        trainer.train()
    finally:
        with contextlib.suppress(Exception):
            sink.close()
    elapsed = max(_time.perf_counter() - t0, 1e-9)
    fps = float(trainer.env_steps) / float(elapsed)
    typer.echo(
        f"[kivski-smoke] DONE episodes={trainer.episode_count} updates={trainer.update_step} "
        f"env_steps={trainer.env_steps} elapsed={elapsed:.2f}s fps={fps:.1f}"
    )


if __name__ == "__main__":
    app()
