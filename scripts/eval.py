"""CLI entry-point for head-to-head evaluation between two policies.

Exposed via the ``kivski-eval`` console script declared in ``pyproject.toml``.
The same module is runnable directly with ``python scripts/eval.py run ...``.

Each "policy" argument can be either:

* A registered baseline name (``random``, ``scripted_hold``, ``scripted_rush``)
* A path to a saved checkpoint (``.pt``) -- loaded via
  :class:`kivski_agents.baselines.frozen_snapshot.FrozenSnapshotBaseline`

The result is printed as a Rich table and optionally serialised to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from kivski_agents.baselines import BASELINE_REGISTRY, get_baseline
from kivski_agents.eval import ALL_SCENARIOS, EvalRunner, EvalResult
from kivski_sim.config import KivskiConfig, load_config


app = typer.Typer(add_completion=False, help="Run the Kivski head-to-head eval suite.")


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------


def _resolve_policy(name_or_path: str, env: Any, map_data: Any, seed: int) -> Any:
    """Translate a CLI policy identifier into an instantiated policy.

    Order of resolution:
        1. If ``name_or_path`` is a registered baseline key, instantiate it.
        2. Otherwise, treat ``name_or_path`` as a checkpoint file and load it
           via :class:`FrozenSnapshotBaseline`.
        3. If neither matches, raise a :class:`typer.BadParameter`.
    """
    if name_or_path in BASELINE_REGISTRY:
        return get_baseline(name_or_path, env, map_data, seed=int(seed))

    ckpt = Path(name_or_path).expanduser()
    if ckpt.is_file():
        try:
            from kivski_agents.baselines.frozen_snapshot import FrozenSnapshotBaseline
        except Exception as exc:  # pragma: no cover - depends on optional torch
            raise typer.BadParameter(
                f"Cannot load checkpoint {name_or_path}: {exc}"
            ) from exc
        return FrozenSnapshotBaseline(ckpt, device=None)

    raise typer.BadParameter(
        f"Unknown policy {name_or_path!r}. Expected one of "
        f"{sorted(BASELINE_REGISTRY)} or a path to a .pt checkpoint."
    )


def _resolve_scenario(name: str) -> Any:
    """Look up a :class:`ScenarioSpec` by name from :data:`ALL_SCENARIOS`."""
    for spec in ALL_SCENARIOS:
        if spec.name == name:
            return spec
    raise typer.BadParameter(
        f"Unknown scenario {name!r}. Available: {[s.name for s in ALL_SCENARIOS]}"
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_result(console: Console, result: EvalResult) -> None:
    """Pretty-print an :class:`EvalResult` using a Rich table."""
    table = Table(title=f"Eval: {result.scenario}", header_style="bold magenta")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Yellow policy", result.policy_yellow)
    table.add_row("Blue policy", result.policy_blue)
    table.add_row("Matches played", str(result.num_matches))
    table.add_row("Yellow match wins", str(result.yellow_match_wins))
    table.add_row("Blue match wins", str(result.blue_match_wins))
    table.add_row("Draws", str(result.draws))
    table.add_row("Yellow winrate", f"{result.yellow_winrate:.3f}")
    table.add_row("Blue winrate", f"{result.blue_winrate:.3f}")
    table.add_row("Avg rounds / match", f"{result.avg_rounds_per_match:.2f}")
    table.add_row("Avg match duration (ticks)", f"{result.avg_match_duration_ticks:.1f}")
    table.add_row("Bomb plant rate (per round)", f"{result.bomb_plant_rate:.3f}")
    table.add_row("Bomb defuse rate (per round)", f"{result.bomb_defuse_rate:.3f}")

    console.print(table)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    policy_yellow: str = typer.Argument(
        ...,
        help="Yellow policy: a baseline name (random|scripted_hold|scripted_rush) or a path to a .pt checkpoint.",
    ),
    policy_blue: str = typer.Argument(
        ...,
        help="Blue policy: same options as policy_yellow.",
    ),
    scenario: str = typer.Option(
        "default_5v5",
        "--scenario",
        "-s",
        help="Scenario name from kivski_agents.eval.scenarios.ALL_SCENARIOS.",
    ),
    matches: int = typer.Option(
        20, "--matches", "-m", help="Number of matches to play."
    ),
    seed: int = typer.Option(42, "--seed", help="Deterministic eval seed."),
    config: str = typer.Option(
        "configs/default.yaml", "--config", "-c", help="Path to a KivskiConfig YAML."
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="If set, write the result as JSON to this path."
    ),
) -> None:
    """Run a head-to-head evaluation between two policies."""
    console = Console()
    cfg: KivskiConfig = load_config(config)
    spec = _resolve_scenario(scenario)

    runner = EvalRunner(spec, cfg)
    yellow = _resolve_policy(policy_yellow, runner.env, runner.map_data, seed=seed)
    blue = _resolve_policy(policy_blue, runner.env, runner.map_data, seed=seed + 1)

    console.print(
        f"[bold green]Running[/] [yellow]{yellow.name}[/] vs [blue]{blue.name}[/] "
        f"on scenario [magenta]{spec.name}[/] for {matches} matches "
        f"(seed={seed})..."
    )
    result = runner.run(yellow, blue, num_matches=matches, seed=seed)
    _render_result(console, result)

    if output:
        out_path = Path(output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[green]Wrote[/] {out_path}")


@app.command()
def list_scenarios() -> None:
    """List the built-in eval scenarios."""
    console = Console()
    table = Table(title="Available scenarios", header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Team size", justify="right")
    table.add_column("Starting money", justify="right")
    table.add_column("Pre-plant?", justify="center")
    table.add_column("Attackers alive", justify="right")
    table.add_column("Defenders alive", justify="right")
    for spec in ALL_SCENARIOS:
        table.add_row(
            spec.name,
            str(spec.team_size),
            "-" if spec.starting_money is None else str(spec.starting_money),
            "yes" if spec.bomb_planted else "no",
            "-" if spec.attackers_alive is None else str(spec.attackers_alive),
            "-" if spec.defenders_alive is None else str(spec.defenders_alive),
        )
    console.print(table)


@app.command()
def list_baselines() -> None:
    """List the registered baseline policies."""
    console = Console()
    console.print("Registered baselines: " + ", ".join(sorted(BASELINE_REGISTRY)))


if __name__ == "__main__":
    app()
