"""kivski-watch CLI: inspect saved replays or watch a single match deterministically.

Usage examples:
    kivski-watch info models/replays/run-xyz.replay
    kivski-watch run --policy-yellow random --policy-blue scripted_rush --seed 7
    kivski-watch replay models/replays/run-xyz.replay --speed 2.0
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kivski_sim.config import load_config
from kivski_sim.map_loader import load_map
from kivski_sim.replay import ReplayReader

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def info(path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True)) -> None:
    """Print header + frame counts of a replay file."""
    reader = ReplayReader(path)
    hdr = reader.header
    actions = sum(1 for _ in reader.iter_actions())
    reader = ReplayReader(path)  # rewind
    events = sum(1 for _ in reader.iter_events())

    tbl = Table(title=f"Replay: {path.name}", show_header=False)
    tbl.add_row("version", str(hdr.version))
    tbl.add_row("seed", str(hdr.seed))
    tbl.add_row("map_name", hdr.map_name)
    tbl.add_row("team_size", str(hdr.team_size))
    tbl.add_row("kivski_version", hdr.kivski_version)
    tbl.add_row("config_hash", hdr.config_hash)
    tbl.add_row("created_at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(hdr.created_at)))
    tbl.add_row("action_frames", str(actions))
    tbl.add_row("event_frames", str(events))
    console.print(tbl)


@app.command()
def run(
    policy_yellow: str = typer.Option("random", help="random | scripted_rush | scripted_hold | <ckpt.pt>"),
    policy_blue: str = typer.Option("random"),
    map_name: str = typer.Option("dustline"),
    seed: int = typer.Option(42),
    rounds: int = typer.Option(6, help="Override max_rounds for a shorter watch."),
    config: str = typer.Option("configs/default.yaml"),
    output: Path | None = typer.Option(None, help="Optional path to save a .replay file."),
) -> None:
    """Run a single deterministic match headlessly and print round summaries."""
    cfg = load_config(config)
    map_data = load_map(map_name)

    try:
        from kivski_sim.engine import Engine
    except Exception as exc:  # pragma: no cover - missing optional dep handling
        console.print(f"[red]Could not import Engine:[/red] {exc}")
        raise typer.Exit(1) from exc

    # Build engine, force short max_rounds for a quick watch session.
    engine = Engine(cfg=cfg, map_data=map_data, seed=seed)
    engine._max_rounds_override = rounds  # type: ignore[attr-defined]

    try:
        from kivski_agents.baselines import get_baseline

        py = get_baseline(policy_yellow, type("E", (), {"action_space": lambda self, _n: None})(), map_data, seed)
        pb = get_baseline(policy_blue, type("E", (), {"action_space": lambda self, _n: None})(), map_data, seed + 1)
    except Exception:
        console.print("[yellow]Falling back to engine-internal random actions (baselines unavailable).[/yellow]")
        py = pb = None  # noqa: F841

    snap = engine.reset(seed=seed)
    console.print(f"[bold]Match started[/bold]  map={map_name}  seed={seed}  rounds={rounds}")

    tbl = Table(title="Round summaries")
    tbl.add_column("round")
    tbl.add_column("outcome")
    tbl.add_column("yellow")
    tbl.add_column("blue")

    done = False
    last_round = -1
    while not done:
        actions = {a.agent_id: None for a in engine.state.agents if a.alive}  # placeholder, replaced by engine defaults
        # For a true watch we'd plug in policy actions here; in V1 fallback we let engine step with no-ops.
        snap, _rewards, done = engine.step({})  # type: ignore[arg-type]
        if snap.round_id != last_round and engine.state.round_summaries:
            rs = engine.state.round_summaries[-1]
            tbl.add_row(
                str(rs.round_id),
                rs.outcome.name,
                str(rs.yellow_score),
                str(rs.blue_score),
            )
            last_round = snap.round_id

    console.print(tbl)
    console.print(f"[green]Match done[/green]  final_score yellow={snap.yellow_score} blue={snap.blue_score}")

    if output is not None:
        json_path = output.with_suffix(".json")
        json_path.write_text(json.dumps(snap.to_json_dict(), indent=2))
        console.print(f"Saved final snapshot to {json_path}")


@app.command()
def replay(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    speed: float = typer.Option(1.0, help="Playback speed multiplier (CLI prints only, no UI)."),
) -> None:
    """Play back a replay file, printing one line per round end."""
    reader = ReplayReader(path)
    hdr = reader.header
    console.print(f"[bold]Replaying[/bold] {path.name}  map={hdr.map_name}  seed={hdr.seed}  team_size={hdr.team_size}")

    last_emit = time.time()
    interval = 1.0 / max(speed, 0.01)
    for ev in reader.iter_events():
        if ev.kind in {"round_end", "plant", "defuse", "detonate"}:
            console.print(f"tick={ev.tick:>5}  {ev.kind:<12}  {json.dumps(ev.data)}")
        now = time.time()
        if now - last_emit < interval:
            time.sleep(max(0.0, interval - (now - last_emit)))
        last_emit = time.time()


if __name__ == "__main__":
    app()
