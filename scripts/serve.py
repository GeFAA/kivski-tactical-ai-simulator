"""CLI entry-point to launch the FastAPI + WebSocket server.

Exposed via the ``kivski-serve`` console script declared in ``pyproject.toml``.
We use Typer so the same command works as ``python scripts/serve.py serve ...``
or as ``kivski-serve serve ...`` after install.
"""

from __future__ import annotations

import os

import typer

app = typer.Typer(add_completion=False, help="Run the Kivski FastAPI + WebSocket server.")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    config: str = typer.Option("configs/default.yaml", help="Path to KivskiConfig YAML."),
    reload: bool = typer.Option(False, help="Enable uvicorn auto-reload (dev only)."),
    log_level: str = typer.Option("info", help="Uvicorn log level."),
) -> None:
    """Start the FastAPI + WebSocket server."""
    os.environ.setdefault("KIVSKI_DEFAULT_CONFIG", config)
    # Import inside the command so ``--help`` stays cheap even without uvicorn.
    import uvicorn

    uvicorn.run(
        "kivski_api.server:app",
        host=host,
        port=port,
        reload=reload,
        ws="auto",
        log_level=log_level,
    )


if __name__ == "__main__":
    app()
