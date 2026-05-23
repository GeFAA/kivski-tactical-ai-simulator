"""Uvicorn entry-point.

Two ways to run:

* ``python -m kivski_api.server`` -- reads host/port from the loaded
  :class:`KivskiConfig` (``configs/default.yaml`` by default).
* ``uvicorn kivski_api.server:app`` -- standard uvicorn invocation; ``app`` is
  a module-level instance for compatibility with ``uvicorn ... --reload``.
"""

from __future__ import annotations

import logging

import uvicorn

from kivski_sim.config import load_config

from kivski_api.app import create_app

app = create_app()


def main() -> None:
    """Run uvicorn using settings from the loaded config."""
    logging.getLogger("kivski_api").info("Starting server via kivski_api.server.main()")
    cfg = load_config()
    uvicorn.run(
        "kivski_api.server:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=False,
        ws="auto",
        log_level="info",
    )


if __name__ == "__main__":
    main()
