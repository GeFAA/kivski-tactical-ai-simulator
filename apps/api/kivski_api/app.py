"""FastAPI application factory.

Construction order is intentional:

1. Load (or accept) a :class:`KivskiConfig` -- it controls CORS, broadcast
   rate, etc.
2. Build the :class:`FastAPI` instance with a lifespan that shuts down every
   live :class:`MatchSession` on exit.
3. Install permissive CORS (configurable via the ``KIVSKI_CORS_ORIGINS`` env
   var -- comma-separated list, defaults to the Vite dev server URLs).
4. Mount every router from :mod:`kivski_api.routes`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from kivski_sim.config import KivskiConfig, load_config
from kivski_sim.utils import now_unix

from kivski_api import __version__
from kivski_api.metrics_broadcaster import MetricsBroadcaster
from kivski_api.routes import checkpoints as checkpoints_routes
from kivski_api.routes import health as health_routes
from kivski_api.routes import maps as maps_routes
from kivski_api.routes import match as match_routes
from kivski_api.routes import system as system_routes
from kivski_api.routes import training as training_routes
from kivski_api.routes import ws as ws_routes
from kivski_api.session import REGISTRY, TrainingWatchdog
from kivski_api.training_clock import get_clock

__all__ = ["create_app"]

_LOG = logging.getLogger("kivski_api.app")

_DEFAULT_CORS = "http://localhost:5173,http://127.0.0.1:5173"

# How often the lifespan ticks the training clock. 15 s is short enough
# that a crash or shutdown loses at most ~15 s of attributed wall-clock,
# but long enough that the persistent JSON file isn't being rewritten
# at HTTP-request cadence.
_TRAINING_CLOCK_INTERVAL_SECONDS: float = 15.0


def _any_training_running() -> bool:
    """True iff any TrainingJob in the registry currently has a live process."""
    return any(job.is_running() for job in REGISTRY.training.values())


async def _training_clock_loop(stop_event: asyncio.Event) -> None:
    """Periodically advance the persistent training clock.

    Runs until ``stop_event`` is set. Catches and logs any per-tick
    error so a transient filesystem hiccup can't kill the loop.
    """
    clock = get_clock()
    while not stop_event.is_set():
        try:
            clock.tick(now_unix(), _any_training_running())
        except Exception:  # pragma: no cover - defensive
            _LOG.exception("training_clock tick raised")
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=_TRAINING_CLOCK_INTERVAL_SECONDS,
            )
        except TimeoutError:
            continue


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("KIVSKI_CORS_ORIGINS", _DEFAULT_CORS)
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["*"]


def create_app(cfg: KivskiConfig | None = None) -> FastAPI:
    """Build a fully-wired :class:`FastAPI` instance ready for uvicorn."""
    if cfg is None:
        try:
            cfg = load_config()
        except Exception:
            # Don't refuse to boot just because the YAML is missing; the
            # routes that need a config will load one on demand.
            cfg = KivskiConfig()

    # Make sure the root logger emits at INFO by default.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.getenv("KIVSKI_LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _LOG.info("Kivski API starting (version=%s, tick_hz=%s)", __version__, cfg.server.tick_broadcast_hz)
        # Start the live metrics broadcaster so any /api/training/start
        # call has a tail loop ready before the trainer writes its first
        # metrics line.
        broadcaster = MetricsBroadcaster(REGISTRY)
        await broadcaster.start()
        # Start the training watchdog so a crashed trainer subprocess
        # auto-restarts from the most recent checkpoint instead of
        # silently leaving the system idle.
        watchdog = TrainingWatchdog(REGISTRY)
        await watchdog.start()
        # Persistent training clock: every 15 s we record whether a
        # trainer was running, attributing the elapsed wall-clock to
        # the cumulative "total trained" counter. Loaded from disk on
        # boot so the counter survives PC restarts.
        clock_stop = asyncio.Event()
        clock_task = asyncio.create_task(
            _training_clock_loop(clock_stop),
            name="training-clock-loop",
        )
        try:
            yield
        finally:
            _LOG.info(
                "Kivski API shutting down -- stopping broadcaster + watchdog + %d match(es)",
                len(REGISTRY.sessions),
            )
            clock_stop.set()
            try:
                await asyncio.wait_for(clock_task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                clock_task.cancel()
            await watchdog.stop()
            await broadcaster.stop()
            await REGISTRY.shutdown()

    app = FastAPI(
        title="Kivski Tactical AI Simulator API",
        description="HTTP + WebSocket bridge between the training process and the live viewer.",
        version=__version__,
        lifespan=lifespan,
    )

    origins = _parse_cors_origins()
    _LOG.info("CORS origins: %s", origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers -- order doesn't matter, but keeping it consistent helps when
    # reading the OpenAPI schema.
    app.include_router(health_routes.router)
    app.include_router(maps_routes.router)
    app.include_router(checkpoints_routes.router)
    app.include_router(training_routes.router)
    app.include_router(match_routes.router)
    app.include_router(system_routes.router)
    app.include_router(ws_routes.router)

    return app
