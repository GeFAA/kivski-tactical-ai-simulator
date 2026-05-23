# Changelog

All notable changes to the Kivski Tactical AI Simulator.

## [0.2.0] — 2026-05-23

### Added — Training throughput
- **`SubprocVecEnv`** (`packages/agents/kivski_agents/training/parallel_vec_env.py`):
  true multi-core parallelism via `torch.multiprocessing.spawn`. Worker hosts a
  fraction of envs behind a duplex Pipe. Windows-spawn compatible. Auto-fallback
  to `SyncVecEnv` if spawn fails.
- **`ThreadedVecEnv`**: lighter alternative using ThreadPoolExecutor (GIL is
  released during numpy/engine ops, gives ~30–50% boost without subprocess overhead).
- **`make_vec_env(kind=…)`** factory selecting `sync` / `thread` / `subproc`.
- **`auto_tune.py`**: `detect_optimal_num_envs` (max(8, min(64, cpu-2))) and
  `detect_optimal_workers` (max(1, min(num_envs, cpu-1))).
- **`Engine.step(..., light_snapshot=True)`** skips per-tick agent / event / message
  serialisation in the trainer hot path (viewer keeps full snapshot via
  `engine.snapshot()`).
- **`scripts/train.py`** CLI flags: `--vec-kind`, `--num-workers`, `--auto-envs`;
  default `vec_kind=subproc`; FPS reported prominently after smoke runs.
- **Config defaults**: `num_envs` 16→32, `minibatch_size` 1024→2048.

Measured env-step throughput (5v5 dustline, no model, hidden=64):
| vec_kind | num_envs | workers | step/s | speed-up |
|----------|----------|---------|--------|----------|
| sync     | 16       | —       | 1 475  | 1.0×     |
| subproc  | 16       | 8       | 4 969  | 3.4×     |
| subproc  | 32       | 16      | 6 693  | 4.5×     |
| subproc  | 64       | 16      | 6 727  | 4.5×     |

### Added — Live observability
- **`JSONLSink`** in `packages/agents/kivski_agents/telemetry.py`: one JSON record
  per metric push, flushed eagerly. `make_sink("csv")` now returns `MultiSink(CSV, JSONL)`
  so the live API broadcaster has a tail-friendly feed while CSV stays for offline
  analytics.
- **`MetricsBroadcaster`** (`apps/api/kivski_api/metrics_broadcaster.py`):
  asyncio lifespan task that polls every active `TrainingJob`'s `metrics.jsonl`,
  parses new lines and broadcasts `metrics_sample` + `training_status` frames to
  every WS subscriber. Thread-safe via `list()` copies; tolerant of corrupted lines.
- **`TrainingPanel`** sparklines now React-state-driven (was refs which silently
  pinned to first sample). Live policy / value / entropy update at every WS
  broadcast.
- **`GET /api/training/configs`** endpoint: lists `configs/*.yaml` so the dropdown
  is populated (returned `404` silently in v0.1).
- **`GET /api/system/info`** endpoint + new **Sys** tab in RightSidebar:
  cpu_count, cpu%, memory_total_gb / memory_used_gb / memory%, load average,
  platform, python, torch_version, cuda_available, uptime, pid.

### Fixed — Wire protocol
- **`postCommand` 404**: rewritten as a typed-union dispatcher that fans the UI
  command out to the correct per-action endpoint (`POST /api/match/{id}/pause`
  etc.). All buttons now hit a real backend route.
- **`store.currentMatchId`** + `setCurrentMatchId` exposed via api-client so
  match-scoped commands don't need the caller to thread the id through every
  layer.
- **Save Checkpoint button**: V1 no-op with hover hint ("trainer auto-saves
  every N episodes") — a dedicated endpoint will land later.

### Tests
- 8 new tests in `tests/integration/test_training_smoke.py`
  (subproc / threaded / make_vec_env / auto-tune). Subproc tests skip on
  CI (multiprocessing in CI runners is fragile).
- Full suite: **209 passed, 1 skipped** on a clean run.

### End-to-end verification
- New `scripts/e2e_smoke.py` drives Chromium via Playwright:
  loads the page, clicks **Start training**, captures WebSocket frames, takes
  full-page screenshots into `models/logs/e2e/{load,training,inspector}.png`.
  Run with `./.venv/Scripts/python.exe scripts/e2e_smoke.py`.

---

## [0.1.0] — 2026-05-23

Initial release. See README for the full feature list — engine, MAPPO + TarMAC,
PettingZoo env, React + PixiJS viewer, FastAPI WebSocket bridge, league self-play,
curriculum, eval suite.
