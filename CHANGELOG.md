# Changelog

All notable changes to the Kivski Tactical AI Simulator.

## [0.3.0] — 2026-05-23

### Fixed — visual bugs reported by user
- **Round display** in header showed `"-"` instead of `"1"` for round 0
  (`{round || "-"}` treats `0` as falsy). Now: `formatRound(round, phase)` →
  "Warmup" / "Round N / max" / "Final"; timer shows `--:--` on match_over.
- **Agent teleport / jumpy motion**: `MapViewer` rebuilt the entire player
  layer on every snapshot with no interpolation. Replaced with a persistent
  per-agent `PlayerRenderState` and frame-accurate `lerp` via `app.ticker`
  (snap-on-round-change so respawns don't slide across the map).
- **Sidebar attackers/defenders flip after round-12 side-switch**: `LeftSidebar`
  grouped by `agent.side` which flips at the switch. Now groups by stable
  `team` identity (yellow / blue) with a "PLAYING AS ATTACKERS" subtitle
  that correctly flips on side-switch. Required adding `team` field to
  `AgentSnapshot` + `wire.ts` decoding.

### Added — see what you trained
- **Auto-load latest checkpoint** in live-match: `SessionRegistry.create_match()`
  with no explicit policy now loads the newest `models/checkpoints/*.pt` for
  both sides (falls back to `RandomPolicy` if none exist). Match response +
  WebSocket snapshot now expose `policy_yellow_name` + `policy_blue_name`.
- **Comparison mode**: `GET /api/checkpoints/recommended` returns the
  available pickable policies (`random` / `scripted_rush` / `scripted_hold` /
  `latest` / `best` / `<ckpt>.pt`). Frontend `MatchSetupModal` (opened via
  new "New Match" button) lets you pick yellow vs blue and start a head-to-head
  match — useful for A/B testing checkpoints against baselines or vs each other.
- **Live winrate strip** in header: when training is running, two small chips
  show `WR vs Random` and `WR vs Scripted` with ↑/↓ delta arrows, fed by the
  `winrate_vs_random` / `winrate_vs_scripted` keys the trainer writes into
  `metrics.jsonl` after each eval batch.
- **Policy badges** in header: `[N] Trained` (yellow) / `[R] Random` (gray) /
  `[S] Scripted` (blue) with tooltip showing the exact policy name; makes it
  obvious whether the viewer is showing learning or random.
- **Latest checkpoint section** in SystemInfo tab: name / episodes / created-at,
  polled every 10s.

### Added — faster convergence
- **Frame-skip** in `KivskiParallelEnv`: agents emit decisions every `N` ticks
  (config `simulation.frame_skip`, default `1` for live viewer / set explicitly
  for training). Same action repeats for the inner steps, rewards accumulate.
  Reduces exploration variance ~2-3×, accelerates wallclock convergence.
- **Reward curriculum** (`packages/agents/.../training/curriculum.py`):
  three stages by default — `killshoot` (0-5k eps, only kill+survive+damage),
  `objective` (5k-20k eps, adds plant/defuse), `full` (everything). Makes
  early training visibly more aggressive instead of randomly wandering.
  Trainer pushes stage changes to all vec_env backends (sync / threaded /
  subproc via Pipe command); `live/reward_curriculum_stage` logged to JSONL.
- **Trainer pushes live winrate** vs baselines into `metrics.jsonl` after
  every eval batch — the `MetricsBroadcaster` picks those up and pushes a
  `metrics_sample` WS frame. End-to-end verified.

### Tests
- 227 unit/integration passing locally (1 skipped); new tests for env
  frame-skip, reward-curriculum gating, recommended-policies endpoint,
  load_policy named variants, match comparison flow.

### End-to-end verification
- Playwright run on the new stack: page loads, "ROUND 1" displays correctly,
  sidebar shows "YELLOW TEAM / PLAYING AS ATTACKERS", policy badges visible,
  "New Match" button present in bottom controls, 0 console errors, 0 failed
  responses, 141 snapshot + 2 training_status WS frames received.

---

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
