# Architecture

This document is the layer-by-layer reference for the Kivski codebase. It
covers the data flow between the simulator, the RL stack, the API, and the
viewer, plus the process / threading model and the major design decisions.

For the algorithmic side - MAPPO, recurrent core, TarMAC, reward shaping -
read [`ML.md`](ML.md). For the observation/action wire format, read
[`OBSERVATION_ACTION_SPEC.md`](OBSERVATION_ACTION_SPEC.md).

---

## 1. Layer overview

```
+-------------------------------------------------------------------+
|                        Browser (apps/web)                         |
|                                                                   |
|  React + Zustand    PixiJS v8 (WebGL)    REST + WebSocket client  |
|        |                  |                       |               |
|        +------- App state / Map / Overlays -------+               |
+-------------------------------|-----------------------------------+
                                | WebSocket frames (msgpack/json)
                                v
+-------------------------------------------------------------------+
|                  FastAPI process (apps/api)                       |
|                                                                   |
|  REST routes       WS routes        MatchSession registry         |
|  /api/health       /ws/match             |                        |
|  /api/maps                               v                        |
|  /api/checkpoints       +-----------------------------------+     |
|  /api/training          | MatchSession                      |     |
|  /api/match             |   Engine + PolicyRunner(s)        |     |
|                         |   periodic snapshot -> WS clients |     |
|                         +-----------------------------------+     |
+-------------------------------|-----------------------------------+
                                | in-process call
                                v
+-------------------------------------------------------------------+
|                Simulator + RL stack (single process)              |
|                                                                   |
|  kivski_sim                       kivski_agents                   |
|    types, config                    networks (actor-critic)       |
|    map_loader, geometry             mappo (loss + trainer)        |
|    visibility (LoS, FoV, sound)     buffer (rollout)              |
|    rng (per-channel seeded)         training (Vec, Roll, League)  |
|    state, engine                    eval (scenarios, runner, Elo) |
|    env (PettingZoo Parallel)        baselines (random/scripted/   |
|    combat, economy                  frozen_snapshot)              |
|    obs_decoder, replay              telemetry, metrics            |
+-------------------------------------------------------------------+
```

The simulator and RL stack live in the same Python process as the API
server. Training runs in its own Python process (started via
`kivski-train`); the API and viewer talk to it over filesystem checkpoints
plus an optional shared snapshot publisher when running together.

---

## 2. Module map

### 2.1 `kivski_sim` - the engine

| Module | Responsibility |
|--------|----------------|
| `types.py` | Enums (`Side`, `Team`, `Phase`, `BombPhase`, `RoundOutcome`, `MatchOutcome`, `WeaponClass`, `MoveIntent`, `MicroAction`, `CommAction`, `BuyChoice`), dataclasses (`ActionBundle`, `Vec2`, `SoundEvent`, `Message`, `CombatEvent`, `RoundSummary`, `FrameMeta`), the `WEAPONS` table. Imports nothing heavy so it is safe to load anywhere. |
| `config.py` | Pydantic models that mirror `configs/default.yaml`. Layered overrides: file -> env var -> CLI flag. |
| `rng.py` | `RngHub` with isolated PCG64 channels (`combat`, `spawn`, `buy_noise`, `sound`, `comm_init`, `drop`). Channels are derived via BLAKE2b so seeds compose deterministically and channels are independent. Supports `snapshot()` / `restore()` for replay. |
| `geometry.py` | Polygon AABB, raycast, segment / polygon intersection helpers. NumPy-vectorised + a numba-JITted hot path. |
| `map_loader.py` | Parses JSON into `MapData` (walls, cover, bombsites, spawns, named areas). Builds spatial indices for fast LoS queries. |
| `visibility.py` | Line-of-sight, field-of-view cone tests, sound propagation with intensity falloff. |
| `combat.py` | Per-tick hit resolution. Pulls reaction-time samples from the `combat` RNG channel; consults the weapon table for damage and range curves. |
| `economy.py` | Buy-phase payouts, equipment carry-over, loss-streak bonuses. All numbers driven by `EconomyConfig`. |
| `state.py` | `AgentState`, `WorldState`. Plain dataclasses; no behaviour. The engine mutates these in place each tick. |
| `engine.py` | The core loop: process intents -> resolve movement -> update visibility -> resolve combat -> economy -> phase transitions. Emits a `Snapshot` per tick. |
| `env.py` | PettingZoo `ParallelEnv` adapter. Builds per-agent flat observation vectors, exposes `MultiDiscrete([9, 6, 9, 8, 2*team_size+1])` action space, computes rewards (sparse + shaped), and emits the per-tick snapshot consumed by the viewer. |
| `obs_decoder.py` | Section-wise schema + decoder for the flat observation vector. Used by tests, the viewer's inspector panel, and by `kivski_agents.factory.build_model` to size the input layer. |
| `replay.py` | Msgpack-based serialisation of `(RngHub.snapshot(), WorldState, action streams)` so a match can be deterministically replayed. |
| `logging_setup.py` | Rich + stdlib logging configuration. |
| `utils.py` | Small helpers. |

### 2.2 `kivski_agents` - the RL stack

| Module | Responsibility |
|--------|----------------|
| `networks/` | `ObservationEncoder` (MLP into hidden), `RecurrentCore` (GRU for temporal credit assignment), `CommEncoder` + `CommAttention` + `CommGate` (TarMAC), `ActorHeads` (autoregressive multi-head), `ValueHead` (centralised critic), `KivskiActorCritic` (top-level). |
| `mappo.py` | `MAPPOLoss` (clipped surrogate + value loss + entropy bonus + comm-gate regulariser) and `MAPPOTrainer` that owns the optimiser and applies gradient clipping. |
| `buffer.py` | `RolloutBuffer` (per-step storage with GRU hidden states) and `RolloutBatch` (the minibatch view fed to the loss). |
| `factory.py` | `build_model`, `build_trainer`, `default_action_dims(team_size)`, `infer_joint_obs_dim`. Single source of truth for "what is the standard model for this config". |
| `policy_runner.py` | `PolicyBundle` + `PolicyRunner` for inference - used by the API to step the live engine without involving the trainer. |
| `metrics.py` | Per-episode aggregators (winrate, plant rate, average buy spend, kill participation, ...). |
| `telemetry.py` | CSV / TensorBoard / W&B backends with a small unified interface. |
| `run_naming.py` | Generates deterministic-ish run ids (`run-2026-05-23-1430-7a91`). |
| `training/` | The trainer subpackage. See 2.3. |
| `eval/` | Standalone eval suite. See 2.4. |
| `baselines/` | Drop-in sparring partners. See 2.5. |

### 2.3 `kivski_agents.training`

| Module | Responsibility |
|--------|----------------|
| `vec_env.py` | `VecEnvWrapper` - sync vectorised env. Holds `N` `KivskiParallelEnv` instances and steps them in lockstep. |
| `rollout_collector.py` | Runs the actor for `rollout_steps`, populates a `RolloutBuffer`, returns the batch for PPO. |
| `league.py` | `LeagueEntry`, `LeagueManager`, `OpponentSampler`. Maintains the snapshot pool and decides per-episode which opponent to fight. |
| `curriculum.py` | `CurriculumManager`. Tracks current stage, advances on episode count, exposes `current_stage_overrides()` so the env is rebuilt with shrunken team size etc. |
| `trainer.py` | `Trainer` + `TrainerConfig`. Top-level orchestration: build envs, build model, loop {collect rollout, PPO update, log, maybe checkpoint, maybe snapshot to league, maybe advance curriculum}. |

### 2.4 `kivski_agents.eval`

| Module | Responsibility |
|--------|----------------|
| `scenarios.py` | `ScenarioSpec` dataclass + `ALL_SCENARIOS` (full pistol, full buy, retake 2v3, save round, default 5v5). `build_scenario()` materialises one into an env. |
| `runner.py` | `EvalRunner` - takes a scenario + two policies, plays N matches, returns `EvalResult` (winrates, plant/defuse rate, average match duration). |
| `elo.py` | `EloRating`, `EloTracker`, `TrueSkillTracker` (optional, requires `trueskill`). |

### 2.5 `kivski_agents.baselines`

| Module | Responsibility |
|--------|----------------|
| `random_policy.py` | `RandomBaseline` - uniformly samples actions. The trivially weakest sparring partner. |
| `scripted.py` | `ScriptedRushBaseline` and `ScriptedHoldBaseline` - simple FSM-style policies. Useful as a non-trivial benchmark. |
| `frozen_snapshot.py` | `FrozenSnapshotBaseline` - loads a saved checkpoint into a frozen network and runs inference only. |
| `registry.py` | `BASELINE_REGISTRY` mapping names -> constructors. `get_baseline(name, env, map_data, seed)` instantiates by name. |

### 2.6 `kivski_api` - the live API

| Module | Responsibility |
|--------|----------------|
| `app.py` | `create_app(cfg)` factory: builds the FastAPI instance, installs CORS, mounts every router, attaches a lifespan that drains running `MatchSession`s on shutdown. |
| `server.py` | uvicorn entry referenced from the CLI. |
| `session.py` | `MatchSession` + global `REGISTRY`. Owns one live engine + policy bundle per session, broadcasts snapshots to its WebSocket subscribers at `server.tick_broadcast_hz`. |
| `policies.py` | Helpers to load checkpoints (or baselines by name) into a `PolicyBundle` usable inside a `MatchSession`. |
| `routes/health.py` | `GET /api/health` - process and config sanity check. |
| `routes/maps.py` | `GET /api/maps`, `GET /api/maps/{name}` - serves the map JSON to the viewer. |
| `routes/checkpoints.py` | `GET /api/checkpoints` - lists saved checkpoints from `KIVSKI_CHECKPOINT_DIR`. |
| `routes/training.py` | `GET /api/training/runs`, etc. - read-only metrics for the dashboard. |
| `routes/match.py` | `POST /api/match` (start), `GET /api/match/{id}` (state), `DELETE /api/match/{id}` (stop). |
| `routes/ws.py` | `WS /ws/match/{id}` - subscriber socket that streams per-tick snapshots and accepts a few playback commands. |

### 2.7 `apps/web` - the viewer

| File | Responsibility |
|------|----------------|
| `src/App.tsx` | Three-column shell + WebSocket wiring. |
| `src/components/MapViewer.tsx` | PixiJS v8 WebGL renderer: map polygons, agents, bomb, FoV cones, sound rings, comm arrows. |
| `src/components/MatchHeader.tsx` | Score / round / phase / timer header strip. |
| `src/components/LeftSidebar.tsx` | Team rosters, per-player HP/money/weapon. |
| `src/components/RightSidebar.tsx` | Tabbed event feed / inspector / comm log. |
| `src/components/BottomControls.tsx` | Playback + training controls. |
| `src/components/DebugToggles.tsx` | Toggles for FoV / sound / comm overlays. |
| `src/lib/store.ts` | Zustand store - holds match state, UI state, debug toggles. |
| `src/lib/api-client.ts` | REST + WS client with auto-reconnect (exponential backoff). |
| `src/lib/map-loader.ts` | Map fetch with a built-in `dustline` placeholder fallback so the viewer still renders when the API is offline. |
| `src/lib/types.ts` | TypeScript types mirroring `types.py`. |

---

## 3. Data flow

### 3.1 Per-tick training data flow

```
configs/default.yaml
        |
        v
   KivskiConfig
        |
        +-- builds N KivskiParallelEnv (in VecEnvWrapper)
        |
        |   for step in range(rollout_steps):
        |       observations  <- env.last_obs
        |       hiddens, comm <- model.forward(observations)
        |       actions       <- ActorHeads.sample(...)
        |       env.step(actions)
        |       buffer.add(obs, actions, logp, value, reward, done, hidden, comm)
        |
        v
   RolloutBuffer.batch()
        |
        v
   for epoch in range(ppo_epochs):
        for minibatch in shuffle(batch):
             MAPPOLoss(...) -> backprop -> optim.step()
        |
        v
   Trainer.log_episode_stats()
        |
        +-- telemetry: csv / tensorboard / wandb
        |
        +-- every K episodes: checkpoint to KIVSKI_CHECKPOINT_DIR
        |
        +-- every L episodes: LeagueManager.snapshot(current_model)
        |
        `-- maybe: CurriculumManager.advance_if_ready()
```

### 3.2 Per-tick live-viewer data flow

```
Browser
   |  WS open -> /ws/match/{id}
   v
FastAPI /ws/match.ws_endpoint()
   |
   v
MatchSession.subscribe(socket)
   ^
   |  every 1 / tick_broadcast_hz seconds:
   |     Engine.tick(actions)
   |     snapshot = engine.snapshot()
   |     payload  = serialize(snapshot, frame_meta, events, comm_log)
   |     for socket in subscribers: socket.send_json(payload)
   |
Engine.tick(actions)
   ^
   |  actions from PolicyRunner.act(observations, received_comms)
   |
PolicyRunner
   |
   +-- learned: KivskiActorCritic.forward(...) -> sample()
   `-- baseline: RandomBaseline / ScriptedRushBaseline / ScriptedHoldBaseline
```

### 3.3 Replay flow

```
during training / live match:
   RngHub.snapshot()                 (state of every RNG channel)
   WorldState.snapshot()             (positions, hp, money, bomb)
   action_stream.append(per-tick actions)

replay file (msgpack):
   header  = { seed, map_name, config_hash }
   frames  = [ (tick, actions, rng_snapshot_or_none) ... ]
   footer  = { round_summaries, match_outcome }

on replay:
   Engine.load_replay(path)
   for frame in frames:
       if frame.rng_snapshot: RngHub.restore(frame.rng_snapshot)
       Engine.tick(frame.actions)
```

Replays are deterministic given the seed and the engine version. If the
engine changes incompatibly we bump the `version` field and refuse to replay
older blobs.

---

## 4. Process model

### 4.1 Training run

A single `python kivski-train` process:

- Main thread runs the trainer loop, the actor forward pass, the PPO update.
- `VecEnvWrapper` steps `N` envs synchronously inside the same process
  (subprocess vec is on the roadmap).
- `Telemetry` flushes CSV / TensorBoard / W&B on a background timer
  (`telemetry.flush_every_seconds`).
- No GPU sharing - the policy lives entirely on `KIVSKI_DEVICE`.

### 4.2 Live viewer

Two processes:

```
[ kivski-serve ] -- uvicorn ASGI loop
       |
       +-- /api/...   (request/response)
       +-- /ws/...    (long-lived asyncio.Task per socket)
       |
       MatchSession (one per active match)
         |
         +-- async tick_loop (period = 1 / tick_broadcast_hz)
         |     Engine.tick(actions)
         |     broadcast snapshot
         `-- PolicyRunner.act (sync call in the executor when learned)


[ npm run dev ] -- Vite dev server, port 5173
       |
       +-- proxies /api/* -> http://127.0.0.1:8000
       `-- proxies /ws/*  -> ws://127.0.0.1:8000
```

The viewer process is purely a static-asset + proxy server in dev. In
production you `npm run build` and serve `apps/web/dist/` directly (any
static host works).

### 4.3 Concurrency

- The engine is **single-threaded** per env. It is safe to call into the
  engine from a single asyncio task at a time.
- The trainer is **single-process** but uses NumPy / Numba / PyTorch BLAS
  which release the GIL inside their hot loops, so a multicore CPU is
  usable.
- The WebSocket layer is fully **async** (FastAPI / uvicorn). Each match
  session has one `asyncio.Task` driving the engine + broadcast loop, plus
  one consumer task per connected client to handle inbound playback
  commands.

---

## 5. Determinism

We promise: given the same `seed`, the same config, and the same code
version, two runs produce bit-exact observations, actions, and outcomes.

This is enforced by:

1. **`RngHub` channels.** Every stochastic call site (combat hit roll,
   spawn picker, buy noise, sound emission, comm init, drop point) uses its
   own named channel. Channels are derived from the master seed via
   BLAKE2b so they are independent.
2. **No `time.time()` / wall-clock dependencies** in the engine. The tick
   counter is the only source of "time".
3. **Sorted iteration** over dicts and sets in any hot path.
4. **Snapshot / restore** round-trips for replays. Tested in
   `tests/unit/test_replay.py`.
5. **PyTorch determinism flags** are *not* forced (we want CUDA throughput).
   Determinism therefore covers the engine end-to-end and the *actions* of
   a CPU model; CUDA models will diverge on different hardware.

---

## 6. Key design decisions and tradeoffs

| Decision | Why | Cost |
|----------|-----|------|
| **2D top-down, not 3D.** | An order of magnitude faster to simulate and reason about; lets us iterate on map design and balance in hours instead of weeks. | No vertical play, no peeker's advantage from elevation, no propagation of audio cues via 3D geometry. |
| **Numba JIT instead of C/Rust binding.** | Single language, easy to debug, easy for contributors to extend. | ~2-3x slower than a C-extension hot path. Acceptable for V1; can swap modules to a Rust extension later without changing the public Python API. |
| **PettingZoo Parallel rather than AEC.** | Lockstep team-game semantics; one observation/action set per tick is the right abstraction. | Some MARL libraries still expect AEC. We provide a thin adapter when needed. |
| **MAPPO (CTDE) instead of QMIX / independent PPO.** | Empirically strongest baseline for cooperative team games in 2022-2025 literature; centralised critic with shared parameters is sample-efficient. | Joint observation grows linearly with team size; the critic is a possible bottleneck above ~10v10. |
| **GRU instead of Transformer for the recurrent core.** | GRUs are fast on CPU, easy to backprop through, robust to long rollouts; transformers add complexity that buys little at our horizon length (~256 ticks). | We lose attention over the full history; the comm channel + observation history fields make up for it. |
| **TarMAC-style comm with Gumbel-Sigmoid gate.** | Gives agents bandwidth-limited, learned communication; the gate keeps the channel sparse so we do not collapse to "everyone shouts every tick". | Adds a hyperparameter (`gumbel_temperature`); requires care to keep the signature-attention temperature stable early in training. |
| **Sparse outcome reward + decaying shaping.** | Outcome reward is the only signal we trust long-term; shaping accelerates early learning but is risky if left on. The linear decay schedule lets us have both. | Choosing `decay_after_episodes` requires tuning per map / team size. |
| **YAML config + Pydantic models.** | Strongly typed config, easy CLI override, easy env-var override, easy to diff. | Slight friction adding new fields (have to touch both YAML and the dataclass). |
| **Snapshot pool league rather than PSRO.** | Cheap, reliably defeats Rock-Paper-Scissors loops, easy to reason about. | Less principled than PSRO; can still get stuck in a small region of strategy space. |
| **In-process API + engine.** | Zero IPC overhead, simple deployment. | The trainer cannot also serve the viewer concurrently with full throughput; we recommend running them in separate processes. |
| **Frontend uses raw PixiJS, not a higher-level game framework.** | We render mostly polygons + sprites; a framework would add API surface for no real win. | Slightly more boilerplate; we manually manage tick interpolation. |
| **No subprocess VecEnv yet.** | Sync vec is simpler to debug and adequate for V1 throughput targets on a laptop. | A real training run on a 16-core box leaves cycles on the table. Slated for V1.1. |

---

## 7. Performance notes

Approximate numbers on a developer laptop (Ryzen 7 5800H, no GPU):

| Workload | Throughput |
|----------|------------|
| Engine ticks per env (Numba warm) | ~1 000 / s |
| `KivskiParallelEnv.step` (5v5, dustline) | ~600 / s |
| `VecEnvWrapper(N=16).step` | ~7 000 step-calls / s |
| MAPPO update (rollout 256 * 16 envs, GRU=256, comm=64) | ~1.2 s / update on CPU; ~0.25 s on a single RTX 4060 |
| End-to-end episodes / hour (sync vec, CPU only) | ~700 |

These are intentionally modest - the simulator is not optimised for raw
throughput. The roadmap items most likely to improve the numbers are
subprocess vec env, Numba inlining of the visibility loop, and a Rust port
of `combat.py`.

---

## 8. Extension points

When you want to ...

- ... **add a weapon**: add a `WeaponStats` entry in `kivski_sim.types.WEAPONS`,
  add a new `WeaponClass` member, regenerate observation widths via the
  schema test.
- ... **add a phase**: extend the `Phase` enum, add a transition rule in
  `engine.py`, update the `_map_ctx_fields` in `obs_decoder.py`.
- ... **change reward shaping**: edit `reward_shaping:` in the config; the
  env reads these values at construction and applies them in
  `KivskiParallelEnv._compute_reward`.
- ... **add an actor head**: extend `ActorHeads`, update
  `default_action_dims`, extend `ActionBundle`, route the head through
  `engine.process_intents`.
- ... **add a comm callout**: add a `CommAction` enum member. The learned
  payload vector is unchanged - only the human-readable label and the
  viewer overlay change.
- ... **add a baseline**: see [`../README.md#adding-a-new-baseline`](../README.md#adding-a-new-baseline).
- ... **add a map**: see [`MAP_FORMAT.md`](MAP_FORMAT.md).
- ... **swap the recurrent core for a Transformer**: replace
  `RecurrentCore` in `kivski_agents.networks` and update the buffer
  hidden-state layout. Keep the input/output shapes the same so the rest of
  the stack does not care.

---

## 9. Glossary

- **Tick** - one logical engine step; `simulation.tick_rate_hz` per second.
- **Round** - one bomb-defuse round (buy -> live -> [post plant] -> round over).
- **Match** - a series of rounds; default 24 with a side switch at round 12.
- **CTDE** - Centralised Training, Decentralised Execution. The critic sees
  the joint observation during training but actors only see their own
  observation at inference.
- **MAPPO** - Multi-Agent PPO. PPO with parameter sharing across agents and
  a centralised critic.
- **TarMAC** - Targeted Multi-Agent Communication. Agents emit
  `(signature, value)` pairs; receivers attend over teammates' signatures
  to decide what to read.
- **Gumbel-Sigmoid** - Reparameterised relaxation of a Bernoulli, used here
  to make the comm gate differentiable.
- **League** - the population of opponents the trainer faces: random,
  scripted, frozen snapshots of past self.
- **Curriculum** - optional staged progression that grows team size and
  enables the economy over a training run.
- **Replay** - msgpack file containing the seed, action stream, and
  periodic RNG snapshots, enough to bit-exactly re-simulate a match.

---

See also: [`ML.md`](ML.md), [`OBSERVATION_ACTION_SPEC.md`](OBSERVATION_ACTION_SPEC.md),
[`MAP_FORMAT.md`](MAP_FORMAT.md), [`ROADMAP.md`](ROADMAP.md).
