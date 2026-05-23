# Kivski Tactical AI Simulator

> A top-down 2D 5v5 bomb-defuse multi-agent reinforcement learning simulator
> with a live match viewer. Real ML, real emergence, no scripted strategies.

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Node](https://img.shields.io/badge/node-%E2%89%A520-339933.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#roadmap)
[![PettingZoo](https://img.shields.io/badge/API-PettingZoo%20Parallel-7c3aed.svg)](https://pettingzoo.farama.org/)

<!-- Screenshot placeholder. Drop a PNG at docs/screenshots/live-viewer.png and
     uncomment the line below. -->
<!-- ![Live viewer](docs/screenshots/live-viewer.png) -->

Kivski is an end-to-end research stack for studying emergent cooperation in
team-based tactical games. Two five-agent teams fight bomb-defuse rounds on
an original 60x40 grid map called *Dustline*. There are no scripted tactics:
all coordination - site holds, retakes, trades, callouts - must emerge from
multi-agent reinforcement learning. A React + PixiJS viewer streams the
ongoing match over a WebSocket so you can watch the policies play in real time.

---

## Highlights

- **Real multi-agent RL.** Recurrent MAPPO (PPO + centralised critic +
  shared parameters) with TarMAC-style learnable agent-to-agent
  communication. The comm channel is a learned vector with attention-based
  routing - not a hand-tuned ping system.
- **Original map "Dustline".** Data-driven JSON, 60x40 grid, two bombsites,
  multiple rotation paths, named tactical areas, configurable cover and walls.
- **Live match viewer.** React + TypeScript + PixiJS v8 (WebGL), 60 fps
  rendering with field-of-view, sound, and comm-flow overlays. Streamed from
  the trainer over a FastAPI WebSocket.
- **Self-play league + curriculum.** Random / scripted / frozen-snapshot
  opponents, Elo or TrueSkill rating, optional 1v1 -> 5v5 curriculum.
- **Deterministic engine.** Per-subsystem seeded RNG channels (combat, spawn,
  buy noise, sound, comm init, drop), msgpack replays, frame-exact rewind.
- **PettingZoo Parallel API.** Standard MARL interface; pluggable into
  Stable-Baselines3, RLlib, Sample Factory and other libraries with thin
  adapters.
- **No hardcoded roles or tactics.** Buy choices, position holds, trades,
  rotations and callouts are all learnable. The baselines (`scripted_rush`,
  `scripted_hold`) exist only as sparring partners and benchmarks.

## What it is (and what it isn't)

It **is**:
- A research-quality MARL environment with a non-trivial action space, partial
  observability, sparse outcome rewards, and a small dense-shaping signal.
- A clean reference implementation of recurrent MAPPO with autoregressive
  multi-head actions plus a learned communication channel.
- A reproducible, seedable simulator with replays and a full eval suite.
- A live spectator UI that makes emergent behaviour easy to inspect.

It **is not**:
- A copy of any commercial tactical shooter. Map, weapons, callouts, sound
  events and UI are original. No copyrighted names, models, icons, or
  trademarks are used.
- A production-ready esports-grade bot. V1 agents are intentionally weak; the
  architecture is what lets them improve over many matches of training.
- A real-time 3D game engine. The world model is a 2D grid with continuous
  positions, top-down rendered. Physics is intentionally light.

---

## Quick start (Windows 10 / 11)

### Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| Python    | 3.10 - 3.12 (3.11 recommended) | The pinned target in `pyproject.toml`. |
| Node.js   | >= 20 | For the React/Vite viewer. |
| RAM       | 8 GB minimum, 16 GB recommended | Scales with `training.num_envs`. |
| GPU       | Optional | CUDA-capable GPU speeds up MAPPO updates; CPU works. |

### Install

```powershell
# Clone
git clone https://github.com/<your-user>/kivski-tactical-ai-simulator.git
cd kivski-tactical-ai-simulator

# Python backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Optional: enable Weights & Biases logging
# pip install -e ".[wandb]"

# Optional: install everything (dev + wandb)
# pip install -e ".[all]"

# Frontend (Node workspace at the repo root)
npm install
```

If `Activate.ps1` is blocked, see
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md#powershell-execution-policy).

### Run a live match (no training required)

```powershell
# Terminal 1 -- start API + WebSocket server (default port 8000)
kivski-serve

# Terminal 2 -- start frontend (Vite dev server on http://localhost:5173)
npm run dev
```

Open <http://localhost:5173>. You will see a top-down view of *Dustline* with
random-policy agents playing a match. The right-hand panel shows the event
feed, the bottom bar exposes playback controls, and the debug toggles enable
field-of-view, sound, and communication overlays.

### Train

```powershell
# Quick smoke run (~1 minute, 2 envs, 2 episodes -- proves the loop boots)
kivski-train smoke --episodes 2

# Full training using the default config
kivski-train --config configs/default.yaml --episodes 50000

# Resume from a saved checkpoint
kivski-train --resume models/checkpoints/run-XXX-ep10000.pt

# Override hyperparameters from the CLI
kivski-train --num-envs 8 --episodes 20000 --device cuda
```

Checkpoints land in `models/checkpoints/`, replays in `models/replays/`,
and logs in `models/logs/` by default (configurable via env vars - see
[Configuration cheat sheet](#configuration-cheat-sheet)).

### Evaluate

```powershell
# Sanity check: random vs random
kivski-eval run random random --matches 20

# Trained policy vs a scripted hold opponent
kivski-eval run models/checkpoints/run-XXX-ep20000.pt scripted_hold --matches 50

# Discover what is available
kivski-eval list-scenarios
kivski-eval list-baselines

# Persist the head-to-head result as JSON
kivski-eval run random scripted_rush --matches 100 --output results/run.json
```

### Tests

```powershell
pytest                          # all unit + integration tests
pytest tests/unit -v            # only unit
pytest tests/integration -v     # API + env smoke
pytest -k "not slow"            # skip the slow smoke matches
pytest --cov=kivski_sim --cov=kivski_agents --cov-report=term-missing
```

---

## Architecture

```
+------------------------+      +-----------------+      +------------------+
| React + PixiJS viewer  | <==> | FastAPI + WS    | <==> | KivskiParallel   |
| (apps/web)             |  WS  | (apps/api)      |      | Env (PettingZoo) |
+------------------------+      +-----------------+      +------------------+
                                         |                       |
                                         v                       v
                                +-----------------+      +------------------+
                                | Trainer + MAPPO |      | Game Engine      |
                                | (kivski_agents) |----->| (kivski_sim)     |
                                +-----------------+      | + map + replays  |
                                         |               +------------------+
                                         v                       |
                                +-----------------+               v
                                | League + Elo +  |      +------------------+
                                | Curriculum      |      | Determinism via  |
                                +-----------------+      | RngHub channels  |
                                                         +------------------+
```

Each layer at a glance:

- **`kivski_sim` (engine).** Pure Python + NumPy + Numba. Deterministic
  tick-based engine, weapons, combat resolution, line-of-sight + sound
  propagation, economy, replays. Imports of PettingZoo/Gymnasium are deferred
  so the engine alone is a tiny dependency footprint suitable for batched
  training or use as a library.
- **`kivski_sim.env` (RL interface).** A PettingZoo `ParallelEnv` adapter on
  top of the engine. Builds per-agent flat observations, exposes a
  `MultiDiscrete([9, 6, 9, 8, 2*team_size+1])` action space, computes shaped
  rewards, and emits a per-tick snapshot for the viewer.
- **`kivski_agents` (RL stack).** Recurrent actor-critic networks with
  autoregressive action heads, TarMAC comm encoders + attention,
  Gumbel-Sigmoid comm gating, MAPPO loss and PPO update loop.
- **`kivski_agents.training`.** Vectorised env wrapper, rollout collector,
  league / opponent sampler, curriculum staging, top-level `Trainer` class.
- **`kivski_agents.eval`.** Standalone scenario specs (full pistol, full buy,
  retake 2v3, save round), head-to-head runner, Elo + TrueSkill trackers.
- **`kivski_agents.baselines`.** Drop-in `random`, `scripted_rush`,
  `scripted_hold` and a `frozen_snapshot` loader for saved checkpoints. All
  baselines share the same minimal `act` / `reset` interface as learned
  policies so the runner is policy-agnostic.
- **`kivski_api`.** FastAPI app with REST (`/api/...`) and WebSocket
  (`/ws/match`) routes. Wraps a live engine in a `MatchSession`, streams
  snapshots, exposes checkpoints + maps + training stats.
- **`apps/web`.** React 18 + TypeScript + Vite + PixiJS v8 + Zustand +
  Tailwind. WebSocket client with auto-reconnect, three-column shell, PixiJS
  WebGL renderer for the map and agents, overlays for FoV / sound / comms.

A deeper layer-by-layer walkthrough lives in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### Why this tech stack?

- **Python + PyTorch backend.** Standard for MARL research, mature MARL
  libraries, easy GPU acceleration, and the path of least resistance for
  collaborators familiar with the field.
- **NumPy-vectorised engine + Numba JIT.** Fast enough to hit ~1000 logical
  ticks/sec/env on a laptop. Pure Python data structures mean replays are
  cheap msgpack blobs that can be deterministically replayed bit-exact.
- **PettingZoo Parallel API.** Standard interface that future-proofs
  integration with Stable-Baselines3, RLlib, Sample Factory and other
  off-the-shelf trainers.
- **MAPPO + recurrent (GRU) + TarMAC attention comms.** The CTDE
  (centralised training, decentralised execution) paradigm: a single shared
  centralised critic sees the joint observation, while each actor only sees
  its own egocentric view. Recurrence handles partial observability across
  ticks; TarMAC-style attention over learned "thought vectors" provides
  bandwidth-limited inter-agent communication.
- **React + PixiJS v8 + Vite.** 60 fps top-down rendering, mature WebGL
  pipeline, fast HMR for iterating on the viewer, no game-engine lock-in.
- **FastAPI WebSocket.** Async-native, type-safe with Pydantic, runs in the
  same Python process tree as the ML stack so we can hand the live engine
  off without any IPC overhead.

---

## Game mechanics

A *match* is a best-of-N series of *rounds* between two five-agent teams.
Teams swap sides at half-time (`simulation.side_switch_round`). The first
team to a majority of rounds wins.

| Phase            | Trigger                       | What happens                                                   |
|------------------|-------------------------------|---------------------------------------------------------------|
| `BUY`            | Round start                   | Agents may spend money on weapons / armor via the `buy` head. |
| `LIVE`           | Buy timer ends                | Combat is enabled. Attackers carry / plant the bomb.          |
| `POST_PLANT`     | Bomb planted                  | Defenders must defuse; bomb timer counts down.                |
| `ROUND_OVER`     | Win condition reached         | Brief delay, payouts assigned, next round starts.             |
| `MATCH_OVER`     | Majority reached or rounds capped | Final scores frozen, match ends.                            |

A round can be won by:

- Eliminating the opposing team before plant (`ATTACKERS_ELIM` / `DEFENDERS_ELIM`).
- The bomb detonating (`BOMB_DETONATED` - attackers win).
- The bomb being defused (`BOMB_DEFUSED` - defenders win).
- The round timer expiring before plant (`TIMEOUT` - defenders win).

### Map system

- Data-driven JSON in `packages/maps/`. Drop a new file in and it becomes
  loadable by name.
- **Walls** are polygons that block both movement and sight.
- **Cover** polygons may block movement and/or sight independently and can be
  marked `low` (peek-able).
- Two **bombsites** (A and B on Dustline) with named centres + polygons.
- **Five spawns per side**, pre-placed in opposite corners.
- **Named areas** (`Mid`, `MidPit`, `A-Long`, `A-Short`, `B-Tunnel`,
  `B-Ramp`, `YellowSpawn`, `BlueSpawn`, ...) for analytics-friendly logging.

See [`docs/MAP_FORMAT.md`](docs/MAP_FORMAT.md) for the full schema.

### Combat model

- **Hit probability** = `base_accuracy * range_falloff * movement_penalty * cover_penalty`.
- **Damage** = `weapon_damage * armor_falloff * cover_multiplier`.
- **Reaction times** sampled per shot from the `combat` RNG channel,
  uniformly in `[reaction_time_min_ticks, reaction_time_max_ticks]`.
- Per-side, per-weapon accuracy and damage curves are tunable in
  `combat:` / weapon table of the config.

### Economy

- Per-round payouts based on outcome plus a loss-streak bonus.
- Plant / defuse / kill bonuses on top.
- Equipment carries over on survival. V1 simplification: all players respawn
  with the sidearm at round start; money persists between rounds.
- All payouts and bonuses live in the `economy:` block of the config and
  are deterministic given the seed.

---

## Multi-Agent RL design

### Observation space (per agent)

Egocentric, partial-observable. Each agent gets a flat
`float32` vector composed of:

- **Self block** - hp, armor, money, current weapon (one-hot over 7 classes),
  has-bomb flag, alive flag.
- **Self position** - normalised x, y, facing.
- **Teammate slots** (`agent.observation.teammate_slots`, default 4) -
  per teammate: alive, hp, relative dx/dy, distance, has-bomb, weapon id,
  money.
- **Last-known enemy slots** (`last_known_enemies`, default 5) -
  per enemy: age of the last sighting, relative dx/dy, weapon id, was-alive,
  distance at last observation. Enemies fade out of memory as their age
  grows.
- **Sound event slots** (`sound_event_slots`, default 6) -
  recent sounds the agent could hear (steps, shots, plant, defuse, pickup)
  with relative dx/dy, intensity, and a kind id.
- **Received message slots** (`received_message_slots`, default 5) -
  the categorical comm id plus a small payload vector from any teammate
  who chose to broadcast last tick.
- **Map context** - normalised displacements to both bombsites, time-in-round,
  one-hot of the current `Phase`.
- **Team context** - fraction of teammates alive, fraction of enemies
  alive (last known), bomb phase, my team's score, enemy's score,
  consecutive losses (for the loss-streak bonus shaping).

The exact width per section is computed by
`kivski_sim.obs_decoder.get_observation_dim(cfg)`. A field-by-field schema is
available via `build_observation_schema(cfg)`. See
[`docs/OBSERVATION_ACTION_SPEC.md`](docs/OBSERVATION_ACTION_SPEC.md) for a
complete slot-by-slot table.

### Action space

`MultiDiscrete([9, 6, 9, 8, 2 * team_size + 1])`:

| Head        | Size | Meaning |
|-------------|------|---------|
| **Move intent** | 9 | `HOLD` + 8 compass directions (N, NE, E, SE, S, SW, W, NW). |
| **Micro action** | 6 | `DEFAULT`, `CROUCH_HOLD`, `PEEK`, `SPRINT`, `FALL_BACK`, `INTERACT` (plant/defuse/pickup). |
| **Comm action** | 9 | `NONE` + 8 callout categories (`PING_LOCATION`, `WARN_DANGER`, ...). |
| **Buy choice** | 8 | `NONE`, `SIDEARM`, `HEAVY_PISTOL`, `SMG`, `SHOTGUN`, `RIFLE`, `PRECISION`, `ARMOR`. Only honored during `Phase.BUY`. |
| **Aim target** | `2*team_size + 1` | `-1` (no target) or one of the currently-visible enemies. |

Heads are sampled **autoregressively** - later heads see an embedding of
earlier choices, so e.g. the `aim_target` head can be conditioned on whether
the agent decided to `PEEK` or `SPRINT`.

### Communication channel (TarMAC-style)

- Each agent's `CommEncoder` produces a learned thought vector
  `(signature, value)` of dimension `ml.comm_embedding_dim` (default 64).
- Receivers attend over teammate signatures using multi-head attention
  (`ml.comm_attention_heads`, default 4) and aggregate values weighted by
  attention scores.
- A `CommGate` (Gumbel-Sigmoid) decides whether to actually broadcast the
  vector this tick, producing a sparse channel under load.
- The **discrete `CommAction` token** the agent picks is a human-readable
  *category* that the viewer renders for debugging. The **payload** is the
  learned vector. Categories are not hardcoded to specific meanings -
  semantics are learned end-to-end.

### Reward design

Sparse outcome rewards (dominant signal) plus a small shaping term that
linearly decays to zero over `reward_shaping.decay_after_episodes`:

| Source                       | Reward                       |
|------------------------------|------------------------------|
| Round win                    | `+1.0`                       |
| Round loss                   | `-1.0`                       |
| Plant / defuse               | `+0.5` / `+0.4` (decayable)  |
| Damage dealt (per HP)        | `+0.005` (decayable)         |
| Damage received (per HP)     | `-0.003` (decayable)         |
| Survival per second          | `+0.001` (decayable)         |
| Bomb pickup                  | `+0.05` (decayable)          |
| Useful trade (avenged kill)  | `+0.15` (decayable)          |
| Pointless death (no info)    | `-0.20` (decayable)          |
| Map control per tile         | `+0.0008` (decayable)        |

The shaping decay schedule prevents long-term reward hacking on heuristics
that may diverge from the true objective once the agent gets stronger.

### Training: League / PBT

- **Population** of opponents at every step: a `random` baseline, the two
  `scripted_*` baselines, and frozen snapshots of past selves.
- **Sampling fractions** for opponent mix configurable in `configs/default.yaml`:

  ```yaml
  league:
    population_size: 4
    snapshot_every_episodes: 1000
    exploit_fraction: 0.25
    random_fraction: 0.10
    scripted_fraction: 0.10
  ```

  The remainder of episodes are spent against the current self (pure
  self-play).
- **Rating**: built-in Elo tracker, or `TrueSkill` (Bayesian, factor-graph
  based) when the optional dependency is present.
- Snapshot freezing every N episodes prevents Rock-Paper-Scissors loops and
  keeps a diverse opponent pool.

### Curriculum (optional)

Disabled by default. When enabled (`training.curriculum.enabled: true`),
the trainer steps through:

| Stage          | Team size | Rounds | Economy | Episodes |
|----------------|-----------|--------|---------|----------|
| `1v1_no_eco`   | 1         | 6      | off     | 2 000    |
| `2v2_no_eco`   | 2         | 8      | off     | 3 000    |
| `3v3_basic_eco`| 3         | 12     | on      | 5 000    |
| `5v5_full`     | 5         | 24     | on      | 40 000   |

---

## Realistic expectations (be honest)

On a laptop, 50k - 100k training episodes will reliably produce:

- Aim and crosshair-placement improvement.
- Trade fragging on contested corners.
- Site-hold and stake-out patterns near choke points.
- Buy patterns that respond to money level and round outcome.

It will probably **not** produce, without significantly more compute:

- Human-level mid-round adaptation.
- Emergent "in-game leader" behaviour or stable team meta-strategies.
- Complex economic warfare across half-time.

A reasonable success bar is **outperforming the `scripted_rush` baseline
within ~20k episodes** in the default `default_5v5` eval scenario. Use the
provided eval suite and Elo tracker to measure rather than trust your gut.

---

## Project layout

```
kivski-tactical-ai-simulator/
|-- apps/
|   |-- web/                       # React + Vite + PixiJS live viewer
|   `-- api/                       # FastAPI + WebSocket bridge
|       `-- kivski_api/
|           |-- app.py             # FastAPI factory + CORS
|           |-- server.py          # uvicorn entry
|           |-- session.py         # MatchSession registry
|           |-- policies.py        # policy loading for live matches
|           `-- routes/            # health, maps, checkpoints, training, match, ws
|-- packages/
|   |-- sim/kivski_sim/            # deterministic game engine
|   |   |-- engine.py              # core tick loop
|   |   |-- env.py                 # PettingZoo ParallelEnv adapter
|   |   |-- combat.py              # weapons + hit resolution
|   |   |-- economy.py             # payouts, buy phase
|   |   |-- geometry.py            # polygons, raycasts
|   |   |-- visibility.py          # LoS, FoV, sound propagation
|   |   |-- map_loader.py          # JSON -> MapData
|   |   |-- obs_decoder.py         # flat-vector schema + decoder
|   |   |-- replay.py              # msgpack record / replay
|   |   |-- rng.py                 # per-channel seeded RNG hub
|   |   |-- state.py               # AgentState, WorldState
|   |   |-- types.py               # enums, dataclasses
|   |   |-- config.py              # pydantic config models
|   |   `-- utils.py
|   |-- agents/kivski_agents/      # MAPPO + TarMAC + training + league + eval
|   |   |-- networks/              # ObservationEncoder, RecurrentCore,
|   |   |                          # ActorHeads, ValueHead, CommEncoder,
|   |   |                          # CommAttention, CommGate, KivskiActorCritic
|   |   |-- mappo.py               # MAPPOLoss, MAPPOTrainer
|   |   |-- buffer.py              # rollout buffer
|   |   |-- factory.py             # build_model / build_trainer
|   |   |-- policy_runner.py       # PolicyBundle for the live API
|   |   |-- training/              # VecEnvWrapper, RolloutCollector,
|   |   |                          # LeagueManager, CurriculumManager, Trainer
|   |   |-- eval/                  # ScenarioSpec, EvalRunner, EloTracker
|   |   `-- baselines/             # random, scripted_rush, scripted_hold,
|   |                              # frozen_snapshot, registry
|   `-- maps/                      # original map JSONs (dustline.json)
|-- configs/                       # YAML configs
|   `-- default.yaml
|-- scripts/                       # console-script entry points
|   |-- eval.py                    # kivski-eval
|   `-- serve.py                   # kivski-serve
|-- tests/
|   |-- unit/                      # ~13 unit test modules
|   |-- integration/               # API + env smoke
|   `-- smoke/                     # end-to-end match runs
|-- models/                        # checkpoints / logs / replays (gitignored)
|-- docs/
|   |-- ARCHITECTURE.md
|   |-- ML.md
|   |-- MAP_FORMAT.md
|   |-- OBSERVATION_ACTION_SPEC.md
|   |-- ROADMAP.md
|   |-- TROUBLESHOOTING.md
|   `-- CONTRIBUTING.md
|-- .github/workflows/ci.yml
|-- .editorconfig
|-- .env.example
|-- .gitignore
|-- LICENSE
|-- README.md
|-- package.json                   # Node workspace root
|-- package-lock.json
`-- pyproject.toml
```

---

## Configuration cheat sheet

The most-tweaked fields in `configs/default.yaml`:

| Field                                       | Default | Why you'd change it |
|---------------------------------------------|---------|---------------------|
| `simulation.team_size`                      | 5       | Shrink for curriculum / debugging. |
| `simulation.tick_rate_hz`                   | 10      | Lower = coarser physics, faster wall-clock. |
| `simulation.max_rounds`                     | 24      | Shorter matches for quick smoke runs. |
| `training.num_envs`                         | 16      | Tune to your RAM and CPU cores. |
| `training.rollout_steps`                    | 256     | PPO rollout length per update. |
| `ml.learning_rate`                          | 3e-4    | Standard PPO default. |
| `ml.entropy_coef`                           | 0.015   | Raise to keep exploration alive longer. |
| `ml.ppo_clip`                               | 0.2     | Tightening lowers update aggressiveness. |
| `ml.comm_attention_heads`                   | 4       | Comm channel attention width. |
| `ml.comm_embedding_dim`                     | 64      | Comm vector width. |
| `reward_shaping.enabled`                    | true    | Disable for pure sparse training. |
| `reward_shaping.decay_after_episodes`       | 20000   | When the shaping reward fades to zero. |
| `league.population_size`                    | 4       | Number of snapshots kept in the pool. |
| `league.exploit_fraction`                   | 0.25    | Fraction of episodes vs frozen exploiters. |

CLI overrides: `--episodes`, `--num-envs`, `--map-name`, `--device`,
`--seed`, `--config`.

Environment-variable overrides (see `.env.example`):

| Variable                | Effect |
|-------------------------|--------|
| `KIVSKI_SEED`           | Master seed for all RNG channels. |
| `KIVSKI_NUM_ENVS`       | Override `training.num_envs`. |
| `KIVSKI_DEVICE`         | `auto` / `cpu` / `cuda`. |
| `KIVSKI_DEFAULT_CONFIG` | Default config path (used by the server too). |
| `KIVSKI_CHECKPOINT_DIR` | Where to write checkpoints. |
| `KIVSKI_REPLAY_DIR`     | Where to write replays. |
| `KIVSKI_LOG_DIR`        | Where to write CSV / TensorBoard logs. |
| `KIVSKI_CORS_ORIGINS`   | Comma-separated CORS allowlist for the API. |
| `WANDB_API_KEY`         | Enables W&B logging (requires the `wandb` extra). |
| `KIVSKI_WANDB_PROJECT`  | W&B project name. |
| `KIVSKI_WANDB_MODE`     | `online` / `offline` / `disabled`. |

---

## Originality / IP

This project is entirely original. No copyrighted maps, weapons, names,
sounds, icons, or UI from any existing commercial title are used. Generic
terminology only:

- **Sides**: "Attackers" / "Defenders" (yellow / blue teams).
- **Bombsites**: A and B.
- **Weapons**: `Blade` (knife), `ZP-9` (sidearm), `Kestrel-50` (heavy pistol),
  `Viper-Repeater` (SMG), `Hex-Rifle` (rifle), `Talon Marksman` (precision /
  sniper), `Maw-12` (shotgun).
- **Map**: `Dustline`.

If you ship a fork that adds maps, weapons, callouts, or assets, please keep
them original or appropriately licensed.

---

## Limitations & next steps

Honest list of what V1 does **not** do:

- Single map only (`Dustline`). Add more by dropping JSON into
  `packages/maps/` and pointing `simulation.map` at the new name.
- No dropped-weapon pickup (planned for V1.1).
- No grenades, smokes, or flash utility yet (planned).
- No subprocess-based vectorised env yet - only sync vec, single Python
  process. The engine releases the GIL through Numba in hot paths, but for
  serious training throughput a subprocess vec env is wanted.
- No distributed training. Single-machine only in V1.
- No on-policy distillation or asymmetric self-play (PSRO-style) - just a
  basic league + snapshot pool.

### Roadmap

Highlights from [`docs/ROADMAP.md`](docs/ROADMAP.md):

- Subprocess `VecEnv` for ~4-8x training throughput.
- Dropped-weapon pickup + smoke / flash utility.
- Second original map (open layout, more long sight lines).
- Distributed rollout via Ray or a custom message queue.
- Improved comm-channel visualisation (animated message routing in viewer).
- ONNX export path for the policy.
- Optional opponent-modelling head.

---

## Development

### Code style

| Language   | Tool   | Command                |
|------------|--------|------------------------|
| Python     | ruff   | `ruff check .`         |
| Python     | mypy   | `mypy packages tests`  |
| TypeScript | ESLint | `npm run lint`         |
| TypeScript | tsc    | `npm run typecheck`    |

Both linters run in CI on every push and PR
(see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

### Adding a new baseline

1. Implement a class with `reset(agent_names)` and
   `act(observations, received_comms=None) -> (actions, comm_payloads)` in
   `packages/agents/kivski_agents/baselines/`.
2. Register it in `baselines/registry.py` so the CLI and config can find it
   by name.
3. Add it to `evaluation.baselines` in your config to include it in the eval
   suite.

### Adding a new map

1. Create `packages/maps/<name>.json` (follow
   [`docs/MAP_FORMAT.md`](docs/MAP_FORMAT.md)).
2. Add a unit test in `tests/unit/test_map_loader.py` to lock in the
   geometry.
3. Train with `--map-name <name>` (or set `simulation.map: <name>` in a
   config).

### Adding a new evaluation scenario

1. Define a `ScenarioSpec` in
   `packages/agents/kivski_agents/eval/scenarios.py`.
2. Add it to `ALL_SCENARIOS`.
3. It will appear in `kivski-eval list-scenarios` automatically.

---

## Contributing

PRs welcome. Please:

1. Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) first to find the
   right layer for your change.
2. Open an issue for anything beyond a one-file change so we can sanity-check
   the direction before you spend time on it.
3. Keep `ruff check .` and `npm run lint` clean, and add unit tests for any
   new behaviour.
4. See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for the full workflow.

---

## License

[MIT](LICENSE). Do what you want, no warranty.

---

## Acknowledgements

- Das et al., *TarMAC: Targeted Multi-Agent Communication*, ICML 2019.
- Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent
  Games*, NeurIPS 2022.
- Terry et al., *PettingZoo: Gym for Multi-Agent Reinforcement Learning*,
  NeurIPS 2021.
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
- Jang et al., *Categorical Reparameterization with Gumbel-Softmax*, ICLR 2017.
