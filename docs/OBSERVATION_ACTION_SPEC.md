# Observation and action space specification

This is the byte-for-byte reference for the per-agent observation vector
and the per-agent action bundle. If something here disagrees with the
code, the code (in particular `kivski_sim.obs_decoder` and
`kivski_sim.env.KivskiParallelEnv`) is authoritative; please open an
issue.

For the algorithmic side of how this is consumed see [`ML.md`](ML.md). For
where these structures live in the codebase see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 1. Observation vector

The observation passed to each agent is a **flat `np.float32` vector**.
The width depends on the config (slot counts), but the section ordering
is fixed:

```
[ self | self_pos | teammates[] | enemies[] | sounds[] | messages[] | map_ctx | team_ctx ]
```

`kivski_sim.obs_decoder.get_observation_dim(cfg)` returns the total
width. `build_observation_schema(cfg)` returns a nested dict with each
field's `(start, end, name)`.

Helper to decode a flat vector into a readable dict:

```python
from kivski_sim.obs_decoder import decode_observation, build_observation_schema
schema = build_observation_schema(cfg)
human = decode_observation(obs_vector, schema)
```

### 1.1 Section widths (default config)

| Section     | Slots                                            | Per-slot width | Total |
|-------------|--------------------------------------------------|----------------|-------|
| `self`      | 1                                                | `7 + 7 = 14`   | 14    |
| `self_pos`  | 1                                                | 3              | 3     |
| `teammates` | `agent.observation.teammate_slots` (default 4)   | 8              | 32    |
| `enemies`   | `agent.observation.last_known_enemies` (default 5)| 6              | 30    |
| `sounds`    | `agent.observation.sound_event_slots` (default 6) | 5              | 30    |
| `messages`  | `agent.observation.received_message_slots` (default 5) | 7         | 35    |
| `map_ctx`   | 1                                                | `6 + 4 = 10`   | 10    |
| `team_ctx`  | 1                                                | 6              | 6     |
| **Total**   |                                                  |                | **160** |

Default totals **160 floats**; the exact width is recomputed from your
config at env construction time.

### 1.2 `self` block - 14 floats

The agent's own state.

| Idx | Name                  | Range       | Meaning                                |
|----:|-----------------------|-------------|----------------------------------------|
|  0  | `hp_norm`             | `[0, 1]`    | HP / max HP.                            |
|  1  | `armor_norm`          | `[0, 1]`    | Armor / max armor.                      |
|  2  | `money_norm`          | `[0, 1]`    | Money / max-credits ceiling.            |
|  3  | `weapon_is_knife`     | `{0, 1}`    | One-hot over `WeaponClass`.             |
|  4  | `weapon_is_sidearm`   | `{0, 1}`    |                                         |
|  5  | `weapon_is_heavy_pistol` | `{0, 1}` |                                         |
|  6  | `weapon_is_smg`       | `{0, 1}`    |                                         |
|  7  | `weapon_is_rifle`     | `{0, 1}`    |                                         |
|  8  | `weapon_is_precision` | `{0, 1}`    |                                         |
|  9  | `weapon_is_shotgun`   | `{0, 1}`    |                                         |
| 10  | `has_bomb`            | `{0, 1}`    | 1 if this agent is the bomb carrier.    |
| 11  | `alive`               | `{0, 1}`    | 0 after death this round.               |
| 12  | (reserved tail)       | -           | Padding to round up section width.      |
| 13  | (reserved tail)       | -           | Same.                                   |

The two reserved tail floats keep the section width consistent in case
new `WeaponClass` values are added. They are zero today.

### 1.3 `self_pos` block - 3 floats

| Idx | Name          | Range       | Meaning |
|----:|---------------|-------------|--------|
|  0  | `pos_x_norm`  | `[0, 1]`    | x / map_width. |
|  1  | `pos_y_norm`  | `[0, 1]`    | y / map_height. |
|  2  | `facing_norm` | `[-1, 1]`   | `facing_radians / pi`. |

### 1.4 `teammates` slots - 8 floats per slot

`teammate_slots` entries, ordered by **stable agent id** so that the same
slot always refers to the same teammate across ticks. A slot is zeroed
out if the corresponding teammate is dead and we are past the dead-frame
fade.

| Idx | Name              | Range     | Meaning                                |
|----:|-------------------|-----------|----------------------------------------|
|  0  | `alive`           | `{0, 1}`  | 1 if alive this tick.                  |
|  1  | `hp_norm`         | `[0, 1]`  | Teammate HP.                            |
|  2  | `dx`              | `[-1, 1]` | `(other.x - self.x) / map_width`.       |
|  3  | `dy`              | `[-1, 1]` | `(other.y - self.y) / map_height`.      |
|  4  | `distance`        | `[0, 1]`  | Euclidean distance / map diagonal.      |
|  5  | `has_bomb`        | `{0, 1}`  | 1 if this teammate carries the bomb.    |
|  6  | `weapon_id_norm`  | `[0, 1]`  | `WeaponClass.value / (num_weapons-1)`.  |
|  7  | `money_norm`      | `[0, 1]`  | Teammate money.                         |

### 1.5 `enemies` slots - 6 floats per slot

`last_known_enemies` entries. Enemies you can currently see fill the
first slots; the rest are filled with last-known sightings, sorted by
recency (smallest `age_norm` first). A slot is zeroed if no sighting
exists.

| Idx | Name                  | Range     | Meaning                                |
|----:|-----------------------|-----------|----------------------------------------|
|  0  | `age_norm`            | `[0, 1]`  | `ticks_since_seen / max_age`.           |
|  1  | `dx`                  | `[-1, 1]` | Relative x (last known).                |
|  2  | `dy`                  | `[-1, 1]` | Relative y (last known).                |
|  3  | `weapon_id_norm`      | `[0, 1]`  | Weapon at last sighting.                |
|  4  | `was_alive`           | `{0, 1}`  | 1 if alive at the last sighting.        |
|  5  | `distance_at_last_obs`| `[0, 1]`  | Distance / map diagonal.                |

### 1.6 `sounds` slots - 5 floats per slot

Recent sounds inside this agent's hearing radius, including the
agent's own footsteps and gunshots (which is intentional - the agent
can learn to use them to predict opponent detection).

| Idx | Name           | Range     | Meaning                                  |
|----:|----------------|-----------|------------------------------------------|
|  0  | `age_norm`     | `[0, 1]`  | `ticks_since_sound / max_age`.           |
|  1  | `dx`           | `[-1, 1]` | Relative x (sound source).               |
|  2  | `dy`           | `[-1, 1]` | Relative y.                              |
|  3  | `intensity`    | `[0, 1]`  | Approximate received loudness.           |
|  4  | `kind_id_norm` | `[0, 1]`  | Sound kind id / num kinds (step/shot/plant/defuse/pickup). |

### 1.7 `messages` slots - 7 floats per slot

Messages received from teammates last tick. The TarMAC payload vector is
**not** in the observation - it is fed through the comm attention path
inside the network. What lives here is just the *meta* of the message.

| Idx | Name                   | Range     | Meaning                                  |
|----:|------------------------|-----------|------------------------------------------|
|  0  | `age_norm`             | `[0, 1]`  | Ticks since the message arrived.         |
|  1  | `sender_team_idx_norm` | `[0, 1]`  | Sender's index in the team.              |
|  2  | `comm_action_norm`     | `[0, 1]`  | `CommAction.value / (num_actions-1)`.    |
|  3  | `dx`                   | `[-1, 1]` | Relative x of sender at send time.       |
|  4  | `dy`                   | `[-1, 1]` | Relative y.                              |
|  5  | `has_payload`          | `{0, 1}`  | 1 if a learned payload was attached.     |
|  6  | `payload_norm`         | `[0, 1]`  | Scalar summary of the payload (norm).    |

### 1.8 `map_ctx` block - 10 floats

| Idx | Name              | Range     | Meaning |
|----:|-------------------|-----------|--------|
|  0  | `bombsite_a_dx`   | `[-1, 1]` | Bombsite A direction. |
|  1  | `bombsite_a_dy`   | `[-1, 1]` | |
|  2  | `bombsite_b_dx`   | `[-1, 1]` | Bombsite B direction. |
|  3  | `bombsite_b_dy`   | `[-1, 1]` | |
|  4  | `time_in_round_norm` | `[0, 1]` | `ticks_in_round / max_ticks_per_round`. |
|  5  | `reserved`        | -         | Reserved for a future "objective progress" scalar. |
|  6  | `phase_buy`       | `{0, 1}`  | One-hot phase. |
|  7  | `phase_live`      | `{0, 1}`  | |
|  8  | `phase_post_plant`| `{0, 1}`  | |
|  9  | `phase_round_over`| `{0, 1}`  | |

### 1.9 `team_ctx` block - 6 floats

| Idx | Name                       | Range     | Meaning |
|----:|----------------------------|-----------|--------|
|  0  | `teammates_alive_frac`     | `[0, 1]`  | (alive teammates) / team_size. |
|  1  | `enemies_alive_known_frac` | `[0, 1]`  | Same but for last-known enemy slots. |
|  2  | `bomb_phase_norm`          | `[0, 1]`  | `BombPhase.value / (num_phases-1)`. |
|  3  | `my_team_score_norm`       | `[0, 1]`  | `team_score / max_rounds`. |
|  4  | `enemy_team_score_norm`    | `[0, 1]`  | Same for the opposing team. |
|  5  | `consecutive_losses_norm`  | `[0, 1]`  | Loss streak / max-streak ceiling. |

---

## 2. Action space

`MultiDiscrete([9, 6, 9, 8, 2 * team_size + 1])` per agent.

```python
from kivski_agents.factory import default_action_dims
default_action_dims(team_size=5)
# -> [9, 6, 9, 8, 11]
```

### 2.1 Heads in order

| Head index | Name        | Size            | Source enum     |
|------------|-------------|-----------------|-----------------|
| 0          | move        | 9               | `MoveIntent`    |
| 1          | micro       | 6               | `MicroAction`   |
| 2          | comm        | 9               | `CommAction`    |
| 3          | buy         | 8               | `BuyChoice`     |
| 4          | aim_target  | `2*team_size+1` | int (see below) |

### 2.2 Head 0 - `MoveIntent`

| Value | Name | Vector (unit-length) |
|------:|------|----------------------|
| 0 | `HOLD` | `( 0,  0)` |
| 1 | `N`    | `( 0, -1)` |
| 2 | `NE`   | `( 0.707, -0.707)` |
| 3 | `E`    | `( 1,  0)` |
| 4 | `SE`   | `( 0.707, 0.707)` |
| 5 | `S`    | `( 0,  1)` |
| 6 | `SW`   | `(-0.707, 0.707)` |
| 7 | `W`    | `(-1,  0)` |
| 8 | `NW`   | `(-0.707, -0.707)` |

The engine multiplies the unit vector by the agent's current movement
speed (modulated by `MicroAction`).

### 2.3 Head 1 - `MicroAction`

| Value | Name           | Effect |
|------:|----------------|--------|
| 0 | `DEFAULT`        | Normal walk, weapon ready. |
| 1 | `CROUCH_HOLD`    | Crouched, accuracy boost (`combat.base_accuracy_crouched`), slower. |
| 2 | `PEEK`           | Brief shoulder peek - exposes for one tick to gather info. |
| 3 | `SPRINT`         | Faster movement, louder steps, worse accuracy (`base_accuracy_moving`). |
| 4 | `FALL_BACK`      | Crouch + walk backwards (engine reverses move vector). |
| 5 | `INTERACT`       | Plant (if attacker on bombsite with bomb) / defuse (if defender on planted bomb) / pickup (if standing on a dropped bomb). |

### 2.4 Head 2 - `CommAction`

| Value | Name              | Suggested viewer label (not enforced semantics) |
|------:|-------------------|------------------------------------------------|
| 0 | `NONE`              | No callout this tick. |
| 1 | `PING_LOCATION`     | "Here." |
| 2 | `WARN_DANGER`       | "Danger." |
| 3 | `REQUEST_SUPPORT`   | "Need support." |
| 4 | `SUGGEST_ROTATE`    | "Rotate." |
| 5 | `SUGGEST_ATTACK`    | "Push." |
| 6 | `SUGGEST_FALLBACK`  | "Fall back." |
| 7 | `CONTACT_ENEMY`     | "Contact." |
| 8 | `BOMBSITE_CLEAR`    | "Site clear." |

The discrete value is a *category label* shown in the viewer for human
debugging. The actual information routed to teammates is the learned
TarMAC payload vector (see [`ML.md`](ML.md#5-tarmac-targeted-multi-agent-communication)),
gated by the Gumbel-Sigmoid comm gate.

### 2.5 Head 3 - `BuyChoice`

Only acted on during `Phase.BUY`. Outside of buy phase the engine ignores
the value.

| Value | Name           | Cost (default) |
|------:|----------------|----------------|
| 0 | `NONE`           | 0 |
| 1 | `SIDEARM`        | 0 (free upgrade from knife) |
| 2 | `HEAVY_PISTOL`   | 700 |
| 3 | `SMG`            | 1500 |
| 4 | `SHOTGUN`        | 1100 |
| 5 | `RIFLE`          | 2700 |
| 6 | `PRECISION`      | 4200 |
| 7 | `ARMOR`          | 1000 |

If the agent cannot afford the chosen item, the buy is silently dropped
(no error, no negative reward beyond the implicit cost of forgoing the
purchase).

### 2.6 Head 4 - `aim_target`

`2 * team_size + 1` possible values. Encoded as:

| Range            | Meaning |
|------------------|---------|
| `0`              | No specific target. |
| `1 ... team_size`| Index into the **enemy slots** (1-based). |
| `team_size+1 ... 2*team_size` | Index into the **teammate slots** (1-based). Used for healing / cover swap in future versions; currently a no-op other than the autoregressive conditioning. |

When the chosen target is not currently visible / valid, the engine
falls back to the agent's free-fire mode (shoot at facing).

### 2.7 Sampling order

Heads are sampled **autoregressively** by `ActorHeads`:

```
move   -> micro -> comm -> buy -> aim_target
```

Each later head receives an embedding of all earlier sampled values
concatenated to the GRU output before its logits are computed. Sample
log-probability is the sum of the per-head log-probabilities; entropy is
the sum of per-head entropies.

---

## 3. Comm channel data flow

The discrete `CommAction` plus a learned payload vector form the message:

```python
@dataclass(slots=True)
class Message:
    tick: int
    sender: AgentId
    receivers: tuple[AgentId, ...]
    action: CommAction
    payload: np.ndarray | None         # learned vector (TarMAC)
    pos: tuple[float, float] | None    # location ping if applicable
```

At each tick:

1. Every alive agent computes `(sig_i, val_i)` via `CommEncoder`.
2. `CommGate` decides whether to actually broadcast (Gumbel-Sigmoid).
3. Receivers run `CommAttention` over teammate signatures and aggregate
   teammate values weighted by attention.
4. The aggregated read vector is concatenated to the receiver's next
   observation embedding.
5. For the observation log + viewer, a compact `messages` slot entry is
   built (see [1.7](#17-messages-slots---7-floats-per-slot)) so the
   discrete category and meta are observable to teammates as plain
   features.

Visualisation in the viewer draws an arrow from sender to receivers
when `CommAction != NONE` and the gate fires; colour encodes the
`CommAction` category.

---

## 4. Partial observability

This section is informative; behaviour is enforced by
`kivski_sim.visibility`.

### 4.1 Field of view (FoV)

- Each agent has a forward cone (default 110 degrees).
- Inside the cone, target visibility is determined by the LoS check
  (segment intersected with walls / cover, taking `blocks_sight` and
  `low` into account).
- A `PEEK` action temporarily widens the cone for that tick.

### 4.2 Sound

- Movement, shots, plant, defuse, pickup all emit `SoundEvent`s with a
  `radius` and an `intensity`.
- Listeners inside the radius receive a sound slot with intensity
  attenuated by distance and by `blocks_sight` walls in between (walls
  also dampen sound).
- Sound events do **not** reveal the agent's identity or weapon;
  only position, kind, and intensity.

### 4.3 Last-known enemies

- When an enemy enters another agent's FoV with LoS, the env writes a
  fresh sighting into the team-wide last-known store.
- The store fades entries over time (`age_norm`). Once `age_norm` reaches
  1.0 the slot is recycled.
- Sightings are **shared across the team**: any teammate seeing an enemy
  populates the slot for everyone. This is the engine-level shortcut
  that mimics what perfect-info comms would otherwise have to convey;
  the *learned* comm channel still has to convey richer information like
  intent.

### 4.4 Dead-frame fade

When a teammate dies, their `teammates` slot keeps producing valid
position + HP for a few ticks before going to zero, so the policy has
time to react to the death rather than instantly losing the feature.

---

## 5. Versioning

This document tracks observation schema version `1` (matches
`OBSERVATION_SCHEMA = build_observation_schema(KivskiConfig())` at the
top of `obs_decoder.py`). Any change that adds, removes, or reorders
fields requires bumping the schema version and invalidating older
replays.
