# Map format

Maps are plain JSON files in `packages/maps/`. The default and currently
only shipped map is `dustline.json`. Drop a new file into that directory
and reference it by name in the config:

```yaml
simulation:
  map: my_new_map
```

The loader, `kivski_sim.map_loader`, parses the JSON into a `MapData`
dataclass and builds the spatial indices used by visibility queries.

---

## 1. Top-level shape

```json
{
  "name": "string",
  "version": 1,
  "width": 60,
  "height": 40,
  "tile_size": 1.0,
  "description": "free-form, viewer-only",

  "spawns": {
    "attacker": [[x, y], ...],
    "defender": [[x, y], ...]
  },

  "bombsites": {
    "A": { "center": [x, y], "polygon": [[x, y], ...] },
    "B": { "center": [x, y], "polygon": [[x, y], ...] }
  },

  "walls":       [ { "polygon": [...], "blocks_movement": bool, "blocks_sight": bool }, ... ],
  "cover":       [ { "polygon": [...], "blocks_movement": bool, "blocks_sight": bool, "low": bool }, ... ],
  "named_areas": [ { "name": "string", "polygon": [...] }, ... ]
}
```

All polygons are arrays of `[x, y]` floats. Coordinates are in world units
(`tile_size` defaults to 1.0). The convention is **closed polygons listed
in CCW order**; the loader will re-orient if you ship CW.

---

## 2. Field reference

### `name` *(string, required)*

Used to look the map up by name. Must match the filename stem (e.g.
`dustline.json` -> `"name": "dustline"`).

### `version` *(int, required)*

Schema version. Currently always `1`. Bumping the value lets us migrate
incompatibly without breaking older replays.

### `width`, `height` *(float, required)*

World dimensions in tiles. The default `dustline` is `60 x 40`. The map is
axis-aligned with `(0, 0)` at the top-left, `+x` right, `+y` down (screen
coordinates).

### `tile_size` *(float, optional, default 1.0)*

Multiplier from world units to renderer units. Most code paths use world
units directly; the viewer uses `tile_size` only when computing pixel
sizes.

### `description` *(string, optional)*

Free-form text shown in the viewer's map inspector. Not consumed by the
engine.

### `spawns` *(required)*

```json
"spawns": {
  "attacker": [[x0, y0], ..., [x4, y4]],
  "defender": [[x0, y0], ..., [x4, y4]]
}
```

Each side must have **exactly `simulation.team_size` spawn points** (5 by
default; for curriculum stages with smaller team sizes, the first N are
used). Coordinates must be inside the map and *not* inside a wall.

### `bombsites` *(required)*

```json
"bombsites": {
  "A": { "center": [x, y], "polygon": [[x, y], ...] },
  "B": { "center": [x, y], "polygon": [[x, y], ...] }
}
```

Each entry has:

- `center` - planted-bomb anchor and the centre used by the observation's
  `bombsite_*_dx/dy` features.
- `polygon` - the area an attacker must stand inside to plant.

A map must have **exactly two bombsites**, keyed `"A"` and `"B"`.

### `walls` *(required, list)*

Solid obstacles. Each entry:

```json
{
  "polygon": [[x, y], ...],
  "blocks_movement": true,
  "blocks_sight": true
}
```

Walls almost always have both blocks set to `true`. The fields exist so
you can model invisible barriers if you ever need to.

### `cover` *(required, list)*

Tactical cover. Each entry:

```json
{
  "polygon": [[x, y], ...],
  "blocks_movement": true,
  "blocks_sight": true,
  "low": false
}
```

- `blocks_movement` - agents cannot path through.
- `blocks_sight` - line-of-sight queries are blocked. A `low: true` piece
  may block sight only when the looker is standing and the target is
  crouched (and vice versa); the engine handles the case.
- `low` - hint to the renderer and to the combat module that this is
  waist-high cover (peek-able).

### `named_areas` *(optional, list)*

Polygons with names for analytics. Not used by the engine for any rule;
the env exposes "agent X is in area Y" in its info dict, and the viewer
can colour or label regions.

```json
{ "name": "Mid", "polygon": [[22.0, 14.0], [38.0, 14.0], [38.0, 26.0], [22.0, 26.0]] }
```

Suggested areas worth including for any new map:

- `YellowSpawn`, `BlueSpawn`
- `BombsiteA`, `BombsiteB`
- `Mid` (the central contested area)
- Per-bombsite approach corridors (`A-Long`, `A-Short`, `B-Tunnel`, etc.)
- Per-team rotation corridors (`YellowFlank`, `BlueRotate`, etc.)

---

## 3. Coordinate system

```
   (0,0)                  (width, 0)
       +---------------------+
       |                     |
       |        Map          |    + x ->
       |                     |
       |                     |    v y
       +---------------------+
   (0, height)            (width, height)
```

- Origin at top-left.
- `+x` is east, `+y` is south.
- Compass `N` is `-y`, `S` is `+y`, `E` is `+x`, `W` is `-x`.
- Angles are measured from `+x` axis, CCW positive.

---

## 4. Constraints (validated on load)

The loader rejects a map if:

1. `name` is missing or does not match the filename stem.
2. `version != 1`.
3. `width <= 0` or `height <= 0`.
4. `spawns.attacker` or `spawns.defender` is empty.
5. A spawn point is outside `[0, width] x [0, height]` or inside a wall.
6. Either bombsite key is missing or has an empty polygon.
7. Any polygon has fewer than 3 vertices.
8. A spawn does not have at least one walkable neighbour (i.e. is fully
   enclosed by walls).

Soft (warned, not rejected) checks:

- Spawn or bombsite is inside a `cover` polygon (probably a typo).
- No path exists between spawn and the corresponding bombsite. (Validated
  by an A* run; warning only because intentionally walled-off maps may be
  useful for unit tests.)

You can validate a map without booting the engine:

```python
from kivski_sim.map_loader import load_map
map_data = load_map("packages/maps/my_new_map.json")
print(map_data.summary())
```

---

## 5. Example snippet (from `dustline.json`)

Header + first few obstacles:

```json
{
  "name": "dustline",
  "version": 1,
  "width": 60,
  "height": 40,
  "tile_size": 1.0,
  "description": "Original Kivski map 'Dustline' - two bombsites (A top-right, B bottom-left), Yellow spawn top-left, Blue spawn bottom-right. Multiple rotations through mid and outer corridors. All corridors connect via doorway gaps.",
  "spawns": {
    "attacker": [
      [3.5, 3.5], [5.0, 3.5], [6.5, 3.5],
      [4.0, 5.0], [6.0, 5.0]
    ],
    "defender": [
      [53.5, 36.5], [55.0, 36.5], [56.5, 36.5],
      [54.0, 35.0], [56.0, 35.0]
    ]
  },
  "bombsites": {
    "A": { "center": [50.0, 9.5], "polygon": [[46.0, 6.0], [54.0, 6.0], [54.0, 13.0], [46.0, 13.0]] },
    "B": { "center": [12.0, 31.5], "polygon": [[8.0, 28.0], [16.0, 28.0], [16.0, 35.0], [8.0, 35.0]] }
  },
  "walls": [
    { "polygon": [[0.0, 0.0], [60.0, 0.0], [60.0, 1.0], [0.0, 1.0]], "blocks_movement": true, "blocks_sight": true }
  ],
  "cover": [
    { "polygon": [[12.0, 10.0], [14.0, 10.0], [14.0, 12.0], [12.0, 12.0]], "blocks_movement": true, "blocks_sight": true, "low": false },
    { "polygon": [[48.0, 8.5], [49.5, 8.5], [49.5, 10.5], [48.0, 10.5]], "blocks_movement": true, "blocks_sight": false, "low": true }
  ],
  "named_areas": [
    { "name": "Mid",         "polygon": [[22.0, 14.0], [38.0, 14.0], [38.0, 26.0], [22.0, 26.0]] },
    { "name": "BombsiteA",   "polygon": [[46.0, 6.0],  [54.0, 6.0],  [54.0, 13.0], [46.0, 13.0]] },
    { "name": "BombsiteB",   "polygon": [[8.0, 28.0],  [16.0, 28.0], [16.0, 35.0], [8.0, 35.0]] }
  ]
}
```

See the full file at `packages/maps/dustline.json` for a complete map
with ~30 wall polygons, 12 cover pieces (mixed `low` / full), and 13
named areas.

---

## 6. Map design tips

A few hard-won rules from iterating on Dustline:

- **At least three approaches per bombsite.** Otherwise the defenders'
  optimal policy is trivially "hold all chokes" and the attackers have
  no learning gradient.
- **Asymmetric rotations.** Defender rotations should be slightly shorter
  than attacker rotations (post-plant balance).
- **Mid control should matter.** If `Mid` does not connect to both
  bombsites, the policy learns to ignore it and you lose the most
  interesting tactical layer.
- **Cover, not walls, near contact points.** Walls are binary; cover lets
  the combat model exercise its damage / accuracy curves.
- **Sound corridors.** Sound propagates around walls. Long unbroken
  corridors leak too much info; break them up with cover polygons that
  block sight.
- **Avoid one-tile-wide doorways.** They cause path-finding pile-ups and
  unrealistic kill cams.

The roadmap calls for a second original map with a more open layout to
stress-test the learned policies against a different sight-line
distribution.
