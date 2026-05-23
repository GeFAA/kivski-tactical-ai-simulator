"""Map loader and queryable ``MapData`` container.

Loads JSON files from ``packages/maps/`` and exposes typed accessors used by
the engine, observation builder, and live viewer encoder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from kivski_sim.geometry import (
    point_in_polygon,
    polygon_aabb,
)
from kivski_sim.types import Side


def _maps_root() -> Path:
    """Return the directory holding map JSON files."""
    here = Path(__file__).resolve()
    # packages/sim/kivski_sim/map_loader.py -> packages/sim/kivski_sim -> packages/sim -> packages
    candidate = here.parents[2] / "maps"
    if candidate.is_dir():
        return candidate
    # Fallback: repo-root/packages/maps (when installed in editable mode from repo)
    for parent in here.parents:
        guess = parent / "packages" / "maps"
        if guess.is_dir():
            return guess
    return candidate  # last resort -- will error in load_map


class Bombsite(NamedTuple):
    """A bombsite has a centroid (for spawn / planting heuristics) and a polygon."""

    name: str
    center: np.ndarray
    polygon: np.ndarray


class Obstacle(NamedTuple):
    """A wall or cover polygon plus its blocking flags."""

    polygon: np.ndarray
    aabb: tuple[float, float, float, float]
    blocks_movement: bool
    blocks_sight: bool
    low: bool  # low cover: blocks bullets but lets sound / vision through


class NamedArea(NamedTuple):
    name: str
    polygon: np.ndarray
    aabb: tuple[float, float, float, float]


@dataclass(slots=True)
class MapData:
    """Loaded, query-ready map representation.

    All polygons are stored as ``np.ndarray`` of shape ``(N, 2)`` (float64) so
    they can be fed straight to the geometry kernels without per-call conversion.
    """

    name: str
    version: int
    width: int
    height: int
    tile_size: float
    spawns: dict[Side, np.ndarray]  # Side -> (5, 2) float array
    bombsites: dict[str, Bombsite]
    walls: list[Obstacle]
    cover: list[Obstacle]
    named_areas: list[NamedArea]

    # Convenience caches built once on load.
    _all_obstacles: list[Obstacle] = field(default_factory=list)
    _sight_obstacles: list[Obstacle] = field(default_factory=list)
    _movement_obstacles: list[Obstacle] = field(default_factory=list)

    # --------------------------------------------------------------
    # Predicates
    # --------------------------------------------------------------

    def is_blocked(self, pos: np.ndarray) -> bool:
        """True if the position falls inside any movement-blocking obstacle."""
        p = np.asarray(pos, dtype=np.float64)
        for ob in self._movement_obstacles:
            x0, y0, x1, y1 = ob.aabb
            if not (x0 - 1e-9 <= p[0] <= x1 + 1e-9 and y0 - 1e-9 <= p[1] <= y1 + 1e-9):
                continue
            if point_in_polygon(p, ob.polygon):
                return True
        # also out-of-bounds counts as blocked
        return bool(
            not (0.0 <= float(p[0]) <= float(self.width) and 0.0 <= float(p[1]) <= float(self.height))
        )

    def is_in_bombsite(self, pos: np.ndarray) -> str | None:
        """Return the bombsite name (``"A"``/``"B"``) containing ``pos`` else ``None``."""
        p = np.asarray(pos, dtype=np.float64)
        for name, site in self.bombsites.items():
            if point_in_polygon(p, site.polygon):
                return name
        return None

    def nearest_spawn(self, side: Side, agent_idx: int) -> np.ndarray:
        """Return the spawn coordinate assigned to ``agent_idx`` on ``side``."""
        arr = self.spawns[side]
        idx = int(agent_idx) % arr.shape[0]
        return arr[idx].copy()

    def area_name(self, pos: np.ndarray) -> str | None:
        """Name of the first named area containing ``pos``, else ``None``."""
        p = np.asarray(pos, dtype=np.float64)
        for area in self.named_areas:
            x0, y0, x1, y1 = area.aabb
            if not (x0 - 1e-9 <= p[0] <= x1 + 1e-9 and y0 - 1e-9 <= p[1] <= y1 + 1e-9):
                continue
            if point_in_polygon(p, area.polygon):
                return area.name
        return None

    # --------------------------------------------------------------
    # Geometry views (read-only handles for callers like visibility)
    # --------------------------------------------------------------

    def sight_blocking_polygons(self) -> list[np.ndarray]:
        """Polygons that block line-of-sight (walls + non-low cover)."""
        return [ob.polygon for ob in self._sight_obstacles]

    def movement_blocking_polygons(self) -> list[np.ndarray]:
        """Polygons that block movement (walls + all cover)."""
        return [ob.polygon for ob in self._movement_obstacles]

    def all_obstacles(self) -> list[Obstacle]:
        """All obstacles (walls + cover), preserving load order."""
        return list(self._all_obstacles)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _to_polygon(raw: list[list[float]]) -> np.ndarray:
    return np.asarray(raw, dtype=np.float64)


def _build_obstacle(raw: dict, default_low: bool = False) -> Obstacle:
    poly = _to_polygon(raw["polygon"])
    return Obstacle(
        polygon=poly,
        aabb=polygon_aabb(poly),
        blocks_movement=bool(raw.get("blocks_movement", True)),
        blocks_sight=bool(raw.get("blocks_sight", True)),
        low=bool(raw.get("low", default_low)),
    )


def list_maps() -> list[str]:
    """Return the sorted list of available map names (without ``.json``)."""
    root = _maps_root()
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.json"))


def load_map(name: str) -> MapData:
    """Load and parse the map JSON named ``name`` (without ``.json``)."""
    root = _maps_root()
    path = root / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Map '{name}' not found at {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    spawns: dict[Side, np.ndarray] = {
        Side.ATTACKER: np.asarray(raw["spawns"]["attacker"], dtype=np.float64),
        Side.DEFENDER: np.asarray(raw["spawns"]["defender"], dtype=np.float64),
    }

    bombsites: dict[str, Bombsite] = {}
    for site_name, site_raw in raw["bombsites"].items():
        bombsites[site_name] = Bombsite(
            name=site_name,
            center=np.asarray(site_raw["center"], dtype=np.float64),
            polygon=_to_polygon(site_raw["polygon"]),
        )

    walls = [_build_obstacle(w, default_low=False) for w in raw.get("walls", [])]
    cover = [_build_obstacle(c, default_low=False) for c in raw.get("cover", [])]

    named_areas: list[NamedArea] = []
    for area in raw.get("named_areas", []):
        poly = _to_polygon(area["polygon"])
        named_areas.append(NamedArea(name=area["name"], polygon=poly, aabb=polygon_aabb(poly)))

    md = MapData(
        name=raw["name"],
        version=int(raw.get("version", 1)),
        width=int(raw["width"]),
        height=int(raw["height"]),
        tile_size=float(raw.get("tile_size", 1.0)),
        spawns=spawns,
        bombsites=bombsites,
        walls=walls,
        cover=cover,
        named_areas=named_areas,
    )
    md._all_obstacles = [*walls, *cover]
    md._sight_obstacles = [ob for ob in md._all_obstacles if ob.blocks_sight]
    md._movement_obstacles = [ob for ob in md._all_obstacles if ob.blocks_movement]
    return md


__all__ = [
    "Bombsite",
    "MapData",
    "NamedArea",
    "Obstacle",
    "list_maps",
    "load_map",
]
