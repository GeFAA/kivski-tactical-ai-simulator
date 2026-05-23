"""Line-of-sight, field-of-view, and sound propagation queries.

These helpers consume a :class:`MapData` (built by :mod:`kivski_sim.map_loader`)
and the engine's agent positions to answer:

* Can agent A see agent B right now?
* Who lies inside an agent's FOV cone?
* How loud does a noise sound to a listener through the local geometry?
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from kivski_sim.geometry import (
    polygon_aabb,
    raycast,
    segment_aabb_hit,
    segment_intersects_polygon,
    vec_distance,
)
from kivski_sim.map_loader import MapData, Obstacle


# ---------------------------------------------------------------------------
# Line of Sight
# ---------------------------------------------------------------------------


def compute_los(
    map_data: MapData,
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    max_range: float,
) -> tuple[bool, float, bool]:
    """Check whether ``pos_a`` has line-of-sight to ``pos_b``.

    Returns ``(visible, distance, through_cover)``.

    * ``visible`` is False if any fully-sight-blocking obstacle (wall or
      non-low cover) intersects the segment, or if the distance exceeds
      ``max_range``.
    * ``through_cover`` is True when the segment crosses at least one *low*
      cover polygon -- vision still passes but the engine should apply a
      cover damage multiplier and accuracy penalty.
    """
    a = np.asarray(pos_a, dtype=np.float64)
    b = np.asarray(pos_b, dtype=np.float64)
    dist = vec_distance(a, b)
    if dist > max_range or dist <= 0.0:
        if dist <= 0.0:
            return True, 0.0, False
        return False, dist, False

    through_cover = False
    sx, sy = float(a[0]), float(a[1])
    ex, ey = float(b[0]), float(b[1])

    for ob in map_data.all_obstacles():
        x0, y0, x1, y1 = ob.aabb
        if not segment_aabb_hit(sx, sy, ex, ey, x0, y0, x1, y1):
            continue
        if not segment_intersects_polygon(a, b, ob.polygon):
            continue
        if ob.blocks_sight:
            return False, dist, False
        if ob.low:
            through_cover = True
    return True, dist, through_cover


# ---------------------------------------------------------------------------
# Field of View cone
# ---------------------------------------------------------------------------


DEFAULT_FOV_RADIANS = math.pi * 0.8   # ~144 degrees, fits the 2D top-down style


def _angular_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference between two radian angles."""
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


def compute_fov(
    map_data: MapData,
    agent_pos: np.ndarray,
    facing_angle: float,
    fov_radians: float,
    max_range: float,
    all_targets: Sequence[tuple[int, np.ndarray]],
) -> list[int]:
    """Return the ids of targets visible to an agent.

    ``all_targets`` is an iterable of ``(target_id, position)`` pairs. The
    function applies the FOV cone test first (cheap) and then a LoS raycast
    against the map's sight-blocking polygons (expensive).
    """
    half = max(0.0, float(fov_radians)) * 0.5
    origin = np.asarray(agent_pos, dtype=np.float64)
    visible: list[int] = []
    for target_id, target_pos in all_targets:
        tp = np.asarray(target_pos, dtype=np.float64)
        dx = float(tp[0] - origin[0])
        dy = float(tp[1] - origin[1])
        if dx * dx + dy * dy < 1e-12:
            visible.append(int(target_id))
            continue
        angle = math.atan2(dy, dx)
        if _angular_diff(angle, float(facing_angle)) > half:
            continue
        seen, _, _ = compute_los(map_data, origin, tp, max_range)
        if seen:
            visible.append(int(target_id))
    return visible


# ---------------------------------------------------------------------------
# Sound propagation
# ---------------------------------------------------------------------------


_COVER_LOW_ATTENUATION = 0.85    # low cover barely muffles
_WALL_ATTENUATION = 0.55         # solid walls muffle a lot (but do not block)
_MIN_AUDIBLE_STRENGTH = 0.04     # below this the listener simply hears nothing


def _sound_attenuation(map_data: MapData, listener: np.ndarray, source: np.ndarray) -> float:
    """Return a multiplicative attenuation factor in (0, 1] for the path."""
    factor = 1.0
    sx, sy = float(source[0]), float(source[1])
    ex, ey = float(listener[0]), float(listener[1])
    for ob in map_data.all_obstacles():
        x0, y0, x1, y1 = ob.aabb
        if not segment_aabb_hit(sx, sy, ex, ey, x0, y0, x1, y1):
            continue
        if not segment_intersects_polygon(source, listener, ob.polygon):
            continue
        if ob.low:
            factor *= _COVER_LOW_ATTENUATION
        elif ob.blocks_sight:
            factor *= _WALL_ATTENUATION
        # if we ever add purely-decorative non-blocking obstacles, ignore them
    return factor


def sound_audible(
    map_data: MapData,
    listener_pos: np.ndarray,
    sound_pos: np.ndarray,
    sound_intensity: float,
    sound_radius: float,
    rng: np.random.Generator | None = None,
) -> tuple[bool, float, tuple[float, float]]:
    """Decide whether a sound is audible to ``listener_pos``.

    Returns ``(heard, perceived_strength, approximate_pos)``.

    * Sound travels through walls (it is not LoS), but each obstacle on the
      ray-path attenuates the perceived strength.
    * The approximate position has gaussian jitter that grows with distance,
      so far-away noises only give a coarse direction.
    """
    listener = np.asarray(listener_pos, dtype=np.float64)
    source = np.asarray(sound_pos, dtype=np.float64)
    dist = vec_distance(listener, source)
    if sound_radius <= 0.0 or dist > sound_radius:
        return False, 0.0, (float(source[0]), float(source[1]))

    # Linear distance falloff inside the radius.
    falloff = max(0.0, 1.0 - dist / sound_radius)
    attenuation = _sound_attenuation(map_data, listener, source)
    strength = float(sound_intensity) * falloff * attenuation

    if strength < _MIN_AUDIBLE_STRENGTH:
        return False, strength, (float(source[0]), float(source[1]))

    # Position jitter: bigger when sound is faint / far away.
    if rng is None:
        rng = np.random.default_rng()
    jitter_sigma = 0.5 + 1.5 * (1.0 - strength)
    jitter = rng.normal(loc=0.0, scale=jitter_sigma, size=2)
    approx = (float(source[0] + jitter[0]), float(source[1] + jitter[1]))
    return True, strength, approx


# ---------------------------------------------------------------------------
# Grid sampling raycast
# ---------------------------------------------------------------------------


def raycast_grid_sample(
    map_data: MapData,
    start: np.ndarray,
    end: np.ndarray,
    step: float = 0.5,
) -> tuple[bool, np.ndarray | None, float]:
    """Sample the segment at ``step`` intervals and return the first blocked point.

    This is the cheap fallback used by some heuristic agents that need to
    approximate "is there cover between me and that spot?" without computing
    polygon intersections. For exact tests use ``geometry.raycast`` against
    :meth:`MapData.sight_blocking_polygons` directly.
    """
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(end, dtype=np.float64)
    dist = vec_distance(a, b)
    if dist <= 0.0:
        return False, None, 0.0
    n = max(2, int(math.ceil(dist / max(step, 1e-3))))
    sight_polys = map_data.sight_blocking_polygons()
    for i in range(1, n + 1):
        t = i / n
        p = a + (b - a) * t
        for poly in sight_polys:
            x0, y0, x1, y1 = polygon_aabb(poly)
            if not (x0 - 1e-6 <= p[0] <= x1 + 1e-6 and y0 - 1e-6 <= p[1] <= y1 + 1e-6):
                continue
            # If this sample is inside a sight-blocking polygon, we hit it.
            from kivski_sim.geometry import point_in_polygon
            if point_in_polygon(p, poly):
                return True, p.copy(), t * dist
    # Fallback to exact raycast for final answer when no grid sample landed inside.
    return raycast(a, b, sight_polys)


def visible_obstacles(
    map_data: MapData,
    pos: np.ndarray,
    max_range: float,
) -> Iterable[Obstacle]:
    """Iterate obstacles whose AABB sits within ``max_range`` of ``pos``.

    Used to prune the obstacle list before per-target LoS checks.
    """
    px, py = float(pos[0]), float(pos[1])
    r2 = max_range * max_range
    for ob in map_data.all_obstacles():
        x0, y0, x1, y1 = ob.aabb
        cx = max(x0, min(px, x1))
        cy = max(y0, min(py, y1))
        if (cx - px) ** 2 + (cy - py) ** 2 <= r2:
            yield ob


__all__ = [
    "DEFAULT_FOV_RADIANS",
    "compute_fov",
    "compute_los",
    "raycast_grid_sample",
    "sound_audible",
    "visible_obstacles",
]
