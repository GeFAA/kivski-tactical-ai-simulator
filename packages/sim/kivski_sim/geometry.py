"""Pure geometry helpers used by the map / visibility layer.

All functions operate on NumPy arrays expressed in *tile coordinates* (the same
units the engine moves agents in). The module is dependency-light on purpose so
it can also be imported by tooling (map preview, baked-LoS exporter, etc.).

Hot paths use ``@njit`` from numba when available; if numba cannot be imported
the same Python implementation is used directly so this module always works.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# numba shim (optional)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised via behaviour, not branches
    from numba import njit  # type: ignore

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        """No-op decorator used when numba is unavailable."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap


_EPS = 1e-9


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def vec_norm(v: np.ndarray) -> float:
    """Euclidean length of a 2D vector."""
    return float(math.hypot(float(v[0]), float(v[1])))


def vec_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))


def vec_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Signed angle in radians from vector ``a`` to vector ``b`` (-pi..pi)."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dot = ax * bx + ay * by
    det = ax * by - ay * bx
    return float(math.atan2(det, dot))


# ---------------------------------------------------------------------------
# Segment / polygon predicates (njit-friendly numeric core)
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _seg_intersect(
    a1x: float,
    a1y: float,
    a2x: float,
    a2y: float,
    b1x: float,
    b1y: float,
    b2x: float,
    b2y: float,
) -> bool:
    """Return True if segment a1-a2 intersects segment b1-b2 (proper or shared point)."""
    rx = a2x - a1x
    ry = a2y - a1y
    sx = b2x - b1x
    sy = b2y - b1y
    denom = rx * sy - ry * sx
    qmpx = b1x - a1x
    qmpy = b1y - a1y
    if abs(denom) < 1e-12:
        # Parallel / collinear -- treat as miss (we model touching as no hit
        # so an agent grazing a wall corner is not considered blocked).
        return False
    t = (qmpx * sy - qmpy * sx) / denom
    u = (qmpx * ry - qmpy * rx) / denom
    return (0.0 <= t <= 1.0) and (0.0 <= u <= 1.0)


def segment_intersects_segment(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    """Public wrapper around the numba kernel."""
    return bool(
        _seg_intersect(
            float(a1[0]),
            float(a1[1]),
            float(a2[0]),
            float(a2[1]),
            float(b1[0]),
            float(b1[1]),
            float(b2[0]),
            float(b2[1]),
        )
    )


@njit(cache=True, fastmath=True)
def _seg_hits_polygon(
    sx: float,
    sy: float,
    ex: float,
    ey: float,
    poly: np.ndarray,
) -> bool:
    """Return True if the segment crosses any edge of the closed polygon ``poly``."""
    n = poly.shape[0]
    for i in range(n):
        j = (i + 1) % n
        if _seg_intersect(sx, sy, ex, ey, poly[i, 0], poly[i, 1], poly[j, 0], poly[j, 1]):
            return True
    return False


def segment_intersects_polygon(seg_start: np.ndarray, seg_end: np.ndarray, polygon: np.ndarray) -> bool:
    """True if the segment crosses any edge of the polygon."""
    poly = np.asarray(polygon, dtype=np.float64)
    return bool(
        _seg_hits_polygon(
            float(seg_start[0]),
            float(seg_start[1]),
            float(seg_end[0]),
            float(seg_end[1]),
            poly,
        )
    )


@njit(cache=True, fastmath=True)
def _point_in_polygon(px: float, py: float, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (boundary points count as inside)."""
    n = poly.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi = poly[i, 0]
        yi = poly[i, 1]
        xj = poly[j, 0]
        yj = poly[j, 1]
        # boundary fast-path
        if (
            abs((yj - yi) * (px - xi) - (py - yi) * (xj - xi)) < 1e-9
            and min(xi, xj) - 1e-9 <= px <= max(xi, xj) + 1e-9
            and min(yi, yj) - 1e-9 <= py <= max(yi, yj) + 1e-9
        ):
            return True
        if (yi > py) != (yj > py):
            x_int = (xj - xi) * (py - yi) / (yj - yi + 1e-18) + xi
            if px < x_int:
                inside = not inside
        j = i
    return inside


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Public wrapper -- boundary points count as inside."""
    poly = np.asarray(polygon, dtype=np.float64)
    return bool(_point_in_polygon(float(point[0]), float(point[1]), poly))


# ---------------------------------------------------------------------------
# AABB helpers
# ---------------------------------------------------------------------------


def polygon_aabb(polygon: np.ndarray) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) of a polygon."""
    p = np.asarray(polygon, dtype=np.float64)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def polygons_overlap_aabb(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """Cheap axis-aligned bounding-box overlap test used as a broad-phase filter."""
    ax0, ay0, ax1, ay1 = polygon_aabb(poly_a)
    bx0, by0, bx1, by1 = polygon_aabb(poly_b)
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def segment_aabb_hit(
    sx: float,
    sy: float,
    ex: float,
    ey: float,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
) -> bool:
    """Liang-Barsky-style early-out: does segment touch the polygon AABB at all?"""
    seg_minx = min(sx, ex)
    seg_maxx = max(sx, ex)
    seg_miny = min(sy, ey)
    seg_maxy = max(sy, ey)
    return not (seg_maxx < minx or seg_minx > maxx or seg_maxy < miny or seg_miny > maxy)


# ---------------------------------------------------------------------------
# Circle / segment distance (used for projectile-radius / hit-box checks)
# ---------------------------------------------------------------------------


def circle_segment_distance(
    circle_center: np.ndarray,
    radius: float,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> float:
    """Shortest distance between a circle's edge and a segment (negative if overlap)."""
    cx, cy = float(circle_center[0]), float(circle_center[1])
    sx, sy = float(seg_start[0]), float(seg_start[1])
    ex, ey = float(seg_end[0]), float(seg_end[1])
    dx = ex - sx
    dy = ey - sy
    denom = dx * dx + dy * dy
    if denom < _EPS:
        return float(math.hypot(cx - sx, cy - sy) - radius)
    t = ((cx - sx) * dx + (cy - sy) * dy) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    closest_x = sx + t * dx
    closest_y = sy + t * dy
    return float(math.hypot(cx - closest_x, cy - closest_y) - radius)


# ---------------------------------------------------------------------------
# Raycasting against a collection of polygonal obstacles
# ---------------------------------------------------------------------------


def _segment_polygon_hit_t(
    sx: float,
    sy: float,
    ex: float,
    ey: float,
    poly: np.ndarray,
) -> float:
    """Return the smallest t in [0,1] where segment hits any edge; -1.0 if none."""
    best = 2.0
    n = poly.shape[0]
    for i in range(n):
        j = (i + 1) % n
        x1 = poly[i, 0]
        y1 = poly[i, 1]
        x2 = poly[j, 0]
        y2 = poly[j, 1]
        rx = ex - sx
        ry = ey - sy
        sx2 = x2 - x1
        sy2 = y2 - y1
        denom = rx * sy2 - ry * sx2
        if abs(denom) < 1e-12:
            continue
        qmpx = x1 - sx
        qmpy = y1 - sy
        t = (qmpx * sy2 - qmpy * sx2) / denom
        u = (qmpx * ry - qmpy * rx) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 and t < best:
            best = t
    return best if best <= 1.0 else -1.0


def raycast(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: Iterable[np.ndarray],
) -> tuple[bool, np.ndarray | None, float]:
    """Cast a ray from ``start`` to ``end`` against polygonal obstacles.

    Returns ``(hit, hit_point, distance)``. If nothing is hit, ``hit`` is False,
    ``hit_point`` is None, and ``distance`` is the full segment length.
    """
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    full = math.hypot(ex - sx, ey - sy)
    if full < _EPS:
        return False, None, 0.0
    best_t = 2.0
    for poly in obstacles:
        p = np.asarray(poly, dtype=np.float64)
        minx, miny, maxx, maxy = polygon_aabb(p)
        if not segment_aabb_hit(sx, sy, ex, ey, minx, miny, maxx, maxy):
            continue
        t = _segment_polygon_hit_t(sx, sy, ex, ey, p)
        if 0.0 <= t < best_t:
            best_t = t
    if best_t > 1.0:
        return False, None, full
    hx = sx + best_t * (ex - sx)
    hy = sy + best_t * (ey - sy)
    return True, np.array([hx, hy], dtype=np.float64), best_t * full


__all__ = [
    "vec_norm",
    "vec_distance",
    "vec_angle",
    "segment_intersects_segment",
    "segment_intersects_polygon",
    "point_in_polygon",
    "polygon_aabb",
    "polygons_overlap_aabb",
    "segment_aabb_hit",
    "circle_segment_distance",
    "raycast",
]
