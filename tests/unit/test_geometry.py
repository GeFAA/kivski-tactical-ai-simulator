"""Unit tests for kivski_sim.geometry and kivski_sim.visibility primitives."""

from __future__ import annotations

import math

import numpy as np
from kivski_sim.geometry import (
    circle_segment_distance,
    point_in_polygon,
    polygons_overlap_aabb,
    raycast,
    segment_intersects_polygon,
    segment_intersects_segment,
    vec_angle,
    vec_distance,
)
from kivski_sim.map_loader import MapData, Obstacle
from kivski_sim.types import Side
from kivski_sim.visibility import (
    DEFAULT_FOV_RADIANS,
    compute_fov,
    compute_los,
    sound_audible,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _square(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Build a CCW axis-aligned square polygon."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def _build_map(walls: list[Obstacle], cover: list[Obstacle] | None = None) -> MapData:
    """Tiny synthetic map for visibility tests (60x40, no spawns/bombsites needed)."""
    md = MapData(
        name="test",
        version=1,
        width=60,
        height=40,
        tile_size=1.0,
        spawns={Side.ATTACKER: np.zeros((1, 2)), Side.DEFENDER: np.zeros((1, 2))},
        bombsites={},
        walls=walls,
        cover=cover or [],
        named_areas=[],
    )
    md._all_obstacles = [*walls, *(cover or [])]
    md._sight_obstacles = [o for o in md._all_obstacles if o.blocks_sight]
    md._movement_obstacles = [o for o in md._all_obstacles if o.blocks_movement]
    return md


def _wall(x0, y0, x1, y1, low: bool = False) -> Obstacle:
    poly = _square(x0, y0, x1, y1)
    return Obstacle(
        polygon=poly,
        aabb=(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
        blocks_movement=True,
        blocks_sight=not low,
        low=low,
    )


# ---------------------------------------------------------------------------
# Point in polygon
# ---------------------------------------------------------------------------


def test_point_in_polygon_inside():
    poly = _square(0, 0, 10, 10)
    assert point_in_polygon(np.array([5.0, 5.0]), poly)


def test_point_in_polygon_outside():
    poly = _square(0, 0, 10, 10)
    assert not point_in_polygon(np.array([11.0, 5.0]), poly)
    assert not point_in_polygon(np.array([-1.0, 5.0]), poly)
    assert not point_in_polygon(np.array([5.0, 20.0]), poly)


def test_point_in_polygon_on_edge():
    poly = _square(0, 0, 10, 10)
    # boundary should count as inside
    assert point_in_polygon(np.array([0.0, 5.0]), poly)
    assert point_in_polygon(np.array([5.0, 10.0]), poly)


def test_point_in_concave_polygon():
    # L-shape (concave): outer rectangle minus a bite from top-right
    poly = np.array(
        [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]],
        dtype=np.float64,
    )
    # in the inside-arm
    assert point_in_polygon(np.array([2.0, 7.0]), poly)
    # in the bite (which is OUTSIDE the L)
    assert not point_in_polygon(np.array([7.0, 7.0]), poly)
    # in the base arm
    assert point_in_polygon(np.array([7.0, 2.0]), poly)


# ---------------------------------------------------------------------------
# Segment intersection
# ---------------------------------------------------------------------------


def test_segment_intersection_basic():
    a1, a2 = np.array([0.0, 0.0]), np.array([10.0, 10.0])
    b1, b2 = np.array([0.0, 10.0]), np.array([10.0, 0.0])
    assert segment_intersects_segment(a1, a2, b1, b2)


def test_segment_intersection_parallel_miss():
    a1, a2 = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    b1, b2 = np.array([0.0, 1.0]), np.array([10.0, 1.0])
    assert not segment_intersects_segment(a1, a2, b1, b2)


def test_segment_intersection_no_overlap():
    a1, a2 = np.array([0.0, 0.0]), np.array([5.0, 5.0])
    b1, b2 = np.array([10.0, 10.0]), np.array([20.0, 20.0])
    assert not segment_intersects_segment(a1, a2, b1, b2)


def test_segment_intersects_polygon():
    poly = _square(4, 4, 6, 6)
    assert segment_intersects_polygon(np.array([0.0, 5.0]), np.array([10.0, 5.0]), poly)
    assert not segment_intersects_polygon(np.array([0.0, 0.0]), np.array([3.0, 0.0]), poly)


# ---------------------------------------------------------------------------
# Raycast
# ---------------------------------------------------------------------------


def test_raycast_hits_wall():
    wall = _square(5, 0, 6, 10)
    hit, point, dist = raycast(np.array([0.0, 5.0]), np.array([10.0, 5.0]), [wall])
    assert hit
    assert point is not None
    assert math.isclose(point[0], 5.0, abs_tol=1e-6)
    assert math.isclose(point[1], 5.0, abs_tol=1e-6)
    assert 0.0 < dist < 10.0


def test_raycast_misses_through_gap():
    # Two walls with a gap from y=4..6
    upper = _square(5, 0, 6, 4)
    lower = _square(5, 6, 6, 10)
    hit, point, dist = raycast(np.array([0.0, 5.0]), np.array([10.0, 5.0]), [upper, lower])
    assert not hit
    assert point is None
    assert math.isclose(dist, 10.0, rel_tol=1e-6)


def test_raycast_picks_nearest_obstacle():
    near = _square(2, 0, 3, 10)
    far = _square(7, 0, 8, 10)
    hit, point, dist = raycast(np.array([0.0, 5.0]), np.array([10.0, 5.0]), [far, near])
    assert hit and point is not None
    assert math.isclose(point[0], 2.0, abs_tol=1e-6)
    assert math.isclose(dist, 2.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# AABB & circle helpers
# ---------------------------------------------------------------------------


def test_polygons_overlap_aabb():
    a = _square(0, 0, 5, 5)
    b = _square(4, 4, 9, 9)
    c = _square(10, 10, 12, 12)
    assert polygons_overlap_aabb(a, b)
    assert not polygons_overlap_aabb(a, c)


def test_circle_segment_distance():
    # Segment along x axis, circle right above the midpoint, radius 1
    d = circle_segment_distance(
        np.array([5.0, 2.0]),
        1.0,
        np.array([0.0, 0.0]),
        np.array([10.0, 0.0]),
    )
    assert math.isclose(d, 1.0, abs_tol=1e-6)
    # Now overlapping (circle radius 3 -> distance to segment 0 is 2, minus 3 = -1)
    d2 = circle_segment_distance(
        np.array([5.0, 2.0]),
        3.0,
        np.array([0.0, 0.0]),
        np.array([10.0, 0.0]),
    )
    assert d2 < 0.0


def test_vec_helpers():
    assert math.isclose(vec_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])), 5.0)
    # 90deg between (1,0) and (0,1) -> +pi/2
    assert math.isclose(
        vec_angle(np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        math.pi / 2,
        abs_tol=1e-6,
    )


# ---------------------------------------------------------------------------
# Line-of-sight
# ---------------------------------------------------------------------------


def test_los_blocked_by_wall():
    md = _build_map(walls=[_wall(4, 0, 6, 10)])
    visible, dist, through_cover = compute_los(md, np.array([0.0, 5.0]), np.array([10.0, 5.0]), 20.0)
    assert not visible
    assert not through_cover
    assert math.isclose(dist, 10.0, abs_tol=1e-6)


def test_los_through_open_space():
    md = _build_map(walls=[_wall(0, 8, 10, 10)])  # wall is far from the ray
    visible, dist, through_cover = compute_los(md, np.array([0.0, 1.0]), np.array([10.0, 1.0]), 20.0)
    assert visible
    assert not through_cover
    assert math.isclose(dist, 10.0, abs_tol=1e-6)


def test_los_through_low_cover_sets_flag():
    md = _build_map(walls=[], cover=[_wall(4, 4, 6, 6, low=True)])
    visible, _, through_cover = compute_los(md, np.array([0.0, 5.0]), np.array([10.0, 5.0]), 20.0)
    assert visible
    assert through_cover


def test_los_out_of_range_returns_false():
    md = _build_map(walls=[])
    visible, _, _ = compute_los(md, np.array([0.0, 0.0]), np.array([50.0, 0.0]), max_range=10.0)
    assert not visible


# ---------------------------------------------------------------------------
# Sound
# ---------------------------------------------------------------------------


def test_sound_attenuation_with_distance():
    md = _build_map(walls=[])
    rng = np.random.default_rng(42)
    heard_near, near_str, _ = sound_audible(
        md,
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        sound_intensity=1.0,
        sound_radius=20.0,
        rng=rng,
    )
    heard_far, far_str, _ = sound_audible(
        md,
        np.array([0.0, 0.0]),
        np.array([18.0, 0.0]),
        sound_intensity=1.0,
        sound_radius=20.0,
        rng=rng,
    )
    assert heard_near
    assert near_str > far_str
    # Beyond radius -> inaudible
    heard_out, _, _ = sound_audible(
        md,
        np.array([0.0, 0.0]),
        np.array([25.0, 0.0]),
        sound_intensity=1.0,
        sound_radius=20.0,
        rng=rng,
    )
    assert not heard_out


def test_sound_passes_through_walls_but_attenuates():
    open_map = _build_map(walls=[])
    walled_map = _build_map(walls=[_wall(4, -1, 6, 1)])
    rng = np.random.default_rng(0)
    _, s_open, _ = sound_audible(
        open_map,
        np.array([0.0, 0.0]),
        np.array([10.0, 0.0]),
        sound_intensity=1.0,
        sound_radius=20.0,
        rng=rng,
    )
    _, s_walled, _ = sound_audible(
        walled_map,
        np.array([0.0, 0.0]),
        np.array([10.0, 0.0]),
        sound_intensity=1.0,
        sound_radius=20.0,
        rng=rng,
    )
    assert s_walled < s_open  # wall attenuates the sound path


# ---------------------------------------------------------------------------
# FOV cone
# ---------------------------------------------------------------------------


def test_fov_cone_filters_correctly():
    md = _build_map(walls=[])
    origin = np.array([5.0, 5.0])
    facing = 0.0  # looking +x
    # DEFAULT_FOV is 144 degrees -> half-angle 72 degrees. So a target 90deg
    # off-axis is OUTSIDE the cone; a target 60deg off-axis is INSIDE.
    targets = [
        (1, np.array([10.0, 5.0])),  # 0deg  -> in
        (2, np.array([0.0, 5.0])),  # 180deg -> out
        (3, np.array([5.0, 0.0])),  # 90deg up -> out
        (4, np.array([10.0, 5.0 - math.tan(math.radians(60)) * 5.0])),  # 60deg up -> in
        (5, np.array([10.0, 5.0 + math.tan(math.radians(60)) * 5.0])),  # 60deg down -> in
    ]
    visible = compute_fov(md, origin, facing, DEFAULT_FOV_RADIANS, max_range=60.0, all_targets=targets)
    assert 1 in visible
    assert 2 not in visible
    assert 3 not in visible
    assert 4 in visible
    assert 5 in visible


def test_fov_narrow_cone_excludes_sides():
    md = _build_map(walls=[])
    origin = np.array([5.0, 5.0])
    facing = 0.0
    narrow = math.pi / 6  # 30 degrees total
    targets = [
        (1, np.array([10.0, 5.0])),  # straight ahead -- visible
        (2, np.array([10.0, 9.0])),  # off-axis -- excluded
    ]
    visible = compute_fov(md, origin, facing, narrow, max_range=30.0, all_targets=targets)
    assert visible == [1]


def test_fov_respects_walls():
    md = _build_map(walls=[_wall(7, 0, 8, 10)])
    origin = np.array([5.0, 5.0])
    targets = [(1, np.array([12.0, 5.0]))]
    visible = compute_fov(md, origin, 0.0, DEFAULT_FOV_RADIANS, max_range=30.0, all_targets=targets)
    assert visible == []
