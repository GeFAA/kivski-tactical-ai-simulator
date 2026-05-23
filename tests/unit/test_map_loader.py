"""Unit tests for kivski_sim.map_loader (Dustline map)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kivski_sim.map_loader import list_maps, load_map
from kivski_sim.types import Side
from kivski_sim.visibility import compute_los


# ---------------------------------------------------------------------------
# Smoke-loading
# ---------------------------------------------------------------------------


def test_load_dustline_succeeds():
    md = load_map("dustline")
    assert md.name == "dustline"
    assert md.width == 60
    assert md.height == 40
    assert math.isclose(md.tile_size, 1.0)
    # Should have a non-trivial geometry set
    assert len(md.walls) >= 15
    assert len(md.cover) >= 8
    assert len(md.named_areas) >= 6


def test_list_maps_contains_dustline():
    assert "dustline" in list_maps()


def test_load_unknown_map_raises():
    with pytest.raises(FileNotFoundError):
        load_map("does-not-exist-map")


# ---------------------------------------------------------------------------
# Bombsites / spawns
# ---------------------------------------------------------------------------


def test_dustline_has_two_bombsites():
    md = load_map("dustline")
    assert set(md.bombsites.keys()) == {"A", "B"}
    for site in md.bombsites.values():
        assert site.polygon.shape[1] == 2
        assert site.polygon.shape[0] >= 3
        # centroid must lie inside the polygon
        assert md.is_in_bombsite(site.center) == site.name


def test_dustline_spawns_count_5_each():
    md = load_map("dustline")
    assert md.spawns[Side.ATTACKER].shape == (5, 2)
    assert md.spawns[Side.DEFENDER].shape == (5, 2)
    # Spawns must be inside the playable area and not blocked.
    for side in (Side.ATTACKER, Side.DEFENDER):
        for i in range(5):
            pos = md.nearest_spawn(side, i)
            assert 0.0 < pos[0] < md.width
            assert 0.0 < pos[1] < md.height
            assert not md.is_blocked(pos), f"spawn {side.name}#{i} at {pos} is blocked"


# ---------------------------------------------------------------------------
# Named areas
# ---------------------------------------------------------------------------


def test_named_areas_resolve():
    md = load_map("dustline")
    names = {a.name for a in md.named_areas}
    # Must have at least the canonical taktical callouts.
    assert "Mid" in names
    assert "A-Long" in names
    assert "B-Tunnel" in names
    # Bombsite centers should resolve to a named area when overlapping one.
    a_area = md.area_name(md.bombsites["A"].center)
    assert a_area is not None


# ---------------------------------------------------------------------------
# Spatial sanity
# ---------------------------------------------------------------------------


def test_is_blocked_inside_wall():
    md = load_map("dustline")
    # Pick a point that is inside one of the perimeter walls (y=0..1 strip).
    assert md.is_blocked(np.array([30.0, 0.5]))
    # Way outside the map -> blocked too.
    assert md.is_blocked(np.array([-5.0, -5.0]))


def test_path_exists_from_yellow_to_a():
    """Sanity check: a multi-segment route from Yellow spawn into A-Long is
    completely open (each segment has clean LoS). We don't test straight LoS
    from inside the spawn into A -- the spawn intentionally forces agents to
    round a doorway corner first."""
    md = load_map("dustline")
    spawn_exit = np.array([8.5, 5.5])     # in the doorway east of yellow spawn
    a_long_mid = np.array([25.0, 4.0])    # inside A-Long corridor
    site_a_entry = np.array([47.0, 9.5])  # just inside Bombsite A
    for seg_start, seg_end in [(spawn_exit, a_long_mid), (a_long_mid, site_a_entry)]:
        visible, _, _ = compute_los(md, seg_start, seg_end, max_range=80.0)
        assert visible, f"segment {seg_start.tolist()} -> {seg_end.tolist()} blocked"
    assert not md.is_blocked(spawn_exit)
    assert not md.is_blocked(a_long_mid)
    assert not md.is_blocked(site_a_entry)


def test_path_exists_from_yellow_to_b():
    """Multi-segment route from Yellow into Bombsite B via YellowFlank ->
    B-Connector -> B-Site."""
    md = load_map("dustline")
    yellow_south_doorway = np.array([5.5, 7.5])
    flank_mid = np.array([8.0, 12.0])      # inside YellowFlank, past doorway, clear of cover block
    flank_through = np.array([12.0, 13.5])  # inside the gap of the y=13 wall row
    b_connector = np.array([11.0, 24.0])  # B-Connector
    site_b_entry = np.array([13.0, 33.5])  # inside Bombsite B polygon, clear of low cover
    for seg_start, seg_end in [
        (yellow_south_doorway, flank_mid),
        (flank_mid, flank_through),
        (flank_through, b_connector),
        (b_connector, site_b_entry),
    ]:
        visible, _, _ = compute_los(md, seg_start, seg_end, max_range=40.0)
        assert visible, f"segment {seg_start.tolist()} -> {seg_end.tolist()} blocked"
    for p in (yellow_south_doorway, flank_mid, flank_through, b_connector, site_b_entry):
        assert not md.is_blocked(p), f"{p.tolist()} is blocked"


def test_bombsite_membership_excludes_outside():
    md = load_map("dustline")
    # Mid of the map should not be a bombsite.
    assert md.is_in_bombsite(np.array([30.0, 20.0])) is None
    # Both site centers are inside their own site.
    assert md.is_in_bombsite(md.bombsites["A"].center) == "A"
    assert md.is_in_bombsite(md.bombsites["B"].center) == "B"
