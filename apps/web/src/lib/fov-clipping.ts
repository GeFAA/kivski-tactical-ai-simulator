/**
 * 2D wall-clipped Field-of-View cone.
 *
 * This is a **purely visual** helper — the engine's real LoS check
 * (``packages/sim/kivski_sim/visibility.compute_fov``) does proper
 * per-target raycasting on the backend. The viewer's "Show FoV" debug
 * overlay used to draw a naive geometric cone that bled through walls
 * and gave the wrong impression that agents see through solid cover.
 *
 * We build a fan of ``rayCount`` rays spread across the cone arc,
 * clip each ray against every wall edge it crosses, and return the
 * resulting polygon (origin + clipped tips). Drop the polygon into
 * ``PIXI.Graphics.poly()`` and you get a cone that hugs walls.
 *
 * Performance: with ~30 wall edges per map and 64 rays this is
 * ~2k segment-segment intersections per redraw. The overlay only
 * updates on snapshot ticks (10 Hz) so the cost is in the microseconds
 * range. If maps ever grow much bigger, gate the per-wall loop with an
 * AABB pre-filter against ``[origin, rayEnd]``.
 */

import type { MapWall, Vec2 } from "./types";

/**
 * Segment-segment intersection.
 *
 * Returns the distance along ``p1 → p2`` at which the ray crosses
 * the wall segment ``p3 → p4``, or ``null`` if the segments do not
 * cross within their finite extents.
 *
 * The classic 2D parametric formulation: solve for ``t`` and ``u``
 * in ``p1 + t * (p2-p1) = p3 + u * (p4-p3)``. The ray hits the wall
 * iff ``t ∈ [0, 1]`` and ``u ∈ [0, 1]``. Parallel / colinear
 * segments (``denom == 0``) return ``null`` — for FoV-clipping the
 * grazing case is fine to ignore because the next ray over will
 * still terminate at the wall corner.
 */
export const segmentIntersect = (
  p1: Vec2,
  p2: Vec2,
  p3: Vec2,
  p4: Vec2,
): { distance: number } | null => {
  const r_x = p2.x - p1.x;
  const r_y = p2.y - p1.y;
  const s_x = p4.x - p3.x;
  const s_y = p4.y - p3.y;

  const denom = r_x * s_y - r_y * s_x;
  if (denom === 0) return null; // parallel or colinear

  const dx = p3.x - p1.x;
  const dy = p3.y - p1.y;
  const t = (dx * s_y - dy * s_x) / denom;
  const u = (dx * r_y - dy * r_x) / denom;

  // Allow tiny epsilon at the segment endpoints so a ray that grazes
  // a corner still registers a hit instead of slipping through.
  const eps = 1e-9;
  if (t < -eps || t > 1 + eps) return null;
  if (u < -eps || u > 1 + eps) return null;

  // Distance from p1 to the intersection point in world units.
  const distance = t * Math.hypot(r_x, r_y);
  return { distance };
};

/**
 * Build a wall-clipped FoV cone polygon.
 *
 * @param origin     Agent position (world units).
 * @param facingRad  Facing yaw (radians; 0 = +x, CCW positive).
 * @param fovRad     Total cone arc (radians; e.g. ``144 * pi / 180``).
 * @param maxRange   Maximum sight range when no wall is hit.
 * @param walls      All occluders (walls + sight-blocking cover).
 * @param rayCount   Cone resolution; 64 is plenty for a smooth visual.
 *
 * @returns Polygon vertices, ordered along the cone arc. The first
 *   entry is ``origin`` and the next ``rayCount + 1`` entries are
 *   the clipped tips, so the polygon is closed by Pixi's ``poly()``
 *   automatically.
 */
export const clippedFovPolygon = (
  origin: Vec2,
  facingRad: number,
  fovRad: number,
  maxRange: number,
  walls: MapWall[],
  rayCount = 64,
): Vec2[] => {
  const points: Vec2[] = [{ x: origin.x, y: origin.y }];
  const halfFov = fovRad / 2;

  for (let i = 0; i <= rayCount; i++) {
    const t = rayCount === 0 ? 0 : i / rayCount;
    const angle = facingRad - halfFov + t * fovRad;
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);

    let hitDist = maxRange;
    const rayEnd: Vec2 = {
      x: origin.x + dx * maxRange,
      y: origin.y + dy * maxRange,
    };

    for (const wall of walls) {
      const poly = wall.poly;
      const n = poly.length;
      if (n < 2) continue;
      for (let j = 0; j < n; j++) {
        const a = poly[j];
        const b = poly[(j + 1) % n];
        const hit = segmentIntersect(origin, rayEnd, a, b);
        if (hit && hit.distance < hitDist) {
          hitDist = hit.distance;
        }
      }
    }

    points.push({
      x: origin.x + dx * hitDist,
      y: origin.y + dy * hitDist,
    });
  }

  return points;
};
