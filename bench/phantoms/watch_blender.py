"""Build a skeletonised mechanical watch movement in Blender; export an STL per material.

Run headless (Blender 4.5 LTS):

    blender -b --factory-startup -P bench/phantoms/watch_blender.py -- OUT_DIR

A 29 mm (13 ligne) hand-wound Swiss-lever movement at 18,000 vibrations an
hour, seen from the bridge side: z is the movement's axis, z = 0 the dial side
of the main plate, lengths are millimetres. Every part is a closed mesh; parts
of one material are fused into one mesh and different materials never overlap,
so a ray's path length through each STL is that material's share of it.

- A titanium skeleton main plate, carried up to the bridges as a casing ring
  with round windows, and filleted titanium bridges (barrel bridge, train
  bridge, pallet bridge, balance cock) on titanium posts, their edges chamfered.
- Barrel: brass drum with 84 teeth and a lid, a steel arbor, and a 0.1 mm
  Nivaflex mainspring in 16 coils, packed at the arbor and the wall.
- Going train: brass centre (80 teeth), third (75) and fourth (75) wheels with
  filleted crossings, on steel arbors with 12-, 10- and 10-leaf pinions; watch
  (ogival) tooth forms after NIHS 20-25. Pivots turn in ruby jewels with oil sinks.
- Escapement: a silicon escape wheel with 20 club teeth on a 10-leaf steel
  pinion, a silicon pallet lever with ruby entry and exit stones, a steel guard
  pin and banking pins.
- Balance: a three-armed Glucydur wheel with twelve 18-carat gold timing
  screws, a steel staff, a roller with a ruby impulse pin, a 35 µm Nivarox
  hairspring of 12.5 turns in a steel collet and stud, and regulator curb pins.
- Keyless top: a skeletonised steel ratchet wheel on the barrel arbor's square,
  a crown wheel, a click with its spring, and slotted steel screws throughout.

The materials and their names are listed in ``MATERIALS``; the hero planes
(``LEVELS``) are the z heights at which the figures cut the movement.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

import bpy  # pyright: ignore[reportMissingImports]
import numpy as np

TAU = 2 * math.pi
MATERIALS = (
    "titanium",
    "brass",
    "steel",
    "silicon",
    "ruby",
    "glucydur",
    "gold",
    "nivarox",
    "nivaflex",
)
LEVELS = {"balance": 2.965, "escapement": 1.85, "plate": 0.7, "third wheel": 3.15}
Obj = Any  # bpy.types.Object


# ---------------------------------------------------------------- 2-D geometry


def d(angle: float) -> np.ndarray:
    """Unit vector at ``angle`` radians."""
    return np.array([math.cos(angle), math.sin(angle)])


def polar(r: np.ndarray | float, a: np.ndarray) -> np.ndarray:
    """Points at radii ``r`` and angles ``a`` (radians)."""
    return np.stack(np.broadcast_arrays(r * np.cos(a), r * np.sin(a)), axis=-1)


SIDES = 256  # every round part: a hole and the part filling it share their vertices


def segments(r: float) -> int:
    """Polygon sides for a circle of radius ``r`` mm: always ``SIDES`` (1 µm at 14.5 mm)."""
    del r
    return SIDES


def circle(r: float, c: np.ndarray | tuple = (0.0, 0.0), n: int | None = None) -> np.ndarray:
    """A closed polygon approximating the circle of radius ``r`` about ``c``."""
    n = n or segments(r)
    return np.asarray(c, float) + polar(r, np.linspace(0, TAU, n, endpoint=False))


def arc(c: np.ndarray, r: float, a0: float, a1: float, n: int | None = None) -> np.ndarray:
    """Points from angle ``a0`` to ``a1`` (either direction), both ends included."""
    n = n or max(3, int(abs(a1 - a0) / TAU * SIDES) + 2)
    return np.asarray(c, float) + polar(r, np.linspace(a0, a1, n))


def ccw(outline: np.ndarray) -> np.ndarray:
    """``outline`` ordered anticlockwise."""
    x, y = outline[:, 0], outline[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return outline if area > 0 else outline[::-1]


def dedupe(outline: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """``outline`` without repeated consecutive points."""
    keep = np.linalg.norm(np.diff(outline, axis=0, append=outline[:1]), axis=1) > tol
    return outline[keep]


def bar(p: np.ndarray, q: np.ndarray, w0: float, w1: float | None = None) -> np.ndarray:
    """A tapered bar from ``p`` (width ``w0``) to ``q`` (width ``w1``)."""
    w1 = w0 if w1 is None else w1
    t = (q - p) / np.linalg.norm(q - p)
    n = np.array([-t[1], t[0]])
    return np.array([p - w0 / 2 * n, q - w1 / 2 * n, q + w1 / 2 * n, p + w0 / 2 * n])


def rounded_bar(p: np.ndarray, q: np.ndarray, w: float) -> np.ndarray:
    """A bar with semicircular ends."""
    t = (q - p) / np.linalg.norm(q - p)
    a = math.atan2(t[1], t[0])
    return np.concatenate(
        [
            arc(q, w / 2, a - math.pi / 2, a + math.pi / 2),
            arc(p, w / 2, a + math.pi / 2, a + 1.5 * math.pi),
        ]
    )


def _fillet_line_circle(
    centre: np.ndarray,
    radius: float,
    *,
    inside: bool,
    line_n: np.ndarray,
    line_c: float,
    rho: float,
    near: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A fillet of radius ``rho`` between a circle and the line ``n @ p = c``.

    The fillet lies on the side ``n @ p > c`` of the line and inside (``inside``)
    or outside the circle; of the two solutions the one nearest ``near`` is kept.
    Returns the fillet centre and its tangent points on the line and the circle.
    """
    offset = line_c + rho
    reach = radius - rho if inside else radius + rho
    # Points on the offset line n @ p = offset at distance ``reach`` from centre.
    foot = centre + (offset - line_n @ centre) * line_n
    along = np.array([-line_n[1], line_n[0]])
    half = math.sqrt(max(reach**2 - np.sum((foot - centre) ** 2), 0.0))
    options = [foot + half * along, foot - half * along]
    f = min(options, key=lambda p: np.linalg.norm(p - near))
    on_line = f - rho * line_n
    radial = (f - centre) / np.linalg.norm(f - centre)
    on_circle = centre + radius * radial
    return f, on_line, on_circle


def crossings(
    count: int,
    hub: float,
    rim: float,
    width: float,
    fillet: float,
    phase: float = 0.0,
    curve: float = 0.0,
) -> list[np.ndarray]:
    """The openings between ``count`` straight spokes, with filleted corners.

    Spokes of ``width`` run from radius ``hub`` to ``rim``; ``curve`` bends the
    openings' sides into arcs (0 for straight spokes).
    """
    holes = []
    o = np.zeros(2)
    for j in range(count):
        s0 = phase + TAU * j / count
        s1 = phase + TAU * (j + 1) / count
        n0 = d(s0 + math.pi / 2)  # side of spoke j facing the opening
        n1 = -d(s1 + math.pi / 2)  # side of spoke j+1 facing the opening
        mid = d((s0 + s1) / 2)
        fa, la, ca = _fillet_line_circle(
            o, hub, inside=False, line_n=n0, line_c=width / 2, rho=fillet, near=hub * mid
        )
        fb, lb, cb = _fillet_line_circle(
            o, rim, inside=True, line_n=n0, line_c=width / 2, rho=fillet, near=rim * mid
        )
        fc, lc, cc = _fillet_line_circle(
            o, rim, inside=True, line_n=n1, line_c=width / 2, rho=fillet, near=rim * mid
        )
        fd, ld, cd = _fillet_line_circle(
            o, hub, inside=False, line_n=n1, line_c=width / 2, rho=fillet, near=hub * mid
        )

        def corner(f: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
            a0 = math.atan2(*(p - f)[::-1])
            a1 = math.atan2(*(q - f)[::-1])
            if a1 - a0 > math.pi:
                a1 -= TAU
            elif a0 - a1 > math.pi:
                a1 += TAU
            return arc(f, fillet, a0, a1, 10)

        def angle(p: np.ndarray) -> float:
            return math.atan2(p[1], p[0])

        rim_arc = arc(o, rim, angle(cb), angle(cb) + (angle(cc) - angle(cb)) % TAU)
        hub_arc = arc(o, hub, angle(cd), angle(cd) - (angle(cd) - angle(ca)) % TAU)
        outline = np.concatenate(
            [
                corner(fa, ca, la),
                corner(fb, lb, cb),
                rim_arc[1:-1],
                corner(fc, cc, lc),
                corner(fd, ld, cd),
                hub_arc[1:-1],
            ]
        )
        if curve:
            bend = curve * np.sin(
                np.pi * np.clip((np.linalg.norm(outline, axis=1) - hub) / (rim - hub), 0, 1)
            )
            angles = np.arctan2(outline[:, 1], outline[:, 0]) + bend
            outline = polar(np.linalg.norm(outline, axis=1), angles)
        holes.append(ccw(dedupe(outline)))
    return holes


def watch_teeth(teeth: int, module: float, *, wheel: bool = True, per_side: int = 9) -> np.ndarray:
    """A watch wheel or pinion outline (NIHS 20-25 forms, simplified).

    Wheels: tooth 1.41 m thick at the pitch circle, ogival addendum 1.41 m,
    dedendum 1.75 m. Pinions: leaf 1.05 m thick, rounded addendum 0.855 m,
    dedendum 1.60 m. Flanks are radial below the pitch circle; gaps have
    round bottoms.
    """
    rp = module * teeth / 2
    thick, add, ded = (1.41, 1.41, 1.75) if wheel else (1.05, 0.855, 1.60)
    half = thick * module / 2 / rp  # half the tooth's angle at the pitch circle
    rf = rp - ded * module
    pitch = TAU / teeth
    points = []
    for k in range(teeth):
        a = k * pitch
        # Gap bottom before this tooth: from the previous tooth's flank to this one.
        g0, g1 = a - pitch + half, a - half
        points.append(arc(np.zeros(2), rf, g0, g1, 6)[1:-1])
        # Left (clockwise) flank up, addendum over the tip, right flank down.
        flank = np.linspace(rf, rp, per_side // 2, endpoint=False)
        tip = np.linspace(0, 1, per_side)
        if wheel:
            w = half * rp * (1 - tip**2) ** 0.62  # ogive
        else:
            w = half * rp * np.sqrt(np.clip(1 - tip**2, 0, 1))  # semicircle
        r_tip = rp + add * module * tip
        left = np.concatenate([polar(flank, a - half), polar(r_tip, a - w / r_tip)])
        right = np.concatenate(
            [polar(r_tip[::-1], a + (w / r_tip)[::-1]), polar(flank[::-1], a + half)]
        )
        points += [left, right[1:]]
    return ccw(dedupe(np.concatenate(points)))


def club_teeth(teeth: int, tip: float, base: float) -> np.ndarray:
    """A Swiss lever escape wheel outline with club teeth leaning anticlockwise."""
    pitch = TAU / teeth
    points = []
    for k in range(teeth):
        a = k * pitch
        # Back of the tooth: a concave curve from the rim up to the heel of the club.
        s = np.linspace(0, 1, 12)
        heel = (tip - 0.11, a + 0.47 * pitch)
        ctrl = (base + 0.55 * (tip - base), a + 0.06 * pitch)
        r = (1 - s) ** 2 * base + 2 * s * (1 - s) * ctrl[0] + s**2 * heel[0]
        t = (1 - s) ** 2 * a + 2 * s * (1 - s) * ctrl[1] + s**2 * heel[1]
        points.append(polar(r, t))
        # Impulse face across the club to the locking corner, then the locking face.
        corner = (tip, a + 0.62 * pitch)
        points.append(
            polar(np.array([tip - 0.06, corner[0]]), np.array([a + 0.555 * pitch, corner[1]]))
        )
        foot = (base, a + 0.38 * pitch)
        face = np.linspace(0, 1, 6)[1:]
        points.append(
            polar(
                corner[0] + (foot[0] - corner[0]) * face, corner[1] + (foot[1] - corner[1]) * face
            )
        )
        points.append(arc(np.zeros(2), base, a + 0.38 * pitch, a + pitch, 8)[1:-1])
    return ccw(dedupe(np.concatenate(points)))


def ratchet_teeth(teeth: int, tip: float, root: float) -> np.ndarray:
    """A sawtooth ratchet outline: steep faces anticlockwise."""
    pitch = TAU / teeth
    points = []
    for k in range(teeth):
        a = k * pitch
        rise = np.linspace(0, 1, 6)
        points.append(polar(root + (tip - root) * rise**0.8, a + 0.88 * pitch * rise))
        points.append(polar(np.array([tip - 0.02, root]), np.array([a + 0.92 * pitch, a + pitch])))
    return ccw(dedupe(np.concatenate(points)))


# ---------------------------------------------------------------- 2-D distance fields


class Field:
    """Signed distances (negative inside) sampled on a square grid of ``step`` mm."""

    def __init__(
        self, lo: tuple[float, float], hi: tuple[float, float], step: float = 0.01
    ) -> None:
        self.step = step
        self.x = np.arange(lo[0] - 0.2, hi[0] + 0.2, step)
        self.y = np.arange(lo[1] - 0.2, hi[1] + 0.2, step)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def disc(self, c: np.ndarray, r: float) -> np.ndarray:
        """A disc of radius ``r`` about ``c``."""
        return np.hypot(self.X - c[0], self.Y - c[1]) - r

    def capsule(
        self, p: np.ndarray, q: np.ndarray, r0: float, r1: float | None = None
    ) -> np.ndarray:
        """A bar of rounded ends from ``p`` (half-width ``r0``) to ``q`` (``r1``)."""
        r1 = r0 if r1 is None else r1
        ba = q - p
        h = np.clip(((self.X - p[0]) * ba[0] + (self.Y - p[1]) * ba[1]) / (ba @ ba), 0, 1)
        dist = np.hypot(self.X - p[0] - h * ba[0], self.Y - p[1] - h * ba[1])
        return dist - (r0 + (r1 - r0) * h)

    def curve(self, points: np.ndarray, r0: float, r1: float | None = None) -> np.ndarray:
        """A bar following a polyline, tapering from ``r0`` to ``r1``."""
        r1 = r0 if r1 is None else r1
        w = np.linspace(r0, r1, len(points))
        out = np.full(self.X.shape, np.inf)
        for k in range(len(points) - 1):
            out = np.minimum(out, self.capsule(points[k], points[k + 1], w[k], w[k + 1]))
        return out

    def loops(self, value: np.ndarray) -> list[np.ndarray]:
        """Closed outlines of ``value < 0`` by marching squares."""
        f = np.pad(value, 1, constant_values=1.0)
        x = np.concatenate([[self.x[0] - self.step], self.x, [self.x[-1] + self.step]])
        y = np.concatenate([[self.y[0] - self.step], self.y, [self.y[-1] + self.step]])
        inside = f < 0
        # Crossing points on horizontal edges (j, i)-(j, i+1) and vertical edges (j, i)-(j+1, i).
        h_cross = inside[:, :-1] != inside[:, 1:]
        v_cross = inside[:-1, :] != inside[1:, :]
        th = f[:, :-1] / (f[:, :-1] - f[:, 1:] + 1e-30)
        tv = f[:-1, :] / (f[:-1, :] - f[1:, :] + 1e-30)
        point = {}
        for j, i in zip(*np.nonzero(h_cross), strict=True):
            point[("h", j, i)] = (x[i] + th[j, i] * self.step, y[j])
        for j, i in zip(*np.nonzero(v_cross), strict=True):
            point[("v", j, i)] = (x[i], y[j] + tv[j, i] * self.step)
        links: dict[tuple, list[tuple]] = {}

        def link(a: tuple, b: tuple) -> None:
            links.setdefault(a, []).append(b)
            links.setdefault(b, []).append(a)

        cells = h_cross[:-1, :] | h_cross[1:, :] | v_cross[:, :-1] | v_cross[:, 1:]
        for j, i in zip(*np.nonzero(cells), strict=True):
            edges = [
                e
                for e, on in (
                    (("h", j, i), h_cross[j, i]),
                    (("v", j, i + 1), v_cross[j, i + 1]),
                    (("h", j + 1, i), h_cross[j + 1, i]),
                    (("v", j, i), v_cross[j, i]),
                )
                if on
            ]
            if len(edges) == 2:
                link(*edges)
            elif len(edges) == 4:  # saddle: decide by the cell centre
                centre = f[j, i] + f[j, i + 1] + f[j + 1, i] + f[j + 1, i + 1]
                b0 = inside[j, i]
                if (centre < 0) == b0:
                    link(edges[0], edges[1])
                    link(edges[2], edges[3])
                else:
                    link(edges[0], edges[3])
                    link(edges[1], edges[2])
        loops = [np.array([point[k] for k in loop]) for loop in _walk(links)]
        return [pts[::2] if len(pts) > 400 else pts for pts in loops if len(pts) >= 8]


def _walk(links: dict[tuple, list[tuple]]) -> list[list[tuple]]:
    """Follow each closed chain of linked crossing points once."""
    loops, seen = [], set()
    for start in links:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [n for n in links[cur] if n != prev]
            if not nxt or nxt[0] == start or nxt[0] in seen:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        loops.append(loop)
    return loops


def smooth_union(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """``min(a, b)`` with junctions rounded over about ``k`` mm."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smooth_cut(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """``a`` minus ``b``, edges rounded over about ``k`` mm."""
    return -smooth_union(-a, b, k)


def bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 24) -> np.ndarray:
    """``n`` points along the quadratic Bezier curve with control point ``p1``."""
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * t * (1 - t) * p1 + t**2 * p2


def _inside(point: np.ndarray, loop: np.ndarray) -> bool:
    x, y = loop[:, 0], loop[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    crosses = ((y > point[1]) != (yn > point[1])) & (
        point[0] < (xn - x) * (point[1] - y) / (yn - y + 1e-30) + x
    )
    return bool(np.count_nonzero(crosses) % 2)


# ---------------------------------------------------------------- 3-D meshes (numpy)


class Solid:
    """A closed triangle-and-polygon mesh: vertices and faces (index tuples)."""

    def __init__(self, v: np.ndarray, f: list[tuple[int, ...]]) -> None:
        self.v = np.asarray(v, float)
        self.f = f

    def moved(self, xy: np.ndarray | tuple = (0, 0), z: float = 0.0, angle: float = 0.0) -> Solid:
        """Turned by ``angle`` about z, then moved by ``(xy, z)``."""
        c, s = math.cos(angle), math.sin(angle)
        v = self.v @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        return Solid(v + np.array([xy[0], xy[1], z]), self.f)

    def turned(self, matrix: np.ndarray, at: np.ndarray) -> Solid:
        """Rotated by ``matrix``, then moved to ``at``."""
        return Solid(self.v @ np.asarray(matrix).T + at, self.f)


def _oriented(v: np.ndarray, faces: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Faces flipped if needed so the enclosed volume is positive."""
    vol = 0.0
    for f in faces:
        for i in range(1, len(f) - 1):
            vol += np.dot(v[f[0]], np.cross(v[f[i]], v[f[i + 1]]))
    return faces if vol > 0 else [tuple(reversed(f)) for f in faces]


def prism(outline: np.ndarray, z0: float, z1: float) -> Solid:
    """``outline`` (a simple polygon) extruded from ``z0`` to ``z1``."""
    p = ccw(dedupe(np.asarray(outline, float)))
    n = len(p)
    v = np.concatenate([np.c_[p, np.full(n, z0)], np.c_[p, np.full(n, z1)]])
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    faces += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return Solid(v, faces)


def lathe(profile: list[tuple[float, float]], n: int | None = None) -> Solid:
    """Revolve a closed ``(r, z)`` profile about z.

    A solid's profile starts and ends on the axis (r = 0); a ring's never
    touches it and closes on itself.
    """
    pts = np.asarray(profile, float)
    solid = pts[0, 0] == 0 and pts[-1, 0] == 0
    n = n or segments(pts[:, 0].max())
    a = np.linspace(0, TAU, n, endpoint=False)
    verts, rings = [], []
    for r, z in pts:
        if r == 0:
            rings.append(len(verts))
            verts.append((0.0, 0.0, z))
        else:
            rings.append(list(range(len(verts), len(verts) + n)))
            verts += [(r * math.cos(t), r * math.sin(t), z) for t in a]
    faces = []
    pairs = list(itertools.pairwise(rings))
    if not solid:
        pairs.append((rings[-1], rings[0]))
    for ra, rb in pairs:
        if isinstance(ra, int) and isinstance(rb, int):
            continue
        if isinstance(ra, int):
            faces += [(ra, rb[(i + 1) % n], rb[i]) for i in range(n)]
        elif isinstance(rb, int):
            faces += [(ra[i], ra[(i + 1) % n], rb) for i in range(n)]
        else:
            faces += [(ra[i], ra[(i + 1) % n], rb[(i + 1) % n], rb[i]) for i in range(n)]
    v = np.asarray(verts)
    return Solid(v, _oriented(v, faces))


def cylinder(r: float, z0: float, z1: float, xy: tuple | np.ndarray = (0, 0)) -> Solid:
    """A solid cylinder about the vertical through ``xy``."""
    return lathe([(0, z0), (r, z0), (r, z1), (0, z1)]).moved(xy)


def tube(r0: float, r1: float, z0: float, z1: float, xy: tuple | np.ndarray = (0, 0)) -> Solid:
    """A ring between radii ``r0`` and ``r1`` about the vertical through ``xy``."""
    return lathe([(r0, z0), (r1, z0), (r1, z1), (r0, z1)]).moved(xy)


def strip(centre: np.ndarray, thickness: float, z0: float, z1: float) -> Solid:
    """A strip of rectangular section along the planar polyline ``centre``."""
    c = np.asarray(centre, float)
    t = np.gradient(c, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.c_[-t[:, 1], t[:, 0]]
    outer, inner = c + thickness / 2 * n, c - thickness / 2 * n
    m = len(c)
    rings = [outer, inner, inner, outer]
    zs = [z0, z0, z1, z1]
    v = np.concatenate([np.c_[r, np.full(m, z)] for r, z in zip(rings, zs, strict=True)])
    faces = []
    for k in range(4):
        a, b = k * m, ((k + 1) % 4) * m
        faces += [(a + i, a + i + 1, b + i + 1, b + i) for i in range(m - 1)]
    faces += [(0, m, 2 * m, 3 * m), (m - 1 + 3 * m, m - 1 + 2 * m, m - 1 + m, m - 1)]
    return Solid(v, _oriented(v, faces))


def along(axis: np.ndarray) -> np.ndarray:
    """A rotation taking +z to the unit ``axis``."""
    z = np.asarray(axis, float) / np.linalg.norm(axis)
    x = np.cross([0.0, 0.0, 1.0], z) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    return np.c_[x, np.cross(z, x), z]


# ---------------------------------------------------------------- Blender layer

_scratch: Any = None
_parts: dict[str, list[Obj]] = {m: [] for m in MATERIALS}


def _object(name: str, solid: Solid) -> Obj:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(solid.v.tolist(), [], [list(f) for f in solid.f])
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def part(name: str, material: str, solid: Solid | Obj) -> Obj:
    """Register ``solid`` (made into an object if need be) as a part of ``material``."""
    obj = solid if not isinstance(solid, Solid) else _object(name, solid)
    obj.name = name
    _parts[material].append(obj)
    return obj


def _boolean(target: Obj, operands: list[Solid | Obj], operation: str) -> Obj:
    if not operands:
        return target
    collection = bpy.data.collections.new(f"ops_{target.name}")
    bpy.context.scene.collection.children.link(collection)
    for k, o in enumerate(operands):
        obj = _object(f"{target.name}_op{k}", o) if isinstance(o, Solid) else o
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        collection.objects.link(obj)
    mod = target.modifiers.new("boolean", "BOOLEAN")
    mod.operation = operation
    mod.solver = "MANIFOLD"
    mod.operand_type = "COLLECTION"
    mod.collection = collection
    graph = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(target.evaluated_get(graph))
    target.modifiers.remove(mod)
    old = target.data
    target.data = mesh
    bpy.data.meshes.remove(old)
    for obj in list(collection.objects):
        data = obj.data
        bpy.data.objects.remove(obj)
        if data.users == 0:
            bpy.data.meshes.remove(data)
    bpy.data.collections.remove(collection)
    return target


def cut(target: Obj, *cutters: Solid | Obj) -> Obj:
    """``target`` with ``cutters`` removed, in place."""
    return _boolean(target, list(cutters), "DIFFERENCE")


def fuse(name: str, solids: list[Solid | Obj]) -> Obj:
    """The union of ``solids``, made in the first (or a new object called ``name``)."""
    first = solids[0]
    target = _object(name, first) if isinstance(first, Solid) else first
    return _boolean(target, solids[1:], "UNION")


def extrude_field(name: str, field: Field, value: np.ndarray, z0: float, z1: float) -> Obj:
    """The region ``value < 0`` extruded from ``z0`` to ``z1``, holes and all."""
    loops = field.loops(value)
    depth = [sum(_inside(a[0], b) for b in loops if b is not a) for a in loops]
    order = sorted(range(len(loops)), key=lambda k: depth[k])
    obj = None
    for k in order:
        solid = (
            prism(loops[k], z0, z1) if depth[k] % 2 == 0 else prism(loops[k], z0 - 0.1, z1 + 0.1)
        )
        if obj is None:
            obj = _object(name, solid)
        elif depth[k] % 2 == 0:
            obj = _boolean(obj, [solid], "UNION")
        else:
            obj = cut(obj, solid)
    return obj


def chamfer(obj: Obj, width: float) -> Obj:
    """Bevel the sharp edges of ``obj`` (the bridges' anglage)."""
    mod = obj.modifiers.new("anglage", "BEVEL")
    mod.width = width
    mod.segments = 1
    mod.limit_method = "ANGLE"
    mod.angle_limit = math.radians(50)
    mod.harden_normals = False
    graph = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(graph))
    obj.modifiers.remove(mod)
    old = obj.data
    obj.data = mesh
    bpy.data.meshes.remove(old)
    return obj


# ---------------------------------------------------------------- the movement

RADIUS = 14.5
PLATE = (0.0, 1.40)
BRIDGES = (4.15, 5.25)
COCK = (3.45, 4.45)
PALLET_BRIDGE = (2.10, 2.38)

BARREL = d(math.pi / 2) * 5.76  # barrel 84 teeth, m 0.12, drives the 12-leaf centre pinion
CENTRE = np.zeros(2)
THIRD = CENTRE + 4.05 * d(math.radians(-45))  # centre 80 teeth, m 0.09; third pinion 10
FOURTH = THIRD + 3.40 * d(math.radians(-75))  # third 75 teeth, m 0.08; fourth pinion 10
ESCAPE = FOURTH + 2.975 * d(math.radians(-165))  # fourth 75 teeth, m 0.07; escape pinion 10
PALLET = ESCAPE + 2.55 * d(math.pi)
BALANCE = PALLET + 3.30 * d(math.pi)  # straight-line lever
ESCAPE_TIP = 2.25
STONE_ANGLE = math.radians(30)  # entry and exit stones either side of the line of centres
LEVER = math.pi  # direction from the pallet arbor to the balance staff

BARREL_POSTS = [BARREL + 5.8 * d(math.radians(a)) for a in (20, 160, 215)]
TRAIN_POSTS = [np.array(q) for q in ((4.9, 2.3), (7.8, -1.5), (6.2, -9.4), (2.6, -10.2))]
PALLET_POSTS = [PALLET + 1.9 * d(math.radians(a)) for a in (115, -115)]
COCK_POST = BALANCE + 5.6 * d(math.radians(150))
BANKING = [PALLET + 1.25 * d(LEVER) + s * 0.30 * d(LEVER + math.pi / 2) for s in (1, -1)]
CROWN = BARREL + 5.73 * d(math.radians(195))
CLICK = BARREL + 4.6 * d(math.radians(-20))
SQUARE = np.array([[-0.42, -0.42], [0.42, -0.42], [0.42, 0.42], [-0.42, 0.42]])


def towards(a: np.ndarray, b: np.ndarray) -> float:
    """Direction (radians) from ``a`` to ``b``."""
    return math.atan2(*(b - a)[::-1])


def at(solid: Solid, xy: np.ndarray, angle: float = 0.0) -> Solid:
    """``solid`` turned by ``angle`` about z, then moved to ``xy``."""
    return solid.moved((0, 0), 0.0, angle).moved(xy)


def shaft(name: str, xy: np.ndarray, stations: list[tuple[float, float]]) -> Obj:
    """A steel arbor through ``(z, r)`` stations, its ends on the axis."""
    profile = [(0.0, stations[0][0])] + [(r, z) for z, r in stations] + [(0.0, stations[-1][0])]
    return part(name, "steel", lathe(profile).moved(xy))


def pivoted(name: str, xy: np.ndarray, pivot: float, body: float, top: float = 4.46) -> Obj:
    """An arbor of radius ``body`` between pivots in the main plate and a bridge at ``top``."""
    return shaft(
        name,
        xy,
        [
            (1.10, pivot),
            (1.45, pivot),
            (1.53, body),
            (top - 0.41, body),
            (top - 0.33, pivot),
            (top, pivot),
        ],
    )


def jewel(
    name: str,
    xy: np.ndarray,
    z: tuple[float, float],
    r_out: float,
    hole: float,
    *,
    sink: str,
    host: Obj,
    clearance: float,
) -> None:
    """A ruby jewel with an oil sink on its ``sink`` face, pressed into ``host``.

    ``host`` is cut to fit, with a ``clearance`` hole in front of the oil sink.
    """
    z0, z1 = z
    depth = 0.45 * (z1 - z0)
    mouth = min(hole + 0.8 * depth, 0.85 * r_out)
    if sink == "top":
        profile = [(hole, z0), (r_out, z0), (r_out, z1), (mouth, z1), (hole, z1 - depth)]
        beyond = (z1, BRIDGES[1] + 1)
    else:
        profile = [(hole, z1), (hole, z0 + depth), (mouth, z0), (r_out, z0), (r_out, z1)]
        beyond = (-1.0, z0)
    part(name, "ruby", lathe(profile).moved(xy))
    cut(host, cylinder(r_out, z0, z1, xy), cylinder(clearance, *beyond, xy))


def screw(
    name: str,
    xy: np.ndarray,
    top: float,
    head: float,
    shank: float,
    length: float,
    *,
    slot: float = 0.3,
    hosts: tuple[Obj, ...] = (),
) -> None:
    """A steel screw with a slotted cheese head standing on ``top``, through ``hosts``."""
    for host in hosts:
        cut(host, cylinder(shank + 0.01, top - length - 0.01, top + 0.01, xy))
    h = 0.62 * head
    profile = [
        (0.0, top - length),
        (shank, top - length),
        (shank, top),
        (head, top),
        (head, top + 0.8 * h),
        (0.85 * head, top + h),
        (0.0, top + h),
    ]
    body = part(name, "steel", lathe(profile).moved(xy))
    blade = bar(np.array([-2 * head, 0]), np.array([2 * head, 0]), 0.22 * head)
    cut(body, at(prism(blade, top + 0.55 * h, top + 2 * h), xy, slot))


def gear(
    name: str,
    material: str,
    xy: np.ndarray,
    z: tuple[float, float],
    outline: np.ndarray,
    *,
    angle: float,
    bore: float,
    holes: list[np.ndarray] = (),
) -> Obj:
    """A wheel of ``outline`` turned by ``angle``, with openings ``holes`` and a ``bore``."""
    wheel = part(name, material, at(prism(outline, *z), xy, angle))
    cutters = [at(prism(h, z[0] - 0.1, z[1] + 0.1), xy, angle) for h in holes]
    cut(wheel, *cutters, cylinder(bore, z[0] - 0.1, z[1] + 0.1, xy))
    return wheel


def pinion(
    name: str, xy: np.ndarray, z: tuple[float, float], leaves: int, module: float, angle: float
) -> None:
    """A steel pinion's leaves (the arbor fills its centre)."""
    part(name, "steel", at(prism(watch_teeth(leaves, module, wheel=False), *z), xy, angle))


def mesh_angle(wheel: np.ndarray, pinion_xy: np.ndarray) -> float:
    """Rotation putting a wheel tooth on the line of centres."""
    return towards(wheel, pinion_xy)


def gap_angle(pinion_xy: np.ndarray, wheel: np.ndarray, leaves: int) -> float:
    """Rotation putting a pinion's gap on the line of centres."""
    return towards(pinion_xy, wheel) + math.pi / leaves


def _arm(
    field: Field,
    shape: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    bend: float,
    w0: float,
    w1: float | None = None,
    k: float = 0.9,
) -> np.ndarray:
    """``shape`` joined smoothly to a bar from ``q0`` to ``q1`` bowed sideways by ``bend``."""
    mid = (q0 + q1) / 2 + bend * np.linalg.norm(q1 - q0) * d(towards(q0, q1) + math.pi / 2)
    return smooth_union(shape, field.curve(bezier(q0, mid, q1), w0, w1), k)


def _rim(angle_deg: float) -> np.ndarray:
    return 13.6 * d(math.radians(angle_deg))


PLATE_ARMS = (  # from, to, bow, half-widths at either end
    (CENTRE, THIRD, 0.12, 0.55, None),
    (THIRD, FOURTH, -0.10, 0.5, None),
    (FOURTH, ESCAPE, 0.10, 0.48, None),
    (ESCAPE, PALLET, 0.0, 0.45, None),
    (PALLET, BALANCE, 0.15, 0.48, None),
    (BALANCE, _rim(-120), 0.18, 0.62, 0.8),
    (BALANCE, COCK_POST, -0.2, 0.5, None),
    (COCK_POST, _rim(160), 0.15, 0.55, 0.75),
    (BARREL_POSTS[2], COCK_POST, -0.25, 0.5, None),
    (CENTRE, BARREL_POSTS[2], -0.2, 0.5, None),
    (TRAIN_POSTS[0], CENTRE, 0.2, 0.5, None),
    (TRAIN_POSTS[0], TRAIN_POSTS[1], -0.15, 0.48, None),
    (TRAIN_POSTS[1], _rim(-5), 0.1, 0.55, 0.75),
    (THIRD, TRAIN_POSTS[1], 0.1, 0.45, None),
    (FOURTH, TRAIN_POSTS[2], 0.12, 0.45, None),
    (TRAIN_POSTS[2], _rim(-58), 0.0, 0.55, 0.75),
    (ESCAPE, TRAIN_POSTS[3], -0.12, 0.45, None),
    (TRAIN_POSTS[3], _rim(-80), 0.1, 0.55, 0.75),
    (BARREL_POSTS[0], _rim(48), 0.2, 0.55, 0.75),
    (BARREL_POSTS[1], _rim(132), -0.2, 0.55, 0.75),
    (PALLET, PALLET_POSTS[0], 0.25, 0.38, None),
    (PALLET, PALLET_POSTS[1], -0.25, 0.38, None),
    (PALLET, BANKING[0], 0.0, 0.3, None),
    (PALLET, BANKING[1], 0.0, 0.3, None),
)


def main_plate() -> Obj:
    """The skeleton main plate, its posts and casing ring, banking pins and lower jewels."""
    field = Field((-RADIUS, -RADIUS), (RADIUS, RADIUS), 0.02)
    rho = np.hypot(field.X, field.Y)
    shape = np.maximum(rho - RADIUS, 13.25 - rho)
    bosses = [
        (BARREL, 5.6),
        (CENTRE, 1.45),
        (THIRD, 1.2),
        (FOURTH, 1.15),
        (ESCAPE, 1.05),
        (PALLET, 1.0),
        (BALANCE, 1.35),
    ]
    bosses += [(q, 1.05) for q in [*BARREL_POSTS, *TRAIN_POSTS, COCK_POST]]
    bosses += [(q, 0.85) for q in PALLET_POSTS] + [(q, 0.55) for q in BANKING]
    for q, r in bosses:
        shape = smooth_union(shape, field.disc(q, r), 0.6)
    for q0, q1, bend, w0, w1 in PLATE_ARMS:
        shape = _arm(field, shape, q0, q1, bend, w0, w1)
    plate = part("main plate", "titanium", extrude_field("main plate", field, shape, *PLATE))
    openings = crossings(5, 1.35, 4.55, 0.85, 0.45, 0.2, curve=0.3)
    cut(
        plate,
        cylinder(5.05, 0.85, PLATE[1] + 0.1, BARREL),
        *[at(prism(h, -0.1, 0.86), BARREL) for h in openings],
    )
    posts = [cylinder(0.55, PLATE[1] - 0.05, BRIDGES[0], q) for q in BARREL_POSTS + TRAIN_POSTS]
    posts += [cylinder(0.40, PLATE[1] - 0.05, PALLET_BRIDGE[0], q) for q in PALLET_POSTS]
    posts += [cylinder(0.55, PLATE[1] - 0.05, COCK[0], COCK_POST)]
    # Casing ring: the plate's rim carried up to the bridges, with round windows.
    wall = lathe(
        [
            (13.55, PLATE[1] - 0.05),
            (RADIUS, PLATE[1] - 0.05),
            (RADIUS, BRIDGES[1]),
            (13.55, BRIDGES[1]),
        ]
    )
    plate = fuse("main plate", [plate, *posts, wall])
    windows = []
    for k in range(18):
        a = TAU * (k + 0.5) / 18
        hole = cylinder(0.42 if k % 2 else 0.55, 13.3, 14.8)
        windows.append(
            hole.turned(along(np.array([*d(a), 0.0])), np.array([0.0, 0.0, 2.1 if k % 2 else 3.85]))
        )
    cut(plate, *windows, *[cylinder(0.09, 1.05, PLATE[1] + 0.1, q) for q in BANKING])
    for q in BANKING:
        part("banking pin", "steel", cylinder(0.09, 1.05, 2.05, q))
    lower = {
        "centre": (CENTRE, 0.60, 0.10),
        "third": (THIRD, 0.55, 0.09),
        "fourth": (FOURTH, 0.50, 0.08),
        "escape": (ESCAPE, 0.48, 0.07),
        "pallet": (PALLET, 0.48, 0.07),
        "balance": (BALANCE, 0.50, 0.05),
    }
    for name, (q, r_out, pivot) in lower.items():
        jewel(
            f"{name} lower jewel",
            q,
            (1.05, PLATE[1]),
            r_out,
            pivot + 0.004,
            sink="bottom",
            host=plate,
            clearance=0.28,
        )
    jewel(
        "barrel lower jewel",
        BARREL,
        (0.45, 0.85),
        0.90,
        0.405,
        sink="bottom",
        host=plate,
        clearance=0.55,
    )
    return plate


def barrel_bridge() -> Obj:
    """A bow over the barrel with arms to its posts, the crown wheel and the click."""
    b = BARREL
    field = Field((-8.0, -1.0), (8.0, 13.0))
    rho = np.hypot(field.X - b[0], field.Y - b[1])
    bow = np.maximum(np.maximum(rho - 5.75, 5.15 - rho), b[1] + 0.9 - field.Y)
    shape = smooth_union(field.disc(b, 1.75), bow, 0.5)
    for q in BARREL_POSTS:
        shape = smooth_union(shape, field.disc(q, 0.95), 0.5)
        shape = _arm(field, shape, b, q, 0.12, 0.62, 0.48, k=0.8)
    for q, r in ((CROWN, 1.25), (CLICK, 0.85)):
        shape = smooth_union(shape, field.disc(q, r), 0.6)
    shape = smooth_union(shape, field.capsule(BARREL_POSTS[2], CROWN, 0.5), 0.6)
    shape = smooth_union(shape, field.capsule(BARREL_POSTS[0], CLICK, 0.45), 0.6)
    return part("barrel bridge", "titanium", extrude_field("barrel bridge", field, shape, *BRIDGES))


def train_bridge() -> Obj:
    """Curved arms joining the centre, third, fourth and escape bearings to four posts."""
    field = Field((-1.6, -11.4), (9.0, 3.5))
    shape = np.full(field.X.shape, 1e3)
    bosses = [(CENTRE, 1.05), (THIRD, 0.95), (FOURTH, 0.9), (ESCAPE, 0.85)]
    for q, r in bosses + [(q, 0.95) for q in TRAIN_POSTS]:
        shape = smooth_union(shape, field.disc(q, r), 0.5)
    posts = TRAIN_POSTS
    for q0, q1, bend, w in (
        (posts[0], CENTRE, 0.15, 0.55),
        (CENTRE, THIRD, -0.1, 0.55),
        (THIRD, posts[1], 0.12, 0.5),
        (THIRD, FOURTH, 0.12, 0.52),
        (FOURTH, posts[2], -0.1, 0.5),
        (FOURTH, ESCAPE, -0.12, 0.48),
        (ESCAPE, posts[3], 0.15, 0.45),
        (posts[1], posts[2], -0.18, 0.42),
    ):
        shape = _arm(field, shape, q0, q1, bend, w, k=0.8)
    return part("train bridge", "titanium", extrude_field("train bridge", field, shape, *BRIDGES))


def pallet_bridge() -> Obj:
    """A small bridge over the pallet arbor on two posts."""
    field = Field((-3.4, -9.6), (0.2, -4.2))
    shape = field.disc(PALLET, 0.65)
    for q in PALLET_POSTS:
        shape = smooth_union(shape, field.disc(q, 0.5), 0.3)
        shape = _arm(field, shape, PALLET, q, 0.25, 0.36, 0.30, 0.5)
    return part(
        "pallet bridge", "titanium", extrude_field("pallet bridge", field, shape, *PALLET_BRIDGE)
    )


def balance_cock() -> Obj:
    """The balance cock: a round head over the staff tapering to one foot."""
    field = Field((-11.4, -9.0), (-2.9, -2.6))
    head = field.disc(BALANCE, 1.75)
    foot = field.disc(COCK_POST, 1.15)
    neck = field.capsule(BALANCE + 1.2 * d(towards(BALANCE, COCK_POST)), COCK_POST, 0.95, 0.85)
    shape = smooth_union(smooth_union(head, neck, 1.2), foot, 0.8)
    return part("balance cock", "titanium", extrude_field("balance cock", field, shape, *COCK))


def bridges(plate: Obj) -> dict[str, Obj]:
    """The four bridges, chamfered, with their jewels and screws into the plate's posts."""
    made = {
        "barrel": barrel_bridge(),
        "train": train_bridge(),
        "pallet": pallet_bridge(),
        "cock": balance_cock(),
    }
    for bridge in made.values():
        chamfer(bridge, 0.06)
    upper = {
        "centre": (CENTRE, 0.60, 0.12),
        "third": (THIRD, 0.55, 0.09),
        "fourth": (FOURTH, 0.50, 0.08),
        "escape": (ESCAPE, 0.48, 0.07),
    }
    for name, (q, r_out, pivot) in upper.items():
        jewel(
            f"{name} upper jewel",
            q,
            (BRIDGES[0], BRIDGES[0] + 0.35),
            r_out,
            pivot + 0.004,
            sink="top",
            host=made["train"],
            clearance=0.30,
        )
    jewel(
        "barrel upper jewel",
        BARREL,
        (BRIDGES[0], BRIDGES[0] + 0.40),
        0.90,
        0.405,
        sink="top",
        host=made["barrel"],
        clearance=0.45,
    )
    jewel(
        "pallet upper jewel",
        PALLET,
        PALLET_BRIDGE,
        0.40,
        0.074,
        sink="top",
        host=made["pallet"],
        clearance=0.22,
    )
    jewel(
        "balance upper jewel",
        BALANCE,
        (COCK[0], COCK[0] + 0.35),
        0.50,
        0.054,
        sink="top",
        host=made["cock"],
        clearance=0.50,
    )
    part("balance cap jewel", "ruby", cylinder(0.45, COCK[0] + 0.38, COCK[0] + 0.50, BALANCE))
    for k, q in enumerate(BARREL_POSTS + TRAIN_POSTS):
        bridge = made["barrel"] if k < len(BARREL_POSTS) else made["train"]
        cut(bridge, cylinder(0.31, BRIDGES[0] - 0.1, BRIDGES[1] + 0.1, q))
        cut(plate, cylinder(0.31, BRIDGES[0] - 0.8, BRIDGES[0] + 0.1, q))
        screw(f"bridge screw {k}", q, BRIDGES[1], 0.52, 0.30, 1.9, slot=0.4 + 0.7 * k)
    for k, q in enumerate(PALLET_POSTS):
        cut(made["pallet"], cylinder(0.21, PALLET_BRIDGE[0] - 0.1, PALLET_BRIDGE[1] + 0.1, q))
        cut(plate, cylinder(0.21, PALLET_BRIDGE[0] - 0.7, PALLET_BRIDGE[0] + 0.1, q))
        screw(f"pallet bridge screw {k}", q, PALLET_BRIDGE[1], 0.30, 0.18, 0.9, slot=1.0 + k)
    cut(made["cock"], cylinder(0.31, COCK[0] - 0.1, COCK[1] + 0.1, COCK_POST))
    cut(plate, cylinder(0.31, COCK[0] - 0.9, COCK[0] + 0.1, COCK_POST))
    screw("cock screw", COCK_POST, COCK[1], 0.55, 0.30, 1.6, slot=2.2)
    return made


def barrel() -> None:
    """The toothed drum and lid, the mainspring in 16 coils and the barrel arbor."""
    b = BARREL
    drum = part(
        "barrel drum",
        "brass",
        lathe(
            [(0.70, 0.90), (4.85, 0.90), (4.85, 3.50), (4.55, 3.50), (4.55, 1.12), (0.70, 1.12)]
        ).moved(b),
    )
    teeth = at(prism(watch_teeth(84, 0.12), 3.05, 3.50), b, mesh_angle(b, CENTRE))
    fuse("barrel drum", [drum, cut(_object("barrel teeth", teeth), cylinder(4.60, 2.9, 3.6, b))])
    part("barrel lid", "brass", tube(0.70, 4.55, 3.20, 3.42, b))
    # Coils packed on the arbor and against the wall, spread between: part wound.
    turns = np.linspace(0, 16, 16 * 200)
    pitch = 0.112 + 0.273 * np.exp(-(((turns - 6.5) / 2.6) ** 2))
    radius = np.concatenate([[0], np.cumsum((pitch[1:] + pitch[:-1]) / 2 * np.diff(turns))])
    radius = 1.42 + radius * (4.47 - 1.42) / radius[-1]
    part("mainspring", "nivaflex", strip(b + polar(radius, TAU * turns + 1.0), 0.10, 1.22, 3.10))
    shaft(
        "barrel arbor",
        b,
        [
            (0.48, 0.40),
            (0.88, 0.40),
            (0.88, 0.69),
            (1.15, 0.69),
            (1.15, 1.30),
            (3.17, 1.30),
            (3.17, 0.69),
            (3.45, 0.60),
            (BRIDGES[0], 0.40),
            (BRIDGES[1], 0.40),
        ],
    )
    part("arbor square", "steel", at(prism(SQUARE, BRIDGES[1], 5.70), b, 0.2))


def going_train() -> None:
    """Centre, third and fourth wheels with their pinions and arbors."""
    shaft(
        "centre arbor",
        CENTRE,
        [
            (1.10, 0.10),
            (1.45, 0.10),
            (1.55, 0.30),
            (3.55, 0.30),
            (3.55, 0.35),
            (3.95, 0.35),
            (3.95, 0.28),
            (4.05, 0.28),
            (4.15, 0.12),
            (4.46, 0.12),
        ],
    )
    pinion("centre pinion", CENTRE, (3.00, 3.55), 12, 0.12, gap_angle(CENTRE, BARREL, 12))
    gear(
        "centre wheel",
        "brass",
        CENTRE,
        (3.75, 3.95),
        watch_teeth(80, 0.09),
        angle=mesh_angle(CENTRE, THIRD),
        bore=0.35,
        holes=crossings(4, 0.78, 3.12, 0.40, 0.26, 0.3, curve=0.18),
    )
    wheels = (
        (
            "third",
            THIRD,
            ((3.05, 3.25), 75, 0.08),
            ((3.70, 4.00), 10, 0.09),
            (0.09, 0.25),
            FOURTH,
            CENTRE,
            (5, 0.62, 2.58, 0.32, 0.20),
        ),
        (
            "fourth",
            FOURTH,
            ((2.80, 3.00), 75, 0.07),
            ((3.02, 3.28), 10, 0.08),
            (0.08, 0.22),
            ESCAPE,
            THIRD,
            (5, 0.56, 2.20, 0.30, 0.17),
        ),
    )
    for name, q, (zw, teeth, m), (zp, leaves, mp), (pivot, body), nxt, prev, cross in wheels:
        pivoted(f"{name} arbor", q, pivot, body)
        pinion(f"{name} pinion", q, zp, leaves, mp, gap_angle(q, prev, leaves))
        gear(
            f"{name} wheel",
            "brass",
            q,
            zw,
            watch_teeth(teeth, m),
            angle=mesh_angle(q, nxt),
            bore=body,
            holes=crossings(*cross, phase=0.5, curve=0.15),
        )


def escapement() -> None:
    """Silicon escape wheel and pallet lever, ruby pallet stones and a steel guard pin."""
    e, p = ESCAPE, PALLET
    pivoted("escape arbor", e, 0.07, 0.16)
    pinion("escape pinion", e, (2.78, 3.02), 10, 0.07, gap_angle(e, FOURTH, 10))
    gear(
        "escape wheel",
        "silicon",
        e,
        (1.75, 1.95),
        club_teeth(20, ESCAPE_TIP, 1.62),
        angle=0.11,
        bore=0.16,
        holes=crossings(5, 0.42, 1.36, 0.20, 0.12, 0.2),
    )
    shaft(
        "pallet arbor",
        p,
        [(1.10, 0.07), (1.45, 0.07), (1.53, 0.18), (2.02, 0.18), (2.08, 0.07), (2.35, 0.07)],
    )
    u, v = d(LEVER), d(LEVER + math.pi / 2)
    outline = [
        circle(0.42, p),
        bar(p, p + 2.45 * u, 0.30, 0.24),
        rounded_bar(p + 2.40 * u - 0.30 * v, p + 2.40 * u + 0.30 * v, 0.30),
    ]
    outline += [
        rounded_bar(p + 2.40 * u + s * 0.30 * v, p + 2.92 * u + s * 0.33 * v, 0.30) for s in (1, -1)
    ]
    stones = []
    for k, sign in enumerate((1, -1)):
        r = d(towards(e, p) + sign * STONE_ANGLE)  # entry and exit, on the line of the tips
        outline += [rounded_bar(p, e + 2.80 * r, 0.34), bar(e + 2.60 * r, e + 3.05 * r, 0.46)]
        n = np.array([-r[1], r[0]])
        inner, outer = e + 2.30 * r, e + 2.92 * r
        slant = 0.06 * sign
        stone = prism(
            np.array(
                [
                    inner - 0.11 * n + slant * r,
                    outer - 0.11 * n,
                    outer + 0.11 * n,
                    inner + 0.11 * n - slant * r,
                ]
            ),
            1.70,
            2.00,
        )
        stones.append(stone)
        part(f"pallet stone {k}", "ruby", stone)
    fork = part(
        "pallet lever", "silicon", fuse("pallet lever", [prism(o, 1.75, 1.95) for o in outline])
    )
    guard = p + 2.48 * u
    cut(
        fork,
        *stones,
        prism(bar(p + 2.55 * u, p + 3.10 * u, 0.30), 1.6, 2.1),
        cylinder(0.18, 1.6, 2.1, p),
        cylinder(0.04, 1.6, 2.1, guard),
    )
    part("guard pin", "steel", cylinder(0.04, 1.75, 2.07, guard))


def balance() -> None:
    """Staff, roller and impulse pin, and the three-armed wheel with twelve gold screws."""
    bal = BALANCE
    shaft(
        "balance staff",
        bal,
        [
            (1.10, 0.05),
            (1.45, 0.05),
            (1.58, 0.20),
            (1.98, 0.20),
            (1.98, 0.22),
            (2.10, 0.22),
            (2.10, 0.25),
            (2.60, 0.25),
            (2.60, 0.30),
            (2.80, 0.30),
            (2.80, 0.22),
            (3.05, 0.22),
            (3.05, 0.20),
            (3.35, 0.20),
            (3.44, 0.05),
            (3.77, 0.05),
        ],
    )
    pin_at = bal + 0.50 * d(LEVER + math.pi)
    roller = part("roller", "steel", tube(0.22, 0.62, 1.98, 2.10, bal))
    cut(
        roller,
        cylinder(0.14, 1.9, 2.2, bal + 0.66 * d(LEVER + math.pi)),
        cylinder(0.085, 1.9, 2.2, pin_at),
    )
    part("impulse pin", "ruby", cylinder(0.085, 1.72, 2.10, pin_at))
    rim = _object("balance rim", tube(4.10, 4.50, 2.60, 3.15, bal))
    arms = _object("balance arms", cylinder(4.12, 2.60, 2.80, bal))
    openings = crossings(3, 0.85, 4.10, 0.55, 0.30, curve=0.25)
    cut(arms, *[at(prism(h, 2.5, 2.9), bal, 0.35) for h in openings], cylinder(0.30, 2.5, 2.9, bal))
    wheel = part("balance wheel", "glucydur", fuse("balance wheel", [rim, arms]))
    holes = []
    for k in range(12):
        a = 0.35 + TAU * (k + 0.5) / 12
        turn = along(np.array([*d(a), 0.0]))
        base = np.array([*(bal + 4.18 * d(a)), 2.875])
        holes.append(cylinder(0.11, 0, 0.33).turned(turn, base))
        head = lathe(
            [(0, 0), (0.11, 0), (0.11, 0.32), (0.21, 0.32), (0.21, 0.52), (0.15, 0.60), (0, 0.60)]
        ).turned(turn, base)
        slot = prism(bar(np.array([-0.3, 0]), np.array([0.3, 0]), 0.05), -0.3, 0.3)
        tip = np.array([*(bal + 4.79 * d(a)), 2.875])
        cut(
            part(f"timing screw {k}", "gold", head),
            Solid(slot.v @ turn[:, [2, 0, 1]].T + tip, slot.f),
        )
    cut(wheel, *holes)


def balance_spring() -> None:
    """The hairspring in its collet and stud, the curb pins and their regulator."""
    bal, start, turns, end_radius = BALANCE, 0.62, 12.5, 3.47
    coils = np.linspace(0, turns, 12 * 160)
    hair = bal + polar(start + (end_radius - start) * coils / turns, TAU * coils + 0.6)
    part("hairspring", "nivarox", strip(hair, 0.035, 2.90, 3.03))
    collet = part("collet", "steel", tube(0.22, 0.55, 2.88, 3.05, bal))
    cut(collet, at(prism(bar(np.zeros(2), np.array([0.7, 0]), 0.08), 2.8, 3.1), bal, 2.8))
    end = TAU * turns + 0.6
    stud = bal + 3.66 * d(end)
    block = prism(bar(np.array([-0.16, 0]), np.array([0.16, 0]), 0.26), 2.84, 3.09)
    part("stud", "steel", at(block, stud, end))
    part("stud pin", "steel", cylinder(0.07, 3.09, COCK[1] + 0.13, stud))
    pins_at = end - 0.42
    r_out = start + (end_radius - start) * (turns - 0.42 / TAU) / turns
    for k, dr in enumerate((-0.095, 0.095)):
        part(
            f"curb pin {k}",
            "steel",
            cylinder(0.045, 2.92, COCK[1] + 0.13, bal + (r_out + dr) * d(pins_at)),
        )
    index = [
        circle(0.85, bal),
        rounded_bar(bal, bal + (r_out + 0.1) * d(pins_at), 0.42),
        rounded_bar(bal, bal + 4.3 * d(pins_at + math.pi), 0.30),
    ]
    regulator = fuse("regulator", [prism(o, COCK[1], COCK[1] + 0.13) for o in index])
    part("regulator", "steel", cut(regulator, cylinder(0.55, COCK[1] - 0.1, COCK[1] + 0.3, bal)))
    arm = [circle(1.15, bal), rounded_bar(bal, stud, 0.36)]
    carrier = fuse("stud carrier", [prism(o, COCK[1] + 0.13, COCK[1] + 0.25) for o in arm])
    part("stud carrier", "steel", cut(carrier, cylinder(0.90, COCK[1], COCK[1] + 0.4, bal)))


def keyless(barrel_bridge: Obj) -> None:
    """Ratchet wheel on the arbor's square, crown wheel, click and click spring."""
    b, top = BARREL, BRIDGES[1] + 0.05
    ratchet = gear(
        "ratchet wheel",
        "steel",
        b,
        (top, 5.70),
        ratchet_teeth(60, 3.55, 3.35),
        angle=0.0,
        bore=0.01,
        holes=crossings(5, 1.05, 2.85, 0.45, 0.30, 0.1, curve=-0.3),
    )
    cut(ratchet, at(prism(SQUARE * 1.001, 5.0, 6.0), b, 0.2))
    screw("ratchet screw", b, 5.70, 0.95, 0.30, 0.45, slot=0.8)
    gear(
        "crown wheel",
        "steel",
        CROWN,
        (top, 5.62),
        watch_teeth(36, 0.11),
        angle=0.0,
        bore=0.30,
        holes=crossings(3, 0.62, 1.55, 0.35, 0.22, 0.4),
    )
    screw("crown wheel screw", CROWN, 5.62, 0.62, 0.30, 0.7, slot=1.3, hosts=(barrel_bridge,))
    tip = b + 3.62 * d(math.radians(-2))
    click = [circle(0.42, CLICK), bar(CLICK, tip, 0.55, 0.18)]
    part("click", "steel", fuse("click", [prism(o, top, 5.55) for o in click]))
    screw("click screw", CLICK, 5.55, 0.40, 0.20, 0.6, slot=0.0, hosts=(barrel_bridge,))
    spring = arc(CLICK, 1.15, math.radians(170), math.radians(300), 40)
    part("click spring", "steel", strip(spring, 0.10, top, 5.45))


def build() -> dict[str, Any]:
    """Model the movement; returns its description."""
    plate = main_plate()
    made = bridges(plate)
    barrel()
    going_train()
    escapement()
    balance()
    balance_spring()
    keyless(made["barrel"])
    named = {
        "barrel": BARREL,
        "centre": CENTRE,
        "third": THIRD,
        "fourth": FOURTH,
        "escape": ESCAPE,
        "pallet": PALLET,
        "balance": BALANCE,
    }
    return {
        "units": "mm",
        "axis": "z",
        "levels_mm": LEVELS,
        "radius_mm": RADIUS,
        "height_mm": [0.0, 5.98],
        "positions_mm": {k: [float(x) for x in q] for k, q in named.items()},
    }


def export(out: Path, description: dict[str, Any]) -> None:
    """Fuse each material's parts and write ``MATERIAL.stl`` and ``watch.json`` to ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    description["materials"] = {}
    for material, objs in _parts.items():
        alive = [o for o in objs if o.name in bpy.data.objects]
        if not alive:
            continue
        merged = fuse(material, alive)
        for o in bpy.context.scene.objects:
            o.select_set(False)
        merged.select_set(True)
        bpy.context.view_layer.objects.active = merged
        bpy.ops.wm.stl_export(
            filepath=str(out / f"{material}.stl"),
            export_selected_objects=True,
            ascii_format=False,
            apply_modifiers=True,
        )
        tris = sum(len(poly.vertices) - 2 for poly in merged.data.polygons)
        description["materials"][material] = {"parts": len(alive), "triangles": tris}
        print(f"{material}: {len(alive)} parts, {tris} triangles")
    (out / "watch.json").write_text(json.dumps(description, indent=2))


def main() -> None:
    """Build the movement into the directory after ``--``."""
    out = Path(sys.argv[sys.argv.index("--") + 1])
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    export(out, build())


if __name__ == "__main__":
    main()
