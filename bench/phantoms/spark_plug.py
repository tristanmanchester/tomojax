"""A spark plug as one closed STL mesh per material, for gVXR.

python spark_plug.py OUT_DIR        (needs numpy, scipy and scikit-image)

A small-engine iridium plug with an M10 x 1 thread, z along its axis, millimeters.

- Steel shell: a 60 degree thread with rounded crests, a chamfered lead-in and a run-out into a
  relief groove; a seat flange with a chamfered edge; a crimp groove; a 16 mm hex whose corners are
  chamfered by a 30 degree cone; and a top rolled over onto the insulator.
- Inside the shell: a copper gasket where the insulator's shoulder sits on the shell's seat, and a
  talc packing and a steel wire ring under the rolled lip.
- Outside: a folded steel sealing washer under the flange.
- Alumina insulator: a rounded nose, a shoulder, a collar under the crimp, a neck, sinusoidal ribs
  and a rounded top; its bore narrows for the center electrode.
- Down the bore: a nickel center electrode with a flared head on the bore step, a copper core and an
  iridium tip; a copper-loaded conductive glass seal with small and a few large bubbles; a carbon
  resistor with pores; a
  second glass seal holding the threaded end of the steel terminal stud; a round terminal nut with
  its groove and an M4 tip.
- A nickel ground electrode with rounded edges, a filleted bend and a weld bead, turned 1.5 degrees,
  carrying a platinum pad under a 0.7 mm gap.
- The insulator, electrode, seals and stud sit 0.02 mm off the shell's axis.

Turned and threaded parts are surfaces of radius against angle and height, so threads are exact
helices; parts with voids, blends or welds are signed distance functions meshed with marching cubes.
Parts are separated by small clearances, so different materials never overlap.
"""

from collections.abc import Callable, Sequence
import json
from pathlib import Path
import struct
import sys

import numpy as np
from skimage.measure import marching_cubes

Mesh = tuple[np.ndarray, np.ndarray]  # vertices (n, 3) and triangles (m, 3)
Radius = Callable[[np.ndarray, np.ndarray], np.ndarray]  # radius from (theta, z)
TAU = 2 * np.pi
N_THETA = 360
OFFSET = np.array([0.02, 0.0, 0.0])  # the insulator assembly's eccentricity in the shell
MU_120KEV = {  # mm^-1 at 120 keV: xraylib CS_Total times density
    "steel": 0.2102,  # Fe 0.98, C 0.01, Mn 0.01, 7.85 g/cm^3
    "nickel": 0.2714,  # Ni 0.96, Cr 0.02, Mn 0.02, 8.75
    "copper": 0.2862,  # Cu, 8.96
    "alumina": 0.0585,  # Al2O3, 3.9
    "conductive_glass": 0.0787,  # borosilicate with 40% copper by mass, 3.6
    "resistor": 0.0280,  # carbon-glass composite, 1.9
    "talc": 0.0291,  # compacted Mg3Si4O10(OH)2, 1.9
    "iridium": 6.8818,  # Ir, 22.56
    "platinum": 6.8914,  # Pt, 21.45
}
HEIGHT = (-22.6, 23.5)


# ---- surfaces of revolution and threads ---------------------------------------------------------
def smoothstep(x: np.ndarray) -> np.ndarray:
    """Return the cubic smoothstep of ``x`` clipped to [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def thread(
    t: np.ndarray,
    z: np.ndarray,
    r_minor: float,
    r_major: float,
    pitch: float,
    start: float | None = None,
    end: float | None = None,
    runout: float = 1.5,
) -> np.ndarray:
    """A 60 degree thread with rounded crests and roots, fading over ``runout`` mm at its end."""
    phase = np.mod((z - pitch * t / TAU) / pitch, 1.0)
    tooth = smoothstep((1.0 - np.abs(phase - 0.5) * 2.0) * 1.5 - 0.25)
    depth = np.ones_like(z)
    if start is not None:
        depth = np.minimum(depth, smoothstep((z - start) / 0.4))
    if end is not None:
        depth = np.minimum(depth, smoothstep((end - z) / runout))
    return r_minor + (r_major - r_minor) * tooth * depth


def hexagon(t: np.ndarray, across_flats: float) -> np.ndarray:
    """Radius of a regular hexagon of the given width across flats."""
    return across_flats / 2 / np.cos(np.mod(t, TAU / 6) - TAU / 12)


def chamfered_hexagon(
    t: np.ndarray, z: np.ndarray, across_flats: float, z0: float, z1: float, chamfer: float = 0.55
) -> np.ndarray:
    """A hexagon whose corners are cut by 30 degree cones at both faces."""
    face = across_flats / 2 + 0.15
    cone = face + np.minimum(z - z0, z1 - z) * np.tan(np.radians(60)) / chamfer * chamfer
    return np.minimum(hexagon(t, across_flats), cone)


def constant(r: float) -> Radius:
    """A radius function that is ``r`` everywhere."""
    return lambda t, _z: np.full_like(t, r)


def orient(v: np.ndarray, t: np.ndarray) -> Mesh:
    """Return the mesh with triangles wound so its signed volume is positive."""
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    return (v, t) if volume > 0 else (v, t[:, ::-1].copy())


def radial_solid(rows: Sequence[tuple], n_theta: int = N_THETA) -> Mesh:
    """Closed solid between outer and inner radius surfaces, from a list of rows.

    Each row is (z, outer) or (z, outer, inner), radius functions of (theta, z). Consecutive rows
    may share a height to make a flat step. Rows without an inner radius reach the axis; mixing the
    two is not allowed.
    """
    theta = np.linspace(0, TAU, n_theta, endpoint=False)
    zs = np.array([row[0] for row in rows], float)
    solid = len(rows[0]) == 2
    assert all((len(row) == 2) == solid for row in rows)

    def ring(r: np.ndarray, z: float) -> np.ndarray:
        return np.stack([r * np.cos(theta), r * np.sin(theta), np.full_like(theta, z)], -1)

    outer = np.stack([row[1](theta, np.full_like(theta, row[0])) for row in rows])
    verts = [np.stack([ring(r, z) for r, z in zip(outer, zs, strict=False)]).reshape(-1, 3)]
    nz = len(zs)
    idx = np.arange(nz * n_theta).reshape(nz, n_theta)
    j1 = np.roll(np.arange(n_theta), -1)
    tris = []
    for i in range(nz - 1):
        a, b, c, d = idx[i], idx[i, j1], idx[i + 1, j1], idx[i + 1]
        tris += [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
    base = nz * n_theta
    if solid:
        verts.append(np.array([[0, 0, zs[0]], [0, 0, zs[-1]]], float))
        tris.append(np.stack([np.full(n_theta, base), idx[0, j1], idx[0]], 1))
        tris.append(np.stack([np.full(n_theta, base + 1), idx[-1], idx[-1, j1]], 1))
    else:
        inner = np.stack([row[2](theta, np.full_like(theta, row[0])) for row in rows])
        assert (inner < outer).all(), "inner radius reaches the outer one"
        verts.append(np.stack([ring(r, z) for r, z in zip(inner, zs, strict=False)]).reshape(-1, 3))
        jdx = idx + base
        for i in range(nz - 1):
            a, b, c, d = jdx[i], jdx[i, j1], jdx[i + 1, j1], jdx[i + 1]
            tris += [np.stack([a, c, b], 1), np.stack([a, d, c], 1)]
        tris += [
            np.stack([idx[0], jdx[0], jdx[0, j1]], 1),
            np.stack([idx[0], jdx[0, j1], idx[0, j1]], 1),
        ]
        tris += [
            np.stack([idx[-1], idx[-1, j1], jdx[-1, j1]], 1),
            np.stack([idx[-1], jdx[-1, j1], jdx[-1]], 1),
        ]
    return orient(np.concatenate(verts), np.concatenate(tris))


def revolve(loop: Sequence[tuple[float, float]], n_theta: int = N_THETA) -> Mesh:
    """Solid of revolution of a closed (r, z) loop; points with r = 0 lie on the axis."""
    loop = np.asarray(loop, float)
    theta = np.linspace(0, TAU, n_theta, endpoint=False)
    verts, rings = [], []
    for r, z in loop:
        if r == 0:
            rings.append(np.full(n_theta, sum(len(v) for v in verts)))
            verts.append(np.array([[0.0, 0.0, z]]))
        else:
            start = sum(len(v) for v in verts)
            rings.append(start + np.arange(n_theta))
            verts.append(np.stack([r * np.cos(theta), r * np.sin(theta), np.full(n_theta, z)], -1))
    j1 = np.roll(np.arange(n_theta), -1)
    tris = []
    for i in range(len(loop)):
        a, d = rings[i], rings[(i + 1) % len(loop)]
        b, c = a[j1], d[j1]
        for tri in (np.stack([a, b, c], 1), np.stack([a, c, d], 1)):
            keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
            tris.append(tri[keep])
    return orient(np.concatenate(verts), np.concatenate(tris))


# ---- signed distance parts ---------------------------------------------------------------------
def sd_cylinder(
    p: Sequence[np.ndarray], r: float, z0: float, z1: float, c: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Signed distance to a cylinder of radius ``r`` along z between ``z0`` and ``z1``."""
    radial = np.hypot(p[0] - c[0], p[1] - c[1]) - r
    along = np.maximum(z0 - p[2], p[2] - z1)
    return np.minimum(np.maximum(radial, along), 0) + np.hypot(
        np.maximum(radial, 0), np.maximum(along, 0)
    )


def sd_box(
    p: Sequence[np.ndarray],
    lo: Sequence[float],
    hi: Sequence[float],
    round_: float = 0.0,
) -> np.ndarray:
    """Signed distance to the box from ``lo`` to ``hi`` with edges rounded by ``round_``."""
    c = [(lo[i] + hi[i]) / 2 for i in range(3)]
    h = [(hi[i] - lo[i]) / 2 - round_ for i in range(3)]
    q = [np.abs(p[i] - c[i]) - h[i] for i in range(3)]
    return (
        np.sqrt(sum(np.maximum(qi, 0) ** 2 for qi in q))
        + np.minimum(np.maximum(np.maximum(q[0], q[1]), q[2]), 0)
        - round_
    )


def sd_ellipsoid(p: Sequence[np.ndarray], c: Sequence[float], r: Sequence[float]) -> np.ndarray:
    """Approximate signed distance to an ellipsoid with center ``c`` and radii ``r``."""
    q = [(p[i] - c[i]) / r[i] for i in range(3)]
    return (np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2) - 1.0) * min(r)


def smooth_union(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth minimum of two signed distance fields with blend width ``k``."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def voids(p: Sequence[np.ndarray], centers: Sequence[tuple], radii: Sequence[float]) -> np.ndarray:
    """Signed distance to the nearest of a set of spheres."""
    return np.min(
        np.stack(
            [
                np.sqrt(sum((p[i] - c[i]) ** 2 for i in range(3))) - r
                for c, r in zip(centers, radii, strict=True)
            ]
        ),
        0,
    )


def random_voids(
    rng: np.random.Generator,
    count: int,
    r_range: tuple[float, float],
    z_range: tuple[float, float],
    radius: float,
    rmin: float = 0.05,
) -> tuple[list, list]:
    """Void centers inside a cylinder of ``radius`` (less the void) between z_range, and radii."""
    centers, radii = [], []
    while len(centers) < count:
        r = rng.uniform(*r_range)
        rho = np.sqrt(rng.uniform(0, 1)) * (radius - r - rmin)
        a = rng.uniform(0, TAU)
        z = rng.uniform(z_range[0] + r + rmin, z_range[1] - r - rmin)
        centers.append((rho * np.cos(a), rho * np.sin(a), z))
        radii.append(r)
    return centers, radii


def marched(
    sdf: Callable[[Sequence[np.ndarray]], np.ndarray],
    lo: Sequence[float],
    hi: Sequence[float],
    step: float,
) -> Mesh:
    """Mesh the zero level set of ``sdf`` on a grid of ``step`` mm over the box ``lo``-``hi``."""
    lo, hi = np.array(lo, float) - 3 * step, np.array(hi, float) + 3 * step
    axes = [np.arange(lo[i], hi[i] + step / 2, step) for i in range(3)]
    field = sdf(np.meshgrid(*axes, indexing="ij")).astype(np.float32)
    field[[0, -1], :, :] = field[:, [0, -1], :] = field[:, :, [0, -1]] = 1.0
    v, f, _, _ = marching_cubes(field, level=0.0, spacing=(step, step, step))
    return orient(v + lo, f)


def moved(mesh: Mesh, offset: np.ndarray) -> Mesh:
    """Translate a mesh by ``offset``."""
    v, t = mesh
    return v + offset, t


def merge(*meshes: Mesh) -> Mesh:
    """Concatenate meshes into one."""
    verts, tris, n = [], [], 0
    for v, t in meshes:
        verts.append(v)
        tris.append(t + n)
        n += len(v)
    return np.concatenate(verts), np.concatenate(tris)


# ---- the plug ----------------------------------------------------------------------------------
BOTTOM, THREAD_END, SEAT, CRIMP, HEX0, HEX1, LIP0, LIP1 = (
    -19.0,
    -9.2,
    -8.0,
    -4.8,
    -4.2,
    3.0,
    3.8,
    4.4,
)
NARROW, BORE, WIDE, LIP_BORE = 3.4, 4.35, 4.75, 4.0


def shell() -> Mesh:
    """The steel shell: thread, seat flange, crimp groove, hex and rolled lip."""

    def m10(t: np.ndarray, z: np.ndarray) -> np.ndarray:
        return thread(t, z, 4.43, 4.97, 1.0, start=BOTTOM, end=THREAD_END)

    def flange(_t: np.ndarray, z: np.ndarray) -> np.ndarray:  # chamfered lower edge
        return np.minimum(7.0, 6.75 + (z - SEAT) * 1.0)

    def hexa(t: np.ndarray, z: np.ndarray) -> np.ndarray:
        return chamfered_hexagon(t, z, 16.0, HEX0, HEX1)

    rows = [
        (z, m10, constant(NARROW))
        for z in np.linspace(BOTTOM, THREAD_END, int((THREAD_END - BOTTOM) * 40) + 1)
    ]
    rows += [
        (THREAD_END, constant(4.22), constant(NARROW)),
        (SEAT, constant(4.22), constant(NARROW)),
        (SEAT, flange, constant(NARROW)),
        (SEAT, flange, constant(BORE)),
    ]
    rows += [(z, flange, constant(BORE)) for z in np.linspace(SEAT, SEAT + 0.3, 7)[1:]]
    rows += [
        (CRIMP, constant(7.0), constant(BORE)),
        (CRIMP, constant(6.0), constant(BORE)),
        (HEX0, constant(6.0), constant(BORE)),
        (HEX0, hexa, constant(BORE)),
    ]
    hex_z = np.linspace(HEX0, HEX1, 49)[1:]
    for z in hex_z:
        if z > 1.0 >= z - (hex_z[1] - hex_z[0]):
            rows += [(1.0, hexa, constant(BORE)), (1.0, hexa, constant(WIDE))]
        rows.append((z, hexa, constant(BORE if z < 1.0 else WIDE)))
    rows += [
        (HEX1, constant(5.6), constant(WIDE)),
        (LIP0, constant(5.55), constant(WIDE)),
        (LIP0, constant(5.55), constant(LIP_BORE)),
    ]
    for z in np.linspace(LIP0, LIP1, 9)[1:]:  # the rolled lip, rounded towards its top
        s = (z - LIP0) / (LIP1 - LIP0)
        rows.append((z, constant(5.55 - 0.85 * s**2), constant(LIP_BORE)))
    return radial_solid(rows)


def internal_gasket() -> Mesh:
    """The copper gasket under the insulator shoulder."""
    return radial_solid(
        [
            (SEAT + 0.02, constant(4.28), constant(3.47)),
            (SEAT + 0.17, constant(4.28), constant(3.47)),
        ]
    )


def washer() -> Mesh:
    """The folded sealing washer: a C-shaped section open towards the thread."""
    return revolve(
        [
            (5.15, -9.02),
            (6.6, -9.02),
            (6.8, -8.82),
            (6.8, -8.22),
            (6.6, -8.02),
            (5.15, -8.02),
            (5.15, -8.17),
            (6.5, -8.17),
            (6.65, -8.32),
            (6.65, -8.72),
            (6.5, -8.87),
            (5.15, -8.87),
        ][::-1]
    )


def talc_and_ring() -> tuple[Mesh, Mesh]:
    """The talc packing and the steel wire ring under the rolled lip."""
    talc = radial_solid(
        [(3.18, constant(4.72), constant(4.0)), (3.45, constant(4.72), constant(4.0))]
    )
    ring = marched(
        lambda p: np.hypot(np.hypot(p[0], p[1]) - 4.36, p[2] - 3.62) - 0.15,
        (-4.6, -4.6, 3.4),
        (4.6, 4.6, 3.85),
        0.025,
    )
    return talc, ring


def insulator() -> Mesh:
    """The alumina insulator, with its stepped bore."""
    ribs = [
        (4.2 + 0.24 * (1 - np.cos(TAU * (z - 5.0) / 2.3)) / 2, z)
        for z in np.linspace(5.0, 16.5, 81)
    ]
    outer = [
        (1.35, -17.5),
        (1.6, -17.42),
        (1.75, -17.22),
        (3.3, -12.0),
        (3.3, SEAT + 0.2),
        (4.3, SEAT + 0.6),
        (4.3, 1.4),
        (4.7, 1.6),
        (4.7, 2.95),
        (3.95, 3.15),
        (3.95, 4.45),
        (4.2, 4.75),
        *ribs,
        (4.2, 16.9),
        (4.1, 17.55),
        (3.8, 17.9),
        (3.4, 18.0),
    ]
    bore = [(1.06, 18.0), (1.06, -11.98), (0.86, -11.98), (0.86, -17.5)]
    return moved(revolve((outer + bore)[::-1]), OFFSET)


def center_electrode() -> tuple[Mesh, Mesh, Mesh]:
    """The nickel center electrode, its copper core and its iridium tip."""
    nickel = revolve(
        [
            (0, -19.6),
            (0.55, -19.6),
            (0.8, -19.35),
            (0.8, -11.96),
            (1.0, -11.96),
            (1.0, -11.4),
            (0.45, -11.4),
            (0.45, -18.8),
            (0, -18.8),
        ][::-1]
    )
    copper = revolve([(0, -18.8), (0.45, -18.8), (0.45, -11.4), (0, -11.4)][::-1])
    iridium = revolve([(0, -20.45), (0.27, -20.45), (0.3, -20.4), (0.3, -19.6), (0, -19.6)][::-1])
    return moved(nickel, OFFSET), moved(copper, OFFSET), moved(iridium, OFFSET)


def bore_fill(seed: int = 7) -> tuple[Mesh, Mesh]:
    """Glass seals with bubbles, the resistor with pores, and the stud's end in the seal."""
    rng = np.random.default_rng(seed)

    def joined(*sets: tuple[list, list]) -> tuple[list, list]:
        return [c for s in sets for c in s[0]], [r for s in sets for r in s[1]]

    # Mostly small bubbles and pores, with a few large bubbles in the seals, as in fired glass.
    lower = joined(
        random_voids(rng, 6, (0.06, 0.16), (-11.38, -9.8), 1.04),
        random_voids(rng, 3, (0.2, 0.3), (-11.38, -9.8), 1.04),
    )
    pores = random_voids(rng, 9, (0.05, 0.11), (-9.78, -3.2), 1.04)
    upper = joined(
        random_voids(rng, 5, (0.06, 0.14), (-3.18, -2.62), 1.04, rmin=0.08),
        random_voids(rng, 2, (0.18, 0.22), (-3.18, -2.62), 1.04, rmin=0.04),
    )

    def stud_end(p: Sequence[np.ndarray]) -> np.ndarray:
        return sd_cylinder(p, 0.97, -2.62, -0.9)

    def glass(p: Sequence[np.ndarray]) -> np.ndarray:
        a = np.maximum(sd_cylinder(p, 1.04, -11.38, -9.8), -voids(p, *lower))
        b = np.maximum(
            np.maximum(sd_cylinder(p, 1.04, -3.18, -1.0), -stud_end(p)), -voids(p, *upper)
        )
        return np.minimum(a, b)

    def resistor(p: Sequence[np.ndarray]) -> np.ndarray:
        return np.maximum(sd_cylinder(p, 1.04, -9.78, -3.2), -voids(p, *pores))

    glass_mesh = marched(glass, (-1.1, -1.1, -11.45), (1.1, 1.1, -0.95), 0.02)
    resistor_mesh = marched(resistor, (-1.1, -1.1, -9.85), (1.1, 1.1, -3.15), 0.02)
    return moved(glass_mesh, OFFSET), moved(resistor_mesh, OFFSET)


def terminal() -> tuple[Mesh, Mesh, Mesh]:
    """The terminal stud, its round nut and the M4 tip."""

    def stud_row(t: np.ndarray, z: np.ndarray) -> np.ndarray:
        return thread(t, z, 0.78, 0.94, 0.5, start=-2.6, end=-1.0, runout=0.3)

    def tip_row(t: np.ndarray, z: np.ndarray) -> np.ndarray:
        return thread(t, z, 1.55, 1.9, 0.7, start=20.2, end=23.3, runout=0.6)

    stud_thread = [(z, stud_row) for z in np.linspace(-2.6, -1.0, 4 * 33 + 1)]
    stud = radial_solid(
        [*stud_thread, (-1.0, constant(0.94)), (-1.0, constant(1.0)), (18.0, constant(1.0))]
    )
    nut = revolve(
        [
            (0, 18.02),
            (2.25, 18.02),
            (2.4, 18.17),
            (2.4, 18.85),
            (2.15, 19.1),
            (2.4, 19.35),
            (2.4, 20.05),
            (2.25, 20.2),
            (0, 20.2),
        ][::-1]
    )
    m4 = radial_solid(
        [(z, tip_row) for z in np.linspace(20.2, 23.3, 4 * 45 + 1)] + [(23.5, constant(1.45))]
    )
    return moved(stud, OFFSET), nut, m4


def ground_electrode() -> tuple[Mesh, Mesh]:
    """The nickel ground electrode with its weld bead, and the platinum pad."""
    turn = np.radians(1.5)

    def nickel(p: Sequence[np.ndarray]) -> np.ndarray:
        x = p[0] * np.cos(turn) + p[1] * np.sin(turn)
        y = -p[0] * np.sin(turn) + p[1] * np.cos(turn)
        q = (x, y, p[2])
        leg = sd_box(q, (3.75, -1.1, -22.0), (4.85, 1.1, -19.04), round_=0.18)
        arm = sd_box(q, (-0.9, -1.1, -22.45), (4.85, 1.1, -21.25), round_=0.18)
        bead = np.maximum(sd_ellipsoid(q, (4.3, 0.0, -19.04), (0.85, 1.3, 0.28)), q[2] + 19.04)
        # The blends bulge past their parts, so clip the weld just below the shell's end face.
        body = np.maximum(smooth_union(smooth_union(leg, arm, 0.45), bead, 0.2), q[2] + 19.03)
        pad = sd_cylinder(q, 0.36, -21.32, -21.15)
        return np.maximum(body, -(pad - 0.03))  # cut 0.03 mm wider than the pad

    def platinum(p: Sequence[np.ndarray]) -> np.ndarray:
        x = p[0] * np.cos(turn) + p[1] * np.sin(turn)
        y = -p[0] * np.sin(turn) + p[1] * np.cos(turn)
        return sd_cylinder((x, y, p[2]), 0.35, -21.30, -21.15)

    return (
        marched(nickel, (-1.0, -1.4, -22.55), (5.0, 1.4, -18.75), 0.025),
        marched(platinum, (-0.45, -0.45, -21.35), (0.45, 0.45, -21.1), 0.012),
    )


def build() -> dict[str, Mesh]:
    """Every material's mesh, by name."""
    nickel, copper, iridium = center_electrode()
    glass, resistor = bore_fill()
    stud, nut, m4 = terminal()
    ground, platinum = ground_electrode()
    talc, ring = talc_and_ring()
    return {
        "steel": merge(shell(), washer(), ring, stud, nut, m4),
        "copper": merge(copper, internal_gasket()),
        "nickel": merge(nickel, ground),
        "iridium": iridium,
        "platinum": platinum,
        "alumina": insulator(),
        "conductive_glass": glass,
        "resistor": resistor,
        "talc": talc,
    }


def closed(mesh: Mesh) -> bool:
    """Whether every edge of the mesh is shared by two triangles of opposite winding."""
    _, t = mesh
    edges = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]).tolist()
    forward = {tuple(e) for e in edges}
    return len(forward) == len(edges) and all((b, a) in forward for a, b in forward)


def write_stl(path: Path, mesh: Mesh) -> None:
    """Write a mesh as a binary STL."""
    v, t = mesh
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    normal = np.cross(b - a, c - a)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    record = np.zeros(
        len(t),
        dtype=[("n", "<f4", 3), ("a", "<f4", 3), ("b", "<f4", 3), ("c", "<f4", 3), ("x", "<u2")],
    )
    record["n"], record["a"], record["b"], record["c"] = normal, a, b, c
    with path.open("wb") as f:
        f.write(b"spark plug".ljust(80, b" ") + struct.pack("<I", len(t)) + record.tobytes())


def main() -> None:
    """Write one STL per material and spark_plug.json into ``sys.argv[1]``."""
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, mesh in build().items():
        write_stl(out / f"{name}.stl", mesh)
        report[name] = {
            "triangles": len(mesh[1]),
            "closed": bool(closed(mesh)),
            "mu_mm": MU_120KEV[name],
        }
        print(name, report[name], flush=True)
    (out / "spark_plug.json").write_text(
        json.dumps(
            {
                "units": "mm",
                "axis": "z",
                "height_mm": HEIGHT,
                "energy_kev": 120,
                "materials": report,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
