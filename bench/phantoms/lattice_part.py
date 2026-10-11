#!/usr/bin/env python3
"""Additively manufactured Ti-6Al-4V lattice part for lab cone-beam CT at 100 keV.

A laser powder-bed fusion cylinder, 10 mm across and 10 mm tall, built along
+z: a 0.5 mm wall, a 0.8 mm base plate and a 0.5 mm top skin enclose an
octet-truss lattice (2.5 mm cubic cell, 0.42 mm struts) whose nodes are
filleted into each other and into the shell. A 1 mm drain hole pierces the top
skin and another the wall. The 12 mm field of view gives 47 µm voxels at
n = 256, a typical lab micro-CT scan of a part this size.

The octet truss is the set of lines along the six <110> directions through the
nodes of a face-centred cubic lattice; the distance to each family is the
distance to the nearest point of a centred rectangular lattice in the plane
across it. Surfaces carry low-amplitude waviness and partly fused powder
particles (16 to 50 µm across), four times as dense on down-facing surfaces,
as LPBF surfaces do.

Defects:

- gas and keyhole pores (30 to 120 µm) through the solid, and a ring of
  contour pores 0.12 mm under the outer wall;
- a lack-of-fusion flaw in the base plate: a flat, irregular void about
  1.5 x 0.8 mm across and up to 100 µm tall, holding unmelted powder;
- a broken strut with rough fracture faces;
- a tungsten inclusion of about 100 µm in a lattice node;
- loose powder trapped against the base plate in one corner of the lattice.

Attenuation is the total linear attenuation coefficient with coherent
scattering at 100 keV (xraylib 4.3 ``CS_Total_CP``, the counterpart of the
NIST XCOM total with coherent scattering): Ti-6Al-4V (Ti 90, Al 6, V 4 wt%) at
4.43 g/cm^3 and tungsten at 19.3. 100 keV is about the mean energy of a 190
kVp tungsten spectrum behind 1 mm of copper. Air is 0. Volumes are attenuation
in mm^-1, ``(x, y, z)`` order.

    python bench/phantoms/lattice_part.py --n 256 --preview /tmp/lattice_part.png
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from _procedural import Grid, WaveNoise, hash_uniform, main, voxelise
import numpy as np
from scipy.spatial import cKDTree

FOV_MM = 12.0
ENERGY_KEV = 100.0
MATERIALS = {"air": 0.0, "Ti-6Al-4V": 0.1181, "tungsten": 8.563}
AIR, TITANIUM, TUNGSTEN = range(3)

RADIUS, WALL, HALF_HEIGHT = 5.0, 0.5, 5.0
PLATE_TOP, SKIN_BOTTOM = -4.2, 4.5
CELL, STRUT = 2.5, 0.21
ORIGIN = np.array([0.0, 0.0, PLATE_TOP + 1.5 * CELL])  # a node plane lies on the base plate
FILLET = 0.12
CHUNK = 1 << 16

# Octet strut directions and, for each, an in-plane axis and the cube axis across it.
_DIRS = []
for axis in range(3):
    e2 = np.eye(3)[axis]
    a, b = [i for i in range(3) if i != axis]
    for sign in (1.0, -1.0):
        d = np.zeros(3)
        d[a], d[b] = 1.0, sign
        d /= np.sqrt(2)
        _DIRS.append((d, np.cross(d, e2), e2))


def _smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth minimum: a fillet of radius about ``k``."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def octet_distance(p: np.ndarray) -> np.ndarray:
    """Smooth distance from points ``(m, 3)`` (lattice frame) to the octet strut axes."""
    su, sw = CELL / (2 * np.sqrt(2)), CELL / 2
    best = None
    for _, e1, e2 in _DIRS:
        u, w = p @ e1, p @ e2
        d2 = np.inf
        for ou, ow in ((0.0, 0.0), (su, sw)):
            du = u - ou - 2 * su * np.round((u - ou) / (2 * su))
            dw = w - ow - 2 * sw * np.round((w - ow) / (2 * sw))
            d2 = np.minimum(d2, du * du + dw * dw)
        dist = np.sqrt(d2)
        best = dist if best is None else _smin(best, dist, FILLET)
    assert best is not None
    return best


def _round_box(
    r: np.ndarray, z: np.ndarray, r1: float, z0: float, z1: float, rounding: float
) -> np.ndarray:
    dr = r - (r1 - rounding)
    dz = np.maximum(z0 + rounding - z, z - (z1 - rounding))
    outside = np.hypot(np.maximum(dr, 0), np.maximum(dz, 0))
    return outside + np.minimum(np.maximum(dr, dz), 0) - rounding


class Part:
    """Signed distance and labels of the lattice part; negative inside solid."""

    def __init__(self, seed: int) -> None:
        """Draw surface relief, pores, particles and powder from ``seed``."""
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.wave = WaveNoise.make(rng, 0.4)
        self.thickness = WaveNoise.make(rng, 1.5)
        self.ripple = WaveNoise.make(rng, 0.12)
        self.flaw_noise = WaveNoise.make(rng, 0.35, waves=6)
        self.fracture = WaveNoise.make(rng, 0.08, waves=6)
        self._pores(rng)
        self._particles(rng)
        nodes = ORIGIN + 0.5 * CELL * np.array([[2.0, 0.0, 2.0], [3.0, 0.0, 3.0]])
        self.strut_mid = nodes.mean(axis=0)
        self.strut_dir = (nodes[1] - nodes[0]) / np.linalg.norm(nodes[1] - nodes[0])
        self.tungsten = ORIGIN + 0.5 * CELL * np.array([-2.0, 0.0, 0.0])
        self.flaw = np.array([-0.3, 0.0, -4.62])

    # Geometry without pores, particles or flaws.
    def base(self, p: np.ndarray) -> np.ndarray:
        """Signed distance to the shell and lattice with surface relief."""
        x, y, z = p.T
        r = np.hypot(x, y)
        outer = _round_box(r, z, RADIUS, -HALF_HEIGHT, HALF_HEIGHT, 0.25)
        cavity = np.maximum(r - (RADIUS - WALL), np.maximum(PLATE_TOP - z, z - SKIN_BOTTOM))
        shell = np.maximum(outer, -cavity)
        lattice = octet_distance(p - ORIGIN) - STRUT - 0.018 * self.thickness(x, y, z)
        solid = _smin(shell, np.maximum(lattice, cavity), FILLET)
        top_hole = np.maximum(r - 0.5, SKIN_BOTTOM - 0.1 - z)
        side_hole = np.maximum(np.hypot(x, z) - 0.5, RADIUS - WALL - 0.1 - y)
        solid = np.maximum(solid, -np.minimum(top_hole, side_hole))
        return solid + 0.008 * self.wave(x, y, z) + 0.005 * self.ripple(x, y, z)

    def _normal(self, p: np.ndarray) -> np.ndarray:
        h = 0.002
        g = np.stack([self.base(p + h * e) - self.base(p - h * e) for e in np.eye(3)], -1)
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)

    def _pores(self, rng: np.random.Generator) -> None:
        candidates = rng.uniform(-RADIUS, RADIUS, (40000, 3))
        candidates = candidates[np.hypot(candidates[:, 0], candidates[:, 1]) < RADIUS]
        radius = rng.uniform(0.015, 0.06, candidates.shape[0])
        inside = self.base(candidates) < -(radius + 0.04)
        centre, radius = candidates[inside][:60], radius[inside][:60]
        phi = rng.uniform(0, 2 * np.pi, 14)
        phi[0] = 0.3
        contour_z = rng.uniform(-0.5, 0.5, 14)
        contour_z[0] = 0.0
        depth = RADIUS - 0.12
        contour = np.stack([depth * np.cos(phi), depth * np.sin(phi), contour_z], -1)
        self.pore_centre = np.vstack([centre, contour])
        self.pore_radius = np.concatenate([radius, rng.uniform(0.03, 0.05, 14)])
        stretch = np.ones((self.pore_centre.shape[0], 3))
        stretch[: centre.shape[0] // 3, 2] = 1.8  # keyhole pores, elongated along the build
        self.pore_stretch = stretch

    def _particles(self, rng: np.random.Generator) -> None:
        """Partly fused powder on the surfaces, denser where they face down."""
        pts = np.column_stack(
            [
                rng.uniform(-RADIUS - 0.1, RADIUS + 0.1, (1_500_000, 2)),
                rng.uniform(-5.1, 5.1, 1_500_000),
            ]
        )
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < RADIUS + 0.1]
        sdf = self.base(pts)
        near = np.abs(sdf) < 0.015
        pts, sdf = pts[near], sdf[near]
        normal = self._normal(pts)
        surface = pts - sdf[:, None] * normal
        accept = rng.uniform(size=surface.shape[0]) < 0.45 * (
            0.25 + 0.75 * np.clip(-normal[:, 2], 0, 1)
        )
        accept &= surface[:, 2] > -HALF_HEIGHT + 0.03  # the base was cut from the build plate
        surface, normal = surface[accept], normal[accept]
        radius = np.clip(0.016 * np.exp(0.3 * rng.normal(size=surface.shape[0])), 0.008, 0.025)
        sink = rng.uniform(0.2, 0.8, surface.shape[0])
        self.particle_centre = surface + normal * (sink * radius)[:, None]
        self.particle_radius = radius
        self.particle_tree = cKDTree(self.particle_centre)

    def _powder(
        self, p: np.ndarray, cell: float, low: float, high: float, chance: float, salt: int
    ) -> np.ndarray:
        """Signed distance to loose powder spheres on two jittered lattices."""
        best = np.full(p.shape[0], np.inf)
        for shift in (0.0, 0.5):
            q = p / cell + shift
            key = np.floor(q)
            h = [hash_uniform(*key.T, salt=salt + 13 * int(2 * shift) + k) for k in range(5)]
            radius = (low + (high - low) * h[3]) / cell
            centre = key + radius[:, None] + (1 - 2 * radius[:, None]) * np.stack(h[:3], -1)
            d = (np.linalg.norm(q - centre, axis=1) - radius) * cell
            best = np.minimum(best, np.where(h[4] < chance, d, np.inf))
        return best

    def __call__(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return labels and margins (mm) at points."""
        labels = np.empty(x.shape, np.int16)
        margin = np.empty(x.shape, np.float32)
        for s in range(0, x.size, CHUNK):
            part = slice(s, s + CHUNK)
            labels[part], margin[part] = self._chunk(np.stack([x[part], y[part], z[part]], -1))
        return labels, margin

    def _chunk(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y, z = p.T
        solid = self.base(p)
        margins = [np.abs(solid)]

        # Broken strut: a rough gap across one strut of the y = 0 plane.
        rel = p - self.strut_mid
        along = rel @ self.strut_dir
        off_axis = np.linalg.norm(rel - along[:, None] * self.strut_dir, axis=1)
        gap = np.maximum(
            np.abs(along - 0.04 * self.fracture(x, y, z)) - 0.12, off_axis - 2.2 * STRUT
        )
        solid = np.maximum(solid, -gap)
        margins.append(np.abs(gap))

        # Gas, keyhole and contour pores.
        d = (p[:, None] - self.pore_centre[None]) / self.pore_stretch[None]
        pore = (np.linalg.norm(d, axis=-1) - self.pore_radius[None]).min(axis=1)
        solid = np.maximum(solid, -pore)
        margins.append(np.abs(pore))

        # Lack-of-fusion flaw: a flat lens between two layers, with unmelted powder.
        f = self.flaw
        rho = np.hypot((x - f[0]) / 0.75, (y - f[1]) / 0.42) * (
            1 + 0.25 * self.flaw_noise(x, y, 0 * z)
        )
        height = (
            0.05 * np.sqrt(np.clip(1 - rho**2, 0, 1)) * (0.75 + 0.25 * self.flaw_noise(y, x, 0 * z))
        )
        lens = np.abs(z - f[2] - 0.012 * self.flaw_noise(x, z, y)) - height
        lens = np.where(rho < 1, lens, np.maximum(lens, 0.02 * (rho - 1)))
        trapped = np.full(p.shape[0], np.inf)
        close = np.flatnonzero(lens < 0.02)
        trapped[close] = self._powder(p[close], 0.04, 0.010, 0.018, 0.7, salt=31)
        flaw = np.maximum(lens, -trapped)
        solid = np.maximum(solid, -flaw)
        margins += [np.abs(lens), np.where(lens < 0.02, np.abs(trapped), np.inf)]

        # Partly fused particles on every surface.
        dist, index = self.particle_tree.query(p, 2, distance_upper_bound=0.06)
        valid = np.isfinite(dist)
        radius = self.particle_radius[np.minimum(index, self.particle_radius.size - 1)]
        particle = np.where(valid, dist - radius, np.inf).min(axis=1)
        solid = np.minimum(solid, particle)
        margins.append(np.abs(particle))

        # Loose powder trapped against the base plate in one corner.
        heap = (
            PLATE_TOP
            + 0.6 * np.clip((-1.0 - x) / 1.6, 0, 1) * np.clip((2.0 - np.abs(y + 0.6)) / 1.0, 0, 1)
            - z
        )
        in_heap = (heap > 0) & (z > PLATE_TOP - 0.03) & (np.hypot(x, y) < RADIUS - WALL)
        powder = np.full(p.shape[0], np.inf)
        heaped = np.flatnonzero(in_heap)
        powder[heaped] = self._powder(p[heaped], 0.045, 0.010, 0.021, 0.8, salt=57)
        solid = np.minimum(solid, powder)
        margins += [np.abs(powder), np.abs(heap)]

        labels = np.where(solid < 0, TITANIUM, AIR).astype(np.int16)
        tungsten = (
            np.linalg.norm(p - self.tungsten, axis=1) - 0.05 - 0.01 * self.ripple(z, x, y)
        ) * 1.0
        labels[tungsten < 0] = TUNGSTEN
        margins.append(np.abs(tungsten))
        return labels, np.min(margins, axis=0)


@lru_cache(maxsize=4)
def _sampler(seed: int) -> Part:
    """One Part per seed, so building several grid sizes draws it once."""
    return Part(seed)


def lattice_part(n: int = 256, *, seed: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """AM titanium lattice part: attenuation at 100 keV in mm^-1 on an ``n``^3 grid.

    The object is fixed in mm (12 mm field of view), so ``n`` only sets the
    voxel size; a given ``seed`` always gives the same part.
    """
    grid = Grid(n, FOV_MM)
    part = _sampler(seed)
    volume = voxelise(part, grid, np.array(list(MATERIALS.values())), fine=3)
    meta = {
        "name": "lattice_part",
        "voxel_mm": grid.voxel,
        "fov_mm": FOV_MM,
        "energy_keV": ENERGY_KEV,
        "materials": dict(MATERIALS),
        "seed": seed,
        "geometry": "cone beam; build direction and rotation axis along z",
        "surface_particles": int(part.particle_radius.size),
    }
    return volume, meta


if __name__ == "__main__":
    main(lattice_part, zoom=("xz", 0.0, (-3.3, 0.9, -5.2, -3.1)), vmax=0.16)
