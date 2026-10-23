#!/usr/bin/env python3
"""Sandstone mini-plug for parallel-beam synchrotron micro-CT at 30 keV.

A 1.0 mm diameter core drilled from a fine-grained arkosic sandstone, imaged
over a 1.1 mm field (4.3 µm voxels at n = 256, the scale of a TOMCAT or I13
micro-CT scan). The core axis is the rotation axis, z.

Grains are packed by random sequential addition of lognormal spheres (median
diameter 110 µm, fine sand) whose centres may approach to 0.8 of the sum of
their radii, then each is given a sub-angular shape: a randomly oriented
ellipsoid (aspect ratios about 1 : 0.85 : 0.72) cut by seven softly rounded
facets, with a few per cent of low-order surface relief. Where two grains
overlap, the point belongs to the grain whose shape function is smaller, so
contacts are flat or gently curved interfaces like pressure-solution contacts,
and the open space left between the grains is a connected intergranular pore
network of about 20% porosity.

About 84% of grains are quartz and 15% K-feldspar, a third of which are
partially dissolved along two cleavage directions (secondary porosity); the
smallest few are zircon, the usual heavy mineral. Microporous clay (kaolinite)
lines the grain contacts as a 2.5 µm seam, bridges them as pendular cement and
fills a few pores; one patch is cemented by poikilotopic calcite. Pyrite
occurs as framboids (clusters of about 4 µm microcrystals, 30 to 50 µm across)
and as euhedral cubes 25 to 50 µm on a side in the pores. Five grains carry
intragranular cracks and one wavy coring-induced microfracture crosses several
grains; cracks are 2 to 3.5 µm wide with an elliptical opening profile,
integrated analytically so they keep their partial-volume contrast below the
voxel size.

Attenuation is the total linear attenuation coefficient with coherent
scattering at 30 keV (xraylib 4.3 ``CS_Total_CP``, the counterpart of the NIST
XCOM total with coherent scattering) for the mineral densities given; clay is
kaolinite at 55% solid fraction (45% air-filled microporosity). Pores are dry
(air, 0). Volumes are attenuation in mm^-1, ``(z, y, x)`` order.

    python bench/phantoms/porous_rock.py --n 256 --preview /tmp/porous_rock.png
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from _procedural import (
    Grid,
    WaveNoise,
    hash_uniform,
    main,
    random_rotations,
    slab_fraction,
    voxelise,
)
import numpy as np
from scipy.spatial import cKDTree

FOV_MM = 1.1
CORE_RADIUS_MM = 0.5
ENERGY_KEV = 30.0

# Linear attenuation at 30 keV in mm^-1 (xraylib CS_Total_CP x density):
# quartz SiO2 2.65 g/cm^3, orthoclase KAlSi3O8 2.56, zircon ZrSiO4 4.65,
# pyrite FeS2 5.01, kaolinite Al2Si2O5(OH)4 2.60 at 55%, calcite CaCO3 2.71.
MATERIALS = {
    "pore (air)": 0.0,
    "quartz": 0.2313,
    "K-feldspar": 0.3065,
    "zircon": 5.915,
    "pyrite": 2.473,
    "clay (kaolinite, 55%)": 0.1094,
    "calcite": 0.5002,
}
PORE, QUARTZ, FELDSPAR, ZIRCON, PYRITE, CLAY, CALCITE = range(7)

FACETS = 7
WAVES = 4
NEIGHBOURS = 8
CHUNK = 1 << 16
FILL = 0.8  # summed sphere volume over the packing volume
SEAM = 0.0025  # mm, width of the clay seam along grain contacts
OVERLAP = 0.8  # centres may approach to this fraction of the radius sum


@dataclass(frozen=True)
class Grains:
    """Packed grains: centres, shapes and minerals."""

    centre: np.ndarray  # (g, 3) mm
    shape: np.ndarray  # (g, 3, 3): maps p - centre to the unit-ellipsoid frame
    quad: np.ndarray  # (g, 6) upper triangle of shape.T @ shape
    facet: np.ndarray  # (g, F, 3) facet normals over their offsets
    wave_dir: np.ndarray  # (g, W, 3)
    wave_amp: np.ndarray  # (g, W)
    wave_freq: np.ndarray  # (g, W)
    wave_phase: np.ndarray  # (g, W)
    mineral: np.ndarray  # (g,)
    dissolved: np.ndarray  # (g,) bool: partially dissolved feldspar
    cleavage: np.ndarray  # (g, 2, 3) cleavage-plane normals
    hole_centre: np.ndarray  # (g, 3) centre of the dissolved region
    hole_radius: np.ndarray  # (g,)


def _pack(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random sequential addition of overlapping lognormal spheres in a cylinder."""
    pad = 0.13
    radius_limit = CORE_RADIUS_MM + pad
    half_height = FOV_MM / 2 + pad
    volume = np.pi * radius_limit**2 * 2 * half_height
    draws = np.clip(0.055 * np.exp(0.33 * rng.normal(size=20000)), 0.02, 0.11)
    count = np.searchsorted(np.cumsum(4 / 3 * np.pi * draws**3), FILL * volume)
    radii = np.sort(draws[:count])[::-1]
    centres = np.zeros((radii.size, 3))
    kept = np.zeros(radii.size)
    placed = 0
    for r in radii:
        for _ in range(200):
            rr = radius_limit * np.sqrt(rng.uniform(size=128))
            ph = rng.uniform(0, 2 * np.pi, 128)
            cand = np.stack(
                [rr * np.cos(ph), rr * np.sin(ph), rng.uniform(-half_height, half_height, 128)], -1
            )
            gap = np.linalg.norm(cand[:, None] - centres[None, :placed], axis=-1)
            gap -= OVERLAP * (r + kept[None, :placed])
            ok = np.flatnonzero(gap.min(axis=1, initial=np.inf) >= 0)
            if ok.size:
                centres[placed] = cand[ok[0]]
                kept[placed] = r
                placed += 1
                break
    return centres[:placed], kept[:placed]


def _grains(rng: np.random.Generator) -> Grains:
    centre, radius = _pack(rng)
    g = centre.shape[0]
    rot = random_rotations(rng, g)
    aspect = np.stack([np.ones(g), rng.uniform(0.75, 0.95, g), rng.uniform(0.6, 0.85, g)], -1)
    aspect /= np.cbrt(aspect.prod(axis=1, keepdims=True))
    semi = 1.07 * radius[:, None] * aspect
    shape = np.einsum("gi,gji->gij", 1.0 / semi, rot)  # diag(1/semi) @ rot.T
    a = np.einsum("gki,gkj->gij", shape, shape)
    quad = np.stack(
        [a[:, 0, 0], a[:, 1, 1], a[:, 2, 2], 2 * a[:, 0, 1], 2 * a[:, 0, 2], 2 * a[:, 1, 2]], -1
    )
    normal = rng.normal(size=(g, FACETS, 3))
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    facet = normal / rng.uniform(0.74, 0.95, (g, FACETS, 1))
    wave_dir = rng.normal(size=(g, WAVES, 3))
    wave_dir /= np.linalg.norm(wave_dir, axis=-1, keepdims=True)
    mineral = np.where(rng.uniform(size=g) < 0.15, FELDSPAR, QUARTZ)
    mineral[np.argsort(radius)[: max(2, g // 300)]] = ZIRCON
    dissolved = (mineral == FELDSPAR) & (rng.uniform(size=g) < 0.3)
    first = rng.normal(size=(g, 3))
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second = np.cross(first, rng.normal(size=(g, 3)))
    second /= np.linalg.norm(second, axis=1, keepdims=True)
    hole_dir = rng.normal(size=(g, 3))
    hole_dir /= np.linalg.norm(hole_dir, axis=1, keepdims=True)
    return Grains(
        centre=centre,
        shape=shape,
        quad=quad,
        facet=facet,
        wave_dir=wave_dir,
        wave_amp=rng.uniform(0.0, 0.035, (g, WAVES)),
        wave_freq=rng.uniform(2.0, 5.0, (g, WAVES)),
        wave_phase=rng.uniform(0, 2 * np.pi, (g, WAVES)),
        mineral=mineral,
        dissolved=dissolved,
        cleavage=np.stack([first, second], 1),
        hole_centre=centre + 0.5 * radius[:, None] * hole_dir,
        hole_radius=radius * rng.uniform(0.55, 0.85, g),
    )


@dataclass(frozen=True)
class Sphere:
    """Spherical regions with a noisy radius."""

    centre: np.ndarray  # (m, 3)
    radius: np.ndarray  # (m,)

    def distance(self, p: np.ndarray, noise: WaveNoise, relief: float) -> np.ndarray:
        """Signed distance, approximately, to the nearest region; negative inside."""
        if self.radius.size == 0:
            return np.full(p.shape[0], np.inf)
        d = np.linalg.norm(p[:, None] - self.centre[None], axis=-1) - self.radius[None]
        return d.min(axis=1) - relief * noise(p[:, 0], p[:, 1], p[:, 2])


@dataclass(frozen=True)
class Cubes:
    """Rotated cubes (euhedral pyrite)."""

    centre: np.ndarray
    rot: np.ndarray
    half: np.ndarray

    def distance(self, p: np.ndarray) -> np.ndarray:
        """Signed distance to the nearest cube; negative inside."""
        best = np.full(p.shape[0], np.inf)
        for c, r, h in zip(self.centre, self.rot, self.half, strict=True):
            q = np.abs((p - c) @ r) - h
            outside = np.linalg.norm(np.maximum(q, 0), axis=1)
            best = np.minimum(best, outside + np.minimum(q.max(axis=1), 0))
        return best


class RockSampler:
    """Material label and boundary margin at points of the sandstone core."""

    def __init__(self, seed: int) -> None:
        """Draw every random feature of the core from ``seed``."""
        rng = np.random.default_rng(seed)
        self.grains = _grains(rng)
        self.tree = cKDTree(self.grains.centre)
        self.seed = seed
        g = self.grains
        inside = np.flatnonzero(np.hypot(g.centre[:, 0], g.centre[:, 1]) < CORE_RADIUS_MM - 0.08)
        self.noise = WaveNoise.make(rng, 0.06)
        self.calcite = Sphere(np.array([[-0.2, 0.17, 0.03]]), np.array([0.11]))
        self.clay = Sphere(
            np.concatenate(
                [g.centre[rng.choice(inside, 5, replace=False)] + rng.normal(0, 0.06, (5, 3))]
            ),
            rng.uniform(0.04, 0.07, 5),
        )
        # Pyrite framboids sit on grain surfaces; one is kept near z = 0 for the preview.
        hosts = rng.choice(inside, 5, replace=False)
        hosts[0] = inside[
            np.argmin(np.abs(g.centre[inside, 2]) + 0.3 * np.abs(g.centre[inside, 0] - 0.25))
        ]
        direction = rng.normal(size=(5, 3))
        direction[0] = [0.0, -1.0, 0.05]
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        self.framboid_radius = rng.uniform(0.014, 0.024, 5)
        self.framboids = Sphere(
            self._surface_point(hosts, direction) + direction * 0.7 * self.framboid_radius[:, None],
            self.framboid_radius,
        )
        hosts = rng.choice(inside, 3, replace=False)
        direction = rng.normal(size=(3, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        half = rng.uniform(0.012, 0.024, 3)
        self.cubes = Cubes(
            self._surface_point(hosts, direction) + direction * half[:, None],
            random_rotations(rng, 3),
            half,
        )

    def _surface_point(self, grains: np.ndarray, direction: np.ndarray) -> np.ndarray:
        g = self.grains
        q = np.einsum("gij,gj->gi", g.shape[grains], direction)
        return g.centre[grains] + direction / np.linalg.norm(q, axis=1, keepdims=True)

    def _shape(self, grain: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Shape function of ``grain`` at frame coordinates ``q``: below 1 inside."""
        g = self.grains
        size = np.linalg.norm(q, axis=1)
        u = q / np.maximum(size, 1e-12)[:, None]
        facets = np.einsum("pfi,pi->pf", g.facet[grain], u)
        rounded = (1.0 + (np.maximum(facets, 0.0) ** 12).sum(axis=1)) ** (1 / 12)
        phase = (
            np.einsum("pwi,pi->pw", g.wave_dir[grain], u) * g.wave_freq[grain] + g.wave_phase[grain]
        )
        relief = 1.0 + (g.wave_amp[grain] * np.sin(phase)).sum(axis=1)
        return size * rounded * relief

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
        g = self.grains
        m = p.shape[0]
        rows = np.arange(m)
        _, near = self.tree.query(p, NEIGHBOURS)
        d = p[:, None, :] - g.centre[near]
        a = g.quad[near]
        ellipse = (
            a[..., 0] * d[..., 0] ** 2
            + a[..., 1] * d[..., 1] ** 2
            + a[..., 2] * d[..., 2] ** 2
            + a[..., 3] * d[..., 0] * d[..., 1]
            + a[..., 4] * d[..., 0] * d[..., 2]
            + a[..., 5] * d[..., 1] * d[..., 2]
        )
        best = np.argpartition(ellipse, 1, axis=1)[:, :2]
        pair = near[rows[:, None], best]
        rho = np.empty((m, 2))
        span = np.empty((m, 2))
        for j in range(2):
            grain = pair[:, j]
            q = np.einsum("pij,pj->pi", g.shape[grain], p - g.centre[grain])
            rho[:, j] = self._shape(grain, q)
            span[:, j] = np.linalg.norm(p - g.centre[grain], axis=1) / np.maximum(rho[:, j], 1e-9)
        swap = rho[:, 1] < rho[:, 0]
        rho[swap] = rho[swap, ::-1]
        span[swap] = span[swap, ::-1]
        pair[swap] = pair[swap, ::-1]
        first, rho1, rho2 = pair[:, 0], rho[:, 0], rho[:, 1]

        labels = np.full(m, PORE, np.int16)
        grain = rho1 < 1.0
        labels[grain] = g.mineral[first[grain]]
        margins = [0.75 * np.abs(rho1 - 1.0) * span[:, 0]]
        # A thin clay seam marks grain-grain contacts inside the overlap lens.
        across = (rho2 - rho1) / (1 / span[:, 0] + 1 / span[:, 1])
        seam = (rho2 < 1.0) & (across < SEAM / 2)
        labels[seam] = CLAY
        margins.append(np.where(rho2 < 1.05, 0.7 * np.abs(across - SEAM / 2), np.inf))

        # Pendular clay at contacts: close to the surfaces of both nearest grains.
        neck = 0.28
        cement = ~grain & (rho1 < 1.0 + neck) & (rho2 < 1.0 + neck)
        labels[cement] = CLAY
        margins.append(
            np.where(rho2 < 1.05 + neck, 0.75 * np.abs(rho1 - 1.0 - neck) * span[:, 0], np.inf)
        )
        margins.append(
            np.where(rho1 < 1.05 + neck, 0.75 * np.abs(rho2 - 1.0 - neck) * span[:, 1], np.inf)
        )

        pore = labels == PORE
        calcite = self.calcite.distance(p, self.noise, 0.012)
        clay = self.clay.distance(p, self.noise, 0.015)
        labels[pore & (clay < 0)] = CLAY
        labels[pore & (calcite < 0)] = CALCITE
        margins += [np.abs(calcite), np.abs(clay)]

        self._dissolve(p, first, labels, margins)
        self._pyrite(p, labels, margins)

        radial = CORE_RADIUS_MM - np.hypot(p[:, 0], p[:, 1])
        labels[radial < 0] = PORE
        margin = np.minimum(np.min(margins, axis=0), np.abs(radial))
        return labels, np.where(radial < 0, -radial, margin)

    def _dissolve(
        self, p: np.ndarray, first: np.ndarray, labels: np.ndarray, margins: list[np.ndarray]
    ) -> None:
        """Open slots along two cleavages inside the dissolved part of feldspars."""
        g = self.grains
        where = np.flatnonzero((labels == FELDSPAR) & g.dissolved[first])
        if where.size == 0:
            return
        grain = first[where]
        q = p[where]
        hole = np.linalg.norm(q - g.hole_centre[grain], axis=1) - g.hole_radius[grain]
        hole -= 0.012 * self.noise(q[:, 0], q[:, 1], q[:, 2])
        slot = np.zeros(where.size, bool)
        edge = np.full(where.size, np.inf)
        third = np.cross(g.cleavage[grain, 0], g.cleavage[grain, 1])
        # The dominant cleavage opens more often than the second one.
        for k, (period, chance) in enumerate([(0.010, 0.5), (0.014, 0.25)]):
            r = q - g.centre[grain]
            s = np.einsum("pi,pi->p", r, g.cleavage[grain, k])
            along = np.einsum("pi,pi->p", r, g.cleavage[grain, 1 - k])
            depth = np.einsum("pi,pi->p", r, third)
            index = np.floor(s / period)
            local = s - index * period
            cells = (index, np.floor(along / 0.025), np.floor(depth / 0.025), grain)
            alive = hash_uniform(*cells, salt=self.seed + k) < chance
            width = period * (0.25 + 0.25 * hash_uniform(index, grain, salt=self.seed + 5 + k))
            slot |= alive & (local < width)
            edge = np.minimum(
                edge, np.minimum(np.abs(local - width), np.minimum(local, period - local))
            )
        open_ = slot & (hole < 0)
        labels[where[open_]] = PORE
        local_margin = np.full(p.shape[0], np.inf)
        local_margin[where] = np.minimum(np.where(hole < 0, edge, np.inf), np.abs(hole))
        margins.append(local_margin)

    def _pyrite(self, p: np.ndarray, labels: np.ndarray, margins: list[np.ndarray]) -> None:
        cube = self.cubes.distance(p)
        labels[cube < 0] = PYRITE
        margins.append(np.abs(cube))
        f = self.framboids
        d = np.linalg.norm(p[:, None] - f.centre[None], axis=-1) - f.radius[None]
        inside = np.flatnonzero(d.min(axis=1) < 0)
        if inside.size:
            # Microcrystals: one sphere per 4 µm cell, jittered by a hash.
            cell = 0.004
            q = p[inside]
            index = np.floor(q / cell)
            jitter = np.stack([hash_uniform(*index.T, salt=7 + k) for k in range(3)], -1)
            crystal = np.linalg.norm(q - (index + 0.3 + 0.4 * jitter) * cell, axis=1) < 0.0019
            labels[inside[crystal]] = PYRITE
        margins.append(np.maximum(np.abs(d.min(axis=1)), 0.0) * (d.min(axis=1) > 0))


@dataclass(frozen=True)
class Crack:
    """Penny-shaped crack with an elliptical opening profile."""

    centre: np.ndarray
    normal: np.ndarray
    radius: float
    aperture: float
    waviness: float

    def open(self, volume: np.ndarray, grid: Grid) -> None:
        """Remove the crack's share of each voxel it crosses, in place."""
        reach = self.radius + self.waviness + grid.voxel
        ranges = [grid.index_range(c - reach, c + reach) for c in self.centre]
        c = grid.centres()
        axes = [c[i0:i1] for i0, i1 in ranges]
        z, y, x = np.meshgrid(*axes[::-1], indexing="ij")
        p = np.stack([x, y, z], -1) - self.centre
        across = p @ self.normal
        in_plane = p - across[..., None] * self.normal
        tangent = np.cross(self.normal, [0.0, 0.0, 1.0])
        tangent /= np.linalg.norm(tangent)
        wave = (
            self.waviness
            * np.sin(2 * np.pi * (in_plane @ tangent) / 0.09)
            * np.cos(2 * np.pi * in_plane[..., 2] / 0.13)
        )
        r = np.linalg.norm(in_plane, axis=-1)
        half = 0.5 * self.aperture * np.sqrt(np.clip(1 - (r / self.radius) ** 2, 0, 1))
        fraction = slab_fraction(across - wave, self.normal, -half, half, grid.voxel)
        region = tuple(slice(i0, i1) for i0, i1 in ranges[::-1])
        volume[region] *= (1.0 - fraction).astype(np.float32)


def _cracks(sampler: RockSampler, seed: int) -> list[Crack]:
    rng = np.random.default_rng([seed, 1])
    g = sampler.grains
    quartz = np.flatnonzero(
        (g.mineral == QUARTZ) & (np.hypot(g.centre[:, 0], g.centre[:, 1]) < CORE_RADIUS_MM - 0.1)
    )
    picks = rng.choice(quartz, 5, replace=False)
    picks[0] = quartz[np.argmin(np.abs(g.centre[quartz, 2]))]
    cracks = []
    for grain in picks:
        normal = rng.normal(size=3)
        normal[2] *= 0.3
        normal /= np.linalg.norm(normal)
        size = 1 / np.linalg.norm(g.shape[grain], ord=2)
        cracks.append(Crack(g.centre[grain], normal, 1.3 * size, rng.uniform(0.002, 0.003), 0.0))
    normal = np.array([np.cos(0.5), np.sin(0.5), 0.12])
    cracks.append(
        Crack(np.array([0.18, -0.12, 0.0]), normal / np.linalg.norm(normal), 0.36, 0.0035, 0.004)
    )
    return cracks


@lru_cache(maxsize=4)
def _sampler(seed: int) -> RockSampler:
    """One RockSampler per seed, so building several grid sizes draws it once."""
    return RockSampler(seed)


def porous_rock(n: int = 256, *, seed: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """Sandstone mini-plug: attenuation at 30 keV in mm^-1 on an ``n``^3 grid.

    The object is fixed in mm (1.1 mm field of view), so ``n`` only sets the
    voxel size; a given ``seed`` always gives the same core.
    """
    grid = Grid(n, FOV_MM)
    sampler = _sampler(seed)
    mu = np.array(list(MATERIALS.values()))
    volume = voxelise(sampler, grid, mu, fine=3)
    for crack in _cracks(sampler, seed):
        crack.open(volume, grid)
    meta = {
        "name": "porous_rock",
        "voxel_mm": grid.voxel,
        "fov_mm": FOV_MM,
        "energy_keV": ENERGY_KEV,
        "materials": dict(MATERIALS),
        "seed": seed,
        "grains": int(sampler.grains.centre.shape[0]),
        "geometry": "parallel beam; core axis along z",
    }
    return volume, meta


if __name__ == "__main__":
    main(porous_rock, zoom=("xy", 0.0, (0.0, 0.4, -0.35, 0.05)), vmax=0.55)
