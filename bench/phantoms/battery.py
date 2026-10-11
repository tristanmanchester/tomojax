#!/usr/bin/env python3
"""Cylindrical lithium-ion cell (a scaled 18650) at 80 keV.

The cell is 8.5 mm in diameter instead of 18 mm, keeping real layer
thicknesses, so the jelly roll has nine turns rather than about twenty. The
9 mm field of view (35 µm voxels at n = 256, a lab cone-beam scan) shows the
upper 5.8 mm of the winding and the header; the cell axis is z.

The jelly roll is an Archimedean spiral, ``r = r0 + p theta / 2 pi`` with a
pitch of 325 µm, of one bundle of four strips: a graphite anode (70 µm
coatings on both sides of 10 µm copper foil), a 20 µm separator, an NMC811
cathode (60 µm coatings on 15 µm aluminium foil) and a second separator. The
separators wrap most of a turn round the hollow core before the anode starts,
the cathode starts a quarter turn later and ends before the anode (anode
overhang), the anode is taller than the cathode and the separators taller than
both, and polyimide tape closes the outer turn. Cathode coatings hold NMC811
secondary particles (6 to 12.6 µm across, 39% by volume) in a carbon-binder
and electrolyte matrix; the anode is homogenised graphite, binder and
electrolyte. An uncoated window midway along the cathode carries the aluminium
tab, which rises through a slot in the polyimide insulator and folds over to
weld to the underside of the vent disc; a nickel tab sits on the outer end of
the anode.

The core holds a split stainless-steel centre pin. The nickel-plated steel can
has a rolled bead above the winding; the header is a polypropylene gasket
clamping an aluminium vent disc with a central dimple and a steel top cap with
a raised terminal and three vent holes, under the crimped lip of the can. Two
defects are present near z = 0: a stainless-steel particle of about 120 µm
lodged across the separator of the seventh turn, and a delamination on the
sixth turn, where the outer cathode coating has lifted up to 80 µm off its
aluminium foil over about 1.9 mm of arc and 1.8 mm of height, pressing into
the separator and the next anode.

Layers are integrated exactly across their thickness (the voxel's projected
footprint on the radial direction) and over their height, so 10 µm foils keep
their partial-volume contrast in 35 µm voxels. Particle volumes are resampled
onto the voxels by box overlaps, and the can, header and defects are
supersampled. Attenuation is the total linear attenuation coefficient with
coherent scattering at 80 keV, monochromatic, as on a high-energy synchrotron
beamline or as the effective energy of a filtered 160 to 180 kVp lab source
(xraylib 4.3 ``CS_Total_CP``, the counterpart of the NIST XCOM total with
coherent scattering). Volumes are attenuation in mm^-1, ``(x, y, z)`` order.

    python bench/phantoms/battery.py --n 256 --preview /tmp/battery.png
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from _procedural import (
    KEEP,
    Grid,
    WaveNoise,
    box_cdf,
    main,
    voxelise,
    workers,
)
import numpy as np

FOV_MM = 9.0
ENERGY_KEV = 80.0

# Linear attenuation at 80 keV in mm^-1 (xraylib CS_Total_CP x density).
# Electrolyte: 1 M LiPF6 in EC/DMC, 1.25 g/cm^3 (0.0217 /mm). Graphite coating:
# 65% graphite (2.26), 5% PVDF (1.78), 30% electrolyte. Separator: 45%
# polyethylene (0.92), 55% electrolyte. Cathode matrix between particles: 25%
# carbon black and PVDF, 75% electrolyte. NMC811 LiNi0.8Mn0.1Co0.1O2 at 4.75.
# Steel: iron 7.87; stainless: 304 (Fe70 Cr19 Ni9 Mn2) at 8.0.
MATERIALS = {
    "gas": 0.0,
    "anode coating (graphite)": 0.0317,
    "copper foil": 0.6837,
    "separator (PE + electrolyte)": 0.0195,
    "cathode matrix (carbon-binder + electrolyte)": 0.0238,
    "NMC811 particles": 0.2310,
    "aluminium (foil, tab, vent)": 0.0545,
    "nickel tab": 0.6502,
    "polyimide (tape, insulator)": 0.0237,
    "steel (can, cap)": 0.4685,
    "stainless steel (pin, particle)": 0.4690,
    "polypropylene (gasket)": 0.0164,
    "cathode coating, homogenised (39% NMC)": 0.1046,
}
(
    GAS,
    ANODE,
    COPPER,
    SEPARATOR,
    CATHODE,
    NMC,
    ALUMINIUM,
    NICKEL,
    POLYIMIDE,
    STEEL,
    STAINLESS,
    POLYPROPYLENE,
    LIFTED,
) = range(13)

R0 = 0.62  # base radius of the spiral at theta = 0, mm
PITCH = 0.325  # one turn of the four-strip bundle, mm
TAB_TURN = 4.0  # cathode tab at phi = 0 on the fifth turn
TAB_HALF = 0.95 / (2 * np.pi * (R0 + PITCH * TAB_TURN))  # 1.9 mm wide tab, in turns
WINDOW_HALF = 1.3 * TAB_HALF  # uncoated foil around it
ANODE_END = 8.5  # at phi = pi
NI_TAB = 0.85 / (2 * np.pi * (R0 + PITCH * ANODE_END))  # 0.85 mm wide, in turns
Z_CATHODE, Z_ANODE, Z_SEPARATOR, Z_TAB = 1.0, 1.15, 1.3, 1.98
BOTTOM = -50.0


@dataclass(frozen=True)
class Layer:
    """One strip of the winding: radial offsets in a turn, turns and height."""

    material: int
    low: float  # mm from the spiral base
    high: float
    turns: tuple[tuple[float, float], ...]  # (start, end) intervals in turns
    top: float  # mm


def _coating(
    start: float, end: float, window: tuple[float, float] | None
) -> tuple[tuple[float, float], ...]:
    if window is None:
        return ((start, end),)
    return ((start, window[0]), (window[1], end))


TAB_WINDOW = (TAB_TURN - WINDOW_HALF, TAB_TURN + WINDOW_HALF)
NI_WINDOW = (ANODE_END - NI_TAB, ANODE_END + 1.0)
LAYERS = (
    Layer(ANODE, 0.000, 0.070, _coating(0.8, ANODE_END, NI_WINDOW), Z_ANODE),
    Layer(COPPER, 0.070, 0.080, ((0.8, ANODE_END),), Z_ANODE),
    Layer(ANODE, 0.080, 0.150, _coating(0.8, ANODE_END, NI_WINDOW), Z_ANODE),
    Layer(SEPARATOR, 0.150, 0.170, ((0.0, 9.2),), Z_SEPARATOR),
    Layer(CATHODE, 0.170, 0.230, _coating(1.05, 8.35, TAB_WINDOW), Z_CATHODE),
    Layer(ALUMINIUM, 0.230, 0.245, ((1.05, 8.35),), Z_CATHODE),
    Layer(CATHODE, 0.245, 0.305, _coating(1.05, 8.35, TAB_WINDOW), Z_CATHODE),
    Layer(SEPARATOR, 0.305, 0.325, ((0.0, 9.2),), Z_SEPARATOR),
    Layer(ALUMINIUM, 0.245, 0.305, ((TAB_TURN - TAB_HALF, TAB_TURN + TAB_HALF),), Z_TAB),
    Layer(NICKEL, 0.080, 0.150, ((ANODE_END - NI_TAB, ANODE_END),), Z_ANODE),
    Layer(POLYIMIDE, 0.325, 0.355, ((8.55, 9.2),), Z_SEPARATOR),
)
CATHODE_LAYERS = (4, 6)


def _polar(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.hypot(x, y), np.mod(np.arctan2(y, x), 2 * np.pi)


def winding_fractions(grid: Grid) -> list[np.ndarray]:
    """Return each layer's area fraction in every xy pixel of ``grid``.

    Across a layer the pixel is integrated exactly (the box-filter of a square
    pixel projected onto the radial direction); along it, layer ends ramp over
    the pixel footprint.
    """
    c = grid.centres()
    x, y = np.meshgrid(c, c, indexing="ij")
    r, phi = _polar(x, y)
    cos, sin = np.abs(np.cos(phi)), np.abs(np.sin(phi))
    widths = grid.voxel * np.stack([cos, sin, np.zeros_like(cos)], -1)
    footprint = grid.voxel * (cos + sin)
    reach = int(np.ceil((0.5 * footprint.max() + 0.36) / PITCH)) + 1
    base = np.floor((r - R0) / PITCH - phi / (2 * np.pi))
    out = [np.zeros_like(r) for _ in LAYERS]
    for shift in range(-reach, reach + 1):
        turn = phi / (2 * np.pi) + base + shift
        s = r - R0 - PITCH * turn
        ramp = footprint / (2 * np.pi * np.maximum(r, 1e-6))
        for k, layer in enumerate(LAYERS):
            across = box_cdf(layer.high - s, widths) - box_cdf(layer.low - s, widths)
            if not np.any(across > 0):
                continue
            along = np.zeros_like(r)
            for start, end in layer.turns:
                along += np.clip((turn - start) / ramp + 0.5, 0, 1) - np.clip(
                    (turn - end) / ramp + 0.5, 0, 1
                )
            out[k] += np.clip(along, 0, 1) * np.maximum(across, 0)
    return out


def layer_at(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the layer index (-1 for none), turn and offset at points."""
    r, phi = _polar(x, y)
    base = np.floor((r - R0) / PITCH - phi / (2 * np.pi))
    index = np.full(x.shape, -1, np.int16)
    turn_at = np.zeros(x.shape)
    offset = np.zeros(x.shape)
    for shift in (0, -1):
        turn = phi / (2 * np.pi) + base + shift
        s = r - R0 - PITCH * turn
        for k, layer in enumerate(LAYERS):
            hit = (index < 0) & (s >= layer.low) & (s < layer.high)
            hit &= np.any([(turn >= a) & (turn < b) for a, b in layer.turns], axis=0)
            index[hit] = k
            turn_at[hit] = turn[hit]
            offset[hit] = s[hit]
    return index, turn_at, offset


def _z_fraction(grid: Grid, top: float) -> np.ndarray:
    z = grid.centres()
    return np.clip((top - (z - grid.voxel / 2)) / grid.voxel, 0, 1) - np.clip(
        (BOTTOM - (z - grid.voxel / 2)) / grid.voxel, 0, 1
    )


PARTICLE_CELL = 0.013  # mm


def _overlap_matrix(edges: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Overlap length of each voxel ``[edges[i], edges[i+1]]`` with each cell, in cells."""
    low = np.maximum(edges[:-1, None], cells[None, :-1])
    high = np.minimum(edges[1:, None], cells[None, 1:])
    return np.maximum(high - low, 0.0) / PARTICLE_CELL


def particle_fraction(grid: Grid, seed: int) -> np.ndarray:
    """Volume fraction of NMC particles in every voxel, as if coating filled it.

    Two lattices of 13 µm cells, offset by half a cell, each hold at most one
    sphere of 6 to 12.6 µm diameter lying wholly inside its cell, so particles
    can touch across lattices. Each cell's particle volume is spread over the
    cell and resampled onto the voxels by exact box overlaps; this is exact
    for cells inside a voxel and smooths below one cell (13 µm) elsewhere.
    Only the annulus that can hold cathode coating, below its top, is filled.
    """
    n, v = grid.n, grid.voxel
    edges = np.arange(n + 1) * v - grid.fov / 2
    out = np.zeros((n, n, n), np.float32)
    scale = (PARTICLE_CELL / v) ** 3
    top = min(Z_CATHODE + v, grid.fov / 2)
    depth = 16
    for shift, salt in ((0.0, 11), (0.5, 23)):

        def axis(low: float, high: float, shift: float = shift) -> tuple[np.ndarray, np.ndarray]:
            first = np.floor(low / PARTICLE_CELL - shift)
            last = np.ceil(high / PARTICLE_CELL - shift)
            index = np.arange(first, last)
            cells = (np.append(index, last) + shift) * PARTICLE_CELL
            return index.astype(np.int64), _overlap_matrix(edges, cells).astype(np.float32)

        ixy, wxy = axis(-CAN_OUT, CAN_OUT)
        iz, wz = axis(edges[0], top)
        cx, cy = np.meshgrid(ixy, ixy, indexing="ij")
        radius_xy = np.hypot(cx + shift + 0.5, cy + shift + 0.5) * PARTICLE_CELL
        inside = (radius_xy > 1.0) & (radius_xy < 3.75)  # where cathode coating can be
        count = int(np.count_nonzero(inside))

        def chunk(
            start: int,
            wxy: np.ndarray = wxy,
            wz: np.ndarray = wz,
            inside: np.ndarray = inside,
            count: int = count,
            salt: int = salt,
        ) -> tuple[int, np.ndarray]:
            # Cells are drawn per block of z-cells from a stream keyed by the
            # block, so every grid sees the same particles.
            zs = slice(start, min(start + depth, wz.shape[1]))
            rng = np.random.default_rng([seed, salt, start])
            draw = rng.random((2, count, depth), np.float32)[:, :, : zs.stop - zs.start]
            radius = (
                np.float32(0.0030 / PARTICLE_CELL) + np.float32(0.0033 / PARTICLE_CELL) * draw[1]
            )
            volume = np.zeros((*inside.shape, draw.shape[2]), np.float32)
            volume[inside] = np.where(draw[0] < 0.9, np.float32(4 / 3 * np.pi) * radius**3, 0)
            rows = np.flatnonzero(wz[:, zs].sum(axis=1) > 0)
            k0, k1 = rows[0], rows[-1] + 1
            part = np.tensordot(wxy, volume, axes=(1, 0))
            part = np.tensordot(wxy, part, axes=(1, 1)).transpose(1, 0, 2)
            return k0, part @ wz[k0:k1, zs].T

        with ThreadPoolExecutor(workers()) as pool:
            for k0, block in pool.map(chunk, range(0, iz.size, depth)):
                out[:, :, k0 : k0 + block.shape[2]] += scale * block
    return out


def _box_sdf(
    u: np.ndarray, v: np.ndarray, u0: float, u1: float, v0: float, v1: float
) -> np.ndarray:
    """Signed distance to an axis-aligned rectangle in a 2D section."""
    du = np.maximum(u0 - u, u - u1)
    dv = np.maximum(v0 - v, v - v1)
    outside = np.hypot(np.maximum(du, 0), np.maximum(dv, 0))
    return outside + np.minimum(np.maximum(du, dv), 0)


def _shell(z: np.ndarray, profile: np.ndarray, slope: np.ndarray, half: float) -> np.ndarray:
    """Signed distance, to first order, to a sheet of half-thickness ``half`` about ``profile``."""
    return np.abs(z - profile) / np.sqrt(1 + slope**2) - half


CAN_OUT, CAN_WALL = 4.25, 0.22
BEAD_Z, BEAD_DEPTH, BEAD_WIDTH = 1.92, 0.36, 0.27
CAN_TOP = 3.42


def hardware(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Can, header, insulator, centre pin and tab fold; KEEP elsewhere."""
    r, phi = _polar(x, y)
    labels = np.full(x.shape, KEEP, np.int16)
    parts: list[tuple[int, np.ndarray]] = []

    # Can wall with a rolled bead, and the crimped lip over the gasket.
    mid = CAN_OUT - CAN_WALL / 2
    bump = np.exp(-(((z - BEAD_Z) / BEAD_WIDTH) ** 2))
    wall_r = mid - BEAD_DEPTH * bump
    slope = BEAD_DEPTH * bump * 2 * (z - BEAD_Z) / BEAD_WIDTH**2
    wall = np.maximum(np.abs(r - wall_r) / np.sqrt(1 + slope**2) - CAN_WALL / 2, z - CAN_TOP)
    lip = _box_sdf(r, z, 3.35 + 0.08, CAN_OUT - 0.08, CAN_TOP - 0.20 + 0.08, CAN_TOP - 0.08) - 0.08
    parts.append((STEEL, np.minimum(wall, lip)))

    # Top cap: flat flange, sloped shoulder with three vent holes, raised terminal.
    profile = np.interp(r, [0.0, 1.55, 2.05, 5.0], [3.55, 3.55, 2.58, 2.58])
    cap_slope = np.where((r > 1.55) & (r < 2.05), (3.55 - 2.58) / 0.5, 0.0)
    cap = np.maximum(_shell(z, profile, cap_slope, 0.125), r - 3.62)
    sector = np.abs(np.mod(phi - 0.6 + np.pi / 3, 2 * np.pi / 3) - np.pi / 3)
    hole = np.maximum(np.maximum(1.62 - r, r - 1.98), (sector - 0.24) * r)
    parts.append((STEEL, np.maximum(cap, -hole)))

    # Vent disc with a central dimple that the tab is welded to.
    dimple = 0.32 * np.clip(1 - (r / 1.15) ** 2, 0, 1) ** 2
    dimple_slope = 0.32 * 2 * np.clip(1 - (r / 1.15) ** 2, 0, 1) * 2 * r / 1.15**2
    vent = np.maximum(_shell(z, 2.32 - dimple, dimple_slope, 0.1), r - 3.62)
    parts.append((ALUMINIUM, vent))

    # Gasket: a C-section ring holding the cap and vent edges.
    outer = _box_sdf(r, z, 3.3, 4.10, 2.12, CAN_TOP - 0.18)
    slot = _box_sdf(r, z, 0.0, 3.72, 2.20, 2.72)
    parts.append((POLYPROPYLENE, np.maximum(outer, -slot)))

    # Insulating washer above the winding with a slot for the tab.
    washer = _box_sdf(r, z, 0.72, 3.9, 1.42, 1.57)
    tab_slot = np.maximum(np.abs(y) - 1.05, np.maximum(1.85 - x, x - 2.55))
    parts.append((POLYIMIDE, np.maximum(washer, -tab_slot)))

    # Split centre pin.
    pin = np.maximum(np.maximum(0.43 - r, r - 0.55), z - 1.1)
    split = np.abs(np.mod(phi - 2.4 + np.pi, 2 * np.pi) - np.pi) * r - 0.06
    parts.append((STAINLESS, np.maximum(pin, -split)))

    # Tab fold: from the top of the riser across to the dimple of the vent disc.
    fold = np.maximum.reduce([np.abs(z - 1.98) - 0.03, np.abs(y) - 0.95, -0.1 - x, x - 2.2])
    parts.append((ALUMINIUM, fold))

    margin = np.full(x.shape, np.inf)
    for material, sdf in reversed(parts):
        labels[sdf < 0] = material
        margin = np.minimum(margin, np.abs(sdf))
    return labels, margin


def _inclusion_centre() -> np.ndarray:
    phi = 2 * np.pi - 0.45
    turn = 6 + phi / (2 * np.pi)
    r = R0 + PITCH * turn + 0.16
    return np.array([r * np.cos(phi), r * np.sin(phi), 0.0])


DELAMINATION = {
    "turn": 5.0 + 0.5 / (2 * np.pi),
    "half_turns": 0.06,
    "z": -0.3,
    "half_z": 0.9,
    "gap": 0.08,
}


class Defects:
    """Metal particle across a separator and a delaminated cathode coating."""

    def __init__(self, seed: int) -> None:
        """Draw the particle's shape from ``seed``."""
        rng = np.random.default_rng([seed, 2])
        self.centre = _inclusion_centre()
        self.semi = np.array([0.065, 0.05, 0.055])
        self.noise = WaveNoise.make(rng, 0.05, waves=6)

    def __call__(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return labels (KEEP outside the defects) and margins."""
        labels = np.full(x.shape, KEEP, np.int16)
        p = np.stack([x, y, z], -1) - self.centre
        rho = np.linalg.norm(p / self.semi, axis=1)
        sdf = (rho - 1 - 0.12 * self.noise(x, y, z)) * 0.05
        labels[sdf < 0] = STAINLESS
        margin = np.abs(sdf)

        # The outer cathode coating of one turn lifts off its foil, opening a
        # lens-shaped gap and pressing into the separator and the next anode.
        d = DELAMINATION
        r, phi = _polar(x, y)
        turn = np.floor(d["turn"]) + phi / (2 * np.pi)
        s = r - R0 - PITCH * turn
        u = (turn - d["turn"]) / d["half_turns"]
        w = (z - d["z"]) / d["half_z"]
        gap = d["gap"] * np.sqrt(np.clip(1 - u**2 - w**2, 0, 1))
        free = labels == KEEP
        lifted = (gap > 0) & (s >= 0.245)
        labels[free & lifted & (s < 0.245 + gap)] = GAS
        labels[free & lifted & (s >= 0.245 + gap) & (s < 0.305 + gap)] = LIFTED
        near = (np.abs(u) < 1.2) & (np.abs(w) < 1.2)
        edges = np.minimum(
            np.abs(s - 0.245), np.minimum(np.abs(s - 0.245 - gap), np.abs(s - 0.305 - gap))
        )
        margin = np.minimum(margin, np.where(near, edges, np.inf))
        return labels, margin


def battery(n: int = 256, *, seed: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """Cylindrical cell: attenuation at 80 keV in mm^-1 on an ``n``^3 grid.

    The object is fixed in mm (9 mm field of view), so ``n`` only sets the
    voxel size. ``seed`` draws the cathode particles and the metal particle's
    shape; the winding itself is fixed.
    """
    grid = Grid(n, FOV_MM)
    mu = np.array(list(MATERIALS.values()))
    fractions = winding_fractions(grid)
    volume = np.zeros((n, n, n), np.float32)
    for layer, area in zip(LAYERS, fractions, strict=True):
        volume += (mu[layer.material] * area)[:, :, None].astype(np.float32) * _z_fraction(
            grid, layer.top
        )[None, None, :].astype(np.float32)

    coating = (fractions[CATHODE_LAYERS[0]] + fractions[CATHODE_LAYERS[1]])[:, :, None]
    coating = coating * _z_fraction(grid, Z_CATHODE)[None, None, :]
    volume += ((mu[NMC] - mu[CATHODE]) * coating * particle_fraction(grid, seed)).astype(np.float32)

    voxelise(hardware, grid, mu, fine=4, under=volume)
    c = _inclusion_centre()
    defects = Defects(seed)
    voxelise(
        defects,
        grid,
        mu,
        fine=4,
        under=volume,
        box=((c[0] - 0.12, c[0] + 0.12), (c[1] - 0.12, c[1] + 0.12), (c[2] - 0.12, c[2] + 0.12)),
    )
    d = DELAMINATION
    phi = 2 * np.pi * (d["turn"] % 1)
    r = R0 + PITCH * d["turn"] + 0.25
    span = 2 * np.pi * r * d["half_turns"] + 0.1
    voxelise(
        defects,
        grid,
        mu,
        fine=4,
        under=volume,
        box=(
            (r * np.cos(phi) - span, r * np.cos(phi) + span),
            (r * np.sin(phi) - span, r * np.sin(phi) + span),
            (d["z"] - d["half_z"] - 0.05, d["z"] + d["half_z"] + 0.05),
        ),
    )
    meta = {
        "name": "battery",
        "voxel_mm": grid.voxel,
        "fov_mm": FOV_MM,
        "energy_keV": ENERGY_KEV,
        "materials": dict(MATERIALS),
        "seed": seed,
        "geometry": "cone or parallel beam; cell axis along z",
        "defects": {"metal particle": c.tolist(), "delamination turn": d["turn"]},
    }
    return volume, meta


if __name__ == "__main__":
    main(battery, zoom=("xy", 0.0, (1.3, 3.5, -1.7, 1.6)), vmax=0.3)
