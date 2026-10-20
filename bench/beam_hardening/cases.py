"""Random polychromatic phantoms for the beam hardening bench, and the held-out split.

A case is a JSON-ready dict: z-aligned cylinders (a container and the rods
inside it, or for laminography a plate and the rods through it), a tube
spectrum, a detector and a dose. Every length is in mm.
Cases containing titanium or using more than 150 kV are held out; the
generator draws development cases without them and held-out cases with at
least one. NumPy only, so the gVXR environment and the scorer share it.
"""

from __future__ import annotations

import numpy as np

# Name: (formula, density g/cm^3).
MATERIALS = {
    "water": ("H2O", 1.00),
    "polyethylene": ("C2H4", 0.94),
    "pmma": ("C5H8O2", 1.18),
    "peek": ("C19H12O3", 1.32),
    "ptfe": ("C2F4", 2.20),
    "magnesium": ("Mg", 1.74),
    "glass": ("SiO2", 2.20),
    "aluminium": ("Al", 2.70),
    "hydroxyapatite": ("Ca5P3O13H", 3.16),
    "titanium": ("Ti", 4.51),
    "iron": ("Fe", 7.87),
    "copper": ("Cu", 8.96),
}
LOW_Z = ("water", "polyethylene", "pmma", "peek", "ptfe")
DENSE = ("magnesium", "glass", "aluminium", "hydroxyapatite", "titanium", "iron", "copper")
HELD_OUT_MATERIAL = "titanium"
HELD_OUT_KV = 150.0
FAMILIES = {"cone": ("single", "inclusions", "tube"), "lamino": ("plate", "plate-rods")}

# Fixed acquisitions, 0.1 mm voxels, 720 views over a full turn: cone-beam CT at
# magnification 2, and parallel-beam laminography of a plate whose rotation axis
# leans 30-40 degrees towards the beam. Each volume holds every ray's path through
# the object, so reprojecting a reconstruction gives whole path lengths.
GEOMETRIES = {
    "cone": {
        "beam": "cone",
        "source_to_axis": 250.0,
        "source_to_detector": 500.0,
        "nu": 400,
        "nv": 24,
        "pixel": 0.2,
        "views": 720,
        "grid": 400,
        "slices": 28,
        "voxel": 0.1,
    },
    "lamino": {
        "beam": "parallel",
        "nu": 200,
        "nv": 150,
        "pixel": 0.1,
        "views": 720,
        "grid": 200,
        "slices": 50,
        "voxel": 0.1,
    },
}
FIELD_RADIUS = 17.0  # every object stays inside this radius, clear of truncation


def held_out(case: dict) -> bool:
    """Whether a case belongs to the held-out families."""
    materials = {shape["material"] for shape in case["shapes"]}
    return HELD_OUT_MATERIAL in materials or case["spectrum"]["kv"] > HELD_OUT_KV


def _spectrum(rng: np.random.Generator, kv: float) -> dict:
    filters = [[], [["Al", rng.uniform(0.25, 3.0)]], [["Cu", rng.uniform(0.1, 1.0)]]]
    filters.append([["Al", rng.uniform(0.25, 1.0)], ["Cu", rng.uniform(0.1, 0.5)]])
    if kv >= 90:
        filters.append([["Sn", rng.uniform(0.1, 0.5)], ["Cu", 0.2], ["Al", 0.5]])
    scintillators = [("CsI", 0.1, 0.6), ("Gd2O2S", 0.05, 0.3)]
    if kv > 120:
        scintillators.append(("CdWO4", 0.5, 2.0))
    name, low, high = scintillators[rng.integers(len(scintillators))]
    nominal_angle = 12.0
    thickness = float(rng.uniform(low, high))
    return {
        "kv": float(kv),
        "generator": ["spekpy", "xpecgen"][rng.integers(2)],
        "anode": "W",
        "anode_angle_deg": nominal_angle,
        "window": ["Be", float(rng.choice([0.15, 0.3, 0.5, 1.0]))],
        "filters": [[m, float(t)] for m, t in filters[rng.integers(len(filters))]],
        "scintillator": [name, thickness],
        # Not recorded by the scanner: what makes a spectrum model approximate.
        "hidden": {
            "anode_angle_deg": float(nominal_angle + rng.uniform(-4.0, 8.0)),
            "extra_al_mm": float(rng.uniform(0.0, 0.3)),
            "scintillator_mm": float(thickness * rng.uniform(0.85, 1.15)),
        },
    }


def _place(rng: np.random.Generator, radius: float, inside: float, taken: list) -> list | None:
    """A centre for a rod of ``radius`` within a disc of ``inside``, clear of ``taken``."""
    for _ in range(200):
        r = np.sqrt(rng.uniform()) * (inside - radius - 0.6)
        angle = rng.uniform(0, 2 * np.pi)
        centre = [float(r * np.cos(angle)), float(r * np.sin(angle))]
        if all(np.hypot(centre[0] - x, centre[1] - y) > radius + q + 0.6 for x, y, q in taken):
            return centre
    return None


def sample(split: str, index: int, geometry: str = "cone") -> dict:
    """Draw case ``index`` of ``split`` (``dev`` or ``heldout``) for a geometry."""
    if split not in ("dev", "heldout"):
        raise ValueError(f"split must be 'dev' or 'heldout', not {split!r}")
    families = FAMILIES[geometry]
    family = families[index % len(families)]
    seed = [index, split == "heldout", 20261020] + ([1] if geometry == "lamino" else [])
    rng = np.random.default_rng(seed)
    for _attempt in range(1000):
        case = (_plate if geometry == "lamino" else _draw)(rng, family, split)
        if case is not None and held_out(case) == (split == "heldout"):
            name = (
                f"{split}-{index:03d}-{family}"
                if geometry == "cone"
                else f"{split}-lamino-{index:03d}-{family}"
            )
            case.update(name=name, split=split, index=index, geometry=geometry)
            return case
    raise RuntimeError(f"could not draw {split} case {index}")


def _mu(material: str, kv: float) -> float:
    """Rough attenuation (mm^-1) at two thirds of the tube voltage, for choosing sizes."""
    z = {
        "water": 7.4,
        "polyethylene": 5.5,
        "pmma": 6.5,
        "peek": 6.6,
        "ptfe": 8.4,
        "magnesium": 12,
        "glass": 11.6,
        "aluminium": 13,
        "hydroxyapatite": 16,
        "titanium": 22,
        "iron": 26,
        "copper": 29,
    }
    energy = 2 * kv / 3
    return MATERIALS[material][1] * (0.17 + 10 * z[material] ** 3 / energy**3) / 10


def _radius(rng: np.random.Generator, material: str, kv: float, low: float, high: float) -> float:
    """A radius giving a central optical depth of 0.5-5, within ``[low, high]``."""
    depth = np.exp(rng.uniform(np.log(0.5), np.log(5.0)))
    return float(np.clip(depth / _mu(material, kv) / 2, low, high))


def _voltage_and_dense(rng: np.random.Generator, split: str) -> tuple[float, list[str]]:
    """A tube voltage and the dense materials allowed with it in this split."""
    kv = rng.uniform(40, 150) if split == "dev" or rng.uniform() < 0.5 else rng.uniform(150, 225)
    dense = [m for m in DENSE if split == "heldout" or m != HELD_OUT_MATERIAL]
    if split == "heldout" and kv <= HELD_OUT_KV:
        dense = [HELD_OUT_MATERIAL]
    return float(kv), dense


def _plate(rng: np.random.Generator, family: str, split: str) -> dict | None:
    """A plate, perhaps with dense rods through it, for laminography."""
    kv, dense = _voltage_and_dense(rng, split)
    tilt = float(rng.uniform(30.0, 40.0))
    material = str(rng.choice([*LOW_Z, "glass", "aluminium", *dense]))
    # Rays cross the plate at the tilt from its face: path 2 h / sin(tilt).
    depth = np.exp(rng.uniform(np.log(0.5), np.log(5.0)))
    half = float(np.clip(depth * np.sin(np.radians(tilt)) / _mu(material, kv) / 2, 0.5, 2.0))
    radius = float(rng.uniform(4.0, 8.0))
    shapes = [{"material": material, "centre": [0.0, 0.0], "radius": radius, "half_height": half}]
    if family == "plate-rods":
        rods = [
            m
            for m in dense
            if m != material and _mu(m, kv) * 2 * half / np.sin(np.radians(tilt)) <= 8.0
        ]
        if not rods:
            return None
        rod_material, taken = str(rng.choice(rods)), []
        for _ in range(int(rng.integers(2, 5))):
            rod = float(rng.uniform(0.3, 1.2))
            centre = _place(rng, rod, radius, taken)
            if centre is None:
                return None
            taken.append((*centre, rod))
            shapes.append(
                {
                    "material": rod_material,
                    "centre": centre,
                    "radius": rod,
                    "parent": 0,
                    "half_height": half,
                }
            )
    return {
        "family": family,
        "shapes": shapes,
        "tilt_deg": tilt,
        "spectrum": _spectrum(rng, kv),
        "flat_quanta": float(np.exp(rng.uniform(np.log(5e3), np.log(5e4)))),
        "blur_px": float(rng.uniform(0.0, 0.7)),
        "seed": int(rng.integers(2**31)),
    }


def _draw(rng: np.random.Generator, family: str, split: str) -> dict | None:
    kv, dense = _voltage_and_dense(rng, split)
    # Rods at least 0.6 mm in radius that the beam still penetrates.
    rods = [m for m in dense if _mu(m, kv) * 1.2 <= 6.0]
    if family == "single":
        material = str(rng.choice([*LOW_Z, *dense] if len(dense) > 1 else dense))
        radius = _radius(rng, material, kv, 1.5, FIELD_RADIUS)
        offset = rng.uniform(0, FIELD_RADIUS - radius)
        shapes = [{"material": material, "centre": [float(offset), 0.0], "radius": radius}]
    elif family == "inclusions":
        if not rods:
            return None
        radius = float(rng.uniform(8.0, FIELD_RADIUS))
        shapes = [{"material": str(rng.choice(LOW_Z)), "centre": [0.0, 0.0], "radius": radius}]
        rod_material = str(rng.choice(rods))
        taken: list = []
        for _ in range(int(rng.integers(2, 5))):
            rod = _radius(rng, rod_material, kv, 0.6, 3.0)
            centre = _place(rng, rod, radius, taken)
            if centre is None:
                return None
            taken.append((*centre, rod))
            shapes.append({"material": rod_material, "centre": centre, "radius": rod, "parent": 0})
        void = float(rng.uniform(0.6, 2.5))
        centre = _place(rng, void, radius, taken)
        if centre is not None and rng.uniform() < 0.5:
            shapes.append({"material": "air", "centre": centre, "radius": void, "parent": 0})
    else:
        radius = float(rng.uniform(6.0, FIELD_RADIUS))
        # A tube wall of a given thickness; its tangent path is about 2 sqrt(2 R wall).
        walls = [
            m
            for m in ("pmma", "glass", "aluminium", *dense)
            if _mu(m, kv) * 2 * np.sqrt(0.8 * radius) <= 6.0
        ]
        if not rods or not walls:
            return None
        wall_material = str(rng.choice(walls))
        wall = float(rng.uniform(0.4, 2.0)) if wall_material in ("pmma", "glass") else 0.4
        shapes = [
            {"material": wall_material, "centre": [0.0, 0.0], "radius": radius},
            {
                "material": str(rng.choice(LOW_Z)),
                "centre": [0.0, 0.0],
                "radius": radius - wall,
                "parent": 0,
            },
        ]
        rod = _radius(rng, rods[0], kv, 0.6, 3.0)
        centre = _place(rng, rod, radius - wall, [])
        if centre is None:
            return None
        shapes.append(
            {"material": str(rng.choice(rods)), "centre": centre, "radius": rod, "parent": 1}
        )
    return {
        "family": family,
        "shapes": shapes,
        "spectrum": _spectrum(rng, kv),
        "flat_quanta": float(np.exp(rng.uniform(np.log(5e3), np.log(5e4)))),
        "blur_px": float(rng.uniform(0.0, 0.7)),
        "seed": int(rng.integers(2**31)),
    }
