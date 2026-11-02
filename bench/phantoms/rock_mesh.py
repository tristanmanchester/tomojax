"""Closed mineral meshes of the procedural sandstone for independent gVXR scans.

Run with NumPy, SciPy and trimesh installed::

    python bench/phantoms/rock_mesh.py OUTPUT --n 192 --seed 0

The fixed 1.1 mm cube is sampled at ``n`` centres per side. Each cube is split into
six tetrahedra shared by every mineral. Within each tetrahedron, one-hot
material fields are interpolated linearly and contoured at 0.501. Since these
fields sum to one, the resulting material interiors cannot overlap. The zero-padded
labels close the finite core at both ends. This is a discretised mesh phantom,
not exact geometry of the procedural sampler: the mesh itself is the corpus
truth, and reconstruction-grid truth must be integrated from these meshes.
The separately applied subvoxel cracks in ``porous_rock()`` are not included.

Marching tetrahedra: Doi and Koide (1991), IEICE E74-D, 214-224:
https://globals.ieice.org/en_transactions/information/10.1587/e74-d_1_214/_p
"""

from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
from porous_rock import ENERGY_KEV, FOV_MM, MATERIALS, RockSampler

if TYPE_CHECKING:
    import trimesh

MINERALS = (
    ("quartz", "SiO2", 2.65),
    ("feldspar", "KAlSi3O8", 2.56),
    ("zircon", "ZrSiO4", 4.65),
    ("pyrite", "FeS2", 5.01),
    ("clay", "Al2Si2O5(OH)4", 2.60 * 0.55),
    ("calcite", "CaCO3", 2.71),
)


def material_mesh(labels: np.ndarray, label: int, spacing: float) -> trimesh.Trimesh:
    """Make a closed, outward-oriented XYZ mesh from a ZYX material mask."""
    import trimesh

    mask = np.pad(labels == label, 1)
    shape = np.array(mask.shape) - 1
    triangles = []
    for order in permutations(range(3)):
        corners = np.vstack([np.zeros(3), np.cumsum(np.eye(3)[list(order)], axis=0)])
        code = np.zeros(tuple(shape), np.uint8)
        for bit, offset in enumerate(corners.astype(int)):
            region = tuple(slice(a, a + n) for a, n in zip(offset, shape, strict=True))
            code |= mask[region].astype(np.uint8) << bit
        for case in range(1, 15):
            cells = np.argwhere(code == case)
            if not len(cells):
                continue
            inside = (case & (1 << np.arange(4))) != 0
            # Points on edges from inside (field 1) to outside (field 0).
            points = np.array(
                [0.501 * a + 0.499 * b for a in corners[inside] for b in corners[~inside]]
            )
            centre = points.mean(axis=0)
            outward = corners[~inside].mean(axis=0) - corners[inside].mean(axis=0)
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            normal *= np.sign(normal @ outward)
            normal /= np.linalg.norm(normal)
            u = points[0] - centre
            v = np.cross(normal, u)
            angle = np.arctan2((points - centre) @ v, (points - centre) @ u)
            points = points[np.argsort(angle)]
            for j in range(1, len(points) - 1):
                tri = cells[:, None, :] + points[[0, j, j + 1]]
                # Transform padded ZYX sample coordinates to physical XYZ.
                tri = (tri[..., ::-1] - (np.array(labels.shape[::-1]) / 2 + 0.5)) * spacing
                # Reversing coordinate order changes handedness.
                triangles.append(tri[:, ::-1].astype(np.float32))
    vertices = np.concatenate(triangles).reshape(-1, 3)
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3), process=True
    )
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError(f"Material {label} did not produce a closed oriented mesh")
    return mesh


def build(output: Path, *, n: int = 192, seed: int = 0) -> dict:
    """Write finite-core mineral STLs and their physical/material provenance."""
    output.mkdir(parents=True, exist_ok=True)
    sampler = RockSampler(seed)
    spacing = FOV_MM / n
    axis = (np.arange(n) - (n - 1) / 2) * spacing
    y, x = np.meshgrid(axis, axis, indexing="ij")
    labels = np.empty((n, n, n), np.uint8)
    for k, z in enumerate(axis):
        labels[k] = sampler(x.ravel(), y.ravel(), np.full(n * n, z))[0].reshape(n, n)
    materials = []
    for label, ((name, compound, density), mu) in enumerate(
        zip(MINERALS, list(MATERIALS.values())[1:], strict=True), start=1
    ):
        if not np.any(labels == label):
            continue
        mesh = material_mesh(labels, label, spacing)
        mesh.export(output / f"{name}.stl")
        materials.append(
            {
                "name": name,
                "compound": compound,
                "density_g_cm3": density,
                "mu_mm": mu,
                "volume_mm3": float(mesh.volume),
                "label_volume_mm3": float(np.count_nonzero(labels == label) * spacing**3),
                "triangles": len(mesh.faces),
            }
        )
    meta = {
        "name": "porous_rock_mesh",
        "seed": seed,
        "mesh_sampling_n": n,
        "mesh_sampling_mm": spacing,
        "fov_mm": FOV_MM,
        "energy_keV": ENERGY_KEV,
        "materials": materials,
        "truth": "closed mineral meshes; voxelise meshes, not the original sampler",
        "limitations": "subvoxel cracks omitted; unresolved mineral detail follows mesh sampling",
    }
    (output / "rock.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    start = time.perf_counter()
    print(json.dumps(build(args.output, n=args.n, seed=args.seed), indent=2))
    print(f"Mesh build: {time.perf_counter() - start:.2f} s")
