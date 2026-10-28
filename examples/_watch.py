"""The watch-movement phantom of the README figures: meshes, projections, truth and noise.

``bench/phantoms/watch_blender.py`` models a 29 mm skeletonised Swiss-lever
movement in Blender as one closed STL mesh per material. gVXR traces the path
length through each mesh along every ray (``bench/phantoms/gvxr_transmission.py``),
so the projections are exact line integrals of the meshes, not of a voxel
grid, and TomoJAX's projector plays no part in them. The truth on a
reconstruction grid is each voxel's exact share of every mesh
(``bench/phantoms/mesh_voxels.py``) times the material's attenuation.

The beam is monochromatic at 120 keV, as on a high-energy synchrotron beamline
such as Diamond I12. Attenuation is xraylib 4.3's total cross-section with
coherent scattering times density; meshes, renders and truths are cached under
``.artifacts/figures/watch``. Blender and gVXR are separate installations,
found at ``--blender`` and ``--gvxr-python``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHANTOMS = ROOT / "bench" / "phantoms"
CACHE = ROOT / ".artifacts" / "figures" / "watch"
BLENDER = ROOT / ".artifacts" / "tools" / "blender" / "blender-4.5.14-linux-x64" / "blender"
GVXR_PYTHON = ROOT / ".artifacts" / "gvxr-env" / "bin" / "python"

ENERGY_KEV = 120.0
MU = {  # mm^-1 at 120 keV; composition (mass fractions) and density in g/cm^3
    "titanium": 0.0917,  # Ti-6Al-4V, 4.43
    "brass": 0.3323,  # CuZn39Pb2, 8.47
    "steel": 0.2102,  # Fe 0.98, C 0.01, Mn 0.01, 7.85
    "silicon": 0.0379,  # Si, 2.33
    "ruby": 0.0597,  # Al2O3, 3.98
    "glucydur": 0.2604,  # CuBe2, 8.25
    "gold": 4.0475,  # Au 0.75, Ag 0.125, Cu 0.125 (18 carat), 15.6
    "nivarox": 0.2297,  # Fe 0.53, Ni 0.38, Cr 0.08, Be 0.01, 8.15
    "nivaflex": 0.3450,  # Co 0.45, Ni 0.21, Cr 0.18, Fe 0.10, W 0.04, Mo 0.02, 8.9
}


def meshes(blender: Path = BLENDER) -> Path:
    """Directory of the movement's STL meshes, built with Blender on first use."""
    script = PHANTOMS / "watch_blender.py"
    out = CACHE / ("meshes-" + hashlib.sha256(script.read_bytes()).hexdigest()[:12])
    if not (out / "watch.json").exists():
        subprocess.run(
            [str(blender), "-b", "--factory-startup", "-P", str(script), "--", str(out)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    return out


def transmission(
    mesh_dir: Path,
    poses: np.ndarray,
    shape: tuple[int, int],
    pixel: float,
    *,
    centre: tuple[float, float] = (0.0, 0.0),
    supersample: int = 2,
    gvxr_python: Path = GVXR_PYTHON,
) -> np.ndarray:
    """Pixel-averaged transmission ``(views, nv, nu)`` for TomoJAX ``poses``."""
    plan = {
        "meshes": str(mesh_dir),
        "names": np.array(list(MU)),
        "mu_mm": np.array(list(MU.values())),
        "shape": np.array(shape),
        "pixel_mm": pixel,
        "centre_mm": np.array(centre, np.float64),
        "supersample": supersample,
        "poses": np.asarray(poses, np.float64),
    }
    digest = hashlib.sha256()
    for key in sorted(plan):
        digest.update(key.encode() + np.asarray(plan[key]).tobytes())
    digest.update((PHANTOMS / "gvxr_transmission.py").read_bytes())
    cached = CACHE / f"render-{digest.hexdigest()[:16]}.npz"
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as work:
            plan_path = Path(work) / "plan.npz"
            np.savez(plan_path, **plan)
            script = PHANTOMS / "gvxr_transmission.py"
            subprocess.run(
                [str(gvxr_python), str(script), str(plan_path), str(cached)],
                check=True,
                stdout=subprocess.DEVNULL,  # gVXR logs its OpenGL setup there
            )
    with np.load(cached) as data:
        return data["transmission"]


def truth(
    mesh_dir: Path,
    shape: tuple[int, int, int],
    voxel: float,
    centre: tuple[float, float, float],
    *,
    supersample: int = 4,
) -> np.ndarray:
    """Mean attenuation (mm^-1) of every voxel of a grid of cubic ``voxel`` mm."""
    sys.path.insert(0, str(PHANTOMS))
    from mesh_voxels import occupancy, read_stl

    origin = tuple(c - (n / 2 - 0.5) * voxel for c, n in zip(centre, shape, strict=True))
    volume = np.zeros(shape, np.float32)
    for name, mu in MU.items():
        triangles = read_stl(mesh_dir / f"{name}.stl")
        volume += mu * occupancy(
            triangles, shape, (voxel,) * 3, origin, supersample=supersample
        ).astype(np.float32)
    return volume


def measure(light: np.ndarray, photons: float, seed: int) -> np.ndarray:
    """Line integrals from Poisson counts of ``photons`` per pixel through ``light``."""
    counts = np.random.default_rng(seed).poisson(photons * np.asarray(light, np.float64))
    return (-np.log(np.maximum(counts, 0.5) / photons)).astype(np.float32)


def levels(mesh_dir: Path) -> dict[str, float]:
    """The z heights (mm) of the planes the figures cut, from the build's record."""
    return json.loads((mesh_dir / "watch.json").read_text())["levels_mm"]
