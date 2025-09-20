from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import jax.numpy as jnp
import jax

from ..core.geometry import Grid, Detector, ParallelGeometry, LaminographyGeometry
from ..core.projector import forward_project_view, forward_project_view_T, get_detector_grid_device
from ..utils.logging import progress_iter
from .phantoms import cube, blobs, shepp_logan_3d, random_cubes_spheres, lamino_disk
from .io_hdf5 import save_nxtomo
from ..utils.memory import default_gather_dtype
import os


@dataclass
class SimConfig:
    nx: int
    ny: int
    nz: int
    nu: int
    nv: int
    n_views: int
    du: float = 1.0
    dv: float = 1.0
    vx: float = 1.0
    vy: float = 1.0
    vz: float = 1.0
    rotation_deg: float | None = None  # total rotation range; defaults by geometry
    geometry: str = "parallel"  # or "lamino"
    tilt_deg: float = 30.0  # lamino
    tilt_about: str = "x"
    phantom: str = "shepp"
    # random_shapes parameters
    n_cubes: int = 8
    n_spheres: int = 7
    min_size: int = 4
    max_size: int = 32
    min_value: float = 0.1
    max_value: float = 1.0
    max_rot_deg: float = 180.0
    noise: str = "none"  # none|poisson|gaussian
    noise_level: float = 0.0  # gaussian sigma or poisson scale
    seed: int = 0
    lamino_thickness_ratio: float = 0.2


def make_phantom(cfg: SimConfig) -> jnp.ndarray:
    if cfg.phantom == "shepp":
        vol = shepp_logan_3d(cfg.nx, cfg.ny, cfg.nz)
    elif cfg.phantom == "cube":
        vol = cube(cfg.nx, cfg.ny, cfg.nz)
    elif cfg.phantom == "blobs":
        vol = blobs(cfg.nx, cfg.ny, cfg.nz, seed=cfg.seed)
    elif cfg.phantom == "random_shapes":
        vol = random_cubes_spheres(
            cfg.nx, cfg.ny, cfg.nz,
            n_cubes=cfg.n_cubes, n_spheres=cfg.n_spheres,
            min_size=cfg.min_size, max_size=cfg.max_size,
            min_value=cfg.min_value, max_value=cfg.max_value,
            max_rot_degrees=cfg.max_rot_deg,
            use_inscribed_fov=True, seed=cfg.seed,
        )
    elif cfg.phantom == "lamino_disk":
        vol = lamino_disk(
            cfg.nx, cfg.ny, cfg.nz,
            thickness_ratio=cfg.lamino_thickness_ratio,
            seed=cfg.seed,
            n_cubes=cfg.n_cubes,
            n_spheres=cfg.n_spheres,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
            min_value=cfg.min_value,
            max_value=cfg.max_value,
            max_rot_degrees=cfg.max_rot_deg,
            tilt_deg=cfg.tilt_deg,
            tilt_about=cfg.tilt_about,
        )
    else:
        raise ValueError(f"unknown phantom {cfg.phantom}")
    return jnp.asarray(vol, dtype=jnp.float32)


def simulate(cfg: SimConfig) -> Dict[str, object]:
    grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.vx, cfg.vy, cfg.vz)
    det = Detector(cfg.nu, cfg.nv, cfg.du, cfg.dv, det_center=(0.0, 0.0))
    # Determine total rotation based on geometry unless overridden
    if cfg.rotation_deg is not None:
        total_deg = float(cfg.rotation_deg)
    else:
        total_deg = 180.0 if cfg.geometry == "parallel" else 360.0
    thetas = np.linspace(0.0, total_deg, cfg.n_views, endpoint=False).astype(np.float32)

    geometry_meta: Dict[str, object] | None = None
    if cfg.geometry == "parallel":
        geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    elif cfg.geometry == "lamino":
        geom = LaminographyGeometry(
            grid=grid, detector=det, thetas_deg=thetas, tilt_deg=cfg.tilt_deg, tilt_about=cfg.tilt_about
        )
        geometry_meta = {"tilt_deg": float(cfg.tilt_deg), "tilt_about": str(cfg.tilt_about)}
    else:
        raise ValueError("geometry must be 'parallel' or 'lamino'")

    vol = make_phantom(cfg)

    # Choose a default mixed-precision gather on accelerators
    gather = default_gather_dtype()

    # Fast path: build all poses and optionally vmapped/jitted projector
    T_all = jnp.stack([jnp.asarray(geom.pose_for_view(i), jnp.float32) for i in range(cfg.n_views)], axis=0)
    det_grid = get_detector_grid_device(det)

    use_fast = (cfg.n_views >= 8) and (os.environ.get("TOMOJAX_PROGRESS", "0") != "1")
    if use_fast:
        @jax.jit
        def project_all(vol_in):
            f = lambda T: forward_project_view_T(
                T,
                grid,
                det,
                vol_in,
                use_checkpoint=True,
                gather_dtype=gather,
                det_grid=det_grid,
            )
            return jax.vmap(f, in_axes=0)(T_all)

        proj = project_all(vol)
    else:
        # Fallback path with per-view progress logging
        projs = []
        for i in progress_iter(range(cfg.n_views), total=cfg.n_views, desc="Simulate: views"):
            p = forward_project_view(geom, grid, det, vol, view_index=i)
            projs.append(p)
        proj = jnp.stack(projs, axis=0)

    rng = np.random.default_rng(cfg.seed)
    if cfg.noise == "gaussian" and cfg.noise_level > 0:
        proj = proj + jnp.asarray(rng.normal(scale=cfg.noise_level, size=proj.shape), dtype=proj.dtype)
    elif cfg.noise == "poisson" and cfg.noise_level > 0:
        s = cfg.noise_level
        lam = np.maximum(0.0, np.asarray(proj)) * s
        noisy = rng.poisson(lam=lam).astype(np.float32) / max(s, 1e-6)
        proj = jnp.asarray(noisy, dtype=proj.dtype)

    return {
        "projections": proj,
        "thetas_deg": thetas,
        "grid": grid.to_dict(),
        "detector": det.to_dict(),
        "geometry_type": cfg.geometry,
        "volume": vol,
        "geometry_meta": geometry_meta,
        "meta": {"seed": cfg.seed, "noise": cfg.noise, "noise_level": cfg.noise_level},
    }


def simulate_to_file(cfg: SimConfig, out_path: str) -> str:
    data = simulate(cfg)
    save_nxtomo(
        out_path,
        projections=np.asarray(data["projections"]),
        thetas_deg=np.asarray(data["thetas_deg"]),
        grid=data["grid"],
        detector=data["detector"],
        geometry_type=data["geometry_type"],
        geometry_meta=data.get("geometry_meta"),
        volume=np.asarray(data["volume"]),
        frame="sample",
    )
    return out_path
