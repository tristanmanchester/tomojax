from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from ..core.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from ..core.geometry.base import DetectorDict, GridDict
from ..core.geometry.views import stack_view_poses
from ..core.projector import (
    forward_project_view,
    forward_project_view_T,
    get_detector_grid_device,
)
from ..utils.memory import default_gather_dtype
from .phantoms import (
    blobs,
    cube,
    lamino_disk,
    random_cubes_spheres,
    rotated_centered_cube,
    shepp_logan_3d,
    sphere,
)
from .artefacts import (
    ArtefactMetadata,
    SimulationArtefacts,
    apply_simulation_artefacts,
    normalise_simulation_artefacts,
)
from .io_hdf5 import NXTomoMetadata, save_nxtomo
from ..utils.logging import progress_iter


class LaminoGeometryMeta(TypedDict):
    tilt_deg: float
    tilt_about: str


class SimMetadata(TypedDict, total=False):
    seed: int
    noise: str
    noise_level: float
    artefacts: ArtefactMetadata


class SimulatedData(TypedDict):
    projections: jnp.ndarray
    thetas_deg: np.ndarray
    grid: GridDict
    detector: DetectorDict
    geometry_type: Literal["parallel", "lamino"]
    volume: jnp.ndarray
    geometry_meta: LaminoGeometryMeta | None
    meta: SimMetadata
    simulation_artefacts: ArtefactMetadata | None


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
    # single-object phantom parameters (for cube/sphere)
    single_size: float = (
        0.5  # relative size (cube side or sphere diameter as fraction of min dim)
    )
    single_value: float = 1.0
    single_rotate: bool = True  # rotate cube randomly (ignored for sphere)
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
    artefacts: SimulationArtefacts | None = None
    seed: int = 0
    lamino_thickness_ratio: float = 0.2


def make_phantom(cfg: SimConfig) -> jnp.ndarray:
    if cfg.phantom == "shepp":
        vol = shepp_logan_3d(cfg.nx, cfg.ny, cfg.nz)
    elif cfg.phantom == "cube":
        if cfg.single_rotate:
            vol = rotated_centered_cube(
                cfg.nx,
                cfg.ny,
                cfg.nz,
                size=float(cfg.single_size),
                value=float(cfg.single_value),
                seed=int(cfg.seed),
            )
        else:
            vol = cube(
                cfg.nx,
                cfg.ny,
                cfg.nz,
                size=float(cfg.single_size),
                value=float(cfg.single_value),
            )
    elif cfg.phantom == "sphere":
        vol = sphere(
            cfg.nx,
            cfg.ny,
            cfg.nz,
            size=float(cfg.single_size),
            value=float(cfg.single_value),
        )
    elif cfg.phantom == "blobs":
        vol = blobs(cfg.nx, cfg.ny, cfg.nz, seed=cfg.seed)
    elif cfg.phantom == "random_shapes":
        vol = random_cubes_spheres(
            cfg.nx,
            cfg.ny,
            cfg.nz,
            n_cubes=cfg.n_cubes,
            n_spheres=cfg.n_spheres,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
            min_value=cfg.min_value,
            max_value=cfg.max_value,
            max_rot_degrees=cfg.max_rot_deg,
            use_inscribed_fov=True,
            seed=cfg.seed,
        )
    elif cfg.phantom == "lamino_disk":
        vol = lamino_disk(
            cfg.nx,
            cfg.ny,
            cfg.nz,
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


def simulate(cfg: SimConfig) -> SimulatedData:
    grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.vx, cfg.vy, cfg.vz)
    det = Detector(cfg.nu, cfg.nv, cfg.du, cfg.dv, det_center=(0.0, 0.0))
    # Determine total rotation based on geometry unless overridden
    if cfg.rotation_deg is not None:
        total_deg = float(cfg.rotation_deg)
    else:
        total_deg = 180.0 if cfg.geometry == "parallel" else 360.0
    thetas = np.linspace(0.0, total_deg, cfg.n_views, endpoint=False).astype(np.float32)

    geometry_meta: LaminoGeometryMeta | None = None
    if cfg.geometry == "parallel":
        geometry_type: Literal["parallel", "lamino"] = "parallel"
        geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    elif cfg.geometry == "lamino":
        geometry_type = "lamino"
        geom = LaminographyGeometry(
            grid=grid,
            detector=det,
            thetas_deg=thetas,
            tilt_deg=cfg.tilt_deg,
            tilt_about=cfg.tilt_about,
        )
        geometry_meta = {
            "tilt_deg": float(cfg.tilt_deg),
            "tilt_about": str(cfg.tilt_about),
        }
    else:
        raise ValueError("geometry must be 'parallel' or 'lamino'")

    vol = make_phantom(cfg)

    # Choose a default mixed-precision gather on accelerators
    gather = default_gather_dtype()

    # Fast path: build all poses and optionally vmapped/jitted projector
    T_all = stack_view_poses(geom, cfg.n_views)
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
        for i in progress_iter(
            range(cfg.n_views), total=cfg.n_views, desc="Simulate: views"
        ):
            p = forward_project_view(
                geom, grid, det, vol, view_index=i, gather_dtype=gather
            )
            projs.append(p)
        proj = jnp.stack(projs, axis=0)

    artefacts = normalise_simulation_artefacts(
        cfg.artefacts,
        noise=cfg.noise,
        noise_level=cfg.noise_level,
    )
    artefact_metadata: ArtefactMetadata | None = None
    if artefacts.has_enabled():
        proj, artefact_metadata = apply_simulation_artefacts(
            proj,
            artefacts,
            seed=cfg.seed,
        )

    meta: SimMetadata = {
        "seed": cfg.seed,
        "noise": cfg.noise,
        "noise_level": cfg.noise_level,
    }
    if artefact_metadata is not None:
        meta["artefacts"] = artefact_metadata

    return {
        "projections": proj,
        "thetas_deg": thetas,
        "grid": grid.to_dict(),
        "detector": det.to_dict(),
        "geometry_type": geometry_type,
        "volume": vol,
        "geometry_meta": geometry_meta,
        "meta": meta,
        "simulation_artefacts": artefact_metadata,
    }


def simulate_to_file(cfg: SimConfig, out_path: str) -> str:
    data = simulate(cfg)
    metadata = NXTomoMetadata.from_dataset(data)
    metadata.image_key = np.zeros((cfg.n_views,), dtype=np.int32)
    metadata.frame = "sample"
    metadata.sample_name = str(cfg.phantom)
    metadata.source_name = "TomoJAX simulator"
    metadata.source_type = "simulation"
    metadata.source_probe = "x-ray"
    save_nxtomo(
        out_path,
        projections=np.asarray(data["projections"]),
        metadata=metadata,
    )
    return out_path
