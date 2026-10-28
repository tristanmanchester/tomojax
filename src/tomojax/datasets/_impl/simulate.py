"""Synthetic data generation and persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Literal, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from tomojax._data import NXTomoMetadata, save_nxtomo
from tomojax.backends import default_gather_dtype
from tomojax.core import progress_iter
from tomojax.core.cone import cone_coefficients, cone_project
from tomojax.core.geometry import (
    ConeBeam,
    ConeGeometry,
    Detector,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
)
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.projector import (
    forward_project_view,
    forward_project_view_T,
    get_detector_grid_device,
)
from tomojax.datasets._impl.artefacts import (
    ArtefactMetadata,
    SimulationArtefacts,
    apply_simulation_artefacts,
    normalise_simulation_artefacts,
)
from tomojax.datasets._impl.phantoms import (
    blobs,
    cube,
    lamino_disk,
    random_cubes_spheres,
    rotated_centered_cube,
    shepp_logan_3d,
    sphere,
)

if TYPE_CHECKING:
    from tomojax.core.geometry.base import DetectorDict, Geometry, GridDict


class SimMetadata(TypedDict, total=False):
    """Simulation provenance metadata."""

    seed: int
    artefacts: ArtefactMetadata


class SimulatedData(TypedDict):
    """Store generated projections, geometry metadata and the phantom.

    Attributes
    ----------
    projections : jax.Array
        Float32 line integrals of shape (n_views, nv, nu), including artefacts.
    angles : numpy.ndarray
        Nominal projection angles in degrees, shape (n_views,).
    grid, detector : dict
        Serialized voxel-grid and detector geometry with physical spacing.
    geometry_type : {'parallel', 'lamino', 'cone'}
        Acquisition geometry family.
    volume : jax.Array
        Ground-truth phantom of shape (nz, ny, nx).
    geometry_meta : dict or None
        Additional acquisition geometry, or None for parallel scans.
    meta : dict
        Simulation seed and artefact provenance.
    simulation_artefacts : dict or None
        Applied artefact parameters and affected sample indices, when present.
    """

    projections: jnp.ndarray
    angles: np.ndarray
    grid: GridDict
    detector: DetectorDict
    geometry_type: Literal["parallel", "lamino", "cone"]
    volume: jnp.ndarray
    geometry_meta: dict[str, object] | None
    meta: SimMetadata
    simulation_artefacts: ArtefactMetadata | None


@dataclass(kw_only=True)
class SimConfig:
    """Configure deterministic synthetic dataset generation.

    Lengths use one common physical unit. Phantom size parameters are in voxels,
    not physical lengths; anisotropic voxel spacing changes their physical shape.

    Parameters
    ----------
    nx, ny, nz : int
        Number of voxels along the object x, y and z axes.
    nu, nv : int
        Detector width and height, in pixels.
    n_views : int
        Number of uniformly spaced views; the final rotation endpoint is excluded.
    du, dv : float, optional
        Detector pixel pitches; both default to 1.
    vx, vy, vz : float, optional
        Object voxel pitches; all default to 1.
    rotation_deg : float or None, optional
        Total angular range in degrees. None gives 180 for parallel geometry
        and 360 for laminography or cone geometry.
    geometry : {'parallel', 'lamino', 'cone'}, optional
        Acquisition geometry; default is 'parallel'.
    tilt_deg : float, optional
        Laminography or cone rotation-axis tilt in degrees; default is 30.
    tilt_about : {'x', 'y'}, optional
        Axis about which the nominal rotation axis is tilted; default is 'x'.
    source_to_axis : float or None, optional
        Cone source-to-axis distance. None uses three times the largest physical
        volume extent.
    source_to_detector : float or None, optional
        Cone source-to-detector distance. None uses 1.5 times source_to_axis.
    phantom : {'shepp', 'cube', 'sphere', 'blobs', 'random_shapes', 'lamino_disk'}, optional
        Phantom family; default is 'shepp'.
    single_size : float, optional
        Cube side or sphere diameter as a fraction of the smallest dimension;
        default is 0.5.
    single_value : float, optional
        Cube or sphere intensity; default is 1.
    single_rotate : bool, optional
        Randomly rotate a single cube; default is True. Ignored for other phantoms.
    n_cubes, n_spheres : int, optional
        Numbers of requested objects for random_shapes and lamino_disk;
        defaults are 8 and 7. Objects too large to fit are skipped.
    min_size, max_size : int, optional
        Bounds on random cube sides and sphere diameters, in voxels;
        defaults are 4 and 32.
    min_value, max_value : float, optional
        Bounds on random object intensities; defaults are 0.1 and 1.
    max_rot_deg : float, optional
        Maximum absolute random cube Euler angle in degrees; default is 180.
    artefacts : SimulationArtefacts or None, optional
        Projection-domain artefacts; None disables them.
    seed : int, optional
        Random seed for the phantom and artefacts; default is 0.
    lamino_thickness_ratio : float, optional
        Central slab thickness relative to nz for lamino_disk; default is 0.2.
    """

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
    geometry: str = "parallel"  # "lamino" or "cone"
    tilt_deg: float = 30.0  # lamino; a cone scan's axis tilt
    tilt_about: str = "x"
    source_to_axis: float | None = None  # cone: default 3x the largest volume extent
    source_to_detector: float | None = None  # cone: default 1.5x source_to_axis
    phantom: str = "shepp"
    # single-object phantom parameters (for cube/sphere)
    single_size: float = 0.5  # relative size (cube side or sphere diameter as fraction of min dim)
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
    artefacts: SimulationArtefacts | None = None
    seed: int = 0
    lamino_thickness_ratio: float = 0.2


def make_phantom(cfg: SimConfig) -> jnp.ndarray:
    """Construct the configured phantom volume.

    Parameters
    ----------
    cfg : SimConfig
        Phantom dimensions, family, seed and shape parameters.

    Returns
    -------
    jax.Array
        Float32 phantom of shape (cfg.nz, cfg.ny, cfg.nx).

    Raises
    ------
    ValueError
        If the phantom name is unsupported.
    """
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
        )
    else:
        raise ValueError(f"unknown phantom {cfg.phantom}")
    return jnp.asarray(vol, dtype=jnp.float32)


def _project_phantom(
    geom: Geometry, grid: Grid, det: Detector, vol: jax.Array, beam: ConeBeam | None, n_views: int
) -> jax.Array:
    """Project one device phantom in native (z, y, x) order."""
    # Choose a default mixed-precision gather on accelerators
    gather = default_gather_dtype()

    # Fast path: build all poses and optionally vmapped/jitted projector
    T_all = stack_view_poses(geom, n_views)
    det_grid = get_detector_grid_device(det)

    use_fast = (n_views >= 8) and (os.environ.get("TOMOJAX_PROGRESS", "0") != "1")
    if beam is not None:
        coefficients = cone_coefficients(T_all, grid, det, beam)
        proj = jnp.concatenate(
            [
                cone_project(vol, coefficients[start : start + 32], grid, det)
                for start in range(0, n_views, 32)
            ]
        )
    elif use_fast:

        @jax.jit
        def project_all(vol_in: jnp.ndarray) -> jnp.ndarray:
            def f(T: jnp.ndarray) -> jnp.ndarray:
                return forward_project_view_T(
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
        for i in progress_iter(range(n_views), total=n_views, desc="Simulate: views"):
            p = forward_project_view(geom, grid, det, vol, view_index=i, gather_dtype=gather)
            projs.append(p)
        proj = jnp.stack(projs, axis=0)

    return proj


def simulate(cfg: SimConfig) -> SimulatedData:
    """Generate projections and metadata from a simulation configuration.

    Parameters
    ----------
    cfg : SimConfig
        Phantom, acquisition geometry and optional line-integral artefacts.

    Returns
    -------
    SimulatedData
        Projections, angles, geometry metadata and the ground-truth volume.

    Raises
    ------
    ValueError
        If the phantom, geometry or artefact settings are unsupported.
    """
    grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.vx, cfg.vy, cfg.vz)
    det = Detector(cfg.nu, cfg.nv, cfg.du, cfg.dv, center=(0.0, 0.0))
    # Determine total rotation based on geometry unless overridden
    if cfg.rotation_deg is not None:
        total_deg = float(cfg.rotation_deg)
    else:
        total_deg = 180.0 if cfg.geometry == "parallel" else 360.0
    thetas = np.linspace(0.0, total_deg, cfg.n_views, endpoint=False).astype(np.float32)

    geometry_meta: dict[str, object] | None = None
    beam: ConeBeam | None = None
    if cfg.geometry == "parallel":
        geometry_type: Literal["parallel", "lamino", "cone"] = "parallel"
        geom = ParallelGeometry(grid=grid, detector=det, angles=thetas)
    elif cfg.geometry == "lamino":
        geometry_type = "lamino"
        geom = LaminographyGeometry(
            grid=grid,
            detector=det,
            angles=thetas,
            tilt_deg=cfg.tilt_deg,
            tilt_about=cfg.tilt_about,
        )
        geometry_meta = {
            "tilt_deg": float(cfg.tilt_deg),
            "tilt_about": str(cfg.tilt_about),
        }
    elif cfg.geometry == "cone":
        geometry_type = "cone"
        extent = max(cfg.nx * cfg.vx, cfg.ny * cfg.vy, cfg.nz * cfg.vz)
        sod = 3.0 * extent if cfg.source_to_axis is None else float(cfg.source_to_axis)
        sdd = 1.5 * sod if cfg.source_to_detector is None else float(cfg.source_to_detector)
        beam = ConeBeam(sod, sdd)
        cone = ConeGeometry(
            grid=grid,
            detector=det,
            angles=thetas,
            beam=beam,
            tilt_deg=cfg.tilt_deg,
            tilt_about=cfg.tilt_about,
        )
        geom = cone
        geometry_meta = cone.geometry_metadata()
    else:
        raise ValueError("geometry must be 'parallel', 'lamino' or 'cone'")

    vol = make_phantom(cfg)
    proj = _project_phantom(geom, grid, det, vol, beam, cfg.n_views)

    artefacts = normalise_simulation_artefacts(cfg.artefacts)
    artefact_metadata: ArtefactMetadata | None = None
    if artefacts.has_enabled():
        proj, artefact_metadata = apply_simulation_artefacts(
            proj,
            artefacts,
            seed=cfg.seed,
        )

    meta: SimMetadata = {"seed": cfg.seed}
    if artefact_metadata is not None:
        meta["artefacts"] = artefact_metadata

    return {
        "projections": proj,
        "angles": thetas,
        "grid": grid.to_dict(),
        "detector": det.to_dict(),
        "geometry_type": geometry_type,
        "volume": vol,
        "geometry_meta": geometry_meta,
        "meta": meta,
        "simulation_artefacts": artefact_metadata,
    }


def simulate_to_file(cfg: SimConfig, out_path: str) -> str:
    """Generate a synthetic dataset and save its projections as NXtomo.

    Parameters
    ----------
    cfg : SimConfig
        Phantom, geometry and artefact configuration.
    out_path : str
        Destination NXtomo file path.

    Returns
    -------
    str
        Destination path. The file contains projections and metadata, not the phantom.

    Raises
    ------
    ValueError
        If simulation settings are unsupported.
    """
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
