"""A synthetic scan of a phantom: :func:`simulate` and :class:`Simulation`."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax._scan import Scan
    from tomojax.datasets import SimConfig
    from tomojax.geometry import Detector, Grid, ScanGeometry

type Phantom = Literal["shepp", "cube", "sphere", "blobs", "random_shapes", "lamino_disk"]
type GeometryName = Literal["parallel", "lamino", "cone"]


@dataclass(frozen=True)
class Simulation:
    """A synthetic scan, its ground-truth phantom and resolved simulation settings.

    Save directly with :func:`tomojax.save` to write the scan together with the
    phantom as its volume, or export the phantom with :func:`tomojax.export`.

    Parameters
    ----------
    scan : Scan
        Simulated projections and their geometry.
    phantom : numpy.ndarray
        Ground-truth phantom, ``(nz, ny, nx)`` on :attr:`grid`.
    info : mapping
        Resolved simulation settings and artefact provenance.
    """

    scan: Scan
    phantom: np.ndarray
    info: Mapping[str, object]

    @property
    def geometry(self) -> ScanGeometry:
        """The simulated scan's geometry.

        Returns
        -------
        ScanGeometry
            Geometry of :attr:`scan`.
        """
        return self.scan.geometry

    @property
    def grid(self) -> Grid:
        """The grid the phantom is on.

        Returns
        -------
        Grid
            Reconstruction voxel grid of :attr:`scan`.
        """
        return self.scan.grid


def default_detector(
    grid: Grid,
    geometry: GeometryName,
    *,
    source_to_axis: float | None = None,
    source_to_detector: float | None = None,
    tilt: float = 0.0,
) -> Detector:
    """Return a default detector sized to cover ``grid``'s physical extent."""
    from tomojax.geometry import Detector

    if geometry == "cone":
        du = dv = min(grid.vx, grid.vy, grid.vz)
        extent = max(grid.nx * grid.vx, grid.ny * grid.vy, grid.nz * grid.vz)
        sod = 3.0 * extent if source_to_axis is None else float(source_to_axis)
        sdd = 1.5 * sod if source_to_detector is None else float(source_to_detector)
        nu, nv = _cone_detector(grid, sod, sdd, du, dv, tilted=abs(float(tilt)) > 1e-12)
        return Detector(nu, nv, du, dv)
    du = min(grid.vx, grid.vy)
    dv = grid.vz
    nu = max(1, math.ceil(max(grid.nx * grid.vx, grid.ny * grid.vy) / du))
    nv = max(1, math.ceil(grid.nz * grid.vz / dv))
    return Detector(nu, nv, du, dv)


def _resolve_sim_grid_detector(
    geometry: GeometryName,
    *,
    size: int | None,
    grid: Grid | None,
    detector: Detector | None,
    source_to_axis: float | None,
    source_to_detector: float | None,
    tilt: float,
    config: SimConfig | None,
) -> tuple[Grid, Detector]:
    """Resolve the physical simulation grid and detector from overrides and ``config``."""
    from tomojax.geometry import Detector, Grid

    if grid is not None:
        resolved_grid = grid
    elif size is not None:
        pitch = (1.0, 1.0, 1.0) if config is None else (config.vx, config.vy, config.vz)
        resolved_grid = Grid(size, size, size, *pitch)
    elif config is not None:
        resolved_grid = Grid(config.nx, config.ny, config.nz, config.vx, config.vy, config.vz)
    else:
        resolved_grid = Grid(64, 64, 64, 1.0, 1.0, 1.0)

    if detector is not None:
        resolved_detector = detector
    elif config is not None and size is None and grid is None:
        resolved_detector = Detector(config.nu, config.nv, config.du, config.dv)
    else:
        resolved_detector = default_detector(
            resolved_grid,
            geometry,
            source_to_axis=source_to_axis,
            source_to_detector=source_to_detector,
            tilt=tilt,
        )
    return resolved_grid, resolved_detector


def _resolve_sim_inputs(
    phantom: Phantom | None,
    *,
    size: int | None,
    n_views: int | None,
    geometry: GeometryName | None,
    rotation: float | None,
    tilt: float | None,
    source_to_axis: float | None,
    source_to_detector: float | None,
    poisson_scale: float | None,
    grid: Grid | None,
    detector: Detector | None,
    seed: int | None,
    config: SimConfig | None,
) -> tuple[Grid, Detector, SimConfig]:
    """Merge explicit overrides with optional ``config`` into grid, detector and ``SimConfig``."""
    from tomojax.datasets import SimConfig, SimulationArtefacts

    base = SimConfig(nx=1, ny=1, nz=1, nu=1, nv=1, n_views=1) if config is None else config
    resolved_phantom = cast(
        "Phantom", base.phantom if phantom is None and config else phantom or "shepp"
    )
    resolved_geometry = cast(
        "GeometryName", base.geometry if geometry is None and config else geometry or "parallel"
    )
    resolved_seed = base.seed if seed is None and config else (0 if seed is None else seed)
    resolved_sod = base.source_to_axis if source_to_axis is None and config else source_to_axis
    resolved_sdd = (
        base.source_to_detector if source_to_detector is None and config else source_to_detector
    )
    if tilt is not None:
        resolved_tilt = tilt
    elif config is not None and (geometry is None or geometry == config.geometry):
        resolved_tilt = config.tilt
    else:
        resolved_tilt = 30.0 if resolved_geometry == "lamino" else 0.0

    resolved_grid, resolved_detector = _resolve_sim_grid_detector(
        resolved_geometry,
        size=size,
        grid=grid,
        detector=detector,
        source_to_axis=resolved_sod,
        source_to_detector=resolved_sdd,
        tilt=resolved_tilt,
        config=config,
    )
    if n_views is not None:
        resolved_n_views = n_views
    elif config is not None and size is None:
        resolved_n_views = config.n_views
    else:
        edge = size if size is not None else max(resolved_grid.shape)
        resolved_n_views = 2 * edge

    resolved_rotation = base.rotation if rotation is None and config else rotation
    resolved_artefacts = base.artefacts
    if poisson_scale is not None:
        artefacts_obj = replace(
            base.artefacts or SimulationArtefacts(), poisson_scale=poisson_scale
        )
        resolved_artefacts = artefacts_obj if artefacts_obj.has_enabled() else None

    cfg = replace(
        base,
        nx=resolved_grid.nx,
        ny=resolved_grid.ny,
        nz=resolved_grid.nz,
        vx=resolved_grid.vx,
        vy=resolved_grid.vy,
        vz=resolved_grid.vz,
        nu=resolved_detector.nu,
        nv=resolved_detector.nv,
        du=resolved_detector.du,
        dv=resolved_detector.dv,
        n_views=resolved_n_views,
        geometry=resolved_geometry,
        tilt=resolved_tilt,
        source_to_axis=resolved_sod,
        source_to_detector=resolved_sdd,
        rotation=resolved_rotation,
        phantom=resolved_phantom,
        seed=resolved_seed,
        artefacts=resolved_artefacts,
    )
    return resolved_grid, resolved_detector, cfg


def simulate(
    phantom: Phantom | None = None,
    *,
    size: int | None = None,
    n_views: int | None = None,
    geometry: GeometryName | None = None,
    rotation: float | None = None,
    tilt: float | None = None,
    source_to_axis: float | None = None,
    source_to_detector: float | None = None,
    poisson_scale: float | None = None,
    grid: Grid | None = None,
    detector: Detector | None = None,
    seed: int | None = None,
    config: SimConfig | None = None,
) -> Simulation:
    """Simulate a synthetic scan and return it with its ground-truth phantom.

    Deterministic for a ``seed``. When ``config`` is given, omitted keywords
    keep ``config``'s values; supplied keywords override them. Explicit
    ``grid`` and ``detector`` placement (``vol_origin``, ``vol_center``,
    ``Detector.center``) is preserved through projection and in the returned
    scan. Save the result directly with :func:`tomojax.save`.

    Parameters
    ----------
    phantom : {"shepp", "cube", "sphere", "blobs", "random_shapes", "lamino_disk"}, optional
        The object. Defaults to ``config.phantom`` or ``"shepp"``.
    size : int or None, optional
        Volume edge in voxels, and the detector size, unless ``grid``,
        ``detector`` or ``config`` say otherwise. Defaults to 64 without ``config``.
    n_views : int or None, optional
        Number of views; defaults to ``config.n_views`` or twice the volume edge.
    geometry : {"parallel", "lamino", "cone"}, optional
        Acquisition geometry. Defaults to ``config.geometry`` or ``"parallel"``.
    rotation : float or None, optional
        Rotation range in degrees; None uses ``config.rotation``, or 180 for
        parallel rays and 360 otherwise.
    tilt : float or None, optional
        Rotation axis tilt in degrees; defaults to ``config.tilt``, or 30 for
        laminography and 0 otherwise.
    source_to_axis, source_to_detector : float or None, optional
        Cone-beam distances; None uses ``config``'s values, or three times the
        volume's physical extent and 1.5 times that.
    poisson_scale : float or None, optional
        Poisson noise, drawing counts of the value times this (higher is quieter);
        0 adds none. None preserves ``config.artefacts``.
    grid : Grid or None, optional
        The volume's grid, including any ``vol_origin`` or ``vol_center``.
    detector : Detector or None, optional
        The detector, including any ``center`` offset.
    seed : int or None, optional
        Random seed for the phantom and noise. Defaults to ``config.seed`` or 0.
    config : SimConfig or None, optional
        Phantom details and detector artefacts (:class:`tomojax.datasets.SimConfig`);
        only explicitly passed keywords replace its fields.

    Returns
    -------
    Simulation
        Named result holding ``scan``, ``phantom`` (``(nz, ny, nx)`` on
        ``scan.grid``) and resolved ``info``.

    Raises
    ------
    ValueError
        If the source is inside the volume, or a setting is unsupported.

    Examples
    --------
    >>> import tomojax as tj
    >>> sim = tj.simulate("shepp", size=64, geometry="lamino")  # doctest: +SKIP
    >>> recon = tj.reconstruct(sim.scan)  # doctest: +SKIP
    >>> tj.save("sim.nxs", sim)  # doctest: +SKIP
    """
    from tomojax import datasets as _datasets
    from tomojax._data import LoadedNXTomo, NXTomoMetadata
    from tomojax._scan import scan_from_record
    from tomojax.io import ProjectionDataset

    resolved_grid, resolved_detector, cfg = _resolve_sim_inputs(
        phantom,
        size=size,
        n_views=n_views,
        geometry=geometry,
        rotation=rotation,
        tilt=tilt,
        source_to_axis=source_to_axis,
        source_to_detector=source_to_detector,
        poisson_scale=poisson_scale,
        grid=grid,
        detector=detector,
        seed=seed,
        config=config,
    )
    has_placement = (
        resolved_grid.vol_origin is not None
        or resolved_grid.vol_center is not None
        or resolved_detector.center != (0.0, 0.0)
    )
    if has_placement:
        data = _datasets.simulate(cfg, grid=resolved_grid, detector=resolved_detector)
    else:
        data = _datasets.simulate(cfg)
    metadata = NXTomoMetadata.from_dataset(cast("Mapping[str, Any]", data))
    metadata.volume = None  # the phantom is the result's volume, not the scan's record
    metadata.sample_name, metadata.source_name = cfg.phantom, "TomoJAX simulator"
    metadata.source_type, metadata.source_probe = "simulation", "x-ray"
    projections = np.asarray(data["projections"], np.float32)
    record = ProjectionDataset.from_nxtomo(LoadedNXTomo(projections, metadata))
    scan = scan_from_record(record, apply_poses=False)
    info: dict[str, object] = {
        "phantom": cfg.phantom,
        "seed": cfg.seed,
        "geometry": cfg.geometry,
        "simulation": data["meta"],
    }
    return Simulation(scan=scan, phantom=np.asarray(data["volume"]), info=info)


def _cone_detector(
    grid: Grid,
    source_to_axis: float,
    source_to_detector: float,
    du: float,
    dv: float,
    *,
    tilted: bool = False,
) -> tuple[int, int]:
    """Detector pixel counts ``(nu, nv)`` that see all of ``grid`` from ``source_to_axis``."""
    from tomojax.geometry import grid_volume_origin

    ox, oy, oz = grid_volume_origin(grid)
    xs = (ox - grid.vx / 2.0, ox + (grid.nx - 0.5) * grid.vx)
    ys = (oy - grid.vy / 2.0, oy + (grid.ny - 0.5) * grid.vy)
    zs = (oz - grid.vz / 2.0, oz + (grid.nz - 0.5) * grid.vz)
    if tilted:
        radius = max(math.hypot(x, y, z) for x in xs for y in ys for z in zs)
        half_z = radius
    else:
        radius = max(math.hypot(x, y) for x in xs for y in ys)
        half_z = max(abs(z) for z in zs)
    if source_to_axis <= radius:
        raise ValueError("the source must be outside the volume: increase source_to_axis")
    across = radius * source_to_detector / math.sqrt(source_to_axis**2 - radius**2)
    up = half_z * source_to_detector / (source_to_axis - radius)
    return 2 * math.ceil(across / du), 2 * math.ceil(up / dv)


__all__ = ["Simulation", "simulate"]
