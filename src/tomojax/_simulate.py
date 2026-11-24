"""A synthetic scan of a phantom: :func:`simulate` and :class:`Simulation`."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple, cast

import numpy as np

from tomojax import progress
from tomojax._parameters import AnnotationModule, GeometryName, Phantom, simulation_distances
from tomojax.errors import InvalidParameter
from tomojax.geometry import ConeBeam, Detector, Grid

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax import datasets
    from tomojax._scan import Scan
    from tomojax.datasets import SimConfig
else:
    datasets = AnnotationModule("tomojax.datasets")


class Simulation(NamedTuple):
    """A synthetic scan and its ground-truth phantom.

    Unpacks as ``scan, phantom = tj.simulate(...)``, exposes named attributes
    ``sim.scan`` and ``sim.phantom``, and saves directly with :func:`tomojax.save`
    or exports with :func:`tomojax.export`.

    Parameters
    ----------
    scan : Scan
        Simulated projections and their geometry.
    phantom : numpy.ndarray
        Ground-truth phantom, ``(nz, ny, nx)`` on ``scan.grid``.
    """

    scan: Scan
    phantom: np.ndarray


def default_detector(
    grid: Grid,
    beam: ConeBeam | None,
    *,
    tilt: float = 0.0,
) -> Detector:
    """Return a default detector sized to cover ``grid``'s physical extent."""
    from tomojax.geometry import Detector

    if beam is not None:
        du = dv = min(grid.vx, grid.vy, grid.vz)
        nu, nv = _cone_detector(
            grid,
            beam.source_to_axis,
            beam.source_to_detector,
            du,
            dv,
            tilted=abs(float(tilt)) > 1e-12,
        )
        return Detector(nu, nv, du, dv)
    du = min(grid.vx, grid.vy)
    dv = grid.vz
    nu = max(1, math.ceil(max(grid.nx * grid.vx, grid.ny * grid.vy) / du))
    nv = max(1, math.ceil(grid.nz * grid.vz / dv))
    return Detector(nu, nv, du, dv)


def build_sim_config(
    phantom: Phantom = "shepp",
    *,
    size: int = 64,
    n_views: int | None = None,
    geometry: GeometryName = "parallel",
    rotation: float | None = None,
    tilt: float | None = None,
    source_to_axis: float | None = None,
    source_to_detector: float | None = None,
    poisson_scale: float = 0.0,
    artefacts: datasets.SimulationArtefacts | None = None,
    grid: Grid | None = None,
    detector: Detector | None = None,
    seed: int = 0,
    phantom_settings: Mapping[str, object] | None = None,
) -> tuple[Grid, Detector, SimConfig]:
    """Resolve grid, detector and ``SimConfig`` once for Python and the CLI."""
    from tomojax.datasets import SimConfig, SimulationArtefacts
    from tomojax.geometry import Grid

    if grid is None and size < 1:
        raise InvalidParameter("size", "size must be a positive voxel count")
    if n_views is not None and n_views < 1:
        raise InvalidParameter("n_views", "n_views must be positive")
    if not math.isfinite(poisson_scale) or poisson_scale < 0:
        raise InvalidParameter("poisson_scale", "poisson_scale must be finite and non-negative")
    resolved_grid = Grid(size, size, size, 1.0, 1.0, 1.0) if grid is None else grid
    resolved_tilt = (30.0 if geometry == "lamino" else 0.0) if tilt is None else tilt
    beam = None
    if geometry == "cone":
        extent = max(
            resolved_grid.nx * resolved_grid.vx,
            resolved_grid.ny * resolved_grid.vy,
            resolved_grid.nz * resolved_grid.vz,
        )
        beam = ConeBeam(*simulation_distances(extent, source_to_axis, source_to_detector))
    resolved_detector = (
        default_detector(resolved_grid, beam, tilt=resolved_tilt) if detector is None else detector
    )
    resolved_n_views = 2 * max(resolved_grid.shape) if n_views is None else n_views
    if poisson_scale > 0.0 and artefacts is not None and artefacts.poisson_scale > 0.0:
        raise InvalidParameter(
            "poisson_scale",
            "pass poisson_scale= or artefacts=SimulationArtefacts(poisson_scale=...), not both",
        )
    resolved_artefacts = artefacts
    if poisson_scale > 0.0:
        resolved_artefacts = replace(
            artefacts or SimulationArtefacts(), poisson_scale=poisson_scale
        )
    if resolved_artefacts is not None and not resolved_artefacts.has_enabled():
        resolved_artefacts = None

    cfg = SimConfig(
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
        geometry=geometry,
        tilt=resolved_tilt,
        source_to_axis=source_to_axis,
        source_to_detector=source_to_detector,
        rotation=rotation,
        phantom=phantom,
        seed=seed,
        artefacts=resolved_artefacts,
    )
    if phantom_settings:
        cfg = replace(cfg, **cast("dict[str, Any]", dict(phantom_settings)))
    return resolved_grid, resolved_detector, cfg


def simulate_config(
    cfg: SimConfig,
    *,
    grid: Grid | None = None,
    detector: Detector | None = None,
) -> Simulation:
    """Run ``tomojax.datasets.simulate`` and wrap the result in :class:`Simulation`."""
    from tomojax import datasets as _datasets
    from tomojax._data import LoadedNXTomo, NXTomoMetadata
    from tomojax._scan import scan_from_record
    from tomojax.core.devices import announce
    from tomojax.io import ProjectionDataset

    with progress.task("simulate", cfg.phantom, None, ""):
        announce()
        has_placement = (
            grid is not None and (grid.vol_origin is not None or grid.vol_center is not None)
        ) or (detector is not None and detector.center != (0.0, 0.0))
        if has_placement:
            data = _datasets.simulate(cfg, grid=grid, detector=detector)
        else:
            data = _datasets.simulate(cfg)
        metadata = NXTomoMetadata.from_dataset(cast("Mapping[str, Any]", data))
        metadata.volume = None
        metadata.sample_name, metadata.source_name = cfg.phantom, "TomoJAX simulator"
        metadata.source_type, metadata.source_probe = "simulation", "x-ray"
        meta: dict[str, Any] = dict(metadata.geometry_meta or {})
        meta["simulation_info"] = {
            "phantom": cfg.phantom,
            "seed": cfg.seed,
            "geometry": cfg.geometry,
            "simulation": data["meta"],
        }
        metadata.geometry_meta = meta
        projections = np.asarray(data["projections"], np.float32)
        record = ProjectionDataset.from_nxtomo(LoadedNXTomo(projections, metadata))
        scan = scan_from_record(record, apply_poses=False)
        return Simulation(scan=scan, phantom=np.asarray(data["volume"]))


def simulate(
    phantom: Phantom = "shepp",
    *,
    size: Annotated[int, "voxel"] = 64,
    n_views: Annotated[int | None, "view"] = None,
    geometry: GeometryName = "parallel",
    rotation: Annotated[float | None, "degree"] = None,
    tilt: Annotated[float | None, "degree"] = None,
    source_to_axis: Annotated[float | None, "geometry_length"] = None,
    source_to_detector: Annotated[float | None, "geometry_length"] = None,
    poisson_scale: Annotated[float, "counts_per_line_integral"] = 0.0,
    artefacts: datasets.SimulationArtefacts | None = None,
    grid: Grid | None = None,
    detector: Detector | None = None,
    seed: int = 0,
) -> Simulation:
    """Simulate a synthetic scan and return it with its ground-truth phantom.

    Deterministic for a ``seed``. Unpacks as ``scan, phantom = tj.simulate(...)``
    or saves directly with :func:`tomojax.save`. Explicit ``grid`` and
    ``detector`` placement (``vol_origin``, ``vol_center``, ``Detector.center``)
    is preserved through projection and in the returned scan. For low-level
    phantom-detail recipes, use :func:`tomojax.datasets.simulate` with
    :class:`tomojax.datasets.SimConfig`.

    Parameters
    ----------
    phantom : {"shepp", "cube", "sphere", "blobs", "random_shapes", "lamino_disk"}, optional
        The object. Default is "shepp".
    size : int, optional
        Volume edge in voxels, and the detector size, unless ``grid`` or
        ``detector`` say otherwise. Default is 64.
    n_views : int or None, optional
        Number of views; None uses twice the longest grid edge.
    geometry : {"parallel", "lamino", "cone"}, optional
        Acquisition geometry. Default is "parallel".
    rotation : float or None, optional
        Rotation range in degrees; None uses 180 for parallel rays and 360 otherwise.
    tilt : float or None, optional
        Rotation axis tilt in degrees; None uses 30 for laminography and 0 otherwise.
    source_to_axis : float or None, optional
        Cone source-to-axis distance. Defaults to three times the largest physical volume extent.
    source_to_detector : float or None, optional
        Cone source-to-detector distance. Defaults to 1.5 times the source-to-axis distance.
    poisson_scale : float, optional
        Poisson noise, drawing counts of the value times this (higher is quieter);
        0 adds none. Default is 0.0.
    artefacts : SimulationArtefacts or None, optional
        Detector artefacts (dead/hot pixels, zingers, stripes, Poisson noise)
        from :class:`tomojax.datasets.SimulationArtefacts`.
    grid : Grid or None, optional
        The volume's grid, including any ``vol_origin`` or ``vol_center``.
    detector : Detector or None, optional
        The detector, including any ``center`` offset.
    seed : int, optional
        Random seed for the phantom and noise. Default is 0.

    Returns
    -------
    Simulation
        Named tuple ``(scan, phantom)`` holding the simulated :class:`Scan` and
        the ground-truth phantom ``(nz, ny, nx)`` on ``scan.grid``.

    Raises
    ------
    ValueError
        If the source is inside the volume, or a setting is unsupported.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan, phantom = tj.simulate("shepp", size=64, geometry="lamino")  # doctest: +SKIP
    >>> recon = tj.reconstruct(scan)  # doctest: +SKIP
    >>> sim = tj.simulate("sphere", size=32)  # doctest: +SKIP
    >>> tj.save("sim.nxs", sim)  # doctest: +SKIP
    """
    resolved_grid, resolved_detector, cfg = build_sim_config(
        phantom,
        size=size,
        n_views=n_views,
        geometry=geometry,
        rotation=rotation,
        tilt=tilt,
        source_to_axis=source_to_axis,
        source_to_detector=source_to_detector,
        poisson_scale=poisson_scale,
        artefacts=artefacts,
        grid=grid,
        detector=detector,
        seed=seed,
    )
    return simulate_config(cfg, grid=resolved_grid, detector=resolved_detector)


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
        raise InvalidParameter(
            "source_to_axis", "the source must be outside the volume: increase source_to_axis"
        )
    across = radius * source_to_detector / math.sqrt(source_to_axis**2 - radius**2)
    up = half_z * source_to_detector / (source_to_axis - radius)
    return 2 * math.ceil(across / du), 2 * math.ceil(up / dv)


__all__ = ["Simulation", "simulate"]
