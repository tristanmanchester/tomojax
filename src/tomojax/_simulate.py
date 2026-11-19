"""A synthetic scan of a phantom: :func:`simulate`."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax._scan import Reconstruction
    from tomojax.datasets import SimConfig
    from tomojax.geometry import Detector, Grid

type Phantom = Literal["shepp", "cube", "sphere", "blobs", "random_shapes", "lamino_disk"]
type GeometryName = Literal["parallel", "lamino", "cone"]


def simulate(
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
    grid: Grid | None = None,
    detector: Detector | None = None,
    seed: int = 0,
    config: SimConfig | None = None,
) -> Reconstruction:
    """A synthetic scan of a phantom, and the phantom: data to try reconstruction on.

    Deterministic for a ``seed``. Lengths are in voxels unless ``grid`` gives a
    voxel size. ``tj.save(path, result, include_scan=True)`` writes what
    ``tomojax simulate`` writes: the scan, and the phantom as the volume.

    Parameters
    ----------
    phantom : {"shepp", "cube", "sphere", "blobs", "random_shapes", "lamino_disk"}, optional
        The object. Default is "shepp".
    size : int, optional
        Volume edge in voxels, and the detector size, unless ``grid`` and
        ``detector`` say otherwise. Default is 64.
    n_views : int or None, optional
        Number of views; None is twice ``size``.
    geometry : {"parallel", "lamino", "cone"}, optional
        Acquisition geometry. Default is "parallel".
    rotation : float or None, optional
        Rotation range in degrees; None is 180 for parallel rays, 360 otherwise.
    tilt : float or None, optional
        Rotation axis tilt in degrees; None is 30 for laminography, 0 otherwise.
    source_to_axis, source_to_detector : float or None, optional
        Cone-beam distances; None is three times the volume's extent, and 1.5 times
        that.
    poisson_scale : float, optional
        Poisson noise, drawing counts of the value times this (higher is quieter);
        0 adds none.
    grid : Grid or None, optional
        The volume's grid; None is ``size`` voxels a side of unit size.
    detector : Detector or None, optional
        The detector; None is ``size`` pixels, or enough to see the whole volume in
        a cone beam.
    seed : int, optional
        Random seed for the phantom and the noise. Default is 0.
    config : SimConfig or None, optional
        The phantom's details and detector artefacts (:class:`tomojax.datasets.SimConfig`);
        the keywords replace its fields.

    Returns
    -------
    Reconstruction
        The phantom as ``volume``, with ``scan`` its simulated projections.

    Raises
    ------
    ValueError
        If the source is inside the volume, or a setting is unsupported.

    Examples
    --------
    >>> import tomojax as tj
    >>> truth = tj.simulate("shepp", size=64, geometry="lamino")  # doctest: +SKIP
    >>> recon = tj.reconstruct(truth.scan)  # doctest: +SKIP
    """
    from tomojax._data import LoadedNXTomo, NXTomoMetadata
    from tomojax._scan import Reconstruction, scan_from_record
    from tomojax.datasets import SimConfig, SimulationArtefacts, simulate as simulate_arrays
    from tomojax.io import ProjectionDataset

    nx, ny, nz = (size, size, size) if grid is None else (grid.nx, grid.ny, grid.nz)
    pitch = (1.0, 1.0, 1.0) if grid is None else (grid.vx, grid.vy, grid.vz)
    if detector is not None:
        nu, nv, du, dv = detector.nu, detector.nv, detector.du, detector.dv
    elif geometry == "cone":
        sod = 3.0 * max(nx, ny, nz) if source_to_axis is None else source_to_axis
        sdd = 1.5 * sod if source_to_detector is None else source_to_detector
        nu = nv = _cone_detector(max(nx, ny, nz), sod, sdd)
        du = dv = 1.0
    else:
        nu, nv, du, dv = max(nx, ny), nz, 1.0, 1.0
    base = SimConfig(nx=1, ny=1, nz=1, nu=1, nv=1, n_views=1) if config is None else config
    artefacts = replace(base.artefacts or SimulationArtefacts(), poisson_scale=poisson_scale)
    cfg = replace(
        base,
        nx=nx,
        ny=ny,
        nz=nz,
        vx=pitch[0],
        vy=pitch[1],
        vz=pitch[2],
        nu=nu,
        nv=nv,
        du=du,
        dv=dv,
        n_views=2 * size if n_views is None else n_views,
        geometry=geometry,
        tilt=(30.0 if geometry == "lamino" else 0.0) if tilt is None else tilt,
        source_to_axis=source_to_axis,
        source_to_detector=source_to_detector,
        rotation_deg=rotation,
        phantom=phantom,
        seed=seed,
        artefacts=artefacts if artefacts.has_enabled() else None,
    )
    data = simulate_arrays(cfg)
    metadata = NXTomoMetadata.from_dataset(cast("Mapping[str, Any]", data))
    metadata.volume = None  # the phantom is the result's volume, not the scan's record
    metadata.sample_name, metadata.source_name = phantom, "TomoJAX simulator"
    metadata.source_type, metadata.source_probe = "simulation", "x-ray"
    projections = np.asarray(data["projections"], np.float32)
    record = ProjectionDataset.from_nxtomo(LoadedNXTomo(projections, metadata))
    scan = scan_from_record(record, apply_poses=False)
    info = {"phantom": phantom, "seed": seed, "simulation": data["meta"]}
    return Reconstruction(np.asarray(data["volume"]), scan.geometry, "simulate", info, scan)


def _cone_detector(size: int, source_to_axis: float, source_to_detector: float) -> int:
    """Detector pixels that see a ``size``-voxel cube from ``source_to_axis``."""
    radius = size / math.sqrt(2.0)
    if source_to_axis <= radius:
        raise ValueError("the source must be outside the volume: increase source_to_axis")
    across = radius * source_to_detector / math.sqrt(source_to_axis**2 - radius**2)
    up = 0.5 * size * source_to_detector / (source_to_axis - radius)
    return 2 * math.ceil(max(across, up))


__all__ = ["simulate"]
