"""Where an FBP/FDK volume is made: on the device, or in host slabs."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import jax.numpy as jnp
    import numpy as np

    from tomojax.geometry import Detector, Geometry, Grid

    from .fbp import FBPConfig
    from .fdk import FDKConfig


def fbp_volume(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    config: FBPConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    out: np.ndarray | None,
    *,
    slab_planes: int | None = None,
) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
    """FBP (FDK for cone beams) and its record, in ``out`` or host slabs when too large.

    ``slab_planes`` fixes the depth of host FDK slabs; None sizes them from free memory.
    """
    from .fbp import fbp

    if out is not None and det_grid is not None:
        raise ValueError("host output does not support explicit detector grids")
    if out is not None or (det_grid is None and too_large_for_the_device(geometry, grid, config)):
        volume = _host_fbp(
            geometry, grid, detector, projections, config, out=out, slab_planes=slab_planes
        )
        return volume, {"config": config, "host_slabs": True}
    volume = fbp(geometry, grid, detector, projections, config=config, det_grid=det_grid)
    return volume, {"config": config}


def first_device(cfg: FBPConfig) -> AbstractContextManager[object]:
    """Make the configuration's first device the default, or leave JAX's own."""
    import jax

    from tomojax.core.devices import as_devices

    devices = as_devices(cfg.devices)
    return jax.default_device(devices[0]) if devices else nullcontext()


def too_large_for_the_device(geometry: Geometry, grid: Grid, cfg: FBPConfig) -> bool:
    """Whether an FBP volume would take more than 40% of free device memory."""
    from tomojax.backends import device_free_memory_bytes

    # View-shared reconstruction holds the whole volume on every device.
    with first_device(cfg):
        free = device_free_memory_bytes()
    return needs_fbp_slabs(geometry, grid, free)


def device_free(cfg: FBPConfig) -> int | None:
    """Free memory on the configuration's first device (or JAX's default), sampled now.

    Parameters
    ----------
    cfg : FBPConfig
        Settings naming the devices, if any.

    Returns
    -------
    int or None
        Free bytes, or None when the allocator does not report them.
    """
    from tomojax.backends import device_free_memory_bytes

    with first_device(cfg):
        return device_free_memory_bytes()


@dataclass(frozen=True)
class AxialRoute:
    """How one axial FBP plane is computed, decided once from a single memory sample.

    Parameters
    ----------
    kind : {"slab", "row", "full"}
        The host slab holding the plane, one detector row on the device, or the whole
        volume.
    host : bool
        Whether the full solver would use host slabs.
    """

    kind: Literal["slab", "row", "full"]
    host: bool


def axial_plane_route(
    geometry: Geometry, grid: Grid, detector: Detector, index: int, free: int | None
) -> AxialRoute:
    """Decide, without solving, how the axial plane ``index`` will be reconstructed.

    Parameters
    ----------
    geometry : Geometry
        Calibrated acquisition.
    grid : Grid
        Full output voxel grid.
    detector : Detector
        Measured detector.
    index : int
        Requested axial plane on the output grid.
    free : int or None
        Free device memory, sampled once by the caller (:func:`device_free`).

    Returns
    -------
    AxialRoute
        The plane's route and the full solver's placement.
    """
    import numpy as np

    from tomojax.geometry import grid_volume_origin

    from ._axial_slabs import independent_planes

    host = needs_fbp_slabs(geometry, grid, free)
    if not independent_planes(geometry, grid, detector):
        return AxialRoute("full", host)
    if host:
        return AxialRoute("slab", host)
    # Compilers can fuse the z multiply-add and replace division by a reciprocal.
    # Prove each operation exact in FP32 before removing interpolation neighbours.
    pitch = float(np.float32(detector.dv))
    z = index * pitch + float(np.float32(grid_volume_origin(grid)[2]))
    centered = z - float(np.float32(detector.center[1]))
    exact = (
        np.frexp(pitch)[0] == 0.5
        and float(np.float32(z)) == z
        and float(np.float32(centered)) == centered
        and centered / pitch == index - (detector.nv - 1) / 2
    )
    return AxialRoute("row" if exact else "full", host)


def needs_fbp_slabs(geometry: Geometry, grid: Grid, free: int | None) -> bool:
    """Use host slabs for a canonical detector when the volume exceeds its device budget."""
    return (
        not getattr(geometry, "detector_roll", 0)
        and free is not None
        and 4 * grid.nx * grid.ny * grid.nz > 0.4 * free
    )


def selected_slab(grid: Grid, index: int) -> tuple[Grid, int]:
    """The full solver's host slab containing an independent axial plane, and its z offset."""
    from dataclasses import replace

    from tomojax.geometry import grid_volume_origin

    from .fbp_host import axial_depth

    depth = axial_depth(grid.nz, None)
    first = index // depth * depth
    ox, oy, oz = grid_volume_origin(grid)
    slab = replace(
        grid,
        nz=min(depth, grid.nz - first),
        vol_origin=(ox, oy, oz + first * grid.vz),
        vol_center=None,
    )
    return slab, first


def check_host_fbp_samples(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    cfg: FBPConfig,
    *,
    axial_index: int | None,
    host: bool | None = None,
) -> None:
    """Check the measured rows each host slab will read, scanning overlapping windows once.

    ``host`` is a placement decided earlier; None decides it from free memory now.
    """
    if not (too_large_for_the_device(geometry, grid, cfg) if host is None else host):
        return

    import numpy as np

    from tomojax.core.devices import as_devices
    from tomojax.core.geometry.cone import is_cone_beam
    from tomojax.core.geometry.views import stack_view_poses
    from tomojax.geometry import ParallelGeometry

    from ._axial_slabs import independent_planes
    from ._host_arrays import check_measurements
    from .fbp_host import axial_depth, axial_row_starts, slab_layout
    from .fdk import host_slab_ranges, storage_rows

    if is_cone_beam(geometry):
        poses = np.asarray(stack_view_poses(geometry, projections.shape[0]), np.float64)
        slabs = host_slab_ranges(grid, as_devices(cfg.devices) or (None,), None)
        windows = [storage_rows(geometry, grid, detector, poses, lo, hi) for lo, hi in slabs]
    elif type(geometry) is ParallelGeometry:
        if axial_index is not None and independent_planes(geometry, grid, detector):
            grid, _ = selected_slab(grid, axial_index)
        depth = axial_depth(grid.nz, None)
        _, local, first, step, integer = slab_layout(grid, detector, depth)
        starts = axial_row_starts(
            grid.nz, depth, detector.nv, local.nv, first, step, integer=integer
        )
        windows = [(row, min(row + local.nv, detector.nv)) for _, row, _ in starts]
    else:
        windows = [(0, detector.nv)]
    checked = 0
    for lower, upper in sorted(windows):
        start = max(lower, checked)
        if start < upper:
            for view in range(0, projections.shape[0], 16):
                check_measurements(projections[view : view + 16, start:upper], "fbp")
            checked = upper


def fdk_config(cfg: FBPConfig) -> FDKConfig:
    """FBP's filter, backend and devices for FDK on cone beams.

    FDK weights cone views exactly, so a constant ``scale`` is refused.
    """
    from tomojax.errors import InvalidParameter

    from .fdk import FDKConfig

    if cfg.scale is not None:
        raise InvalidParameter(
            "config.scale", "fbp: cone beams use FDK's exact view weights; leave scale as None"
        )
    return FDKConfig(filter=cfg.filter, backend=cfg.backend, devices=cfg.devices)


def _host_fbp(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    cfg: FBPConfig,
    *,
    out: np.ndarray | None = None,
    slab_planes: int | None = None,
) -> np.ndarray:
    """Bounded slabs for either beam type, keeping the result on the host."""
    import numpy as np

    from tomojax.core.geometry.cone import is_cone_beam

    from .fbp_host import FBPHostConfig, fbp_host
    from .fdk import FDKHostConfig, fdk_host

    logging.getLogger(__name__).info(
        "Reconstructing %dx%dx%d in z-slabs on the host", grid.nx, grid.ny, grid.nz
    )
    if is_cone_beam(geometry):
        data = np.asarray(projections, np.float32)
        host = FDKHostConfig(slices_per_batch=slab_planes, fdk=fdk_config(cfg))
        return fdk_host(geometry, grid, detector, data, config=host, out=out)
    host = FBPHostConfig(fbp=cfg)
    # Slabs are not shared among devices; the first selected one does the work.
    with first_device(cfg):
        return fbp_host(geometry, grid, detector, projections, config=host, out=out)
