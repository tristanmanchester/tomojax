"""Where an FBP/FDK volume is made: on the device, or in host slabs."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import jax.numpy as jnp
    import numpy as np

    from tomojax.geometry import Detector, Geometry, Grid, ParallelGeometry

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
    if out is not None or (det_grid is None and too_large_for_the_device(grid, config)):
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


def too_large_for_the_device(grid: Grid, cfg: FBPConfig) -> bool:
    """Whether an FBP volume would take more than 40% of free device memory."""
    from tomojax.backends import device_free_memory_bytes

    # View-shared reconstruction holds the whole volume on every device.
    with first_device(cfg):
        free = device_free_memory_bytes()
    return free is not None and 4 * grid.nx * grid.ny * grid.nz > 0.4 * free


def fdk_config(cfg: FBPConfig) -> FDKConfig:
    """Use the same filter and CUDA backend for public FBP on cone beams."""
    from .fdk import FDKConfig

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

    logging.info("Reconstructing %dx%dx%d in z-slabs on the host", grid.nx, grid.ny, grid.nz)
    if is_cone_beam(geometry):
        data = np.asarray(projections, np.float32)
        host = FDKHostConfig(slices_per_batch=slab_planes, fdk=fdk_config(cfg))
        return fdk_host(geometry, grid, detector, data, config=host, out=out)
    host = FBPHostConfig(filter=cfg.filter, scale=cfg.scale, backend=cfg.backend)
    # Slabs are not shared among devices; the first selected one does the work.
    with first_device(cfg):
        return fbp_host(
            cast("ParallelGeometry", geometry), grid, detector, projections, config=host, out=out
        )
