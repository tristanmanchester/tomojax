"""Estimate the storage of an inversion from its resolved acquisition and memory sample."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from tomojax.wire.resources import Placement, Snapshot

if TYPE_CHECKING:
    from tomojax.geometry import ScanGeometry
    from tomojax.recon.api import MethodConfig

    from .cgls import CGLSConfig
    from .mlem import MLEMConfig
    from .sirt import SIRTConfig


def inversion_storage(
    method: str,
    cfg: MethodConfig,
    geometry: ScanGeometry,
    data: Any,
    snapshot: tuple[Snapshot, ...],
    *,
    warm_start: bool,
) -> Placement:
    """Estimate host and device bytes from the arrays and the current placement policy."""
    grid, detector = geometry.grid, geometry.detector
    views, rows, cols = data.shape
    hosts = int(data.nbytes)
    projections = 4 * views * rows * cols
    devices = tuple(dict.fromkeys(s.device for s in snapshot))
    room = snapshot[0].available if snapshot else None
    gpu = bool(devices and devices[0].startswith("gpu:"))
    pool = next((s.available for s in snapshot if s.scope == "cupy"), room)
    volume = 4 * math.prod(grid.shape)
    resident = _recon_bytes(method, cfg, volume, projections, grid, data.shape)
    if method == "fourier":
        from ._fourier_backend import fourier_slab_depth
        from ._fourier_grid import transform_grid

        cpu = all(s.scope == "host" for s in snapshot)
        depth = fourier_slab_depth(
            grid, detector, getattr(cfg, "slices_per_batch", None), "numpy" if cpu else "cupy"
        )
        shape, _ = transform_grid(grid, detector)
        buffers = depth * (16 * math.prod(shape) + 4 * views * cols)
        return Placement(
            devices,
            "slabbed",
            hosts + volume + (buffers if cpu else 0),
            0 if cpu else buffers,
            snapshot,
        )
    from .cgls import CGLSConfig
    from .fbp import FBPConfig
    from .fista_tv import FISTAConfig, fista_view_batch
    from .mlem import MLEMConfig
    from .sirt import SIRTConfig

    if isinstance(cfg, FBPConfig):
        from ._fbp_storage import needs_fbp_slabs

        slabbed = needs_fbp_slabs(geometry, grid, room)
    elif isinstance(cfg, SIRTConfig | MLEMConfig):
        slabbed = _algebraic_slabs(cfg, geometry, views, room, pool, gpu=gpu, warm_start=warm_start)
    elif isinstance(cfg, CGLSConfig):
        slabbed = _cgls_slabs(cfg, geometry, data, room, pool, gpu=gpu, warm_start=warm_start)
    elif isinstance(cfg, FISTAConfig):
        from ._fista_host import fista_host_required

        slabbed = gpu and fista_host_required(
            geometry, grid, data, cfg, room, warm_start=warm_start
        )
    else:
        # SPDHG can stream its measurements and data duals; the primal volume is resident.
        slabbed = False
    batch = (
        fista_view_batch(cfg, views)
        if isinstance(cfg, FISTAConfig)
        else min(views, getattr(cfg, "views_per_batch", getattr(cfg, "views_per_block", views)))
    )
    working = (
        min(resident, max(volume // grid.nz, 4 * batch * rows * cols) * 4) if slabbed else resident
    )
    return Placement(
        devices,
        "slabbed" if slabbed else "resident",
        hosts + (volume if slabbed else 0),
        working,
        snapshot,
    )


def _algebraic_slabs(
    cfg: SIRTConfig | MLEMConfig,
    geometry: ScanGeometry,
    views: int,
    free: int | None,
    pool: int | None,
    *,
    gpu: bool,
    warm_start: bool,
) -> bool:
    from tomojax.core.cone import cone_model

    from ._axial_slabs import axial_slab_depth
    from ._backend import reconstruction_backend
    from ._cone_slabs import (
        algebraic_sweep,
        cone_windows,
        cuda_slab_depth,
        needs_volume_sweeps,
        resident_sweep_depth,
    )

    grid, detector = geometry.grid, geometry.detector
    model = cone_model(geometry, detector)
    if model is None:
        return axial_slab_depth(geometry, grid, detector, views, cfg, free) is not None
    if (
        not gpu
        or not needs_volume_sweeps(grid, cfg, free)
        or reconstruction_backend(cfg.backend, cone=True, context="reconstruct") != "cuda"
    ):
        return False
    windows = cone_windows(geometry, grid, detector, model, views, common_height=False)
    sweep = algebraic_sweep(cfg, views, host_output=False, warm_start=warm_start)
    if pool is None:
        return True
    return (
        resident_sweep_depth(
            grid, sweep["resident"], lambda: cuda_slab_depth(windows, grid, free=pool, **sweep)
        )
        is None
    )


def _cgls_slabs(
    cfg: CGLSConfig,
    geometry: ScanGeometry,
    data: Any,
    free: int | None,
    pool: int | None,
    *,
    gpu: bool,
    warm_start: bool,
) -> bool:
    from tomojax.core.geometry.cone import is_cone_beam

    from ._backend import reconstruction_backend
    from ._cgls_cuda import available, native_fits
    from ._cgls_host import host_sweep_supported
    from ._cone_slabs import needs_volume_sweeps
    from ._fbp_storage import needs_fbp_slabs

    grid = geometry.grid
    cone = is_cone_beam(geometry)
    if (
        not gpu
        or not needs_volume_sweeps(grid, cfg, free)
        or reconstruction_backend(cfg.backend, cone=cone, context="cgls") != "cuda"
    ):
        return False
    canonical = not getattr(geometry, "detector_roll", 0.0)
    host_initial = not warm_start or needs_fbp_slabs(geometry, grid, free)
    if available(
        geometry, data, cfg, host_initial=host_initial, canonical=canonical
    ) and native_fits(cfg, grid, data.shape, pool):
        return False
    return host_sweep_supported(cfg, shared=cfg.devices is not None, cone=cone, parallel=canonical)


def _recon_bytes(
    method: str, config: Any, volume: int, projections: int, grid: Any, shape: tuple[int, ...]
) -> int:
    """Resident working set including the iterate, auxiliary volumes and measured views."""
    if method == "cgls":
        from tomojax.recon.cgls import resident_bytes

        return resident_bytes(grid, shape)
    if method in {"sirt", "mlem"}:
        from ._axial_slabs import axial_workspace

        return int(axial_workspace(config, volume, projections))
    if method == "spdhg":
        return (3 + 3 * (config.tv_weight != 0)) * volume + 2 * projections
    if method == "fista":
        return (4 + 4 * (config.tv_weight != 0)) * volume + 2 * projections
    return volume + 2 * projections
