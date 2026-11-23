"""Estimate the storage of an inversion from its resolved acquisition and memory sample."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from tomojax.wire.resources import Placement, Snapshot

if TYPE_CHECKING:
    from tomojax.geometry import ScanGeometry
    from tomojax.recon.api import MethodConfig


def inversion_storage(
    method: str,
    cfg: MethodConfig,
    geometry: ScanGeometry,
    data: Any,
    snapshot: tuple[Snapshot, ...],
) -> Placement:
    """Estimate host and device bytes from the arrays and the current placement policy."""
    grid, detector = geometry.grid, geometry.detector
    views, rows, cols = data.shape
    hosts = int(data.nbytes)
    projections = 4 * views * rows * cols
    devices = tuple(s.device for s in snapshot)
    room = min((s.available for s in snapshot if s.available is not None), default=None)
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
    # FBP reserves the remainder for projection, filtering and backprojection workspace.
    if method == "fbp":
        from ._fbp_storage import needs_fbp_slabs

        slabbed = needs_fbp_slabs(geometry, grid, room)
    elif method in {"sirt", "mlem"}:
        from tomojax.core.geometry.cone import is_cone_beam

        from ._axial_slabs import independent_planes

        eligible = is_cone_beam(geometry) or (
            grid.nz > 1 and independent_planes(geometry, grid, detector)
        )
        slabbed = eligible and room is not None and resident > room
    else:
        slabbed = bool(getattr(cfg, "stream", False)) or (room is not None and resident > room)
    from .fista_tv import FISTAConfig, fista_view_batch

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


def _recon_bytes(
    method: str, config: Any, volume: int, projections: int, grid: Any, shape: tuple[int, ...]
) -> int:
    """Resident working set including the iterate, auxiliary volumes and measured views."""
    if method == "cgls":
        from tomojax.recon.cgls import resident_bytes

        return resident_bytes(grid, shape)
    if method == "sirt":
        return int(
            (3 + min(config.subsets, 16) + 2 * config.acceleration) * volume
            + (2 + 2 / config.subsets) * projections
        )
    if method == "spdhg":
        return (3 + 3 * (config.tv_weight != 0)) * volume + 2 * projections
    if method == "fista":
        return (4 + 4 * (config.tv_weight != 0)) * volume + 2 * projections
    return (3 if method == "mlem" else 1) * volume + 2 * projections
