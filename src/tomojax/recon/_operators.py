"""The forward projector and its exact transpose, for any geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.core.geometry.views import stack_view_poses
from tomojax.geometry.api import detector_grid_from_geometry_inputs
from tomojax.recon._projection import (
    projection_operators,
    resolve_geometry_projector,
    view_split,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jaxlib._jax import Device  # jax.Device, as a type
    import numpy as np

    from tomojax.core.geometry.base import Geometry, Grid

_BATCH = 32


def _operators(
    geometry: Geometry, grid: Grid | None, context: str, devices: Sequence[Device] | None = None
):
    """The grid, the split of the views among ``devices`` and the operators on them."""
    grid = geometry.grid if grid is None else grid
    detector = geometry.detector
    det_grid = detector_grid_from_geometry_inputs(detector, geometry)
    model, backend = resolve_geometry_projector(
        geometry, "auto", "auto", detector=detector, det_grid=det_grid, context=context
    )
    n_views = len(geometry.thetas_deg)  # pyright: ignore[reportAttributeAccessIssue]
    poses = stack_view_poses(geometry, n_views)
    batch = max(1, min(_BATCH, n_views))
    split = view_split(devices, n_views)
    operators = projection_operators(
        poses, grid, detector, det_grid, backend, batch, model, split=split
    )
    return grid, split, operators


def project(
    geometry: Geometry,
    volume: jax.Array | np.ndarray,
    *,
    grid: Grid | None = None,
    devices: Sequence[Device] | None = None,
) -> jax.Array:
    """Project ``volume`` through ``geometry``: line integrals ``(views, rows, columns)``.

    ``volume`` is ``(nx, ny, nz)`` on ``grid`` (the geometry's own by default),
    in the reciprocal of the geometry's length unit. Works for parallel,
    laminography and cone-beam geometries, with per-view poses and a rolled
    detector; :func:`backproject` is its exact transpose. ``devices`` shares
    the views among several GPUs, each holding the whole volume.
    """
    grid, split, (forward, _) = _operators(geometry, grid, "project", devices)
    if tuple(volume.shape) != (grid.nx, grid.ny, grid.nz):
        raise ValueError(
            f"project: volume shape {tuple(volume.shape)} does not match the grid "
            f"{(grid.nx, grid.ny, grid.nz)}"
        )
    # Compiled whole, the loop fills its output in place: run op by op, the zero
    # start and the result would be two sinogram-sized arrays.
    projections = jax.jit(forward)(jnp.asarray(volume, jnp.float32))
    return projections if split is None else projections[: split.n]


def backproject(
    geometry: Geometry,
    projections: jax.Array | np.ndarray,
    *,
    grid: Grid | None = None,
    devices: Sequence[Device] | None = None,
) -> jax.Array:
    """Apply the transpose of :func:`project` to ``(views, rows, columns)`` projections.

    This is the exact adjoint, not a reconstruction: use
    :func:`tomojax.reconstruct` for a volume. ``devices`` shares the views
    among several GPUs and sums their backprojections.
    """
    grid, split, (_, adjoint) = _operators(geometry, grid, "backproject", devices)
    detector = geometry.detector
    expected = (len(geometry.thetas_deg), detector.nv, detector.nu)  # pyright: ignore[reportAttributeAccessIssue]
    if tuple(projections.shape) != expected:
        raise ValueError(
            f"backproject: projections shape {tuple(projections.shape)} does not match the "
            f"geometry {expected} (views, rows, columns)"
        )
    data = jnp.asarray(projections, jnp.float32) if split is None else split.place(projections)
    return jax.jit(adjoint)(data)


__all__ = ["backproject", "project"]
