"""Matched held-out projection from a host volume, one bounded z slab at a time."""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tomojax import progress
from tomojax.core.cone import ConeModel, cone_project
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.host_volume import release_planes
from tomojax.core.joseph import forward_project_planes, plane_coefficients
from tomojax.geometry.api import detector_grid_from_geometry_inputs
from tomojax.recon._cone_slabs import window_coefficients
from tomojax.recon._projection import resolve_geometry_projector

_SLAB_BYTES = 32 << 20


@partial(jax.jit, static_argnames=("grid", "detector", "backend", "interpolation", "cone"))
def add_slab(total, slab, coefficients, *, grid, detector, backend, interpolation, cone):
    if cone:
        return total + cone_project(slab, coefficients, grid, detector, backend=backend)
    return total + forward_project_planes(
        coefficients, slab, grid, detector, backend=backend, interpolation=interpolation
    )


def host_prediction(geometry, volume, grid, *, backend, interpolation):
    detector = geometry.detector
    lattice = detector_grid_from_geometry_inputs(detector, geometry)
    model, backend = resolve_geometry_projector(
        geometry, backend, detector=detector, det_grid=lattice, context="evaluate"
    )
    poses = stack_view_poses(geometry, len(geometry.angles))
    cone = isinstance(model, ConeModel)
    coeff = np.asarray(
        model.coefficients(poses, grid)
        if cone
        else plane_coefficients(poses, grid, detector, lattice)
    )
    depth = max(1, _SLAB_BYTES // (4 * grid.nx * grid.ny))
    total = jnp.zeros((len(poses), detector.nv, detector.nu), jnp.float32)
    for start in range(0, grid.nz, depth):
        progress.check_cancelled()
        stop = min(grid.nz, start + depth)
        if cone:
            local = window_coefficients(coeff, 0, start)
        else:
            # Map the same Joseph planes to local voxel indices. For rays stepping
            # in z, local plane k is global k + start. Otherwise z is an interpolant.
            axis = coeff[:, 0]
            local = coeff.copy()
            local[:, 4] += start * np.where(axis == 2, coeff[:, 3], -(axis == 1).astype(np.float32))
            local[:, 8] += start * np.where(axis == 2, coeff[:, 7], -(axis == 0).astype(np.float32))
        slab_grid = replace(grid, nz=stop - start, vol_origin=None, vol_center=None)
        total = add_slab(
            total,
            jnp.asarray(volume[start:stop], jnp.float32),
            local,
            grid=slab_grid,
            detector=detector,
            backend=backend,
            interpolation=interpolation,
            cone=cone,
        ).block_until_ready()
        release_planes(volume, start, stop)
    return total
