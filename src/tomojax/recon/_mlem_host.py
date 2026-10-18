"""Host-owned cone EM volumes with coupled, bounded device workspaces.

All slab projections are summed before a subset's Poisson ratio is formed.
Its adjoint then visits every slab, so this is one coupled reconstruction,
not independent slice reconstruction. Caller inputs are never donated.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.backends import device_free_memory_bytes
from tomojax.core.cone import ConeModel, cone_project
from tomojax.geometry import grid_volume_origin

from ._host_arrays import validate_host_views

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Geometry, Grid


type _Slab = tuple[int, int, Grid, jax.Array]


def volume_depth(
    grid: Grid, detector: Detector, batch: int, *, subset_views: int = 0
) -> int | None:
    """Keep the resident solve when its working set fits; otherwise bound z slabs."""
    free = device_free_memory_bytes()
    plane = 4 * grid.nx * grid.ny
    fixed = 2 * batch * detector.nu * detector.nv * 4 + 256 * 2**20
    if free is None or 5 * plane * grid.nz + fixed <= 0.85 * free:
        return None
    ratio_bytes = subset_views * detector.nu * detector.nv * 4
    reserve = ratio_bytes if ratio_bytes <= 0.45 * free else 0
    fitting = max(1, int((0.75 * free - fixed - reserve) / (4 * plane)))
    # Device capacity alone permits slabs whose host staging and pinned copy
    # pools add several volume-sized temporaries over a solve. Bound transfers
    # independently, even when the GPU can hold a much larger working slab.
    fitting = min(fitting, max(1, (128 * 2**20) // plane))
    return min(grid.nz, 2 ** (fitting.bit_length() - 1))


def slab_grid(grid: Grid, start: int, stop: int) -> Grid:
    """A z window in the original object frame, without changing physical sampling."""
    origin = grid_volume_origin(grid)
    return replace(
        grid,
        nz=stop - start,
        vol_origin=(*origin[:2], origin[2] + start * grid.vz),
        vol_center=None,
    )


@partial(jax.jit, static_argnames=("grid", "detector", "backend"))
def _forward(
    volume: jax.Array, coefficients: jax.Array, *, grid: Grid, detector: Detector, backend: str
) -> jax.Array:
    return cone_project(volume, coefficients, grid, detector, backend=backend)


@partial(
    jax.jit,
    static_argnames=("grid", "detector", "backend", "model"),
    donate_argnums=(2, 3),
)
def _adjoint(
    ratio: jax.Array,
    coefficients: jax.Array,
    correction: jax.Array,
    norm: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    model: ConeModel,
) -> tuple[jax.Array, jax.Array]:
    return model.backproject_pair(
        (ratio, jnp.ones_like(ratio)), coefficients,
        grid=grid, detector=detector, backend=backend, accumulate=(correction, norm),
    )  # fmt: skip


@jax.jit
def _updated(
    volume: jax.Array, correction: jax.Array, norm: jax.Array
) -> tuple[jax.Array, jax.Array]:
    positive = norm > 0
    return volume * jnp.where(positive, correction / jnp.where(positive, norm, 1), 1), positive


def _parts(grid: Grid, model: ConeModel, poses: jax.Array, depth: int) -> list[_Slab]:
    result = []
    for start in range(0, grid.nz, depth):
        stop = min(start + depth, grid.nz)
        local = slab_grid(grid, start, stop)
        coefficients = model.coefficients(poses, local)
        # Coefficients already carry the physical origin. Equal-sized kernels
        # share one executable rather than recompiling for each slab position.
        kernel_grid = replace(local, vol_origin=None, vol_center=None)
        result.append((start, stop, kernel_grid, coefficients))
    return result


@partial(jax.jit, donate_argnums=(0,))
def _add_predictions(expected: jax.Array, values: jax.Array, start: jax.Array) -> jax.Array:
    offset = (start, 0, 0)
    previous = jax.lax.dynamic_slice(expected, offset, values.shape)
    return jax.lax.dynamic_update_slice(expected, previous + values, offset)


@partial(jax.jit, donate_argnums=(0,))
def _divide_predictions(expected: jax.Array, measured: jax.Array, start: jax.Array) -> jax.Array:
    offset = (start, 0, 0)
    prediction = jax.lax.dynamic_slice(expected, offset, measured.shape)
    positive = prediction > 0
    ratio = jnp.where(positive, measured / jnp.where(positive, prediction, 1), 0)
    return jax.lax.dynamic_update_slice(expected, ratio, offset)


def _device_slab(volume: np.ndarray, start: int, stop: int) -> jax.Array:
    """Transfer a contiguous ZYX view, retaining XYZ logical coordinates on device."""
    return jax.device_put(volume[:, :, start:stop].T).T


def _store_slab(volume: np.ndarray, start: int, stop: int, value: jax.Array) -> None:
    """Write a slab without a host-side C/F packing array."""
    volume[:, :, start:stop].T[:] = np.asarray(value.T)


def _ratios(
    volume: np.ndarray,
    projections: np.ndarray,
    indices: np.ndarray,
    slabs: list[_Slab],
    detector: Detector,
    batch: int,
    backend: str,
) -> jax.Array | np.ndarray:
    shape = (len(indices), detector.nv, detector.nu)
    free = device_free_memory_bytes()
    ratio_bytes = int(np.prod(shape)) * 4
    largest = max(grid.nx * grid.ny * grid.nz for _, _, grid, _ in slabs)
    scratch = 4 * 4 * largest + 2 * batch * detector.nv * detector.nu * 4 + 256 * 2**20
    resident = free is not None and ratio_bytes + scratch <= 0.85 * free
    expected = jnp.zeros(shape, jnp.float32) if resident else np.zeros(shape, np.float32)
    for start, stop, grid, coefficients in slabs:
        piece = _device_slab(volume, start, stop)
        for first in range(0, len(indices), batch):
            selected = indices[first : first + batch]
            values = _forward(
                piece, coefficients[selected], grid=grid, detector=detector, backend=backend
            )
            if isinstance(expected, jax.Array):
                expected = _add_predictions(expected, values, jnp.int32(first))
            else:
                expected[first : first + len(selected)] += np.asarray(values)
        if isinstance(expected, jax.Array):
            expected.block_until_ready()  # Release this slab's host staging before the next.
        del piece, values
    for first in range(0, len(indices), batch):
        selected = indices[first : first + batch]
        measured = np.asarray(projections[selected], np.float32)
        if isinstance(expected, jax.Array):
            expected = _divide_predictions(expected, jax.device_put(measured), jnp.int32(first))
            expected.block_until_ready()  # At most one advanced-indexed measurement copy lives.
            del measured
            continue
        prediction = expected[first : first + len(selected)]
        positive = prediction > 0
        np.divide(measured, prediction, out=prediction, where=positive)
        prediction[~positive] = 0
    return expected


def _update(
    volume: np.ndarray,
    seen: np.ndarray,
    ratios: jax.Array | np.ndarray,
    indices: np.ndarray,
    slabs: list[_Slab],
    detector: Detector,
    batch: int,
    backend: str,
    model: ConeModel,
    *,
    apply: bool,
) -> None:
    for start, stop, grid, coefficients in slabs:
        shape = (grid.nx, grid.ny, grid.nz)
        correction, norm = jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32)
        for first in range(0, len(indices), batch):
            selected = indices[first : first + batch]
            ratio = jax.device_put(ratios[first : first + len(selected)])
            correction, norm = _adjoint(
                ratio, coefficients[selected], correction, norm,
                grid=grid, detector=detector, backend=backend, model=model,
            )  # fmt: skip
        if apply:
            piece = _device_slab(volume, start, stop)
            value, visible = _updated(piece, correction, norm)
            _store_slab(volume, start, stop, value)
            del piece, value
        else:
            visible = norm > 0
        seen[:, :, start:stop] |= np.packbits(np.asarray(visible.T).T, axis=0)
        del visible, correction, norm, ratio


def solve(
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    model: ConeModel,
    backend: str,
    poses: jax.Array,
    order: np.ndarray,
    subsets: int,
    iterations: int,
    batch: int,
    depth: int,
    initial: jax.Array | np.ndarray | None,
    support: jax.Array | np.ndarray | None,
    warm_geometry: Geometry | None = None,
) -> np.ndarray:
    """Run the validated EM configuration with one host subset and device slab at a time."""
    data = np.asarray(projections)
    validate_host_views(data)
    if np.min(data) < 0 or not np.isfinite(np.asarray(poses)).all():
        raise ValueError("mlem: projections must be finite and nonnegative, and poses finite")
    shape = (grid.nx, grid.ny, grid.nz)
    if warm_geometry is not None:
        volume = initial_volume(warm_geometry, grid, detector, data, depth=depth)
    else:
        volume = (
            np.ones(shape, np.float32, order="F")
            if initial is None
            else np.array(initial, np.float32, order="F")
        )
    if support is not None:
        mask = np.broadcast_to(np.asarray(support, bool), shape)
        for first in range(0, grid.nx, 16):
            np.copyto(volume[first : first + 16], 0, where=~mask[first : first + 16])
    validate_host_views(volume)
    if np.min(volume) < 0:
        raise ValueError("mlem: initial volume must be finite and nonnegative")
    seen = np.zeros(((grid.nx + 7) // 8, grid.ny, grid.nz), np.uint8)
    slabs = _parts(grid, model, poses, depth)
    for _ in range(max(1, iterations)):
        for group in range(subsets):
            indices = order[group::subsets]
            ratios = (
                _ratios(volume, data, indices, slabs, detector, batch, backend)
                if iterations
                else np.zeros((len(indices), detector.nv, detector.nu), np.float32)
            )
            _update(
                volume, seen, ratios, indices, slabs, detector, batch, backend, model,
                apply=iterations > 0,
            )  # fmt: skip
            del ratios
    for start, stop, _, _ in slabs:
        visible = np.unpackbits(seen[:, :, start:stop], axis=0, count=grid.nx)
        np.multiply(volume[:, :, start:stop], visible, out=volume[:, :, start:stop])
    return volume


def initial_volume(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    depth: int,
) -> np.ndarray:
    """The per-view FDK initializer, assembled without a full device volume."""
    from .fdk import FDKHostConfig, fdk_host

    output = np.empty((grid.nx, grid.ny, grid.nz), np.float32, order="F")
    fdk_host(
        geometry, grid, detector, np.asarray(projections),
        config=FDKHostConfig(slices_per_batch=depth), out=output,
    )  # fmt: skip
    np.maximum(output, 0, out=output)
    return output
