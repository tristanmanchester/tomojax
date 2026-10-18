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
from tomojax.core.host_transfer import copy_to_host, upload_host
from tomojax.geometry import grid_volume_origin

from ._host_arrays import validate_host_views

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Geometry, Grid


type _Slab = tuple[int, int, Grid, jax.Array]

_MEASURED_BATCH_BYTES = 32 * 2**20


def volume_depth(grid: Grid, detector: Detector, batch: int) -> int | None:
    """Keep the resident solve when its working set fits; otherwise bound z slabs."""
    free = device_free_memory_bytes()
    plane = 4 * grid.nx * grid.ny
    fixed = 2 * batch * detector.nu * detector.nv * 4 + 256 * 2**20
    if free is None or 5 * plane * grid.nz + fixed <= 0.85 * free:
        return None
    fitting = max(1, int((0.75 * free - fixed) / (4 * plane)))
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


def _ratios(
    volume: np.ndarray,
    projections: np.ndarray,
    indices: np.ndarray,
    slabs: list[_Slab],
    detector: Detector,
    batch: int,
    backend: str,
) -> np.ndarray:
    expected = np.zeros((len(indices), detector.nv, detector.nu), np.float32)
    for start, stop, grid, coefficients in slabs:
        piece = upload_host(volume[:, :, start:stop], next(iter(coefficients.devices())))
        for first in range(0, len(indices), batch):
            selected = indices[first : first + batch]
            values = _forward(
                piece, coefficients[selected], grid=grid, detector=detector, backend=backend
            )
            expected[first : first + len(selected)] += np.asarray(values)
        del piece, values
    # Measurement copies need not share the projector's numerical batch size.
    view_bytes = detector.nv * detector.nu * max(4, projections.dtype.itemsize)
    measured_batch = max(1, min(batch, _MEASURED_BATCH_BYTES // view_bytes))
    for first in range(0, len(indices), measured_batch):
        selected = indices[first : first + measured_batch]
        prediction = expected[first : first + len(selected)]
        measured = np.asarray(projections[selected], np.float32)
        positive = prediction > 0
        np.divide(measured, prediction, out=prediction, where=positive)
        prediction[~positive] = 0
        del measured, positive
    return expected


def _update(
    volume: np.ndarray,
    seen: np.ndarray,
    ratios: np.ndarray,
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
            ratio = upload_host(
                ratios[first : first + len(selected)], next(iter(coefficients.devices()))
            )
            correction, norm = _adjoint(
                ratio, coefficients[selected], correction, norm,
                grid=grid, detector=detector, backend=backend, model=model,
            )  # fmt: skip
        if apply:
            piece = upload_host(volume[:, :, start:stop], next(iter(coefficients.devices())))
            value, visible = _updated(piece, correction, norm)
            copy_to_host(value, volume[:, :, start:stop])
            del piece, value
        else:
            visible = norm > 0
        seen[:, :, start:stop] |= np.asarray(visible)
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
) -> np.ndarray:
    """Run the validated EM configuration with one host subset and device slab at a time."""
    data = np.asarray(projections)
    validate_host_views(data)
    if np.min(data) < 0 or not np.isfinite(np.asarray(poses)).all():
        raise ValueError("mlem: projections must be finite and nonnegative, and poses finite")
    shape = (grid.nx, grid.ny, grid.nz)
    volume = np.ones(shape, np.float32) if initial is None else np.array(initial, np.float32)
    if support is not None:
        mask = np.broadcast_to(np.asarray(support, bool), shape)
        for first in range(0, grid.nx, 16):
            np.copyto(volume[first : first + 16], 0, where=~mask[first : first + 16])
    validate_host_views(volume)
    if np.min(volume) < 0:
        raise ValueError("mlem: initial volume must be finite and nonnegative")
    seen = np.zeros(shape, bool)
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
    np.multiply(volume, seen, out=volume)
    return volume


def initial_volume(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    depth: int,
) -> np.ndarray:
    """The unchanged segmented FDK initializer, assembled without a full device volume."""
    from ._warm_start import fbp_start
    from .fbp import FBPConfig

    output = np.empty((grid.nx, grid.ny, grid.nz), np.float32)
    for start in range(0, grid.nz, depth):
        stop = min(start + depth, grid.nz)
        local = slab_grid(grid, start, stop)
        image = fbp_start(geometry, local, detector, projections, config=FBPConfig(), det_grid=None)
        output[:, :, start:stop] = np.maximum(np.asarray(image), 0)
        del image
    return output
