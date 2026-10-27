"""One private volume representation throughout a resident cone solve."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import jax

from tomojax.core.cone import ConeModel
from tomojax.core.volume_layout import transpose_volume

if TYPE_CHECKING:
    from tomojax.geometry import Grid


def resident_model(model: str | ConeModel, backend: str) -> str | ConeModel:
    """Select z-contiguous internal state only for resident CUDA cone volumes."""
    return (
        replace(model, _z_fast=True)
        if isinstance(model, ConeModel) and backend == "cuda"
        else model
    )


def _private(model: str | ConeModel) -> bool:
    return isinstance(model, ConeModel) and model.private_storage


def volume_shape(model: str | ConeModel, grid: Grid) -> tuple[int, int, int]:
    """Buffer dimensions, without changing physical geometry or coordinates."""
    return model.volume_shape(grid) if isinstance(model, ConeModel) else (grid.nz, grid.ny, grid.nx)


def pack_volume[T: jax.Array | None](volume: T, model: str | ConeModel) -> T:
    """Make an owned private initial or support inside the compiled solve."""
    return cast("T", transpose_volume(volume)) if volume is not None and _private(model) else volume


def restore_volume(volume: jax.Array, model: str | ConeModel) -> jax.Array:
    """Return public zyx after releasing the recurrence's transient workspace."""
    return transpose_volume(jax.block_until_ready(volume)) if _private(model) else volume


def voxel_spacing(model: str | ConeModel, grid: Grid) -> tuple[float, float, float]:
    """Physical spacings in the quadratic penalty's reversed-array-axis order."""
    return (grid.vz, grid.vy, grid.vx) if _private(model) else (grid.vx, grid.vy, grid.vz)
