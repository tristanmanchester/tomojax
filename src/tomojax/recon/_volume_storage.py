"""One private volume representation throughout a resident CUDA solve."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from tomojax.core.cone import ConeModel
from tomojax.core.joseph import PrivateJoseph
from tomojax.core.volume_layout import transpose_volume

if TYPE_CHECKING:
    from tomojax.geometry import Grid


type Model = str | ConeModel | PrivateJoseph


def resident_model(model: Model, backend: str) -> Model:
    """Select z-contiguous internal state for resident CUDA cone and Joseph volumes."""
    if backend != "cuda":
        return model
    if isinstance(model, ConeModel):
        return replace(model, _z_fast=True)
    return PrivateJoseph() if model == "joseph" else model


def is_joseph(model: Model) -> bool:
    """Whether ``model`` is the Joseph plane model, in either storage."""
    return model == "joseph" or isinstance(model, PrivateJoseph)


def _private(model: Model) -> bool:
    return isinstance(model, ConeModel | PrivateJoseph) and model.private_storage


def volume_shape(model: Model, grid: Grid) -> tuple[int, int, int]:
    """Buffer dimensions, without changing physical geometry or coordinates."""
    if isinstance(model, ConeModel | PrivateJoseph):
        return model.volume_shape(grid)
    return (grid.nz, grid.ny, grid.nx)


def pack_volume[T: jax.Array | None](volume: T, model: Model) -> T:
    """Make an owned private initial or support inside the compiled solve."""
    return cast("T", transpose_volume(volume)) if volume is not None and _private(model) else volume


def restore_volume(volume: jax.Array, model: Model) -> jax.Array:
    """Return public zyx after the compiled recurrence has completed."""
    return transpose_volume(jax.block_until_ready(volume)) if _private(model) else volume


def voxel_spacing(model: Model, grid: Grid) -> tuple[float, float, float]:
    """Physical spacings in the quadratic penalty's reversed-array-axis order."""
    return (grid.vz, grid.vy, grid.vx) if _private(model) else (grid.vx, grid.vy, grid.vz)


def initialize_volume(
    initial: jax.Array | None, model: Model, grid: Grid
) -> tuple[jax.Array | None, jax.Array]:
    """Pack a warm start and make a zero iterate without losing the zero-start flag."""
    packed = pack_volume(initial, model)
    return packed, jnp.zeros(volume_shape(model, grid), jnp.float32) if packed is None else packed
