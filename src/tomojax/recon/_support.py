"""Reference reconstruction support masks."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from tomojax.geometry import CORE_X_AXIS, CORE_Y_AXIS, CORE_Z_AXIS

VolumeSupportKind = Literal["cylindrical", "spherical"]


def centered_volume_support(
    shape: tuple[int, int, int],
    *,
    kind: VolumeSupportKind = "cylindrical",
    radius_fraction: float = 0.48,
) -> jax.Array:
    """Return a centered binary volume support mask."""
    if len(shape) != 3:
        raise ValueError("shape must be a 3D volume shape")
    if any(int(dim) <= 0 for dim in shape):
        raise ValueError("shape dimensions must be positive")
    if not 0.0 < float(radius_fraction) <= 1.0:
        raise ValueError("radius_fraction must be in (0, 1]")
    x_size = int(shape[CORE_X_AXIS])
    y_size = int(shape[CORE_Y_AXIS])
    z_size = int(shape[CORE_Z_AXIS])
    x = _centered_axis(x_size)
    y = _centered_axis(y_size)
    z = _centered_axis(z_size)
    radius = float(radius_fraction)
    if kind == "cylindrical":
        xx, yy = jnp.meshgrid(x, y, indexing="ij")
        support_2d = (xx * xx + yy * yy) <= radius * radius
        return jnp.broadcast_to(support_2d[:, :, None], shape)
    if kind == "spherical":
        xx3, yy3, zz3 = jnp.meshgrid(x, y, z, indexing="ij")
        return (xx3 * xx3 + yy3 * yy3 + zz3 * zz3) <= radius * radius
    raise ValueError(f"unknown volume support kind {kind!r}")


def _centered_axis(size: int) -> jax.Array:
    center = (float(size) - 1.0) / 2.0
    half_width = max(float(size) / 2.0, 1.0)
    return (jnp.arange(size, dtype=jnp.float32) - center) / half_width
