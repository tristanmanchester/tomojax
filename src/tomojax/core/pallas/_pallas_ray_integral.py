"""Shared ray integration for inline and cached Pallas projectors."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ._pallas_loop import paired_fori_loop, static_fori_loop
from ._pallas_sampling import _trilinear_load_active


def integrate_rays(
    volume_ref: Any,
    start: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    delta: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    n_steps_ray: jnp.ndarray,
    *,
    shape: tuple[int, int, int],
    in_detector: jnp.ndarray,
    step_size: float,
    n_steps: int,
    kernel_variant_id: int,
    unroll: int | None,
    mask_detector: bool = False,
) -> jnp.ndarray:
    """Integrate a tile, preserving float32 recurrence and sample order.

    ``in_detector`` limits the dynamic loop bound. Inline forward kernels also
    mask each sample; cached traversals already give padded rays zero steps,
    while residual kernels mask their final output instead.
    """
    nx, ny, nz = shape
    ix0, iy0, iz0 = start
    dix, diy, diz = delta
    step_size32 = jnp.float32(step_size)

    def body(step_idx, carry):
        acc, ix, iy, iz = carry
        active = step_idx < n_steps_ray
        if mask_detector:
            active = in_detector & active
        sample = _trilinear_load_active(
            volume_ref,
            ix,
            iy,
            iz,
            nx=nx,
            ny=ny,
            nz=nz,
            active=active,
            kernel_variant_id=kernel_variant_id,
        )
        return (
            acc + sample.astype(jnp.float32) * active.astype(jnp.float32) * step_size32,
            ix + dix,
            iy + diy,
            iz + diz,
        )

    init = (jnp.zeros_like(ix0, dtype=jnp.float32), ix0, iy0, iz0)
    if unroll is None:
        tile_steps = jnp.minimum(
            jnp.max(jnp.where(in_detector, n_steps_ray, 0)),
            jnp.asarray(n_steps, dtype=jnp.int32),
        )
        acc, _, _, _ = paired_fori_loop(tile_steps, body, init)
    else:
        acc, _, _, _ = static_fori_loop(n_steps, body, init, unroll=unroll)
    return acc
