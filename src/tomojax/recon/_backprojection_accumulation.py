"""Chunked backprojection accumulation for reference reconstruction paths."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.core.projector import sum_backproject_views_T

if TYPE_CHECKING:
    from tomojax.forward.api import CoreProjectionGeometry


def sum_backproject_views_chunked(
    core: CoreProjectionGeometry,
    projections: jax.Array,
) -> jax.Array:
    """Accumulate one-view adjoints without materialising a stack of volumes."""
    proj = jnp.asarray(projections, dtype=jnp.float32)
    n_views = int(proj.shape[0])
    if n_views == 0:
        return jnp.zeros((core.grid.nx, core.grid.ny, core.grid.nz), dtype=jnp.float32)

    def body(acc: jax.Array, view_index: jax.Array) -> tuple[jax.Array, None]:
        start = jnp.asarray(view_index, dtype=jnp.int32)
        t_view = jax.lax.dynamic_slice(core.t_all, (start, 0, 0), (1, 4, 4))
        image = jax.lax.dynamic_slice(
            proj,
            (start, 0, 0),
            (1, core.detector.nv, core.detector.nu),
        )
        update = sum_backproject_views_T(
            t_view,
            core.grid,
            core.detector,
            image,
            step_size=core.step_size,
            n_steps=core.n_steps,
            unroll=core.projector_unroll,
            gather_dtype=core.gather_dtype,
            det_grid=core.det_grid,
        )
        return acc + update, None

    init = jnp.zeros((core.grid.nx, core.grid.ny, core.grid.nz), dtype=jnp.float32)
    total, _ = jax.lax.scan(body, init, jnp.arange(n_views, dtype=jnp.int32))
    return total
