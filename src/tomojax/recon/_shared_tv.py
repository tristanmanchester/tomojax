"""Share FISTA's TV update in the same collective that sums its gradient."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.backend_policy import cuda_backend
from tomojax.core.devices import VIEWS

from ._tv_ops import cuda_tv_proximal, cuda_tv_workspace_bytes

if TYPE_CHECKING:
    from .fista_tv import FistaRuntime


@dataclass(frozen=True)
class TVSlabs:
    """Overlapping gradient slabs with disjoint output interiors."""

    axis: int
    span: int
    width: int
    starts: tuple[int, ...]

    def update(
        self, z: jax.Array, gradient: jax.Array, *, runtime: FistaRuntime, lipschitz: jax.Array
    ) -> jax.Array:
        """Sum this device's gradient slab, update it and gather disjoint interiors."""
        axis, length = self.axis, z.shape[self.axis]
        extra = len(self.starts) * self.span - length

        i = jax.lax.axis_index(VIEWS)
        start = jax.lax.clamp(0, i * self.span - (self.width - self.span) // 2, length - self.width)
        contributions = jnp.stack(
            [jax.lax.slice_in_dim(gradient, s, s + self.width, axis=axis) for s in self.starts]
        )
        result = self._proximal_slab(
            z, jax.lax.psum_scatter(contributions, VIEWS), start, runtime, lipschitz
        )
        # The final interior may be short. Pad before dynamic_slice so XLA's
        # high-end clamping cannot shift that interior backwards.
        if extra:
            result = jax.lax.pad(
                result, np.float32(0), [(0, extra if a == axis else 0, 0) for a in range(3)]
            )
        core = jax.lax.dynamic_slice_in_dim(result, i * self.span - start, self.span, axis=axis)
        return jax.lax.slice_in_dim(
            jax.lax.all_gather(core, VIEWS, axis=axis, tiled=True), 0, length, axis=axis
        )

    def _proximal_slab(
        self,
        z: jax.Array,
        g: jax.Array,
        start: jax.Array,
        runtime: FistaRuntime,
        lipschitz: jax.Array,
    ) -> jax.Array:
        metric = runtime.metric
        assert metric is not None
        constraints = runtime.constraints
        proximal = cuda_tv_proximal(
            z.shape,
            runtime.config.tv_prox_iterations,
            nonnegative=False,
            lower_bound=constraints.lower_bound,
            upper_bound=constraints.upper_bound,
            gradient_scale_xyz=metric.gradient_scale_xyz,
        )
        tile = jax.lax.dynamic_slice_in_dim(z, start, self.width, axis=self.axis)
        support = runtime.volume_mask
        if support is not None:
            support = support.reshape((1,) * (z.ndim - support.ndim) + support.shape)
            if support.shape[self.axis] != 1:
                support = jax.lax.dynamic_slice_in_dim(support, start, self.width, axis=self.axis)
            g = g * support
            support = support != 0
        return proximal(
            tile - (1.0 / lipschitz) * g,
            metric.weight / lipschitz,
            jax.lax.convert_element_type(runtime.huber_delta, z.dtype)
            if runtime.regularizer == "huber_tv"
            else None,
            support,
        )


def tv_slabs(runtime: FistaRuntime, shape: tuple[int, ...]) -> TVSlabs | None:
    """Partition a tiled CUDA proximal when each owner needs a proper subvolume."""
    split = runtime.split
    if runtime.metric is None or split is None or not cuda_backend(split.first.platform):
        return None
    steps = runtime.config.tv_prox_iterations
    if cuda_tv_workspace_bytes(shape, np.dtype(np.float32), steps) is None:
        return None
    axis = max(range(len(shape)), key=shape.__getitem__)
    length, count = shape[axis], split.mesh.size
    span = -(-length // count)
    width = span + 2 * steps
    # Bound the packed reduction to at most twice the padded gradient.
    # A fine device partition must not replicate a full halo on arbitrarily many peers.
    if width >= length or 2 * steps > span:
        return None
    starts = tuple(min(max(i * span - steps, 0), length - width) for i in range(count))
    return TVSlabs(axis, span, width, starts)
