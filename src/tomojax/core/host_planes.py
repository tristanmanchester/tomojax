"""Host-volume Joseph products with bounded device arrays."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .joseph_regions import PlaneRegion


class HostPlaneOperator:
    """Apply one full-grid Joseph matrix to host volume slabs.

    The caller owns all host input and output arrays. Device accumulators are
    private and donated between slabs. A forward pass keeps detector groups on
    device; a transpose keeps the complete detector array there when it fits.
    Yielded transpose slabs belong to the caller and must be consumed or deleted
    before requesting the next slab.

    ``slab_depth`` bounds volume transfers and ``detector_bytes`` bounds resident
    detector groups. Neither changes the global grid or sampling coordinates.
    Transfer counters include FP32 volume and detector arrays, not coefficients.
    """

    def __init__(
        self,
        coefficients: jax.Array,
        *,
        region: PlaneRegion,
        slab_depth: int,
        detector_bytes: int,
    ) -> None:
        self.coefficients = jnp.asarray(coefficients, dtype=jnp.float32)
        self.region = region
        self.shape = (region.grid.nx, region.grid.ny, region.grid.nz)
        self.slab_depth = slab_depth
        self.detector_bytes = detector_bytes
        if detector_bytes < region.image_shape[0] * region.image_shape[1] * 4:
            raise ValueError("detector_bytes must hold at least one detector window")
        self.host_to_device_bytes = 0
        self.device_to_host_bytes = 0

    @property
    def projection_shape(self) -> tuple[int, int, int]:
        """The full (view, v, u) projection shape."""
        return (self.coefficients.shape[0], *self.region.image_shape)

    def slabs(self) -> Iterator[tuple[int, int]]:
        """Yield disjoint [first, last) z-plane intervals."""
        for first in range(0, self.shape[2], self.slab_depth):
            yield first, min(first + self.slab_depth, self.shape[2])

    def put(self, value: np.ndarray) -> jax.Array:
        """Copy a host array to an owned, C-contiguous FP32 device buffer."""
        self.host_to_device_bytes += value.size * 4
        return jax.device_put(np.asarray(value, dtype=np.float32, order="C"), may_alias=False)

    def get(self, value: jax.Array) -> np.ndarray:
        """Copy a device array to independent host storage."""
        self.device_to_host_bytes += value.size * 4
        return np.array(value, dtype=np.float32, order="C")

    def _groups(self, directions: int = 0) -> Iterator[tuple[int, int]]:
        n_views, nv, nu = self.projection_shape
        group = self.detector_bytes // ((directions + 1) * nv * nu * 4)
        if group < 1:
            raise ValueError("detector_bytes must hold one view's prediction and columns")
        for first in range(0, n_views, group):
            yield first, min(first + group, n_views)

    def _region(self, first: int) -> PlaneRegion:
        return replace(self.region, volume_origin=(0, 0, first))

    def project(
        self, volume: np.ndarray, *, out: np.ndarray, support: np.ndarray | None = None
    ) -> None:
        """Write the full forward product into ``out``, without modifying inputs."""
        self._forward(volume, out=out, tangent=None, support=support)

    def columns(
        self,
        volume: np.ndarray,
        *,
        tangent: jax.Array,
        out: np.ndarray,
        support: np.ndarray | None = None,
    ) -> None:
        """Write prediction and pose columns into (view, 1+direction, v, u) ``out``."""
        self._forward(volume, out=out, tangent=tangent, support=support)

    def _forward(
        self,
        volume: np.ndarray,
        *,
        out: np.ndarray,
        tangent: jax.Array | None,
        support: np.ndarray | None,
    ) -> None:
        directions = 0 if tangent is None else tangent.shape[1]
        if support is not None:
            support = np.broadcast_to(support, self.shape)
        for start, stop in self._groups(directions):
            accumulator = jnp.zeros(out[start:stop].shape, jnp.float32)
            for first, last in self.slabs():
                host_tile = volume[:, :, first:last]
                if support is not None:
                    host_tile = host_tile * support[:, :, first:last]
                tile = self.put(host_tile)
                region = self._region(first)
                if tangent is None:
                    accumulator = _add_projection(
                        self.coefficients[start:stop], tile, accumulator, region=region
                    )
                else:
                    accumulator = _add_columns(
                        self.coefficients[start:stop],
                        tile,
                        tangent[start:stop],
                        accumulator,
                        region=region,
                    )
                accumulator.block_until_ready()
                tile.delete()
            out[start:stop] = self.get(accumulator)
            accumulator.delete()

    def transpose_slabs(self, images: np.ndarray) -> Iterator[tuple[int, int, jax.Array]]:
        """Yield owned device slabs of the exact transpose, in increasing z order."""
        resident = self.put(images) if images.size * 4 <= self.detector_bytes else None
        try:
            for first, last in self.slabs():
                region = self._region(first)
                accumulator = jnp.zeros((*self.shape[:2], last - first), jnp.float32)
                if resident is not None:
                    accumulator = _add_transpose(
                        self.coefficients, resident, accumulator, region=region
                    )
                else:
                    for start, stop in self._groups():
                        group = self.put(images[start:stop])
                        accumulator = _add_transpose(
                            self.coefficients[start:stop], group, accumulator, region=region
                        )
                        accumulator.block_until_ready()
                        group.delete()
                yield first, last, accumulator
        finally:
            if resident is not None:
                resident.delete()


@partial(jax.jit, static_argnames=("region",), donate_argnums=(2,))
def _add_projection(
    coefficients: jax.Array, tile: jax.Array, out: jax.Array, *, region: PlaneRegion
) -> jax.Array:
    return region.project(coefficients, tile, accumulate=out)


@partial(jax.jit, static_argnames=("region",), donate_argnums=(3,))
def _add_columns(
    coefficients: jax.Array,
    tile: jax.Array,
    tangent: jax.Array,
    out: jax.Array,
    *,
    region: PlaneRegion,
) -> jax.Array:
    return region.columns(coefficients, tile, tangent=tangent, accumulate=out)


@partial(jax.jit, static_argnames=("region",), donate_argnums=(2,))
def _add_transpose(
    coefficients: jax.Array, images: jax.Array, out: jax.Array, *, region: PlaneRegion
) -> jax.Array:
    return region.transpose(coefficients, images, accumulate=out)


__all__ = ["HostPlaneOperator"]
