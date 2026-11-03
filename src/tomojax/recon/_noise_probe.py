"""Batch-independent measurement-noise probes for automatic physical TV."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._data_term import project_residual


def gaussian_probe(view_indices, detector_shape: tuple[int, int], *, weights=None):
    """One reproducible centered iid probe, weighted for a data-gradient transpose.

    Each full detector view has its own seed derived from its global index.
    Batching, shifted tails and host row windows therefore use the same draw.
    Positive weights describe the data objective, not an inverse noise variance.
    """
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(0), i))(view_indices)
    noise = jax.vmap(lambda key: jax.random.normal(key, detector_shape, dtype=jnp.float32))(keys)
    centered = project_residual(noise, "per_view_contrast", weights=weights)
    return centered if weights is None else weights * centered


def probe_gradient_rms(
    poses, grid, detector, backproject, *, batch_size, support=None, read_weights=None
):
    """RMS of one full-scan contrast-noise transpose through the solver's BP.

    ``backproject(images, start, poses, total)`` is the solver's matched batch
    transpose, including its configured interpolation and detector coordinates.
    Only one image batch and one gradient volume are materialized. A support
    mask is part of the data operator, so it also masks the noise gradient.
    ``read_weights(start, size)`` optionally supplies whole-view objective
    weights per batch, without requiring resident storage for the full stack.
    """
    n = poses.shape[0]
    batch = min(n, batch_size)

    def accumulate(k, total):
        start = jnp.minimum(k * batch, n - batch)
        indices = start + jnp.arange(batch)
        valid = indices >= k * batch
        selected = jax.lax.dynamic_slice_in_dim(poses, start, batch)
        batch_weights = None if read_weights is None else read_weights(start, batch)
        noise = gaussian_probe(indices, (detector.nv, detector.nu), weights=batch_weights)
        noise = noise * valid[:, None, None]
        return backproject(noise, start, selected, total)

    gradient = jax.lax.fori_loop(
        0,
        (n + batch - 1) // batch,
        accumulate,
        jnp.zeros((grid.nz, grid.ny, grid.nx), jnp.float32),
    )
    if support is not None:
        gradient = gradient * support
    return jnp.sqrt(jnp.mean(gradient * gradient))
