"""The diagonal of A.T W A for the linear or Keys-cubic Joseph plane projector.

Each voxel's entry is the sum over views and detector pixels of its squared
plane weight times the pixel's weight squared, so it is positive wherever the
operator reaches. This is a Jacobi preconditioner input, not a transpose: the
matched forward and transpose stay the projector's own. CPU scatters one plane
at a time; CUDA runs the transpose gather with squared weights on the squared
pixel weights.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ._plane_interpolation import cubic_weights_and_derivatives
from .joseph import select_axis

if TYPE_CHECKING:
    from .geometry import Detector, Grid


def _scatter(coeff, sqrtweights, accumulator, *, grid, detector, interpolation):
    shape = (grid.nx, grid.ny, grid.nz)
    u, v = jnp.meshgrid(
        jnp.arange(detector.nu, dtype=jnp.float32), jnp.arange(detector.nv, dtype=jnp.float32)
    )

    def view(i, out):
        cf = coeff[i]
        axis = cf[0].astype(jnp.int32)
        na = select_axis(axis, *shape)
        nb = select_axis(axis, shape[1], shape[2], shape[0])
        nc = select_axis(axis, shape[2], shape[0], shape[1])

        def plane(k, target):
            kf = k.astype(jnp.float32)
            qb = (cf[1] * u + cf[2] * v) + (cf[3] * kf + cf[4])
            qc = (cf[5] * u + cf[6] * v) + (cf[7] * kf + cf[8])
            ib, ic = jnp.floor(qb).astype(jnp.int32), jnp.floor(qc).astype(jnp.int32)
            tb, tc = qb - ib, qc - ic
            if interpolation == "cubic":
                taps = (-1, 0, 1, 2)
                weights_b, weights_c = (cubic_weights_and_derivatives(t)[0] for t in (tb, tc))
            else:
                taps = (0, 1)
                weights_b, weights_c = (1 - tb, tb), (1 - tc, tc)
            for (db, wb), (dc, wc) in itertools.product(
                zip(taps, weights_b, strict=True), zip(taps, weights_c, strict=True)
            ):
                b, c = ib + db, ic + dc
                ix = select_axis(axis, k, c, b)
                iy = select_axis(axis, b, k, c)
                iz = select_axis(axis, c, b, k)
                valid = (k < na) & (b >= 0) & (b < nb) & (c >= 0) & (c < nc)
                contribution = (wb * wc * cf[9] * sqrtweights[i]) ** 2
                # Out-of-range indices are clipped onto voxels that receive zero.
                target = target.at[
                    jnp.clip(iz, 0, shape[2] - 1),
                    jnp.clip(iy, 0, shape[1] - 1),
                    jnp.clip(ix, 0, shape[0] - 1),
                ].add(jnp.where(valid, contribution, 0.0))
            return target

        return jax.lax.fori_loop(0, max(shape), plane, out)

    return jax.lax.fori_loop(0, coeff.shape[0], view, accumulator)


def weighted_diagonal(
    coeff: jax.Array,
    sqrtweights: jax.Array,
    accumulator: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str,
    interpolation: str = "linear",
) -> jax.Array:
    """Add diag(A.T W A) for one view batch to ``accumulator`` (nz, ny, nx).

    ``coeff`` holds ``plane_coefficients`` rows and ``sqrtweights`` the square
    roots of the pixel weights, shape (views, nv, nu). ``interpolation`` is the
    projector's, so the diagonal covers exactly the voxels it reaches.
    ``backend`` is ``"cuda"`` for the GPU gather or ``"jax"`` for the plane scatter.
    CUDA uses float32 for all buffers and the result, like the projector;
    the JAX scatter retains the accumulator dtype.
    """
    if backend == "jax":
        return _scatter(
            coeff,
            sqrtweights,
            accumulator,
            grid=grid,
            detector=detector,
            interpolation=interpolation,
        )
    if backend != "cuda":
        raise ValueError("weighted diagonal backend must be 'jax' or 'cuda'")
    from .cuda_projector import _joseph_generic as generic

    weights = jnp.square(sqrtweights.astype(jnp.float32))
    return generic.adjoint(
        coeff.astype(jnp.float32),
        weights,
        accumulator.astype(jnp.float32),
        grid,
        detector,
        interpolation=interpolation,
        absolute_weights=False,
        squared=True,
    )
