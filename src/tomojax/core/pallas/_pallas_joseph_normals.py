"""Per-view least-squares normal equations with constant per-ray workspace."""

from __future__ import annotations

from functools import partial

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from ._pallas_joseph_derivatives import _ray_derivatives


def _kernel(
    coeff_ref,
    volume_ref,
    target_ref,
    directions_ref,
    partials_ref,
    residual_ref,
    *,
    shape,
    det_shape,
    block,
    parameters,
    interpolation,
):
    u, v, valid, sums = _ray_derivatives(
        coeff_ref, volume_ref, shape, det_shape, block, interpolation
    )
    view = pl.program_id(0)
    gb, gkb, gc, gkc, gw, prediction = sums
    target = plt.load(target_ref.at[view, u, v], mask=valid, other=0.0)
    residual = prediction - target
    plt.store(residual_ref.at[0, jnp.arange(block)], residual, mask=valid)
    columns = []
    for index in range(parameters):

        def dc(coefficient, index=index):
            return plt.load(directions_ref.at[view, jnp.int32(coefficient), jnp.int32(index)])

        # Each coordinate is affine in u,v,k. Five running sums suffice for
        # every direction, independent of the number of voxel planes.
        column = (
            gb * ((dc(1) * u + dc(2) * v) + dc(4))
            + gkb * dc(3)
            + gc * ((dc(5) * u + dc(6) * v) + dc(8))
            + gkc * dc(7)
            + gw * dc(9)
        )
        columns.append(column)
        plt.store(partials_ref.at[0, 0, jnp.int32(index + 1)], jnp.sum(column * residual))
    offset = 1 + parameters
    for row in range(parameters):
        for col in range(row, parameters):
            plt.store(
                partials_ref.at[0, 0, jnp.int32(offset)], jnp.sum(columns[row] * columns[col])
            )
            offset += 1
    plt.store(partials_ref.at[0, 0, jnp.int32(0)], 0.5 * jnp.sum(residual**2))
    # The enclosing reduction reads the padded lanes too.
    while offset < partials_ref.shape[-1]:
        plt.store(partials_ref.at[0, 0, jnp.int32(offset)], jnp.float32(0.0))
        offset += 1


def pose_normal_equations(coeff, volume, target, directions, grid, detector, *, interpolation):
    """Return per-view loss, gradient, Gauss-Newton matrix and residual images."""
    block = 64
    parameters = directions.shape[-1]
    triangle_size = parameters * (parameters + 1) // 2
    width = 1 << (parameters + triangle_size).bit_length()
    views = coeff.shape[0]
    count = detector.nv * detector.nu
    tiles = (count + block - 1) // block
    partials, residual = pl.pallas_call(
        partial(
            _kernel,
            shape=(grid.nx, grid.ny, grid.nz),
            det_shape=(detector.nv, detector.nu),
            block=block,
            parameters=parameters,
            interpolation=interpolation,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((views, tiles, width), jnp.float32),
            jax.ShapeDtypeStruct((views, count), jnp.float32),
        ),
        grid=(views, tiles),
        in_specs=[pl.no_block_spec] * 4,
        out_specs=(
            pl.BlockSpec((1, 1, width), lambda view, tile: (view, tile, 0)),
            pl.BlockSpec((1, block), lambda view, tile: (view, tile)),
        ),
        compiler_params=plt.CompilerParams(num_warps=1),
    )(coeff, volume, target.transpose(0, 2, 1), directions)
    sums = jnp.sum(partials, axis=1)
    upper = sums[:, 1 + parameters : 1 + parameters + triangle_size]
    rows, cols = jnp.triu_indices(parameters)
    matrix = jnp.zeros((views, parameters, parameters), jnp.float32)
    matrix = matrix.at[:, rows, cols].set(upper).at[:, cols, rows].set(upper)
    residual = residual.reshape(views, detector.nu, detector.nv).transpose(0, 2, 1)
    return sums[:, 0], sums[:, 1 : 1 + parameters], matrix, residual
