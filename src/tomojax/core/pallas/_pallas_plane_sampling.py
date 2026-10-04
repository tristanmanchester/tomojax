"""Cubic plane samples and their coordinate derivatives for CUDA kernels."""

from __future__ import annotations

from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from tomojax.core._plane_interpolation import cubic_weights_and_derivatives
from tomojax.core.joseph import select_axis


def cubic_sample(volume_ref, axis, k, qb, qc, nb, nc, valid):
    ib, ic = jnp.floor(qb).astype(jnp.int32), jnp.floor(qc).astype(jnp.int32)
    wb, db = cubic_weights_and_derivatives(qb - ib)
    wc, dc = cubic_weights_and_derivatives(qc - ic)
    value = grad_b = grad_c = jnp.zeros_like(qb)
    for row in range(4):
        b = ib + row - 1
        row_value = row_gradient = jnp.zeros_like(qb)
        for col in range(4):
            c = ic + col - 1
            ix = select_axis(axis, k, c, b)
            iy = select_axis(axis, b, k, c)
            iz = select_axis(axis, c, b, k)
            mask = valid & (b >= 0) & (b < nb) & (c >= 0) & (c < nc)
            sample = plt.load(volume_ref.at[ix, iy, iz], mask=mask, other=0.0)
            row_value = row_value + sample * wc[col]
            row_gradient = row_gradient + sample * dc[col]
        value = value + row_value * wb[row]
        grad_b = grad_b + row_value * db[row]
        grad_c = grad_c + row_gradient * wb[row]
    return value, grad_b, grad_c
