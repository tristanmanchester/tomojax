"""Joseph derivatives with constant per-ray state, without a sample tape.

The coefficient pullback recomputes the four or sixteen samples at each plane. Its five
running sums suffice because each coordinate is affine in detector u, v and
plane k. Only tile reductions, not ray-by-plane intermediates, reach memory.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from tomojax.core.joseph import select_axis

from ._pallas_joseph import adjoint_pallas
from ._pallas_plane_sampling import cubic_sample

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid


def _samples(volume_ref, axis, k, ib, ic, nb, nc, valid):
    def sample(b, c):
        ix = select_axis(axis, k, c, b)
        iy = select_axis(axis, b, k, c)
        iz = select_axis(axis, c, b, k)
        mask = valid & (b >= 0) & (b < nb) & (c >= 0) & (c < nc)
        return plt.load(volume_ref.at[ix, iy, iz], mask=mask, other=0.0)

    return sample(ib, ic), sample(ib, ic + 1), sample(ib + 1, ic), sample(ib + 1, ic + 1)


def _ray_derivatives(coeff_ref, volume_ref, shape, det_shape, block, interpolation):
    nx, ny, nz = shape
    nv, nu = det_shape
    view = pl.program_id(0)
    p = pl.program_id(1) * block + jnp.arange(block, dtype=jnp.int32)
    u, v, valid = p // nv, p % nv, p < nv * nu

    def cf(i):
        return plt.load(coeff_ref.at[view, jnp.int32(i)])

    axis = cf(0).astype(jnp.int32)
    ub, vb, kb, cb, uc, vc, kc, cc, weight = [cf(i) for i in range(1, 10)]
    na = select_axis(axis, nx, ny, nz)
    nb = select_axis(axis, ny, nz, nx)
    nc = select_axis(axis, nz, nx, ny)

    def step(k, accum):
        qb = (ub * u + vb * v) + (kb * k + cb)
        qc = (uc * u + vc * v) + (kc * k + cc)
        ib, ic = jnp.floor(qb).astype(jnp.int32), jnp.floor(qc).astype(jnp.int32)
        wb, wc = qb - ib, qc - ic
        if interpolation == "linear":
            f00, f01, f10, f11 = _samples(volume_ref, axis, k, ib, ic, nb, nc, valid)
            low, high = f00 * (1 - wc) + f01 * wc, f10 * (1 - wc) + f11 * wc
            value = low * (1 - wb) + high * wb
            db = (high - low) * weight
            dc = ((f01 - f00) * (1 - wb) + (f11 - f10) * wb) * weight
        else:
            value, db, dc = cubic_sample(volume_ref, axis, k, qb, qc, nb, nc, valid)
            db, dc = db * weight, dc * weight
        gb, gkb, gc, gkc, gw, prediction = accum
        return (
            gb + db,
            gkb + k * db,
            gc + dc,
            gkc + k * dc,
            gw + value,
            prediction + value * weight,
        )

    zero = jnp.zeros((block,), jnp.float32)
    sums = jax.lax.fori_loop(0, na, step, (zero,) * 6)
    return u, v, valid, sums


def _store_partials(partials_ref, u, v, derivatives, cotangent, loss):
    gb, gkb, gc, gkc, gw = (value * cotangent for value in derivatives)
    zero = jnp.float32(0.0)
    # The discrete dominant axis and the inverse map do not affect the forward
    # map within a fixed branch. Their coefficient derivatives are exactly zero.
    values = (
        zero,
        jnp.sum(gb * u),
        jnp.sum(gb * v),
        jnp.sum(gkb),
        jnp.sum(gb),
        jnp.sum(gc * u),
        jnp.sum(gc * v),
        jnp.sum(gkc),
        jnp.sum(gc),
        jnp.sum(gw),
        zero,
        zero,
        zero,
        zero,
        zero,
        loss,
    )
    for i, value in enumerate(values):
        plt.store(partials_ref.at[0, 0, jnp.int32(i)], value)


def _vjp_kernel(
    coeff_ref, volume_ref, cotangent_ref, partials_ref, *, shape, det_shape, block, interpolation
):
    u, v, valid, sums = _ray_derivatives(
        coeff_ref, volume_ref, shape, det_shape, block, interpolation
    )
    cot = plt.load(cotangent_ref.at[pl.program_id(0), u, v], mask=valid, other=0.0)
    _store_partials(partials_ref, u, v, sums[:5], cot, jnp.float32(0.0))


def _loss_kernel(
    coeff_ref,
    volume_ref,
    target_ref,
    partials_ref,
    residual_ref,
    *,
    shape,
    det_shape,
    block,
    interpolation,
):
    u, v, valid, sums = _ray_derivatives(
        coeff_ref, volume_ref, shape, det_shape, block, interpolation
    )
    target = plt.load(target_ref.at[pl.program_id(0), u, v], mask=valid, other=0.0)
    residual = sums[5] - target
    plt.store(residual_ref.at[0, jnp.arange(block)], residual, mask=valid)
    _store_partials(partials_ref, u, v, sums[:5], residual, 0.5 * jnp.sum(residual**2))


def coefficient_vjp(
    coeff: jax.Array,
    volume: jax.Array,
    cotangent: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
) -> jax.Array:
    """Pull a sinogram cotangent back to the 14 prepared coefficients per view."""
    block = 128
    tiles = (detector.nu * detector.nv + block - 1) // block
    partials = pl.pallas_call(
        partial(
            _vjp_kernel,
            shape=(grid.nx, grid.ny, grid.nz),
            det_shape=(detector.nv, detector.nu),
            block=block,
            interpolation=interpolation,
        ),
        out_shape=jax.ShapeDtypeStruct((coeff.shape[0], tiles, 16), jnp.float32),
        grid=(coeff.shape[0], tiles),
        in_specs=[pl.no_block_spec] * 3,
        out_specs=pl.BlockSpec((1, 1, 16), lambda view, tile: (view, tile, 0)),
        compiler_params=plt.CompilerParams(num_warps=4),
    )(coeff, volume, cotangent.transpose(0, 2, 1))
    return jnp.sum(partials, axis=1)[:, :14]


def l2_loss_and_grads(
    coeff: jax.Array,
    volume: jax.Array,
    target: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fuse half squared error and coefficient gradients, then transpose residuals.

    Return the scalar loss, coefficient gradient and matched volume gradient.
    This explicit first-order operation does not define higher derivatives.
    """
    block = 128
    views = coeff.shape[0]
    count = detector.nu * detector.nv
    tiles = (count + block - 1) // block
    partials, residual = pl.pallas_call(
        partial(
            _loss_kernel,
            shape=(grid.nx, grid.ny, grid.nz),
            det_shape=(detector.nv, detector.nu),
            block=block,
            interpolation=interpolation,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((views, tiles, 16), jnp.float32),
            jax.ShapeDtypeStruct((views, count), jnp.float32),
        ),
        grid=(views, tiles),
        in_specs=[pl.no_block_spec] * 3,
        out_specs=(
            pl.BlockSpec((1, 1, 16), lambda view, tile: (view, tile, 0)),
            pl.BlockSpec((1, block), lambda view, tile: (view, tile)),
        ),
        compiler_params=plt.CompilerParams(num_warps=4),
    )(coeff, volume, target.transpose(0, 2, 1))
    sums = jnp.sum(partials, axis=1)
    residual = residual.reshape(views, detector.nu, detector.nv).transpose(0, 2, 1)
    return (
        jnp.sum(sums[:, 15]),
        sums[:, :14],
        adjoint_pallas(coeff, residual, grid, detector, interpolation=interpolation),
    )


def _jvp_kernel(
    coeff_ref,
    volume_ref,
    dcoeff_ref,
    dvolume_ref,
    out_ref,
    *,
    shape,
    det_shape,
    block,
    has_dcoeff,
    has_dvolume,
    interpolation,
):
    nx, ny, nz = shape
    nv, nu = det_shape
    view = pl.program_id(0)
    p = pl.program_id(1) * block + jnp.arange(block, dtype=jnp.int32)
    u, v, valid = p // nv, p % nv, p < nv * nu

    def cf(i):
        return plt.load(coeff_ref.at[view, jnp.int32(i)])

    def dcf(i):
        return plt.load(dcoeff_ref.at[view, jnp.int32(i)]) if has_dcoeff else jnp.float32(0.0)

    axis = cf(0).astype(jnp.int32)
    ub, vb, kb, cb, uc, vc, kc, cc, weight = [cf(i) for i in range(1, 10)]
    dub, dvb, dkb, dcb, duc, dvc, dkc, dcc, dweight = [dcf(i) for i in range(1, 10)]
    na = select_axis(axis, nx, ny, nz)
    nb = select_axis(axis, ny, nz, nx)
    nc = select_axis(axis, nz, nx, ny)

    def step(k, total):
        qb = (ub * u + vb * v) + (kb * k + cb)
        qc = (uc * u + vc * v) + (kc * k + cc)
        ib, ic = jnp.floor(qb).astype(jnp.int32), jnp.floor(qc).astype(jnp.int32)
        wb, wc = qb - ib, qc - ic
        tangent = jnp.zeros((block,), jnp.float32)
        if has_dcoeff:
            if interpolation == "linear":
                f00, f01, f10, f11 = _samples(volume_ref, axis, k, ib, ic, nb, nc, valid)
                low, high = f00 * (1 - wc) + f01 * wc, f10 * (1 - wc) + f11 * wc
                value = low * (1 - wb) + high * wb
                grad_b = high - low
                grad_c = (f01 - f00) * (1 - wb) + (f11 - f10) * wb
            else:
                value, grad_b, grad_c = cubic_sample(volume_ref, axis, k, qb, qc, nb, nc, valid)
            db = (dub * u + dvb * v) + (dkb * k + dcb)
            dc = (duc * u + dvc * v) + (dkc * k + dcc)
            tangent = (grad_b * db + grad_c * dc) * weight + value * dweight
        if has_dvolume:
            if interpolation == "linear":
                f00, f01, f10, f11 = _samples(dvolume_ref, axis, k, ib, ic, nb, nc, valid)
                value = (f00 * (1 - wc) + f01 * wc) * (1 - wb) + (f10 * (1 - wc) + f11 * wc) * wb
            else:
                value, _, _ = cubic_sample(dvolume_ref, axis, k, qb, qc, nb, nc, valid)
            tangent = tangent + value * weight
        return total + tangent

    output = jax.lax.fori_loop(0, na, step, jnp.zeros((block,), jnp.float32))
    plt.store(out_ref.at[0, jnp.arange(block)], output, mask=valid)


def projection_jvp(
    coeff: jax.Array,
    volume: jax.Array,
    dcoeff: jax.Array | None,
    dvolume: jax.Array | None,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
) -> jax.Array:
    """Apply a joint coefficient/volume tangent without saving plane samples."""
    block = 128
    count = detector.nu * detector.nv
    views = coeff.shape[0]
    output = pl.pallas_call(
        partial(
            _jvp_kernel,
            shape=(grid.nx, grid.ny, grid.nz),
            det_shape=(detector.nv, detector.nu),
            block=block,
            interpolation=interpolation,
            has_dcoeff=dcoeff is not None,
            has_dvolume=dvolume is not None,
        ),
        out_shape=jax.ShapeDtypeStruct((views, count), jnp.float32),
        grid=(views, (count + block - 1) // block),
        in_specs=[pl.no_block_spec] * 4,
        out_specs=pl.BlockSpec((1, block), lambda view, tile: (view, tile)),
        compiler_params=plt.CompilerParams(num_warps=4),
    )(coeff, volume, coeff if dcoeff is None else dcoeff, volume if dvolume is None else dvolume)
    return output.reshape(views, detector.nu, detector.nv).transpose(0, 2, 1)
