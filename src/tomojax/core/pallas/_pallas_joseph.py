"""CUDA plane sampling and its gather transpose, with no scattered atomics.

The detector is stored internally with v contiguous, matching the volume's
contiguous z coordinate. The adjoint inverts each plane's affine ray map to
bound its footprint, then evaluates the same interpolation weights as the forward.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from tomojax.core._plane_interpolation import cubic_weight, validate_interpolation
from tomojax.core.joseph import select_axis as _select

from ._pallas_plane_sampling import cubic_sample

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid


def _fp_kernel(
    coeff_ref: Any,
    volume_ref: Any,
    output_ref: Any,
    *,
    shape: tuple[int, int, int],
    det_shape: tuple[int, int],
    block: int,
    interpolation: str = "linear",
) -> None:
    nx, ny, nz = shape
    nv, nu = det_shape
    view = pl.program_id(0)
    tile = pl.program_id(1)
    p = tile * block + jnp.arange(block, dtype=jnp.int32)
    u = p // nv
    v = p % nv
    valid = p < nv * nu

    def cf(i):
        return plt.load(coeff_ref.at[view, jnp.int32(i)])

    a = cf(0).astype(jnp.int32)
    ub, vb, kb, cb, uc, vc, kc, cc, weight = [cf(i) for i in range(1, 10)]
    na = _select(a, nx, ny, nz)
    nb = _select(a, ny, nz, nx)
    nc = _select(a, nz, nx, ny)

    def step(k, accum):
        qb = (ub * u + vb * v) + (kb * k + cb)
        qc = (uc * u + vc * v) + (kc * k + cc)
        ib = jnp.floor(qb).astype(jnp.int32)
        ic = jnp.floor(qc).astype(jnp.int32)
        wb = qb - ib
        wc = qc - ic

        def sample(b, c):
            ix = _select(a, k, c, b)
            iy = _select(a, b, k, c)
            iz = _select(a, c, b, k)
            mask = valid & (b >= 0) & (b < nb) & (c >= 0) & (c < nc)
            return plt.load(volume_ref.at[ix, iy, iz], mask=mask, other=0.0)

        if interpolation == "linear":
            val = (sample(ib, ic) * (1 - wc) + sample(ib, ic + 1) * wc) * (1 - wb) + (
                sample(ib + 1, ic) * (1 - wc) + sample(ib + 1, ic + 1) * wc
            ) * wb
        else:
            val, _, _ = cubic_sample(volume_ref, a, k, qb, qc, nb, nc, valid)
        return accum + val * weight

    output = jax.lax.fori_loop(0, na, step, jnp.zeros((block,), jnp.float32))
    plt.store(output_ref.at[0, jnp.arange(block)], output, mask=valid)


def forward_pallas(
    coeff: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    block: int = 128,
    interpret: bool = False,
    interpolation: str = "linear",
) -> jax.Array:
    """Project voxel-centre planes with the selected interpolation and masked loads."""
    validate_interpolation(interpolation)
    views = coeff.shape[0]
    count = detector.nu * detector.nv
    out = pl.pallas_call(
        partial(
            _fp_kernel,
            shape=(grid.nx, grid.ny, grid.nz),
            det_shape=(detector.nv, detector.nu),
            block=block,
            interpolation=interpolation,
        ),
        out_shape=jax.ShapeDtypeStruct((views, count), jnp.float32),
        grid=(views, (count + block - 1) // block),
        in_specs=[pl.no_block_spec, pl.no_block_spec],
        out_specs=pl.BlockSpec((1, block), lambda view, tile: (view, tile)),
        compiler_params=plt.CompilerParams(num_warps=4),
        interpret=interpret,
    )(coeff, volume)
    return out.reshape(views, detector.nu, detector.nv).transpose(0, 2, 1)


def _bp_kernel(
    coeff_ref: Any,
    images_ref: Any,
    out_ref: Any,
    *,
    shape: tuple[int, int, int],
    det_shape: tuple[int, int],
    nviews: int,
    block: int,
    interpolation: str = "linear",
    absolute_weights: bool = False,
) -> None:
    nx, ny, nz = shape
    nv, nu = det_shape
    p = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    valid = p < nx * ny * nz
    ix = p // (ny * nz)
    iy = (p // nz) % ny
    iz = p % nz

    def view_step(view, accum):
        def cf(i):
            return plt.load(coeff_ref.at[view, jnp.int32(i)])

        a = cf(0).astype(jnp.int32)
        ub, vb, kb, cb, uc, vc, kc, cc, weight, iu_b, iu_c, iv_b, iv_c = [
            cf(i) for i in range(1, 14)
        ]
        k = _select(a, ix, iy, iz)
        b = _select(a, iy, iz, ix)
        c = _select(a, iz, ix, iy)
        hb = kb * k + cb
        hc = kc * k + cc
        center_u = iu_b * (b - hb) + iu_c * (c - hc)
        center_v = iv_b * (b - hb) + iv_c * (c - hc)
        radius = 1 if interpolation == "linear" else 2
        ru = radius * (jnp.abs(iu_b) + jnp.abs(iu_c))
        rv = radius * (jnp.abs(iv_b) + jnp.abs(iv_c))
        start_u = jnp.floor(center_u - ru).astype(jnp.int32) + 1
        start_v = jnp.floor(center_v - rv).astype(jnp.int32) + 1
        count_u = jnp.max(jnp.ceil(center_u + ru).astype(jnp.int32) - start_u)
        count_v = jnp.max(jnp.ceil(center_v + rv).astype(jnp.int32) - start_v)

        # Keep u outside the v loop: dividing a flattened dynamic loop index by
        # count_v otherwise adds integer division/modulo to every detector load.
        # The footprint and accumulation order are unchanged.
        def gather_u(iu, subtotal):
            u = start_u + iu

            def gather_v(iv, total):
                v = start_v + iv
                qb = (ub * u + vb * v) + hb
                qc = (uc * u + vc * v) + hc
                if interpolation == "linear":
                    w = (
                        jnp.maximum(1 - jnp.abs(qb - b), 0.0)
                        * jnp.maximum(1 - jnp.abs(qc - c), 0.0)
                        * weight
                    )
                else:
                    w = cubic_weight(qb - b) * cubic_weight(qc - c) * weight
                if absolute_weights:
                    w = jnp.abs(w)
                mask = valid & (u >= 0) & (u < nu) & (v >= 0) & (v < nv) & (w != 0)
                val = plt.load(images_ref.at[view, u, v], mask=mask, other=0.0)
                return total + val * w

            return jax.lax.fori_loop(0, count_v, gather_v, subtotal)

        return jax.lax.fori_loop(0, count_u, gather_u, accum)

    output = jax.lax.fori_loop(0, nviews, view_step, jnp.zeros((block,), jnp.float32))
    plt.store(out_ref, output, mask=valid)


def adjoint_pallas(
    coeff: jax.Array,
    images: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    block: int = 32,
    interpret: bool = False,
    interpolation: str = "linear",
    absolute_weights: bool = False,
) -> jax.Array:
    """Gather weights per voxel; optionally use their magnitudes for error bounds."""
    validate_interpolation(interpolation)
    count = grid.nx * grid.ny * grid.nz
    out = pl.pallas_call(
        partial(
            _bp_dispatch_kernel,
            shape=(grid.nx, grid.ny, grid.nz),
            det_shape=(detector.nv, detector.nu),
            nviews=coeff.shape[0],
            block=block,
            interpolation=interpolation,
            absolute_weights=absolute_weights,
        ),
        out_shape=jax.ShapeDtypeStruct((count,), jnp.float32),
        grid=((count + block - 1) // block,),
        in_specs=[pl.no_block_spec, pl.no_block_spec, pl.no_block_spec],
        out_specs=pl.BlockSpec((block,), lambda tile: (tile,)),
        # A single warp keeps the footprint maxima within a warp and avoids
        # cross-warp synchronization on every view. Larger explicit tiles retain
        # the original four-warp launch configuration.
        compiler_params=plt.CompilerParams(num_warps=1 if block == 32 else 4),
        interpret=interpret,
    )(coeff, jnp.transpose(images, (0, 2, 1)), _has_unit_rows(coeff))
    return out.reshape((grid.nx, grid.ny, grid.nz))


def _has_unit_rows(coeff: jax.Array) -> jax.Array:
    # Test the actual prepared map exactly, not a tolerance on nominal poses:
    # tiny tilts or row-spacing changes must retain their general interpolation.
    diagonal = (coeff[:, 2] == 0) & (coeff[:, 5] == 0) & (jnp.abs(coeff[:, 6]) == 1)
    antidiagonal = (coeff[:, 1] == 0) & (coeff[:, 6] == 0) & (jnp.abs(coeff[:, 2]) == 1)
    return jnp.all(diagonal | antidiagonal)


def _bp_unit_rows_kernel(
    coeff_ref,
    images_ref,
    out_ref,
    *,
    shape: tuple[int, int, int],
    det_shape: tuple[int, int],
    nviews: int,
    block: int,
    interpolation: str = "linear",
    absolute_weights: bool = False,
) -> None:
    nx, ny, nz = shape
    nv, nu = det_shape
    p = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    valid = p < nx * ny * nz
    ix, iy, iz = p // (ny * nz), (p // nz) % ny, p % nz

    def view_step(view, accum):
        def cf(i):
            return plt.load(coeff_ref.at[view, jnp.int32(i)])

        axis = cf(0).astype(jnp.int32)
        ub, vb, kb, cb, uc, vc, kc, cc, weight, iu_b, iu_c = [cf(i) for i in range(1, 12)]
        k, b, c = _select(axis, ix, iy, iz), _select(axis, iy, iz, ix), _select(axis, iz, ix, iy)
        hb, hc = kb * k + cb, kc * k + cc
        center_u = iu_b * (b - hb) + iu_c * (c - hc)
        radius = 1 if interpolation == "linear" else 2
        ru = radius * (jnp.abs(iu_b) + jnp.abs(iu_c))
        start_u = jnp.floor(center_u - ru).astype(jnp.int32) + 1
        count_u = jnp.max(jnp.ceil(center_u + ru).astype(jnp.int32) - start_u)

        diagonal = (vb == 0) & (uc == 0) & (jnp.abs(vc) == 1)
        u_scale, u_shift, u_target = (
            jnp.where(diagonal, ub, uc),
            jnp.where(diagonal, hb, hc),
            jnp.where(diagonal, b, c),
        )
        v_scale, v_shift, v_target = (
            jnp.where(diagonal, vc, vb),
            jnp.where(diagonal, hc, hb),
            jnp.where(diagonal, c, b),
        )
        center_v = (v_target - v_shift) * v_scale
        v0 = jnp.floor(center_v).astype(jnp.int32) - (radius - 1)
        rows = tuple(v0 + i for i in range(2 * radius))
        vertical = tuple(
            jnp.maximum(1 - jnp.abs((v_scale * v + v_shift) - v_target), 0.0)
            if interpolation == "linear"
            else cubic_weight((v_scale * v + v_shift) - v_target)
            for v in rows
        )

        def gather_u(i, total):
            u = start_u + i
            delta = (u_scale * u + u_shift) - u_target
            wu = (
                jnp.maximum(1 - jnp.abs(delta), 0.0)
                if interpolation == "linear"
                else cubic_weight(delta)
            )
            # Vertical weights are independent of u. Keep their physical
            # b/c multiplication order and accumulate the same ascending rows.
            mask = valid & (u >= 0) & (u < nu)
            for v, wv in zip(rows, vertical, strict=True):
                combined = jnp.where(diagonal, wu * wv, wv * wu) * weight
                if absolute_weights:
                    combined = jnp.abs(combined)
                value = plt.load(
                    images_ref.at[view, u, v],
                    mask=mask & (v >= 0) & (v < nv) & (combined != 0),
                    other=0.0,
                )
                total = total + value * combined
            return total

        return jax.lax.fori_loop(0, count_u, gather_u, accum)

    result = jax.lax.fori_loop(0, nviews, view_step, jnp.zeros((block,), jnp.float32))
    plt.store(out_ref, result, mask=valid)


def _bp_dispatch_kernel(
    coeff_ref,
    images_ref,
    unit_rows_ref,
    out_ref,
    *,
    shape,
    det_shape,
    nviews,
    block,
    interpolation="linear",
    absolute_weights=False,
) -> None:
    def unit_rows():
        _bp_unit_rows_kernel(
            coeff_ref,
            images_ref,
            out_ref,
            shape=shape,
            det_shape=det_shape,
            nviews=nviews,
            block=block,
            interpolation=interpolation,
            absolute_weights=absolute_weights,
        )

    def general():
        _bp_kernel(
            coeff_ref,
            images_ref,
            out_ref,
            shape=shape,
            det_shape=det_shape,
            nviews=nviews,
            block=block,
            interpolation=interpolation,
            absolute_weights=absolute_weights,
        )

    # Select once per voxel tile. A per-view branch adds needless work to every
    # tilted view; reducing the exact predicate once also avoids host transfers.
    jax.lax.cond(plt.load(unit_rows_ref.at[()]), unit_rows, general)
