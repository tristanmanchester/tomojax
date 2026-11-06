"""The diagonal of A.T W A for the linear Joseph plane projector.

Each voxel's entry is the sum over views and detector pixels of its squared
bilinear plane weight times the pixel's weight squared. This is a Jacobi
preconditioner input, not a transpose: the matched forward and transpose stay
the projector's own. CPU scatters one plane at a time; CUDA gathers with one
owner per voxel and no atomics.
"""

from __future__ import annotations

from functools import cache, partial
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from .joseph import select_axis

if TYPE_CHECKING:
    from .geometry import Detector, Grid

_SOURCE = r"""
extern "C" __global__ void weighted_diagonal(
    const float* cf, const float* sw, float* out,
    int views, int nx, int ny, int nz, int nu, int nv) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long size = (long long)nx * ny * nz;
    if (index >= size) return;
    int x = index % nx;
    int y = (index / nx) % ny;
    int z = index / ((long long)nx * ny);
    float total = 0.0f;
    for (int i = 0; i < views; ++i) {
        const float* c = cf + 14 * i;
        int axis = (int)c[0];
        float a = (float)(axis == 0 ? x : (axis == 1 ? y : z));
        float b = (float)(axis == 0 ? y : (axis == 1 ? z : x));
        float d = (float)(axis == 0 ? z : (axis == 1 ? x : y));
        float b0 = b - (c[3] * a + c[4]);
        float d0 = d - (c[7] * a + c[8]);
        float u0 = c[10] * b0 + c[11] * d0;
        float v0 = c[12] * b0 + c[13] * d0;
        float ru = fabsf(c[10]) + fabsf(c[11]);
        float rv = fabsf(c[12]) + fabsf(c[13]);
        // A conservative pixel box from the inverse map; the forward
        // coordinate arithmetic below decides actual support.
        int ulo = max(0, (int)floorf(u0 - ru) - 2);
        int uhi = min(nu - 1, (int)ceilf(u0 + ru) + 2);
        int vlo = max(0, (int)floorf(v0 - rv) - 2);
        int vhi = min(nv - 1, (int)ceilf(v0 + rv) + 2);
        for (int v = vlo; v <= vhi; ++v) for (int u = ulo; u <= uhi; ++u) {
            float qb = (c[1] * (float)u + c[2] * (float)v) + (c[3] * a + c[4]);
            float qc = (c[5] * (float)u + c[6] * (float)v) + (c[7] * a + c[8]);
            float wb = 1.0f - fabsf(qb - b);
            float wc = 1.0f - fabsf(qc - d);
            if (wb <= 0.0f || wc <= 0.0f) continue;
            float w = sw[((long long)i * nv + v) * nu + u] * c[9] * wb * wc;
            total += w * w;
        }
    }
    out[index] += total;
}
"""


@cache
def _kernel() -> Any:
    import cupy as cp

    return cp.RawModule(code=_SOURCE, options=("--fmad=false",)).get_function("weighted_diagonal")


def _launch(context, out, coeff, sqrtweights, initial, *, grid, detector):
    import cupy as cp

    from ._cuda_joseph import xla_stream

    with xla_stream(context, out):
        target, old = cp.asarray(out), cp.asarray(initial)
        if target.data.ptr != old.data.ptr:
            target[...] = old
        size = grid.nx * grid.ny * grid.nz
        _kernel()(
            ((size + 127) // 128,),
            (128,),
            (
                cp.asarray(coeff),
                cp.asarray(sqrtweights),
                target,
                np.int32(coeff.shape[0]),
                np.int32(grid.nx),
                np.int32(grid.ny),
                np.int32(grid.nz),
                np.int32(detector.nu),
                np.int32(detector.nv),
            ),
        )


def _scatter(coeff, sqrtweights, accumulator, *, grid, detector):
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
            wb, wc = qb - ib, qc - ic
            for db, dc in ((0, 0), (1, 0), (0, 1), (1, 1)):
                b, c = ib + db, ic + dc
                ix = select_axis(axis, k, c, b)
                iy = select_axis(axis, b, k, c)
                iz = select_axis(axis, c, b, k)
                valid = (k < na) & (b >= 0) & (b < nb) & (c >= 0) & (c < nc)
                w = (wb if db else 1 - wb) * (wc if dc else 1 - wc) * cf[9]
                contribution = (w * sqrtweights[i]) ** 2
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
) -> jax.Array:
    """Add diag(A.T W A) for one view batch to ``accumulator`` (nz, ny, nx).

    ``coeff`` holds ``plane_coefficients`` rows and ``sqrtweights`` the square
    roots of the pixel weights, shape (views, nv, nu). ``backend`` is ``"cuda"``
    for the voxel-owned GPU gather or ``"jax"`` for the plane scatter.
    """
    if backend == "jax":
        return _scatter(coeff, sqrtweights, accumulator, grid=grid, detector=detector)
    if backend != "cuda":
        raise ValueError("weighted diagonal backend must be 'jax' or 'cuda'")
    call = buffer_callback(
        partial(_launch, grid=grid, detector=detector),
        jax.ShapeDtypeStruct(accumulator.shape, jnp.float32),
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )
    return call(coeff.astype(jnp.float32), sqrtweights, accumulator)
