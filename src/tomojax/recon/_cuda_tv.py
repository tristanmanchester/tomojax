"""Stream-owned CUDA storage for the unchanged finite-iteration TV proximal.

A primal update depends on input within coordinate radius one per dual step.
Overlapping slabs therefore need only ``iterations`` input halo samples. A saved
original-input overlap makes output aliasing safe without a second full volume.
"""

from __future__ import annotations

from functools import cache, partial
import math
from typing import Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core.cone import xla_stream

_SOURCE = r"""
__device__ __forceinline__ float clipped(float x, float lo, float hi) {
    return fminf(fmaxf(x,lo),hi);
}
extern "C" __global__ void primal(
    const float* x, const float* px, const float* py, const float* pz,
    float* u, int nx, int ny, int nz, float lo, float hi)
{
    INDEX t=(INDEX)blockIdx.x*blockDim.x+threadIdx.x;
    INDEX n=(INDEX)nx*ny*nz;
    if(t>=n)return;
    int iz=t%nz, iy=(t/nz)%ny, ix=t/((INDEX)ny*nz);
    float dx=0.f, dy=0.f, dz=0.f;
    if(nx>1) dx=ix==0?px[t]:(ix==nx-1?-px[t-(INDEX)ny*nz]:px[t]-px[t-(INDEX)ny*nz]);
    if(ny>1) dy=iy==0?py[t]:(iy==ny-1?-py[t-nz]:py[t]-py[t-nz]);
    if(nz>1) dz=iz==0?pz[t]:(iz==nz-1?-pz[t-1]:pz[t]-pz[t-1]);
    u[t]=clipped(x[t]+((dx+dy)+dz),lo,hi);
}
extern "C" __global__ void dual(
    const float* u, float* px, float* py, float* pz,
    int nx, int ny, int nz, float tau, const float* lam)
{
    INDEX t=(INDEX)blockIdx.x*blockDim.x+threadIdx.x;
    INDEX n=(INDEX)nx*ny*nz;
    if(t>=n)return;
    int iz=t%nz, iy=(t/nz)%ny, ix=t/((INDEX)ny*nz);
    float v=u[t];
    float gx=ix+1<nx?u[t+(INDEX)ny*nz]-v:0.f;
    float gy=iy+1<ny?u[t+nz]-v:0.f;
    float gz=iz+1<nz?u[t+1]-v:0.f;
    float x=px[t]+tau*gx, y=py[t]+tau*gy, z=pz[t]+tau*gz;
    float radius=fmaxf(lam[0],0.f);
    float scale=fmaxf(fmaxf(fmaxf(fabsf(x),fabsf(y)),fabsf(z)),radius);
    if(scale==0.f)scale=1.f;
    float qx=x/scale, qy=y/scale, qz=z/scale;
    float norm=sqrtf((qx*qx+qy*qy)+qz*qz);
    if(norm==0.f)norm=1.f;
    if(norm<=radius/scale){px[t]=x;py[t]=y;pz[t]=z;}
    else {float length=radius/norm;px[t]=qx*length;py[t]=qy*length;pz[t]=qz*length;}
}
"""


@cache
def _kernels(*, narrow: bool, device: int) -> tuple[Any, Any]:
    import cupy as cp

    with cp.cuda.Device(device):
        module = cp.RawModule(code=_SOURCE.replace("INDEX", "int" if narrow else "long"))
        return module.get_function("primal"), module.get_function("dual")


def _plan(shape: tuple[int, ...], iterations: int) -> tuple[int, int, int]:
    axis = max(range(3), key=shape.__getitem__)
    size, plane = shape[axis], math.prod(shape) // shape[axis]
    # Target bounded scratch, but do not pay more than 50% extra halo work.
    budget = max(
        min(64 << 20, 4 * math.prod(shape)),
        4 * plane * (5 * min(size, 6 * iterations) + min(size, iterations)),
    )
    width = min(size, max(2 * iterations + 1, (budget // (4 * plane) - iterations) // 5))
    core = max(1, width - 2 * iterations) if width < size else size
    return axis, core, min(size, core + 2 * iterations)


def _window(array, axis: int, start: int, stop: int):
    selected = [slice(None)] * 3
    selected[axis] = slice(start, stop)
    return array[tuple(selected)]


def _iterate(
    x, duals, primal, out, lam, *, kernels, iterations: int, tau: float, lower: float, upper: float
) -> None:
    primal_kernel, dual_kernel = kernels
    dims = tuple(np.int32(size) for size in x.shape)
    blocks = (-(-x.size // 256),)
    for _ in range(iterations):
        primal_kernel(blocks, (256,), (x, *duals, primal, *dims, lower, upper))
        dual_kernel(blocks, (256,), (primal, *duals, *dims, tau, lam))
    primal_kernel(blocks, (256,), (x, *duals, out, *dims, lower, upper))


def _tiles(
    source,
    target,
    lam,
    *,
    axis: int,
    core: int,
    width: int,
    kernels,
    iterations: int,
    tau: float,
    lower: float,
    upper: float,
) -> None:
    import cupy as cp

    shape, halo = source.shape, iterations
    plane = math.prod(shape) // shape[axis]
    arena = cp.empty((5, width * plane), cp.float32)
    saved = cp.empty((min(halo, shape[axis]) * plane,), cp.float32)
    alias = source.data.ptr == target.data.ptr
    saved_count = 0
    for first in range(0, shape[axis], core):
        last = min(shape[axis], first + core)
        start, stop = max(0, first - halo), min(shape[axis], last + halo)
        local = list(shape)
        local[axis] = stop - start
        count = math.prod(local)
        tile = arena[0, :count].reshape(local)
        tile[...] = _window(source, axis, start, stop)
        if alias and saved_count:
            left = _window(tile, axis, 0, saved_count)
            left[...] = saved[: left.size].reshape(left.shape)
        if alias:
            keep = _window(tile, axis, max(0, last - halo) - start, last - start)
            saved[: keep.size].reshape(keep.shape)[...] = keep
            saved_count = keep.shape[axis]
        duals = tuple(arena[i, :count] for i in (1, 2, 3))
        for dual in duals:
            dual.fill(0)
        primal = arena[4, :count]
        _iterate(
            tile,
            duals,
            primal,
            primal,
            lam,
            kernels=kernels,
            iterations=iterations,
            tau=tau,
            lower=lower,
            upper=upper,
        )
        _window(target, axis, first, last)[...] = _window(
            primal.reshape(local), axis, first - start, last - start
        )


def _launch(
    context, out, x, lam, *, iterations: int, tau: float, lower: float, upper: float
) -> None:
    import cupy as cp

    with xla_stream(context, out):
        source, target, radius = cp.asarray(x), cp.asarray(out), cp.asarray(lam)
        axis, core, width = _plan(source.shape, iterations)
        count = width * math.prod(source.shape) // source.shape[axis]
        options = {
            "kernels": _kernels(narrow=count < 2**31, device=cp.cuda.Device().id),
            "iterations": iterations,
            "tau": np.float32(tau),
            "lower": np.float32(lower),
            "upper": np.float32(upper),
        }
        if core == source.shape[axis]:
            # No overlap exists: preserve x until the final point-local primal write.
            arena = cp.zeros((4, source.size), cp.float32)
            _iterate(source, tuple(arena[i] for i in range(3)), arena[3], target, radius, **options)
        else:
            _tiles(source, target, radius, axis=axis, core=core, width=width, **options)


def proximal(
    x: jax.Array, lam: jax.Array, *, iterations: int, tau: float, lower: float, upper: float
) -> jax.Array:
    """Apply the finite-iteration TV update, borrowing or owning ``x`` as XLA permits."""
    call = buffer_callback(
        partial(_launch, iterations=iterations, tau=tau, lower=lower, upper=upper),
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        input_output_aliases={0: 0},
        vmap_method="sequential",
    )
    return jax.lax.cond(lam > 0, lambda _: call(x, lam), lambda _: jnp.clip(x, lower, upper), None)
