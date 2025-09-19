from __future__ import annotations

 

import jax
import jax.numpy as jnp

from ..core.geometry import Grid, Detector, Geometry
from ..core.projector import forward_project_view_T
from .filters import get_filter
from ..utils.logging import progress_iter


def _fft_filter_rows(rows: jnp.ndarray, du: float, filter_name: str) -> jnp.ndarray:
    """Filter last axis of (..., nu) rows in frequency domain (rfft/irfft)."""
    nu = rows.shape[-1]
    H = get_filter(filter_name, int(nu), du)
    Hc = jnp.asarray(H, dtype=rows.dtype)
    F = jnp.fft.rfft(rows, axis=-1)
    n_r = F.shape[-1]
    H_r = Hc[:n_r]
    if n_r > 2:
        H_r = H_r.at[1:-1].set(2.0 * H_r[1:-1])
    out = jnp.fft.irfft(F * H_r, n=int(nu), axis=-1)
    return out


def _bp_one(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    filtered: jnp.ndarray,
    *,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
) -> jnp.ndarray:
    """Backproject one view using VJP of the pose-aware forward projector."""
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    zero_vol = jnp.zeros((nx, ny, nz), dtype=jnp.float32)

    def fwd(vol):
        return forward_project_view_T(
            T,
            grid,
            detector,
            vol,
            use_checkpoint=checkpoint_projector,
            unroll=int(projector_unroll),
            gather_dtype=gather_dtype,
        )

    _, vjp = jax.vjp(fwd, zero_vol)
    return vjp(filtered.astype(jnp.float32))[0]


_bp_one_jit = jax.jit(
    _bp_one,
    static_argnames=(
        "grid",
        "detector",
        "projector_unroll",
        "checkpoint_projector",
        "gather_dtype",
    ),
)


def fbp(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    filter_name: str = "ramp",
    scale: float | None = None,
    views_per_batch: int = 0,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
) -> jnp.ndarray:
    """Filtered backprojection for parallel-ray geometry using VJP backproject.

    projections: (n_views, nv, nu) -> volume (nx, ny, nz).
    Memory-safe: filters and backprojects per view-batch.
    """
    proj = jnp.asarray(projections, dtype=jnp.float32)
    n_views, nv, nu = proj.shape
    # Precompute poses once
    T_all = jnp.stack(
        [jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32) for i in range(n_views)],
        axis=0,
    )

    acc = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    b = int(views_per_batch) if int(views_per_batch) > 0 else n_views

    def bp_batch(T_chunk, filt_chunk):
        # Stream over views in the chunk to bound memory instead of materializing
        # a (b, nx, ny, nz) array. This keeps peak memory ~ O(nx*ny*nz).
        def body(accum, inputs):
            T, F = inputs
            bp = _bp_one_jit(
                T,
                grid,
                detector,
                F,
                projector_unroll=projector_unroll,
                checkpoint_projector=checkpoint_projector,
                gather_dtype=gather_dtype,
            )
            return accum + bp, None

        init = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
        acc_chunk, _ = jax.lax.scan(body, init, (T_chunk, filt_chunk))
        return acc_chunk

    # Dynamic backoff on OOM: start with batch size b and shrink on RESOURCE_EXHAUSTED
    total_batches = (n_views + b - 1) // b
    s = 0
    pb_iter = progress_iter(range(total_batches), total=total_batches, desc="FBP: batches")
    for _ in pb_iter:
        cur = min(b, n_views - s)
        T_chunk = T_all[s : s + cur]
        y_chunk = proj[s : s + cur]
        # Filter only the current chunk (reshape rows to 2D)
        rows = y_chunk.reshape((-1, nu))
        rows_f = _fft_filter_rows(rows, du=float(detector.du), filter_name=filter_name)
        filt_chunk = rows_f.reshape((T_chunk.shape[0], nv, nu))
        try:
            acc = acc + bp_batch(T_chunk, filt_chunk)
            s += cur
        except Exception as e:
            msg = str(e)
            if ("RESOURCE_EXHAUSTED" in msg or "Out of memory" in msg) and b > 1:
                # Halve the batch and retry this segment
                b = max(1, b // 2)
                # Recompute total batches for progress bar remaining; leave s unchanged
                remaining = (n_views - s + b - 1) // b
                pb_iter.total = (s // b) + remaining
                continue
            raise

    if scale is None:
        # Heuristic scaling: divide by number of views
        acc = acc / float(n_views)
    else:
        acc = acc * float(scale)
    return acc
