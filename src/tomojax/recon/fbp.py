from __future__ import annotations

from collections import OrderedDict
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp

from ..core.geometry import Grid, Detector, Geometry
from ..core.projector import forward_project_view_T
from .filters import get_filter_np
from ..utils.logging import progress_iter

_RFFT_FILTER_CACHE: "OrderedDict[Tuple[str, int, float, str], np.ndarray]" = OrderedDict()
_RFFT_FILTER_CACHE_CAP = 8


def _default_fbp_scale(n_views: int) -> float:
    """Return the default angular weighting for the current parallel-ray FBP.

    TomoJAX's built-in CT and laminography geometries both use parallel rays,
    so the discrete filtered backprojection sum should be weighted by the
    180-degree angular spacing ``pi / n_views``. Callers with custom angular
    coverage can override this via ``scale=...``.
    """
    if int(n_views) <= 0:
        raise ValueError("n_views must be positive")
    return float(np.pi / float(n_views))


def _get_rfft_filter_cached(filter_name: str, nu: int, du: float, dtype: jnp.dtype) -> jnp.ndarray:
    """Return the one-sided RFFT filter H_r cached by key.

    `jax.numpy.fft.rfft` already returns the non-redundant half-spectrum. Its
    interior bins should be multiplied by the one-sided filter values directly;
    doubling those coefficients would incorrectly double-count positive-frequency
    energy during `irfft` reconstruction.

    - Key: (name, nu, du, dtype)
    - Returns a JAX array of shape (nu//2 + 1,) with dtype matching rows dtype.
    """
    key = (str(filter_name).lower(), int(nu), float(du), str(dtype))
    if key in _RFFT_FILTER_CACHE:
        _RFFT_FILTER_CACHE.move_to_end(key)
        Hr_np = _RFFT_FILTER_CACHE[key]
    else:
        H_np = get_filter_np(filter_name, int(nu), float(du))  # np.float32 length nu
        n_r = int(nu) // 2 + 1
        Hr_np = H_np[:n_r].astype(np.float32, copy=False)
        # Promote to float64 if rows dtype requests it
        if str(dtype) in ("float64", "complex128"):
            Hr_np = Hr_np.astype(np.float64, copy=False)
        _RFFT_FILTER_CACHE[key] = Hr_np
        if len(_RFFT_FILTER_CACHE) > _RFFT_FILTER_CACHE_CAP:
            _RFFT_FILTER_CACHE.popitem(last=False)
    return jnp.asarray(Hr_np, dtype=dtype)


def _fft_filter_rows(rows: jnp.ndarray, du: float, filter_name: str) -> jnp.ndarray:
    """Filter last axis of (..., nu) rows in frequency domain (rfft/irfft)."""
    nu = int(rows.shape[-1])
    H_r = _get_rfft_filter_cached(filter_name, nu, du, rows.dtype)
    F = jnp.fft.rfft(rows, axis=-1)
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
    views_per_batch: int = 1,
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
    requested_b = int(views_per_batch) if int(views_per_batch) > 0 else n_views
    b = max(1, min(requested_b, n_views))

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

    fft_filter_rows_jit = jax.jit(
        lambda rows: _fft_filter_rows(rows, du=float(detector.du), filter_name=filter_name)
    )
    bp_batch_jit = jax.jit(bp_batch)

    # Dynamic backoff on OOM: start with batch size b and shrink on
    # memory pressure. Progress is tracked per-view so retries do not prematurely
    # exhaust a fixed batch iterator.
    s = 0
    view_progress = iter(progress_iter(range(n_views), total=n_views, desc="FBP: views"))
    while s < n_views:
        cur = min(b, n_views - s)
        T_chunk = T_all[s : s + cur]
        y_chunk = proj[s : s + cur]
        try:
            pad_views = b - cur
            if pad_views:
                T_pad = jnp.repeat(T_chunk[-1:], pad_views, axis=0)
                y_pad = jnp.zeros((pad_views, nv, nu), dtype=y_chunk.dtype)
                T_chunk = jnp.concatenate((T_chunk, T_pad), axis=0)
                y_chunk = jnp.concatenate((y_chunk, y_pad), axis=0)

            # Pad the tail chunk and mask its extra views so JAX only needs to
            # compile once per active batch size.
            valid_mask = (jnp.arange(b) < cur)[:, None, None]
            # Filter only the current chunk (reshape rows to 2D)
            rows = y_chunk.reshape((b * nv, nu))
            rows_f = fft_filter_rows_jit(rows)
            filt_chunk = rows_f.reshape((b, nv, nu))
            filt_chunk = jnp.where(valid_mask, filt_chunk, 0.0)
            acc = acc + bp_batch_jit(T_chunk, filt_chunk)
            s += cur
            for _ in range(cur):
                next(view_progress, None)
        except Exception as e:
            msg = str(e).lower()
            if ("resource_exhausted" in msg or "out of memory" in msg) and b > 1:
                # Halve the batch and retry this segment without advancing s.
                b = max(1, b // 2)
                continue
            raise

    if scale is None:
        acc = acc * _default_fbp_scale(n_views)
    else:
        acc = acc * float(scale)
    return acc
