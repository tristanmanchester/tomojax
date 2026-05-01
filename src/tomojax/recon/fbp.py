from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Iterator, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from ..core.geometry.base import Grid, Detector, Geometry, _grid_volume_origin
from ..core.geometry.parallel import ParallelGeometry
from ..core.geometry.views import stack_view_poses
from ..core.projector import backproject_view_T
from ..core.validation import (
    validate_grid,
    validate_detector_grid,
    validate_pose_stack,
    validate_projection_stack,
)
from .filters import get_filter_np
from ..utils.logging import progress_iter

_RFFT_FILTER_CACHE: "OrderedDict[Tuple[str, int, float, str], np.ndarray]" = OrderedDict()
_RFFT_FILTER_CACHE_CAP = 8
_UNSET: Any = object()


@dataclass(frozen=True, slots=True)
class FBPConfig:
    """Configuration for filtered backprojection."""

    filter_name: str = "ramp"
    scale: float | None = None
    views_per_batch: int = 1
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"


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


_fft_filter_rows_jit = jax.jit(_fft_filter_rows, static_argnames=("du", "filter_name"))


def _bp_one(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    filtered: jnp.ndarray,
    *,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Backproject one view with the explicit discrete adjoint."""
    del checkpoint_projector
    return backproject_view_T(
        T,
        grid,
        detector,
        filtered.astype(jnp.float32),
        unroll=int(projector_unroll),
        gather_dtype=gather_dtype,
        det_grid=det_grid,
    )


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


def _bp_batch_sum(
    T_chunk: jnp.ndarray,
    filt_chunk: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Backproject a fixed-size chunk while keeping peak memory at one volume."""

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
            det_grid=det_grid,
        )
        return accum + bp, None

    init = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    acc_chunk, _ = jax.lax.scan(body, init, (T_chunk, filt_chunk))
    return acc_chunk


_bp_batch_sum_jit = jax.jit(
    _bp_batch_sum,
    static_argnames=(
        "grid",
        "detector",
        "projector_unroll",
        "checkpoint_projector",
        "gather_dtype",
    ),
)


def _run_parallel_fbp_direct(
    T_all: jnp.ndarray,
    proj: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    filter_name: str,
) -> jnp.ndarray:
    """Run parallel-beam FBP with direct voxel-domain backprojection.

    The generic adjoint in ``projector.py`` backprojects by walking detector rays
    through the volume, which is necessary for arbitrary posed ray models.  For
    ``ParallelGeometry`` every voxel maps to one detector coordinate per view, so
    FBP can use the standard slice-wise parallel-beam backprojection directly.
    """
    n_views, nv, nu = map(int, proj.shape)
    origin_x, origin_y, origin_z = _grid_volume_origin(grid)
    x = jnp.arange(int(grid.nx), dtype=jnp.float32) * jnp.float32(grid.vx) + jnp.float32(origin_x)
    y = jnp.arange(int(grid.ny), dtype=jnp.float32) * jnp.float32(grid.vy) + jnp.float32(origin_y)
    z = jnp.arange(int(grid.nz), dtype=jnp.float32) * jnp.float32(grid.vz) + jnp.float32(origin_z)

    X = x[:, None]
    Y = y[None, :]
    Z = z
    u_offset = jnp.float32(float(detector.nu) / 2.0 - 0.5)
    v_offset = jnp.float32(float(detector.nv) / 2.0 - 0.5)
    inv_du = jnp.float32(1.0 / float(detector.du))
    inv_dv = jnp.float32(1.0 / float(detector.dv))
    det_cx = jnp.float32(float(detector.det_center[0]))
    det_cz = jnp.float32(float(detector.det_center[1]))
    step_weight = jnp.float32(float(grid.vy))

    rows = proj.reshape((n_views * nv, nu))
    rows_f = _fft_filter_rows_jit(rows, du=float(detector.du), filter_name=filter_name)
    filt = rows_f.reshape((n_views, nv, nu))

    def gather2(image: jnp.ndarray, iu: jnp.ndarray, iv: jnp.ndarray) -> jnp.ndarray:
        iu0 = jnp.floor(iu).astype(jnp.int32)
        iv0 = jnp.floor(iv).astype(jnp.int32)
        iu1 = iu0 + 1
        iv1 = iv0 + 1
        wu1 = iu - iu0.astype(jnp.float32)
        wv1 = iv - iv0.astype(jnp.float32)
        wu0 = jnp.float32(1.0) - wu1
        wv0 = jnp.float32(1.0) - wv1
        flat = image.reshape((-1,))

        def take(iv_idx: jnp.ndarray, iu_idx: jnp.ndarray) -> jnp.ndarray:
            inb = (
                (iu_idx[:, :, None] >= 0)
                & (iu_idx[:, :, None] < int(detector.nu))
                & (iv_idx[None, None, :] >= 0)
                & (iv_idx[None, None, :] < int(detector.nv))
            )
            idx = iv_idx[None, None, :] * int(detector.nu) + iu_idx[:, :, None]
            values = jnp.take(flat, jnp.clip(idx, 0, int(detector.nu * detector.nv) - 1))
            return jnp.where(inb, values, jnp.float32(0.0))

        c00 = take(iv0, iu0) * wu0[:, :, None] * wv0[None, None, :]
        c01 = take(iv1, iu0) * wu0[:, :, None] * wv1[None, None, :]
        c10 = take(iv0, iu1) * wu1[:, :, None] * wv0[None, None, :]
        c11 = take(iv1, iu1) * wu1[:, :, None] * wv1[None, None, :]
        return c00 + c01 + c10 + c11

    def body(accum: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, None]:
        T, image = inputs
        x_world = T[0, 0] * X + T[0, 1] * Y + T[0, 3]
        z_world = T[2, 2] * Z + T[2, 3]
        iu = (x_world - det_cx) * inv_du + u_offset
        iv = (z_world - det_cz) * inv_dv + v_offset
        return accum + gather2(image, iu, iv) * step_weight, None

    init = jnp.zeros((int(grid.nx), int(grid.ny), int(grid.nz)), dtype=jnp.float32)
    acc, _ = jax.lax.scan(body, init, (T_all, filt))
    return acc


_run_parallel_fbp_direct_jit = jax.jit(
    _run_parallel_fbp_direct,
    static_argnames=("grid", "detector", "filter_name"),
)


def _can_use_direct_parallel_fbp(
    geometry: Geometry,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> bool:
    """Return true only for the built-in z-axis ``ParallelGeometry`` convention."""
    return type(geometry) is ParallelGeometry and det_grid is None


def _is_fbp_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "resource_exhausted" in msg or "out of memory" in msg


def _run_fbp_fast_path(
    T_all: jnp.ndarray,
    proj: jnp.ndarray,
    *,
    batch_size: int,
    grid: Grid,
    detector: Detector,
    filter_name: str,
    projector_unroll: int,
    checkpoint_projector: bool,
    gather_dtype: str,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> jnp.ndarray:
    """Run FBP as one compiled scan over padded view chunks."""
    n_views, nv, nu = map(int, proj.shape)
    num_chunks = (n_views + batch_size - 1) // batch_size
    total_views = num_chunks * batch_size
    pad_views = total_views - n_views
    if pad_views:
        T_pad = jnp.repeat(T_all[-1:], pad_views, axis=0)
        y_pad = jnp.zeros((pad_views, nv, nu), dtype=proj.dtype)
        T_all = jnp.concatenate((T_all, T_pad), axis=0)
        proj = jnp.concatenate((proj, y_pad), axis=0)

    T_chunks = T_all.reshape((num_chunks, batch_size, 4, 4))
    y_chunks = proj.reshape((num_chunks, batch_size, nv, nu))
    valid_mask = (jnp.arange(total_views) < n_views).reshape((num_chunks, batch_size, 1, 1))

    def scan_chunks(T_chunks_in, y_chunks_in, valid_mask_in, det_grid_in):
        rows = y_chunks_in.reshape((num_chunks, batch_size * nv, nu))
        rows_f = jax.vmap(
            lambda chunk_rows: _fft_filter_rows_jit(
                chunk_rows,
                du=float(detector.du),
                filter_name=filter_name,
            )
        )(rows)
        filt_chunks = rows_f.reshape((num_chunks, batch_size, nv, nu))
        filt_chunks = jnp.where(valid_mask_in, filt_chunks, 0.0)

        def body(accum, inputs):
            T_chunk, filt_chunk = inputs
            acc_chunk = _bp_batch_sum_jit(
                T_chunk,
                filt_chunk,
                grid=grid,
                detector=detector,
                projector_unroll=projector_unroll,
                checkpoint_projector=checkpoint_projector,
                gather_dtype=gather_dtype,
                det_grid=det_grid_in,
            )
            return accum + acc_chunk, None

        init = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
        acc, _ = jax.lax.scan(body, init, (T_chunks_in, filt_chunks))
        return acc

    return jax.jit(scan_chunks)(T_chunks, y_chunks, valid_mask, det_grid)


def _run_fbp_with_backoff(
    T_all: jnp.ndarray,
    proj: jnp.ndarray,
    *,
    batch_size: int,
    grid: Grid,
    detector: Detector,
    filter_name: str,
    projector_unroll: int,
    checkpoint_projector: bool,
    gather_dtype: str,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    view_progress: Iterator[int],
) -> jnp.ndarray:
    """Fallback path that retries smaller chunks after OOM without skipping views."""
    n_views, nv, nu = map(int, proj.shape)
    acc = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    b = int(batch_size)
    s = 0

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

            valid_mask = (jnp.arange(b) < cur)[:, None, None]
            rows = y_chunk.reshape((b * nv, nu))
            rows_f = _fft_filter_rows_jit(
                rows,
                du=float(detector.du),
                filter_name=filter_name,
            )
            filt_chunk = rows_f.reshape((b, nv, nu))
            filt_chunk = jnp.where(valid_mask, filt_chunk, 0.0)
            acc = acc + _bp_batch_sum_jit(
                T_chunk,
                filt_chunk,
                grid=grid,
                detector=detector,
                projector_unroll=projector_unroll,
                checkpoint_projector=checkpoint_projector,
                gather_dtype=gather_dtype,
                det_grid=det_grid,
            )
            s += cur
            for _ in range(cur):
                next(view_progress, None)
        except Exception as exc:
            if _is_fbp_oom_error(exc) and b > 1:
                b = max(1, b // 2)
                continue
            raise

    return acc


def fbp(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    config: FBPConfig | None = None,
    filter_name: str | object = _UNSET,
    scale: float | None | object = _UNSET,
    views_per_batch: int | object = _UNSET,
    projector_unroll: int | object = _UNSET,
    checkpoint_projector: bool | object = _UNSET,
    gather_dtype: str | object = _UNSET,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Filtered backprojection for parallel-ray geometry using the explicit adjoint.

    projections: (n_views, nv, nu) -> volume (nx, ny, nz).
    Memory-safe: filters and backprojects per view-batch.
    """
    cfg = FBPConfig() if config is None else config
    updates: dict[str, object] = {}
    if filter_name is not _UNSET:
        updates["filter_name"] = str(filter_name)
    if scale is not _UNSET:
        updates["scale"] = None if scale is None else float(scale)
    if views_per_batch is not _UNSET:
        updates["views_per_batch"] = int(views_per_batch)
    if projector_unroll is not _UNSET:
        updates["projector_unroll"] = int(projector_unroll)
    if checkpoint_projector is not _UNSET:
        updates["checkpoint_projector"] = bool(checkpoint_projector)
    if gather_dtype is not _UNSET:
        updates["gather_dtype"] = str(gather_dtype)
    if updates:
        cfg = replace(cfg, **updates)

    validate_grid(grid, "fbp grid")
    n_views, _, _ = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="fbp projections",
    )
    validate_detector_grid(det_grid, detector, context="fbp det_grid")
    proj = jnp.asarray(projections, dtype=jnp.float32)
    # Precompute poses once
    T_all = stack_view_poses(geometry, n_views)
    validate_pose_stack(T_all, n_views, context="fbp geometry")
    requested_b = int(cfg.views_per_batch) if int(cfg.views_per_batch) > 0 else n_views
    b = max(1, min(requested_b, n_views))
    view_progress = iter(progress_iter(range(n_views), total=n_views, desc="FBP: views"))

    def run_generic_path() -> jnp.ndarray:
        try:
            generic_acc = _run_fbp_fast_path(
                T_all,
                proj,
                batch_size=b,
                grid=grid,
                detector=detector,
                filter_name=cfg.filter_name,
                projector_unroll=cfg.projector_unroll,
                checkpoint_projector=cfg.checkpoint_projector,
                gather_dtype=cfg.gather_dtype,
                det_grid=det_grid,
            )
            for _ in range(n_views):
                next(view_progress, None)
            return generic_acc
        except Exception as exc:
            if not _is_fbp_oom_error(exc):
                raise
            return _run_fbp_with_backoff(
                T_all,
                proj,
                batch_size=b,
                grid=grid,
                detector=detector,
                filter_name=cfg.filter_name,
                projector_unroll=cfg.projector_unroll,
                checkpoint_projector=cfg.checkpoint_projector,
                gather_dtype=cfg.gather_dtype,
                det_grid=det_grid,
                view_progress=view_progress,
            )

    if _can_use_direct_parallel_fbp(geometry, det_grid):
        try:
            acc = _run_parallel_fbp_direct_jit(
                T_all,
                proj,
                grid=grid,
                detector=detector,
                filter_name=cfg.filter_name,
            )
            for _ in range(n_views):
                next(view_progress, None)
        except Exception as exc:
            if not _is_fbp_oom_error(exc):
                raise
            acc = run_generic_path()
    else:
        acc = run_generic_path()

    if cfg.scale is None:
        acc = acc * _default_fbp_scale(n_views)
    else:
        acc = acc * float(cfg.scale)
    return acc
