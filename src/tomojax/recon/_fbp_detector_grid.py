"""FBP on explicit detector pixel positions, with the ray model's adjoint.

:func:`tomojax.recon.fbp` takes this path for an explicit ``det_grid``: a ramp
along u with uniform weights, compiled as one scan over view chunks, which
retries smaller chunks after running out of memory without skipping views.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.core import progress_iter
from tomojax.core.projector import backproject_view_T

from .filters import fft_filter_rows, rfft_filter_array

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from tomojax.core.geometry import Detector, Grid

    from .fbp import FBPConfig

_fft_filter_rows_jit = jax.jit(fft_filter_rows)


def is_oom_error(exc: Exception) -> bool:
    """Whether ``exc`` is a device running out of memory, which smaller batches may avoid."""
    msg = str(exc).lower()
    return "resource_exhausted" in msg or "out of memory" in msg


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
    discrete_adjoint = backproject_view_T(
        T,
        grid,
        detector,
        filtered.astype(jnp.float32),
        unroll=int(projector_unroll),
        gather_dtype=gather_dtype,
        det_grid=det_grid,
    )
    # Convert the Euclidean transpose into a physical backprojection: detector
    # area / voxel volume cancels the ray integration length and sampling density.
    return discrete_adjoint * (detector.du * detector.dv / (grid.vx * grid.vy * grid.vz))


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

    def body(
        accum: jnp.ndarray,
        inputs: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, None]:
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


def _run_fbp_fast_path(
    T_all: jnp.ndarray,
    proj: jnp.ndarray,
    *,
    batch_size: int,
    grid: Grid,
    detector: Detector,
    filter: str,
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
    rfft_filter = rfft_filter_array(filter, nu, float(detector.du), proj.dtype)

    def scan_chunks(
        T_chunks_in: jnp.ndarray,
        y_chunks_in: jnp.ndarray,
        valid_mask_in: jnp.ndarray,
        rfft_filter_in: jnp.ndarray,
        det_grid_in: tuple[jnp.ndarray, jnp.ndarray] | None,
    ) -> jnp.ndarray:
        rows = y_chunks_in.reshape((num_chunks, batch_size * nv, nu))
        rows_f = jax.vmap(lambda chunk_rows: _fft_filter_rows_jit(chunk_rows, rfft_filter_in))(rows)
        filt_chunks = rows_f.reshape((num_chunks, batch_size, nv, nu))
        filt_chunks = jnp.where(valid_mask_in, filt_chunks, 0.0)

        def body(
            accum: jnp.ndarray,
            inputs: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, None]:
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

    return jax.jit(scan_chunks)(T_chunks, y_chunks, valid_mask, rfft_filter, det_grid)


def _run_fbp_with_backoff(
    T_all: jnp.ndarray,
    proj: jnp.ndarray,
    *,
    batch_size: int,
    grid: Grid,
    detector: Detector,
    filter: str,
    projector_unroll: int,
    checkpoint_projector: bool,
    gather_dtype: str,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    view_progress: Iterator[int],
) -> jnp.ndarray:
    """Fallback path that retries smaller chunks after OOM without skipping views."""
    n_views, nv, nu = map(int, proj.shape)
    acc = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    rfft_filter = rfft_filter_array(filter, nu, float(detector.du), proj.dtype)
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
            rows_f = _fft_filter_rows_jit(rows, rfft_filter)
            filt_chunk = rows_f.reshape((b, nv, nu))
            filt_chunk = jnp.where(valid_mask, filt_chunk, 0.0)
            candidate = acc + _bp_batch_sum_jit(
                T_chunk,
                filt_chunk,
                grid=grid,
                detector=detector,
                projector_unroll=projector_unroll,
                checkpoint_projector=checkpoint_projector,
                gather_dtype=gather_dtype,
                det_grid=det_grid,
            )
            # Device work is asynchronous. Commit the accumulator and progress
            # only once this chunk succeeds, so OOM retries neither skip nor
            # double-count views and never reuse a failed device buffer.
            candidate.block_until_ready()
            acc = candidate
            s += cur
            for _ in range(cur):
                next(view_progress, None)
        except Exception as exc:
            if is_oom_error(exc) and b > 1:
                b = max(1, b // 2)
                continue
            raise

    return acc


def _run_fbp_generic_with_oom_fallback(
    *,
    fast_path: Callable[[], jnp.ndarray],
    backoff_path: Callable[[], jnp.ndarray],
    view_progress: Iterator[int],
    n_views: int,
) -> jnp.ndarray:
    """Run the compiled generic FBP path, falling back after asynchronous OOMs."""
    try:
        generic_acc = fast_path()
        generic_acc.block_until_ready()
        for _ in range(n_views):
            next(view_progress, None)
        return generic_acc
    except Exception as exc:
        if not is_oom_error(exc):
            raise
        return backoff_path()


def fbp_on_detector_grid(
    T_all: jnp.ndarray,
    proj: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    cfg: FBPConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Backproject with the ray-model adjoint on explicit detector pixel positions.

    Uses a ramp along u with uniform weights, ``pi / n`` unless ``cfg.scale`` is set.
    """
    n_views = int(proj.shape[0])
    requested_b = int(cfg.views_per_batch) if int(cfg.views_per_batch) > 0 else n_views
    b = max(1, min(requested_b, n_views))
    view_progress = iter(progress_iter(range(n_views), total=n_views, desc="FBP: views"))

    def fast_path() -> jnp.ndarray:
        return _run_fbp_fast_path(
            T_all,
            proj,
            batch_size=b,
            grid=grid,
            detector=detector,
            filter=cfg.filter,
            projector_unroll=cfg.projector_unroll,
            checkpoint_projector=cfg.checkpoint_projector,
            gather_dtype=cfg.gather_dtype,
            det_grid=det_grid,
        )

    def backoff_path() -> jnp.ndarray:
        return _run_fbp_with_backoff(
            T_all,
            proj,
            batch_size=b,
            grid=grid,
            detector=detector,
            filter=cfg.filter,
            projector_unroll=cfg.projector_unroll,
            checkpoint_projector=cfg.checkpoint_projector,
            gather_dtype=cfg.gather_dtype,
            det_grid=det_grid,
            view_progress=view_progress,
        )

    acc = _run_fbp_generic_with_oom_fallback(
        fast_path=fast_path,
        backoff_path=backoff_path,
        view_progress=view_progress,
        n_views=n_views,
    )
    from .fbp import default_fbp_scale

    return acc * default_fbp_scale(n_views) if cfg.scale is None else acc * float(cfg.scale)
