"""Filtered-backprojection reconstruction routines."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core import progress_iter
from tomojax.core.geometry import Detector, Geometry, Grid, grid_volume_origin
from tomojax.core.geometry.parallel import ParallelGeometry
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.projector import backproject_view_T
from tomojax.core.validation import (
    validate_detector_grid,
    validate_grid,
    validate_pose_stack,
    validate_projection_stack,
)

from .filters import get_fbp_filter_np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass(frozen=True, slots=True)
class FBPConfig:
    """Configuration for filtered backprojection.

    ``backprojector='auto'`` uses the voxel-driven Pallas kernel for built-in
    parallel geometry on CUDA, and JAX otherwise. ``'jax'`` forces the reference;
    ``'pallas'`` requires a supported CUDA input and never silently changes backend.
    """

    filter_name: str = "ramp"
    scale: float | None = None
    views_per_batch: int = 1
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"
    backprojector: Literal["auto", "jax", "pallas"] = "auto"


def default_fbp_scale(n_views: int) -> float:
    """Return the default angular weighting for the current parallel-ray FBP.

    TomoJAX's built-in CT and laminography geometries both use parallel rays,
    so the discrete filtered backprojection sum should be weighted by the
    180-degree angular spacing ``pi / n_views``. Callers with custom angular
    coverage can override this via ``scale=...``.
    """
    if int(n_views) <= 0:
        raise ValueError("n_views must be positive")
    return float(np.pi / float(n_views))


def _rfft_filter_array(filter_name: str, nu: int, du: float, dtype: jnp.dtype) -> jnp.ndarray:
    """Return the padded discrete FBP filter without doubling RFFT bins."""
    Hr_np = get_fbp_filter_np(filter_name, int(nu), float(du), str(dtype))
    return jnp.asarray(Hr_np, dtype=dtype)


def _fft_filter_rows(rows: jnp.ndarray, rfft_filter: jnp.ndarray) -> jnp.ndarray:
    """Zero-pad detector rows, filter, and crop to prevent circular wraparound."""
    nu = int(rows.shape[-1])
    n_fft = 2 * (int(rfft_filter.shape[0]) - 1)
    F = jnp.fft.rfft(rows, n=n_fft, axis=-1)
    return jnp.fft.irfft(F * rfft_filter, n=n_fft, axis=-1)[..., :nu]


def _pad_detector_rows(rows: jnp.ndarray, width: int) -> jnp.ndarray:
    padding = (width - rows.shape[-1]) // 2
    return jnp.pad(rows, ((0, 0), (0, 0), (padding, padding))) if padding else rows


_fft_filter_rows_jit = jax.jit(_fft_filter_rows)


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


def _run_parallel_fbp_direct(
    T_all: jnp.ndarray,
    proj: jnp.ndarray,
    rfft_filter: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
) -> jnp.ndarray:
    """Run parallel-beam FBP with direct voxel-domain backprojection.

    The generic adjoint in ``projector.py`` backprojects by walking detector rays
    through the volume, which is necessary for arbitrary posed ray models.  For
    ``ParallelGeometry`` every voxel maps to one detector coordinate per view, so
    FBP can use the standard slice-wise parallel-beam backprojection directly.
    """
    n_views, nv, nu = map(int, proj.shape)
    origin_x, origin_y, origin_z = grid_volume_origin(grid)
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

    rows = proj.reshape((n_views * nv, nu))
    rows_f = _fft_filter_rows_jit(rows, rfft_filter)
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

    def body(
        accum: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, None]:
        T, image = inputs
        x_world = T[0, 0] * X + T[0, 1] * Y + T[0, 3]
        z_world = T[2, 2] * Z + T[2, 3]
        iu = (x_world - det_cx) * inv_du + u_offset
        iv = (z_world - det_cz) * inv_dv + v_offset
        return accum + gather2(image, iu, iv), None

    init = jnp.zeros((int(grid.nx), int(grid.ny), int(grid.nz)), dtype=jnp.float32)
    acc, _ = jax.lax.scan(body, init, (T_all, filt))
    return acc


_run_parallel_fbp_direct_jit = jax.jit(
    _run_parallel_fbp_direct,
    static_argnames=("grid", "detector"),
)


@jax.jit(static_argnames=("grid", "detector", "z_integer"))
def _run_parallel_fbp_pallas(
    T_all: jnp.ndarray,
    proj: jnp.ndarray,
    rfft_filter: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    z_integer: bool,
) -> jnp.ndarray:
    from ._fbp_pallas import backproject_filtered_pallas

    return backproject_filtered_pallas(
        T_all,
        _fft_filter_rows(proj, rfft_filter),
        grid=grid,
        detector=detector,
        z_integer=z_integer,
    )


def _parallel_filter_batch_size(n_views: int, nv: int, n_fft: int) -> int:
    """Use a conservative 512 MiB FFT-workspace estimate, not a VRAM guarantee."""
    capacity = max(1, (512 * 1024**2) // (16 * nv * n_fft))
    return n_views if capacity >= n_views else 1 << (capacity.bit_length() - 1)


@jax.jit(static_argnames=("grid", "detector", "backend", "batch_size", "z_integer"))
def _run_parallel_fbp_streamed(
    poses: jnp.ndarray,
    projections: jnp.ndarray,
    rfft_filter: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    z_integer: bool,
) -> jnp.ndarray:
    """Pad, filter and backproject bounded view batches into one accumulator."""
    n, nv, nu = projections.shape
    b = min(batch_size, n)

    def step(chunk: jnp.ndarray, accum: jnp.ndarray) -> jnp.ndarray:
        start = jnp.minimum(chunk * b, n - b)
        batch_poses = jax.lax.dynamic_slice(poses, (start, 0, 0), (b, 4, 4))
        rows = jax.lax.dynamic_slice(projections, (start, 0, 0), (b, nv, nu))
        # Shift the final fixed-size batch back into range and exclude overlaps.
        valid = start + jnp.arange(b) >= chunk * b
        rows = _pad_detector_rows(jnp.where(valid[:, None, None], rows, 0.0), detector.nu)
        if backend == "pallas":
            update = _run_parallel_fbp_pallas(
                batch_poses, rows, rfft_filter, grid=grid, detector=detector, z_integer=z_integer
            )
        else:
            update = _run_parallel_fbp_direct_jit(
                batch_poses, rows, rfft_filter, grid=grid, detector=detector
            )
        return accum + update

    return jax.lax.fori_loop(
        0,
        (n + b - 1) // b,
        step,
        jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32),
    )


def supports_parallel_fbp_z_integer(grid: Grid, detector: Detector) -> bool:
    """Return whether the detector rows align with z-slices for direct Pallas FBP."""
    tol = 1e-5
    origin_z = float(grid_volume_origin(grid)[2])
    first = (origin_z - float(detector.det_center[1])) / float(detector.dv)
    first += float(detector.nv) / 2.0 - 0.5
    step = float(grid.vz) / float(detector.dv)
    # A small per-slice mismatch can accumulate over a large volume.
    return abs(first - round(first)) + (grid.nz - 1) * abs(step - round(step)) <= tol


def run_parallel_fbp_direct_pallas(
    T_all: jnp.ndarray,
    proj: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    filter_name: str,
) -> jnp.ndarray:
    """Run unscaled z-parallel FBP, retaining filtered tails through the volume."""
    validate_grid(grid, "run_parallel_fbp_direct_pallas")
    n_views, _, _ = validate_projection_stack(
        proj, detector, context="run_parallel_fbp_direct_pallas"
    )
    validate_pose_stack(T_all, n_views, context="run_parallel_fbp_direct_pallas")
    if not supports_parallel_fbp_z_integer(grid, detector):
        raise ValueError(
            "Pallas integer-row FBP requires detector rows aligned with voxel z-slices"
        )
    poses = np.asarray(T_all)
    if not (
        np.allclose(poses[:, 2, :], [0, 0, 1, 0], atol=1e-7, rtol=0)
        and np.allclose(poses[:, :2, 2], 0, atol=1e-7, rtol=0)
    ):
        raise ValueError("Pallas integer-row FBP requires z-axis rotations without z translation")
    ox, oy, _ = grid_volume_origin(grid)
    xx, yy = np.meshgrid([ox, ox + (grid.nx - 1) * grid.vx], [oy, oy + (grid.ny - 1) * grid.vy])
    coordinates = (
        poses[:, 0, 0, None] * xx.ravel() + poses[:, 0, 1, None] * yy.ravel() + poses[:, 0, 3, None]
    )
    required_radius = np.max(np.abs(coordinates - detector.det_center[0])) / detector.du
    padding = max(0, math.ceil(required_radius - (detector.nu - 1) / 2))
    detector = replace(detector, nu=detector.nu + 2 * padding)
    ramp = _rfft_filter_array(filter_name, detector.nu, float(detector.du), jnp.float32)
    return _run_parallel_fbp_streamed(
        jnp.asarray(T_all, dtype=jnp.float32),
        jnp.asarray(proj, dtype=jnp.float32),
        ramp,
        grid=grid,
        detector=detector,
        backend="pallas",
        batch_size=_parallel_filter_batch_size(n_views, detector.nv, 2 * (ramp.shape[0] - 1)),
        z_integer=True,
    )


def _can_use_direct_parallel_fbp(
    geometry: Geometry,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> bool:
    """Return true only for the built-in z-axis ``ParallelGeometry`` convention."""
    return type(geometry) is ParallelGeometry and det_grid is None


def _parallel_filter_detector(grid: Grid, detector: Detector) -> Detector:
    """Keep the zero-extended ramp convolution at every reconstructed voxel.

    Raw samples outside the measured row are zero, but filtered samples are not.
    Cropping the filter output to the measured detector discards negative tails
    needed by voxels outside its inscribed field of view. Bound all z rotations
    of voxel centres, with symmetric padding preserving the acquired coordinates.
    This does not infer missing attenuation in genuinely truncated acquisitions.
    """
    ox, oy, _ = grid_volume_origin(grid)
    rx = max(abs(ox), abs(ox + (grid.nx - 1) * grid.vx))
    ry = max(abs(oy), abs(oy + (grid.ny - 1) * grid.vy))
    required_half_width = (math.hypot(rx, ry) + abs(detector.det_center[0])) / detector.du
    padding = max(0, math.ceil(required_half_width - (detector.nu - 1) / 2))
    return replace(detector, nu=detector.nu + 2 * padding) if padding else detector


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
    rfft_filter = _rfft_filter_array(filter_name, nu, float(detector.du), proj.dtype)

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
    rfft_filter = _rfft_filter_array(filter_name, nu, float(detector.du), proj.dtype)
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
            if _is_fbp_oom_error(exc) and b > 1:
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
        if not _is_fbp_oom_error(exc):
            raise
        return backoff_path()


def fbp(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    config: FBPConfig | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Filtered backprojection for uniformly sampled parallel-ray geometry.

    Projections: (n_views, nv, nu) -> attenuation volume (nx, ny, nz).
    Built-in parallel geometry uses voxel-driven backprojection and retains
    filtered tails through the full output volume, assuming zero raw attenuation
    beyond the measured detector. This is not a correction for truncated objects.
    Other geometries
    use a physically normalized discrete adjoint. The default angular weight
    assumes a uniformly sampled half turn; supply ``scale`` for other coverage.
    """
    cfg = FBPConfig() if config is None else config
    if cfg.backprojector not in ("auto", "jax", "pallas"):
        raise ValueError("FBP backprojector must be 'auto', 'jax', or 'pallas'")

    validate_grid(grid, "fbp grid")
    n_views, _, _ = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="fbp projections",
    )
    validate_detector_grid(det_grid, detector, context="fbp det_grid")
    proj = jnp.asarray(projections, dtype=jnp.float32)
    direct_parallel = _can_use_direct_parallel_fbp(geometry, det_grid)
    pallas_eligible = direct_parallel and all(
        device.platform == "gpu" and device.client.platform_version.lower().startswith("cuda")
        for device in proj.devices()
    )
    if cfg.backprojector == "pallas" and not pallas_eligible:
        raise ValueError(
            "Pallas FBP requires CUDA arrays and built-in parallel geometry without det_grid"
        )
    if direct_parallel:
        detector = _parallel_filter_detector(grid, detector)
    # Precompute poses once
    T_all = stack_view_poses(geometry, n_views)
    validate_pose_stack(T_all, n_views, context="fbp geometry")
    requested_b = int(cfg.views_per_batch) if int(cfg.views_per_batch) > 0 else n_views
    b = max(1, min(requested_b, n_views))
    view_progress = iter(progress_iter(range(n_views), total=n_views, desc="FBP: views"))

    def run_generic_path() -> jnp.ndarray:
        generic_proj = _pad_detector_rows(proj, detector.nu)

        def fast_path() -> jnp.ndarray:
            return _run_fbp_fast_path(
                T_all,
                generic_proj,
                batch_size=b,
                grid=grid,
                detector=detector,
                filter_name=cfg.filter_name,
                projector_unroll=cfg.projector_unroll,
                checkpoint_projector=cfg.checkpoint_projector,
                gather_dtype=cfg.gather_dtype,
                det_grid=det_grid,
            )

        def backoff_path() -> jnp.ndarray:
            return _run_fbp_with_backoff(
                T_all,
                generic_proj,
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

        return _run_fbp_generic_with_oom_fallback(
            fast_path=fast_path,
            backoff_path=backoff_path,
            view_progress=view_progress,
            n_views=n_views,
        )

    if direct_parallel:
        rfft_filter = _rfft_filter_array(
            cfg.filter_name, detector.nu, float(detector.du), proj.dtype
        )
        filter_batch = _parallel_filter_batch_size(
            n_views, detector.nv, 2 * (rfft_filter.shape[0] - 1)
        )
        try:
            acc = _run_parallel_fbp_streamed(
                T_all,
                proj,
                rfft_filter,
                grid=grid,
                detector=detector,
                backend="pallas" if pallas_eligible and cfg.backprojector != "jax" else "jax",
                batch_size=filter_batch,
                z_integer=supports_parallel_fbp_z_integer(grid, detector),
            )
            acc.block_until_ready()
            for _ in range(n_views):
                next(view_progress, None)
        except Exception as exc:
            if cfg.backprojector == "pallas" or not _is_fbp_oom_error(exc):
                raise
            acc = run_generic_path()
    else:
        acc = run_generic_path()

    return acc * default_fbp_scale(n_views) if cfg.scale is None else acc * float(cfg.scale)
