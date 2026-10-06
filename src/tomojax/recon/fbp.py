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
from tomojax.core.geometry.cone import beam_of
from tomojax.core.geometry.parallel import ParallelGeometry
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.projector import backproject_view_T
from tomojax.core.validation import (
    validate_detector_grid,
    validate_grid,
    validate_pose_stack,
    validate_projection_stack,
)

from ._fbp_weights import fbp_weights
from .filters import get_fbp_filter_np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass(frozen=True, slots=True)
class FBPConfig:
    """Configuration for filtered backprojection.

    ``backprojector='auto'`` uses the voxel-driven Pallas kernel on CUDA and JAX
    otherwise. ``'jax'`` forces the reference; ``'pallas'`` requires CUDA input
    and never silently changes backend.

    ``scale=None`` weights each view exactly for the scanned arc (see
    :func:`fbp`). A number instead weights every view by that constant and
    filters along u only, the convention before exact weights; ``pi / n`` gives
    the uniform half-turn result. ``views_per_batch``, ``projector_unroll``,
    ``checkpoint_projector`` and ``gather_dtype`` apply only with an explicit
    ``det_grid``, which uses the ray-model adjoint.
    """

    filter_name: str = "ramp"
    scale: float | None = None
    views_per_batch: int = 1
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"
    backprojector: Literal["auto", "jax", "pallas"] = "auto"


def default_fbp_scale(n_views: int) -> float:
    """Return the uniform half-turn angular weight ``pi / n_views``.

    :func:`fbp` derives exact per-view weights by default; this constant is the
    uniform weight for an untilted half turn and the fallback for scans with too
    few views to define a rotation.
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


def _bilinear_detector(image: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Sample a (nv, nu) image at fractional pixel coordinates, zero outside."""
    nv, nu = image.shape
    iu, iv = jnp.floor(u).astype(jnp.int32), jnp.floor(v).astype(jnp.int32)
    wu, wv = u - iu.astype(jnp.float32), v - iv.astype(jnp.float32)
    flat = image.reshape(-1)

    def take(row: jnp.ndarray, col: jnp.ndarray) -> jnp.ndarray:
        inside = (col >= 0) & (col < nu) & (row >= 0) & (row < nv)
        index = jnp.clip(row, 0, nv - 1) * nu + jnp.clip(col, 0, nu - 1)
        return jnp.where(inside, jnp.take(flat, index), 0.0)

    row0 = take(iv, iu) * (1.0 - wu) + take(iv, iu + 1) * wu
    row1 = take(iv + 1, iu) * (1.0 - wu) + take(iv + 1, iu + 1) * wu
    return row0 * (1.0 - wv) + row1 * wv


def _backproject_voxels_jax(
    poses: jnp.ndarray, filtered: jnp.ndarray, *, grid: Grid, detector: Detector
) -> jnp.ndarray:
    """Sum views at each voxel centre's detector position, for any rigid poses."""
    ox, oy, oz = grid_volume_origin(grid)
    x = (jnp.arange(grid.nx, dtype=jnp.float32) * grid.vx + ox)[:, None, None]
    y = (jnp.arange(grid.ny, dtype=jnp.float32) * grid.vy + oy)[None, :, None]
    z = (jnp.arange(grid.nz, dtype=jnp.float32) * grid.vz + oz)[None, None, :]
    cu, cv = detector.det_center

    def body(
        accum: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, None]:
        pose, image = inputs
        world_x = pose[0, 0] * x + pose[0, 1] * y + pose[0, 2] * z + pose[0, 3]
        world_z = pose[2, 0] * x + pose[2, 1] * y + pose[2, 2] * z + pose[2, 3]
        u = (world_x - cu) / detector.du + (detector.nu - 1) / 2.0
        v = (world_z - cv) / detector.dv + (detector.nv - 1) / 2.0
        return accum + _bilinear_detector(image, u, v), None

    init = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    return jax.lax.scan(body, init, (poses, filtered))[0]


def _fft_length(n: int) -> int:
    """Linear-convolution length matching ``get_fbp_filter_np``'s padding."""
    return max(64, 1 << (2 * int(n) - 1).bit_length())


def _filter_views(
    rows: jnp.ndarray,
    view_scale: jnp.ndarray,
    params: jnp.ndarray,
    spectrum: jnp.ndarray,
    arc_length: jnp.ndarray,
    *,
    detector: Detector,
    separable: bool,
) -> jnp.ndarray:
    """Apply each view's exact FBP filter to rows already padded to ``detector``.

    Separable scans multiply a 1D ramp along u by a per-view scale. Otherwise the
    redundancy count N depends on both detector frequencies; see ``_fbp_weights``.
    """
    if separable:
        return _fft_filter_rows(rows * view_scale[:, None, None], spectrum)
    n_u, n_v = 2 * (spectrum.shape[0] - 1), _fft_length(detector.nv)
    transformed = jnp.fft.rfft2(rows, s=(n_v, n_u))
    fu = jnp.fft.rfftfreq(n_u, detector.du).astype(jnp.float32)[None, None, :]
    fv = jnp.fft.fftfreq(n_v, detector.dv).astype(jnp.float32)[None, :, None]
    t_u, t_v, c_u, c_v, phase, weight = (params[:, i, None, None] for i in range(6))
    along = t_u * fu + t_v * fv
    axial = c_u * fu + c_v * fv
    # A zero ramp coordinate is the limit of nearby frequencies, not a double root.
    other = jnp.mod(phase - 2 * jnp.arctan2(jnp.where(along == 0, 1e-30, along), axial), 2 * np.pi)
    count = 1.0 + (other < arc_length).astype(jnp.float32)
    row_ramp = jnp.abs(t_v) <= 1e-6 * jnp.abs(t_u)
    window = spectrum / jnp.maximum(jnp.abs(fu[0, 0]), spectrum[0])
    # A rolled detector needs the ramp along a u-v diagonal; keep the finite-kernel DC term.
    oblique = jnp.where(
        (fu == 0) & (fv == 0), jnp.hypot(t_u, t_v) * spectrum[0], jnp.abs(along) * window
    )
    ramp = jnp.where(row_ramp, jnp.abs(t_u) * spectrum, oblique)
    filtered = jnp.fft.irfft2(transformed * (ramp * weight / count), s=(n_v, n_u))
    return filtered[:, : detector.nv, : detector.nu]


def _pad_detector(rows: jnp.ndarray, detector: Detector) -> jnp.ndarray:
    """Zero-extend measured rows symmetrically to the filter detector."""
    pad_v, pad_u = (detector.nv - rows.shape[1]) // 2, (detector.nu - rows.shape[2]) // 2
    return jnp.pad(rows, ((0, 0), (pad_v, pad_v), (pad_u, pad_u))) if pad_u or pad_v else rows


def _parallel_filter_batch_size(n_views: int, nv: int, n_fft: int) -> int:
    """Use a conservative 512 MiB FFT-workspace estimate, not a VRAM guarantee."""
    capacity = max(1, (512 * 1024**2) // (16 * nv * n_fft))
    return n_views if capacity >= n_views else 1 << (capacity.bit_length() - 1)


@jax.jit(static_argnames=("grid", "detector", "backend", "batch_size", "z_integer", "separable"))
def _run_fbp_streamed(
    poses: jnp.ndarray,
    projections: jnp.ndarray,
    view_scale: jnp.ndarray,
    params: jnp.ndarray,
    spectrum: jnp.ndarray,
    arc_length: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    z_integer: bool,
    separable: bool,
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
        rows = _pad_detector(jnp.where(valid[:, None, None], rows, 0.0), detector)
        filtered = _filter_views(
            rows,
            jax.lax.dynamic_slice(view_scale, (start,), (b,)),
            jax.lax.dynamic_slice(params, (start, 0), (b, params.shape[1])),
            spectrum,
            arc_length,
            detector=detector,
            separable=separable,
        )
        return _backproject_into(
            accum,
            batch_poses,
            filtered,
            grid=grid,
            detector=detector,
            backend=backend,
            z_integer=z_integer,
        )

    return jax.lax.fori_loop(
        0,
        (n + b - 1) // b,
        step,
        jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32),
    )


def _backproject_into(
    accum: jnp.ndarray,
    poses: jnp.ndarray,
    filtered: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    z_integer: bool,
) -> jnp.ndarray:
    """Add filtered views to ``accum``; the CUDA kernel adds in place."""
    if backend == "pallas":
        from ._fbp_pallas import backproject_filtered_pallas

        return backproject_filtered_pallas(
            poses, filtered, grid=grid, detector=detector, z_integer=z_integer, accumulate=accum
        )
    return accum + _backproject_voxels_jax(poses, filtered, grid=grid, detector=detector)


@jax.jit(
    static_argnames=("grid", "detector", "backend", "separable", "z_integer"),
    donate_argnums=(0,),
)
def _fbp_accumulate_batch(
    accum: jnp.ndarray,
    poses: jnp.ndarray,
    rows: jnp.ndarray,
    view_scale: jnp.ndarray,
    params: jnp.ndarray,
    spectrum: jnp.ndarray,
    arc_length: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    separable: bool,
    z_integer: bool,
) -> jnp.ndarray:
    """Filter one batch of views and add its backprojection to ``accum`` in place."""
    filtered = _filter_views(
        _pad_detector(rows, detector),
        view_scale,
        params,
        spectrum,
        arc_length,
        detector=detector,
        separable=separable,
    )
    return _backproject_into(
        accum, poses, filtered, grid=grid, detector=detector, backend=backend, z_integer=z_integer
    )


def _fbp_from_host(
    poses: np.ndarray,
    projections: np.ndarray,
    view_scale: np.ndarray,
    params: np.ndarray,
    spectrum: jnp.ndarray,
    arc_length: float,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    separable: bool,
    z_integer: bool = False,
    check_finite: bool = False,
) -> jnp.ndarray:
    """Stream host projections through the device in fixed-size view batches.

    Only the output volume, two batches and one filtering workspace occupy the
    device. The final batch is padded with zero-weight views so every batch
    reuses one compiled step, and the next batch is read and transferred while
    the current one runs.
    """
    n = projections.shape[0]
    b = min(batch_size, n)

    def batch(start: int) -> tuple[jnp.ndarray, ...]:
        stop = min(start + b, n)
        pad = b - (stop - start)
        rows = np.asarray(projections[start:stop], dtype=np.float32)
        if check_finite and not np.isfinite(rows).all():
            raise ValueError("fbp_host: projections must be finite in FP32")
        arrays = [poses[start:stop], rows, view_scale[start:stop], params[start:stop]]
        if pad:
            # Zero-weight views repeat the last pose and contribute nothing.
            arrays = [
                np.concatenate([a, np.repeat(a[-1:], pad, axis=0) * (0 if i in (1, 2) else 1)])
                for i, a in enumerate(arrays)
            ]
        return tuple(jax.device_put(np.asarray(a, np.float32)) for a in arrays)

    accum = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
    pending = batch(0)
    for start in range(0, n, b):
        current = pending
        # Dispatch is asynchronous: read and transfer the next batch while this one runs.
        accum = _fbp_accumulate_batch(
            accum,
            *current,
            spectrum,
            jnp.float32(arc_length),
            grid=grid,
            detector=detector,
            backend=backend,
            separable=separable,
            z_integer=z_integer,
        )
        if start + b < n:
            pending = batch(start + b)
    return accum


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
    return _run_fbp_streamed(
        jnp.asarray(T_all, dtype=jnp.float32),
        jnp.asarray(proj, dtype=jnp.float32),
        jnp.ones((n_views,), jnp.float32),
        jnp.zeros((n_views, 6), jnp.float32),
        ramp,
        jnp.float32(0),
        grid=grid,
        detector=detector,
        backend="pallas",
        batch_size=_parallel_filter_batch_size(n_views, detector.nv, 2 * (ramp.shape[0] - 1)),
        z_integer=True,
        separable=True,
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


def _filter_detector(grid: Grid, detector: Detector, poses: np.ndarray, *, pad_v: bool) -> Detector:
    """Extend the detector to every view's projected voxel centres.

    Filtered values are nonzero beyond the measured detector. Padding keeps
    those tails for voxels projecting outside it, as ``_parallel_filter_detector``
    does for z-axis scans. Rows need padding only for two-dimensional filters.
    """
    origin = np.asarray(grid_volume_origin(grid), dtype=np.float64)
    far = origin + (np.asarray([grid.nx, grid.ny, grid.nz]) - 1) * [grid.vx, grid.vy, grid.vz]
    corners = np.array(np.meshgrid(*zip(origin, far, strict=True), indexing="ij")).reshape(3, -1)
    rows = np.asarray(poses, dtype=np.float64)
    u = np.einsum("ni,ic->nc", rows[:, 0, :3], corners) + rows[:, 0, 3, None]
    v = np.einsum("ni,ic->nc", rows[:, 2, :3], corners) + rows[:, 2, 3, None]
    reach_u = np.max(np.abs(u - detector.det_center[0])) / detector.du
    reach_v = np.max(np.abs(v - detector.det_center[1])) / detector.dv
    extra_u = max(0, math.ceil(reach_u - (detector.nu - 1) / 2))
    extra_v = max(0, math.ceil(reach_v - (detector.nv - 1) / 2)) if pad_v else 0
    return replace(detector, nu=detector.nu + 2 * extra_u, nv=detector.nv + 2 * extra_v)


def _view_weights(
    poses: jnp.ndarray, scale: float | None
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Return per-view scales, filter parameters, arc length and separability.

    An explicit ``scale`` keeps the uniform weighting: one constant per view
    times a ramp along u. Otherwise weights are exact for the fitted circular
    scan; scans with too few views to define one fall back to ``pi / n``.
    """
    n = int(poses.shape[0])
    if scale is None:
        try:
            weights = fbp_weights(np.asarray(poses))
        except ValueError:
            scale = default_fbp_scale(n)
        else:
            return weights.view_scale, weights.params, weights.arc_length, weights.separable
    return np.full(n, scale, np.float32), np.zeros((n, 6), np.float32), 0.0, True


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
    projections: jnp.ndarray | np.ndarray,
    *,
    config: FBPConfig | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Filtered backprojection for parallel rays rotating about a fixed axis.

    Cone-beam geometries are reconstructed with :func:`tomojax.recon.fdk.fdk`
    using the same filter; ``backprojector="pallas"`` selects its CUDA kernel.

    Projections: (n_views, nv, nu) -> attenuation volume (nx, ny, nz).
    The rotation axis, arc and angular spacing are fitted from the view poses,
    so tilted (laminography) axes, partial or full turns and irregular angles
    are weighted exactly on every measured frequency. Frequencies no view
    measures, such as laminography's missing cone, reconstruct as zero.

    Backprojection is voxel-driven with bilinear detector interpolation and
    retains filtered tails through the full output volume, assuming zero raw
    attenuation beyond the measured detector. This is not a correction for
    truncated objects. An explicit ``det_grid`` instead uses the ray-model
    adjoint with uniform weights.
    """
    cfg = FBPConfig() if config is None else config
    if cfg.backprojector not in ("auto", "jax", "pallas"):
        raise ValueError("FBP backprojector must be 'auto', 'jax', or 'pallas'")
    if beam_of(geometry) is not None:
        # Cone beams: FDK, with the same filter and the CUDA kernel for "pallas".
        if det_grid is not None:
            raise ValueError("fbp: cone-beam geometry needs the canonical detector grid")
        from tomojax.recon.fdk import FDKConfig, fdk

        backend = {"auto": "auto", "jax": "jax", "pallas": "cuda"}[cfg.backprojector]
        return fdk(
            geometry,
            grid,
            detector,
            projections,
            config=FDKConfig(filter_name=cfg.filter_name, backend=backend),
        )

    validate_grid(grid, "fbp grid")
    n_views, _, _ = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="fbp projections",
    )
    validate_detector_grid(det_grid, detector, context="fbp det_grid")
    # Host arrays (including memmaps) stream through the device batch by batch.
    on_host = not isinstance(projections, jax.Array) and det_grid is None
    proj = projections if on_host else jnp.asarray(projections, dtype=jnp.float32)
    # Precompute poses once
    T_all = stack_view_poses(geometry, n_views)
    validate_pose_stack(T_all, n_views, context="fbp geometry")
    if det_grid is not None:
        return _fbp_explicit_detector_grid(T_all, proj, grid, detector, cfg, det_grid)
    cuda = all(
        device.platform == "gpu" and device.client.platform_version.lower().startswith("cuda")
        for device in (T_all.devices() if on_host else proj.devices())
    )
    if cfg.backprojector == "pallas" and not cuda:
        raise ValueError("Pallas FBP requires CUDA arrays")
    view_scale, params, arc_length, separable = _view_weights(T_all, cfg.scale)
    parallel = _can_use_direct_parallel_fbp(geometry, det_grid)
    if parallel:
        filter_detector = _parallel_filter_detector(grid, detector)
    else:
        filter_detector = _filter_detector(grid, detector, np.asarray(T_all), pad_v=not separable)
    spectrum = _rfft_filter_array(
        cfg.filter_name, filter_detector.nu, float(detector.du), jnp.float32
    )
    n_fft_v = 1 if separable else _fft_length(filter_detector.nv)
    batch = _parallel_filter_batch_size(
        n_views, filter_detector.nv * n_fft_v, 2 * (spectrum.shape[0] - 1)
    )
    backend = "pallas" if cuda and cfg.backprojector != "jax" else "jax"
    z_integer = parallel and supports_parallel_fbp_z_integer(grid, detector)
    while True:
        try:
            if on_host:
                acc = _fbp_from_host(
                    np.asarray(T_all, np.float32),
                    proj,
                    np.asarray(view_scale, np.float32),
                    np.asarray(params, np.float32),
                    spectrum,
                    float(arc_length),
                    grid=grid,
                    detector=filter_detector,
                    backend=backend,
                    batch_size=batch,
                    separable=separable,
                    z_integer=z_integer,
                )
                acc.block_until_ready()
                break
            acc = _run_fbp_streamed(
                T_all,
                proj,
                jnp.asarray(view_scale),
                jnp.asarray(params),
                spectrum,
                jnp.float32(arc_length),
                grid=grid,
                detector=filter_detector,
                backend=backend,
                batch_size=batch,
                z_integer=z_integer,
                separable=separable,
            )
            # Surface asynchronous allocation failures here, where a retry can use smaller batches.
            acc.block_until_ready()
            break
        except Exception as exc:
            if batch == 1 or not _is_fbp_oom_error(exc):
                raise
            batch //= 2
    for _ in progress_iter(range(n_views), total=n_views, desc="FBP: views"):
        pass
    return acc


def _fbp_explicit_detector_grid(
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
            filter_name=cfg.filter_name,
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
            filter_name=cfg.filter_name,
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
    return acc * default_fbp_scale(n_views) if cfg.scale is None else acc * float(cfg.scale)
