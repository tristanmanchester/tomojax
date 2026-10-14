"""Filtered-backprojection reconstruction routines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import math
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.backends.api import is_oom_error
from tomojax.core import progress_iter
from tomojax.core.devices import as_devices, view_split
from tomojax.core.geometry import Detector, Geometry, Grid, grid_volume_origin
from tomojax.core.geometry.cone import beam_of
from tomojax.core.geometry.parallel import ParallelGeometry
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import (
    validate_detector_grid,
    validate_grid,
    validate_pose_stack,
    validate_projection_stack,
)

from ._fbp_weights import fbp_weights, sparse_view_scales
from .filters import fft_filter_rows, rfft_filter_array

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax._typed_arrays import Device

    from ._host_arrays import ProjectionRows


@dataclass(frozen=True, slots=True, kw_only=True)
class FBPConfig:
    """Configuration for filtered backprojection.

    ``backprojector='auto'`` uses the voxel-driven Pallas kernel on CUDA and JAX
    otherwise. ``'jax'`` forces the reference; ``'pallas'`` requires CUDA execution
    and never silently changes backend.

    ``scale=None`` weights each view exactly for the scanned arc (see
    :func:`fbp`). A number instead weights every view by that constant and
    filters along u only, the convention before exact weights; ``pi / n`` gives
    the uniform half-turn result. ``views_per_batch``, ``projector_unroll``,
    ``checkpoint_projector`` and ``gather_dtype`` apply only with an explicit
    ``det_grid``, which uses the ray-model adjoint.

    ``devices`` (one or several JAX devices) shares the views among them: each
    filters and backprojects its own, weighted for the whole scan, and the
    volumes are summed on the first. Not with an explicit ``det_grid``.
    """

    filter: str = "ramp"
    scale: float | None = None
    views_per_batch: int = 1
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"
    backprojector: Literal["auto", "jax", "pallas"] = "auto"
    devices: Device | Sequence[Device] | None = None  # kept as a tuple

    def __post_init__(self) -> None:
        object.__setattr__(self, "devices", as_devices(self.devices))


def default_fbp_scale(n_views: int) -> float:
    """Return the uniform half-turn angular weight ``pi / n_views``.

    :func:`fbp` derives exact per-view weights by default; this constant is the
    uniform weight for an untilted half turn and the fallback for scans with too
    few views to define a rotation.
    """
    if int(n_views) <= 0:
        raise ValueError("n_views must be positive")
    return float(np.pi / float(n_views))


def validate_fbp_geometry(
    geometry: Geometry, grid: Grid, detector: Detector, scale: float | None, context: str
) -> tuple[int, int, int]:
    """Check the grid, placement, scale and angles before reading projections or poses."""
    shape = validate_grid(grid, f"{context} grid")
    if scale is not None and not math.isfinite(scale):
        raise ValueError(f"{context}: scale must be finite")
    if not np.isfinite([*grid_volume_origin(grid), *detector.center]).all():
        raise ValueError(f"{context}: grid and detector placement must be finite")
    thetas = getattr(geometry, "angles", None)
    if thetas is not None and not np.isfinite(np.asarray(thetas)).all():
        raise ValueError(f"{context}: angles must be finite")
    return shape


def _bilinear_detector(image: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Sample at centred pixel coordinates, excluding the integer image origin."""
    nv, nu = image.shape
    iu, iv = jnp.floor(u).astype(jnp.int32), jnp.floor(v).astype(jnp.int32)
    wu, wv = u - iu.astype(jnp.float32), v - iv.astype(jnp.float32)
    iu, iv = iu + (nu - 1) // 2, iv + (nv - 1) // 2
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
    cu, cv = detector.center

    def body(
        accum: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, None]:
        pose, image = inputs
        world_x = pose[0, 0] * x + pose[0, 1] * y + pose[0, 2] * z + pose[0, 3]
        world_z = pose[2, 0] * x + pose[2, 1] * y + pose[2, 2] * z + pose[2, 3]
        # Symmetric zero-padding changes integer indices, not physical fractions.
        u = (world_x - cu) / detector.du + ((detector.nu - 1) % 2) / 2.0
        v = (world_z - cv) / detector.dv + ((detector.nv - 1) % 2) / 2.0
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
        return fft_filter_rows(rows * view_scale[:, None, None], spectrum)
    n_u, n_v = 2 * (spectrum.shape[0] - 1), _fft_length(detector.nv)
    transformed = jnp.fft.rfft2(rows, s=(n_v, n_u))
    fu = jnp.fft.rfftfreq(n_u, detector.du).astype(jnp.float32)[None, None, :]
    fv = jnp.fft.fftfreq(n_v, detector.dv).astype(jnp.float32)[None, :, None]
    t_u, t_v, c_u, c_v, phase, weight = (params[:, i, None, None] for i in range(6))
    row_ramp = jnp.abs(t_v) <= 1e-6 * jnp.abs(t_u)
    window = spectrum / jnp.maximum(jnp.abs(fu[0, 0]), spectrum[0])

    def response(vertical: jnp.ndarray) -> jnp.ndarray:
        along = t_u * fu + t_v * vertical
        axial = c_u * fu + c_v * vertical
        # A zero ramp coordinate is the limit of nearby frequencies, not a double root.
        other = jnp.mod(
            phase - 2 * jnp.arctan2(jnp.where(along == 0, 1e-30, along), axial), 2 * np.pi
        )
        # FP32 modulo can round a root just below zero to 2*pi. Complete
        # circles still contain both roots, including at that circular seam.
        count = 1.0 + ((arc_length >= 2 * np.pi) | (other < arc_length)).astype(jnp.float32)
        # Keep the existing smoothing window and finite-kernel DC term.
        oblique = jnp.where(
            (fu == 0) & (vertical == 0),
            jnp.hypot(t_u, t_v) * spectrum[0],
            jnp.abs(along) * window,
        )
        return jnp.where(row_ramp, jnp.abs(t_u) * spectrum, oblique) * weight / count

    factor = response(fv)
    nyquist = slice(n_v // 2, n_v // 2 + 1)
    # A real Nyquist mode aliases both physical frequency signs. Keeping only
    # fftfreq's negative v sign makes a camera-frame reflection change the
    # reconstructed volume. Average the filter (including redundancy) over
    # both aliases; the u face also needs the paired v frequencies. At the
    # corner these two averages include all four physical sign choices.
    factor = factor.at[:, nyquist, :].set(
        0.5 * (factor[:, nyquist, :] + response(-fv[:, nyquist, :]))
    )
    reverse_v = (-jnp.arange(n_v)) % n_v
    factor = factor.at[:, :, -1].set(0.5 * (factor[:, :, -1] + factor[:, reverse_v, -1]))
    filtered = jnp.fft.irfft2(transformed * factor, s=(n_v, n_u))
    return filtered[:, : detector.nv, : detector.nu]


def _pad_detector(rows: jnp.ndarray, detector: Detector) -> jnp.ndarray:
    """Zero-extend measured rows symmetrically to the filter detector."""
    pad_v, pad_u = (detector.nv - rows.shape[1]) // 2, (detector.nu - rows.shape[2]) // 2
    return jnp.pad(rows, ((0, 0), (pad_v, pad_v), (pad_u, pad_u))) if pad_u or pad_v else rows


def _parallel_filter_batch_size(n_views: int, nv: int, n_fft: int) -> int:
    """Use a conservative 512 MiB FFT-workspace estimate, not a VRAM guarantee."""
    capacity = max(1, (512 * 1024**2) // (16 * nv * n_fft))
    return n_views if capacity >= n_views else 1 << (capacity.bit_length() - 1)


@jax.jit(static_argnames=("grid", "detector", "backend", "batch_size", "z_mode", "separable"))
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
    z_mode: str,
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
        return _filter_and_backproject(
            accum,
            batch_poses,
            rows,
            jax.lax.dynamic_slice(view_scale, (start,), (b,)),
            jax.lax.dynamic_slice(params, (start, 0), (b, params.shape[1])),
            spectrum,
            arc_length,
            grid=grid,
            detector=detector,
            backend=backend,
            separable=separable,
            z_mode=z_mode,
        )

    return jax.lax.fori_loop(
        0,
        (n + b - 1) // b,
        step,
        jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32),
    )


def _pack_fixed_rows(
    poses: jnp.ndarray,
    rows: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    integer: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, Grid, Detector]:
    """Sample common axial rows once per view, rather than per xy voxel.

    Match the sampler's centred-coordinate decomposition before adding its
    integer row origin; padding must not change the interpolation fractions.
    Horizontal poses and detector coordinates remain unchanged. Integer mode
    preserves the nearest-row convention, including its alignment tolerance.
    """
    ox, oy, oz = grid_volume_origin(grid)
    z = jnp.arange(grid.nz, dtype=jnp.float32) * jnp.float32(grid.vz) + jnp.float32(oz)
    world_z = z if integer else poses[0, 2, 2] * z + poses[0, 2, 3]
    v = (world_z - jnp.float32(detector.center[1])) / jnp.float32(detector.dv)
    v = v + jnp.float32(((detector.nv - 1) % 2) / 2)
    lower = jnp.floor(v + 0.5) if integer else jnp.floor(v)
    index = lower.astype(jnp.int32) + (detector.nv - 1) // 2

    def take(row: jnp.ndarray) -> jnp.ndarray:
        values = jnp.take(rows, jnp.clip(row, 0, detector.nv - 1), axis=1)
        valid = (row >= 0) & (row < detector.nv)
        return jnp.where(valid[None, :, None], values, 0.0)

    packed = take(index)
    if not integer:
        weight = v - lower
        packed = packed * (1 - weight[None, :, None])
        packed = packed + take(index + 1) * weight[None, :, None]
    local_grid = replace(grid, vol_origin=(ox, oy, 0.0), vol_center=None)
    local_detector = replace(
        detector, nv=grid.nz, dv=grid.vz, center=(detector.center[0], (grid.nz - 1) * grid.vz / 2)
    )
    local_poses = poses.at[:, 2, :].set(jnp.array([0.0, 0.0, 1.0, 0.0], jnp.float32))
    return local_poses, packed, local_grid, local_detector


def _backproject_into(
    accum: jnp.ndarray,
    poses: jnp.ndarray,
    filtered: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    z_mode: str,
) -> jnp.ndarray:
    """Add filtered views to ``accum``; the CUDA kernel adds in place."""
    if backend == "pallas":
        from ._fbp_pallas import backproject_filtered_pallas

        # Compact working sets keep the existing direct sampler: packing adds
        # a launch and a plane workspace. Bound that workspace to twice the
        # source rows even for a strongly upsampled or retained output grid.
        working_bytes = filtered.size * filtered.dtype.itemsize + 4 * grid.nx * grid.ny * grid.nz
        if z_mode == "fixed" and grid.nz <= 2 * detector.nv and working_bytes > 64 * 1024**2:
            poses, filtered, grid, detector = _pack_fixed_rows(
                poses, filtered, grid=grid, detector=detector
            )
            z_mode = "integer"
        return backproject_filtered_pallas(
            poses, filtered, grid=grid, detector=detector, z_mode=z_mode, accumulate=accum
        )
    return accum + _backproject_voxels_jax(poses, filtered, grid=grid, detector=detector)


def _filter_and_backproject(
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
    z_mode: str,
) -> jnp.ndarray:
    """Filter padded rows, packing axial samples first when it saves FFT work."""
    # Axial interpolation commutes with the separable u filter. Keep compact
    # FFTs on the existing path: an extra gather needs a launch and workspace.
    # Use the same conservative FFT-byte estimate as the view-batch planner.
    filtering_bytes = 16 * rows.shape[0] * detector.nv * _fft_length(detector.nu)
    if (
        backend == "pallas"
        and separable
        and z_mode in ("integer", "fixed")
        and grid.nz < detector.nv
        and filtering_bytes > 64 * 1024**2
    ):
        poses, rows, grid, detector = _pack_fixed_rows(
            poses, rows, grid=grid, detector=detector, integer=z_mode == "integer"
        )
        z_mode = "integer"
    filtered = _filter_views(
        rows,
        view_scale,
        params,
        spectrum,
        arc_length,
        detector=detector,
        separable=separable,
    )
    return _backproject_into(
        accum, poses, filtered, grid=grid, detector=detector, backend=backend, z_mode=z_mode
    )


@jax.jit(
    static_argnames=("grid", "detector", "backend", "separable", "z_mode"),
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
    z_mode: str,
) -> jnp.ndarray:
    """Filter one batch of views and add its backprojection to ``accum`` in place."""
    return _filter_and_backproject(
        accum,
        poses,
        _pad_detector(rows, detector),
        view_scale,
        params,
        spectrum,
        arc_length,
        grid=grid,
        detector=detector,
        backend=backend,
        separable=separable,
        z_mode=z_mode,
    )


@jax.jit
def _finite_projections(projections: jax.Array) -> jax.Array:
    return jnp.all(jnp.isfinite(projections))


def _fbp_from_resident(
    poses: np.ndarray,
    projections: jax.Array,
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
    z_mode: str = "general",
    check_finite: bool = False,
) -> jnp.ndarray:
    """Run an already-budgeted resident slab without a host round trip."""
    if check_finite and not bool(_finite_projections(projections)):
        raise ValueError("fbp_host: projections must be finite in FP32")
    device = next(iter(spectrum.devices()))
    metadata = [
        jax.device_put(np.asarray(a, np.float32), device) for a in (poses, view_scale, params)
    ]
    return _run_fbp_streamed(
        metadata[0],
        projections,
        metadata[1],
        metadata[2],
        spectrum,
        jnp.float32(arc_length),
        grid=grid,
        detector=detector,
        backend=backend,
        batch_size=batch_size,
        z_mode=z_mode,
        separable=separable,
    )


def _fbp_from_host(
    poses: np.ndarray,
    projections: np.ndarray | jax.Array | ProjectionRows,
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
    z_mode: str = "general",
    check_finite: bool = False,
    resident_slab: bool = False,
) -> jnp.ndarray:
    """Stream host projections through the device in fixed-size view batches.

    CUDA may queue two computations with a third prefetched input batch when
    the additional batch and estimated filtering workspace fit the memory
    budget. Otherwise one computation and at most two input batches are live.
    The final batch is padded with zero-weight views, reusing one compiled step.
    """
    if resident_slab:
        assert isinstance(projections, jax.Array)
        return _fbp_from_resident(
            poses,
            projections,
            view_scale,
            params,
            spectrum,
            arc_length,
            grid=grid,
            detector=detector,
            backend=backend,
            batch_size=batch_size,
            separable=separable,
            z_mode=z_mode,
            check_finite=check_finite,
        )
    n = projections.shape[0]
    b = min(batch_size, n)
    compute_window = 1
    if backend == "pallas" and n > b:
        from tomojax.backends import device_free_memory_bytes

        free = device_free_memory_bytes()
        per_view = (
            math.prod(projections.shape[1:])
            + math.prod(poses.shape[1:])
            + math.prod(params.shape[1:])
            + 1  # Angular scale.
        )
        fft_rows = detector.nv if separable else _fft_length(detector.nv)
        workspace = 16 * b * fft_rows * (spectrum.shape[0] + detector.nu)
        if free is not None and 4 * b * per_view + workspace <= 0.05 * free:
            compute_window = 2

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
            z_mode=z_mode,
        )
        if start + b < n:
            pending = batch(start + b)
            # Bound submitted work while overlapping transfers. Previous
            # accumulators were donated, so wait only on the latest result.
            if (start // b + 1) % compute_window == 0:
                accum.block_until_ready()
    return accum


def supports_parallel_fbp_z_integer(grid: Grid, detector: Detector) -> bool:
    """Return whether the detector rows align with z-slices for direct Pallas FBP."""
    tol = 1e-5
    origin_z = float(grid_volume_origin(grid)[2])
    first = (origin_z - float(detector.center[1])) / float(detector.dv)
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
    filter: str,
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
    required_radius = np.max(np.abs(coordinates - detector.center[0])) / detector.du
    padding = max(0, math.ceil(required_radius - (detector.nu - 1) / 2))
    detector = replace(detector, nu=detector.nu + 2 * padding)
    ramp = rfft_filter_array(
        filter, detector.nu, float(detector.du), jnp.float32, source_nu=proj.shape[2]
    )
    from ._fbp_pallas import filtered_view_batch_size

    T_all = jnp.asarray(T_all, dtype=jnp.float32)
    batch = _parallel_filter_batch_size(n_views, detector.nv, _fft_length(detector.nu))
    batch = filtered_view_batch_size(
        batch, grid=grid, detector=detector, device=next(iter(T_all.devices()))
    )
    return _run_fbp_streamed(
        T_all,
        jnp.asarray(proj, dtype=jnp.float32),
        jnp.ones((n_views,), jnp.float32),
        jnp.zeros((n_views, 6), jnp.float32),
        ramp,
        jnp.float32(0),
        grid=grid,
        detector=detector,
        backend="pallas",
        batch_size=batch,
        z_mode="integer",
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
    required_half_width = (math.hypot(rx, ry) + abs(detector.center[0])) / detector.du
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
    reach_u = np.max(np.abs(u - detector.center[0])) / detector.du
    reach_v = np.max(np.abs(v - detector.center[1])) / detector.dv
    extra_u = max(0, math.ceil(reach_u - (detector.nu - 1) / 2))
    extra_v = max(0, math.ceil(reach_v - (detector.nv - 1) / 2)) if pad_v else 0
    return replace(detector, nu=detector.nu + 2 * extra_u, nv=detector.nv + 2 * extra_v)


def _view_weights(
    poses: jnp.ndarray, scale: float | None
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Return per-view scales, filter parameters, arc length and separability.

    An explicit ``scale`` keeps the uniform weighting: one constant per view
    times a ramp along u. Otherwise weights are exact for the fitted circular
    scan; scans with too few directions use uniform plane cells, sharing
    repeated exposures' measure. This remains a sparse u-ramp approximation.
    """
    n = int(poses.shape[0])
    if scale is None:
        try:
            weights = fbp_weights(np.asarray(poses))
        except ValueError:
            return sparse_view_scales(np.asarray(poses)), np.zeros((n, 6), np.float32), 0.0, True
        else:
            return weights.view_scale, weights.params, weights.arc_length, weights.separable
    return np.full(n, scale, np.float32), np.zeros((n, 6), np.float32), 0.0, True


def canonical_fbp_detector_axes(
    poses: jnp.ndarray,
    projections: jnp.ndarray | np.ndarray,
    *,
    detector: Detector,
    params: np.ndarray,
    separable: bool,
) -> tuple[jnp.ndarray, jnp.ndarray | np.ndarray, Detector]:
    """Re-express a column ramp as a row ramp, without interpolating measurements.

    A quarter-turned camera has its ramp along v. Keeping that orientation in
    the 2-D path incorrectly applies the u smoothing window and finite-ramp
    length to it. A rigid camera-coordinate change exchanges the pixel axes,
    centres and spacings exactly; the object grid and ray directions stay put.
    Explicit uniform scaling keeps its documented u-only filter unchanged.
    """
    if separable or not np.all(np.abs(params[:, 0]) <= 1e-6 * np.abs(params[:, 1])):
        return poses, projections, detector
    return (
        jnp.stack([-poses[:, 2], poses[:, 1], poses[:, 0], poses[:, 3]], axis=1),
        projections.swapaxes(1, 2)[:, :, ::-1],
        replace(
            detector,
            nu=detector.nv,
            nv=detector.nu,
            du=detector.dv,
            dv=detector.du,
            center=(-detector.center[1], detector.center[0]),
        ),
    )


@dataclass(frozen=True)
class _Plan:
    """One scan's FBP: its exact view weights, filter and backprojector (see :func:`_plan`)."""

    poses: jnp.ndarray
    projections: jnp.ndarray | np.ndarray  # host arrays stream batch by batch
    view_scale: np.ndarray
    params: np.ndarray
    arc_length: float
    separable: bool
    spectrum: jnp.ndarray
    grid: Grid
    detector: Detector  # the filter's, padded
    backend: str
    batch: int
    z_mode: str

    def backprojected(self, device: Device | None, views: range) -> jnp.ndarray:
        """The weighted backprojection of ``views``, on ``device`` (JAX's default for None).

        Batches halve after an out-of-memory error until one view fits.
        """
        part, size = slice(views.start, views.stop), self.batch
        put = partial(jax.device_put, device=device)
        if not self.separable or (self.backend == "pallas" and self.z_mode != "general"):
            from ._fbp_pallas import fft_2d_view_batch_size, filtered_view_batch_size

            resident = (
                self.projections
                if isinstance(self.projections, jax.Array) and self.projections.committed
                else self.poses
            )
            target = device if device is not None else next(iter(put(resident).devices()))
            if self.separable:
                size = filtered_view_batch_size(
                    size, grid=self.grid, detector=self.detector, device=target
                )
            else:
                size = fft_2d_view_batch_size(
                    size,
                    fft_shape=(_fft_length(self.detector.nv), 2 * (self.spectrum.shape[0] - 1)),
                    device=target,
                )
        options = {
            "grid": self.grid,
            "detector": self.detector,
            "backend": self.backend,
            "separable": self.separable,
            "z_mode": self.z_mode,
        }
        while True:
            try:
                if isinstance(self.projections, jax.Array):
                    acc = _run_fbp_streamed(
                        put(self.poses[part]),
                        put(self.projections[part]),
                        put(jnp.asarray(self.view_scale[part])),
                        put(jnp.asarray(self.params[part])),
                        put(self.spectrum),
                        jnp.float32(self.arc_length),
                        batch_size=size,
                        **options,
                    )
                else:
                    acc = _fbp_from_host(
                        np.asarray(self.poses, np.float32)[part],
                        self.projections[part],
                        self.view_scale[part],
                        self.params[part],
                        put(self.spectrum),
                        self.arc_length,
                        batch_size=size,
                        **options,
                    )
                # Surface asynchronous allocation failures here, where a retry
                # can use smaller batches.
                acc.block_until_ready()
                return acc
            except Exception as exc:
                if size == 1 or not is_oom_error(exc):
                    raise
                size //= 2


def _plan(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    poses: jnp.ndarray,
    projections: jnp.ndarray | np.ndarray,
    cfg: FBPConfig,
    *,
    cuda: bool,
) -> _Plan:
    """The weights, filter and backprojector for :func:`fbp` of a parallel-beam scan."""
    n_views = int(poses.shape[0])
    view_scale, params, arc_length, separable = _view_weights(poses, cfg.scale)
    original = poses
    poses, projections, detector = canonical_fbp_detector_axes(
        poses, projections, detector=detector, params=params, separable=separable
    )
    if poses is not original:
        view_scale, params, arc_length, separable = _view_weights(poses, cfg.scale)
    parallel = _can_use_direct_parallel_fbp(geometry, None)
    if parallel:
        filter_detector = _parallel_filter_detector(grid, detector)
    else:
        filter_detector = _filter_detector(grid, detector, np.asarray(poses), pad_v=not separable)
    spectrum = rfft_filter_array(
        cfg.filter,
        filter_detector.nu,
        float(detector.du),
        jnp.float32,
        source_nu=projections.shape[2] if separable else None,
    )
    # Row-wise FFTs keep nv independent rows; a 2D FFT replaces that dimension
    # with its padded length, rather than multiplying by it again.
    fft_rows = filter_detector.nv if separable else _fft_length(filter_detector.nv)
    return _Plan(
        poses=poses,
        projections=projections,
        view_scale=np.asarray(view_scale, np.float32),
        params=np.asarray(params, np.float32),
        arc_length=float(arc_length),
        separable=bool(separable),
        spectrum=spectrum,
        grid=grid,
        detector=filter_detector,
        backend="pallas" if cuda and cfg.backprojector != "jax" else "jax",
        # Preserve the conservative output-width batch bound: a shorter FFT
        # does not make a larger filtered view stack cache-resident.
        batch=_parallel_filter_batch_size(n_views, fft_rows, _fft_length(filter_detector.nu)),
        z_mode=("integer" if supports_parallel_fbp_z_integer(grid, detector) else "fixed")
        if parallel
        else "general",
    )


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
    are weighted exactly on every measured frequency. Repeated directions
    share their angular weight equally. Frequencies no view
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
            config=FDKConfig(filter=cfg.filter, backend=backend, devices=cfg.devices),
        )

    validate_fbp_geometry(geometry, grid, detector, cfg.scale, "fbp")
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
        if cfg.devices is not None:
            raise ValueError("fbp: an explicit det_grid cannot be shared among devices")
        from ._fbp_detector_grid import fbp_on_detector_grid

        return fbp_on_detector_grid(T_all, proj, grid, detector, cfg, det_grid)
    resident = isinstance(proj, jax.Array) and proj.committed
    execution_devices = as_devices(cfg.devices) or (proj.devices() if resident else T_all.devices())
    cuda = all(
        device.platform == "gpu"
        and getattr(getattr(device, "client", None), "platform_version", "")
        .lower()
        .startswith("cuda")
        for device in execution_devices
    )
    if cfg.backprojector == "pallas" and not cuda:
        raise ValueError("Pallas FBP requires CUDA execution devices")
    plan = _plan(geometry, grid, detector, T_all, proj, cfg, cuda=cuda)
    split = view_split(cfg.devices, n_views)
    whole = range(n_views)
    if split is None:
        device = next(iter(execution_devices))
        with jax.default_device(device):
            # Keep a committed distributed input's existing sharding instead
            # of gathering the whole scan onto this compilation-context device.
            acc = plan.backprojected(device if len(execution_devices) == 1 else None, whole)
    else:
        acc = split.summed(plan.backprojected)
    for _ in progress_iter(range(n_views), total=n_views, desc="FBP: views"):
        pass
    return acc
