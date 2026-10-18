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

from ._fbp_weights import fbp_weights
from .filters import fft_filter_rows, rfft_filter_array

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax._typed_arrays import Device

    from ._host_arrays import ProjectionRows


@dataclass(frozen=True, slots=True, kw_only=True)
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

    ``devices`` (one or several JAX devices) shares the views among them: each
    filters and backprojects its own, weighted for the whole scan, and the
    volumes are summed on the first. Not with an explicit ``det_grid``.

    Parameters
    ----------
    filter : str, optional
        Frequency filter: "ramp", "shepp-logan" or "hann". Default is 'ramp'.
    scale : float | None, optional
        Constant angular weight per view, or None for geometry-derived weights.
    views_per_batch : int, optional
        Number of views processed together to bound temporary storage. Default is 1.
    projector_unroll : int, optional
        Loop unrolling for the explicit-grid or ray-reference projector. Default is
        1.
    checkpoint_projector : bool, optional
        Recompute intermediate projector values during differentiation on the
        reference path. Default is True.
    gather_dtype : str, optional
        Gather precision on the reference path: "auto", "fp32", "fp16" or "bf16". Default is 'fp32'.
    backprojector : Literal['auto', 'jax', 'pallas'], optional
        Backprojection backend: "auto", "jax" or "pallas". Auto uses CUDA
        acceleration when available. Default is 'auto'.
    devices : jax.Device or sequence of jax.Device or None, optional
        JAX device or sequence of devices. None uses the current default; multiple
        devices divide the views and each hold the full volume.
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

    Parameters
    ----------
    n_views : int
        Positive number of views in a uniform half turn.

    Returns
    -------
    float
        Angular weight ``pi / n_views``.

    Raises
    ------
    ValueError
        If n_views is not positive.
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
        return _filter_rows_bounded(rows, view_scale, spectrum)
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


_FILTER_WORKSPACE_BYTES = 128 * 1024**2


def _filter_rows_bounded(
    rows: jnp.ndarray, view_scale: jnp.ndarray, spectrum: jnp.ndarray
) -> jnp.ndarray:
    """Bound FFT scratch without changing backprojection batches or their sums."""
    n = rows.shape[0]
    capacity = max(1, _FILTER_WORKSPACE_BYTES // (16 * rows.shape[1] * 2 * (spectrum.shape[0] - 1)))
    if capacity >= n:
        return fft_filter_rows(rows * view_scale[:, None, None], spectrum)
    size = 1 << (capacity.bit_length() - 1)
    batches = (n + size - 1) // size
    padded = jnp.pad(rows, ((0, batches * size - n), (0, 0), (0, 0)))
    weights = jnp.pad(view_scale, ((0, batches * size - n),))

    def step(_: None, index: jnp.ndarray) -> tuple[None, jnp.ndarray]:
        block = jax.lax.dynamic_slice_in_dim(padded, index * size, size)
        scale = jax.lax.dynamic_slice_in_dim(weights, index * size, size)
        return None, fft_filter_rows(block * scale[:, None, None], spectrum)

    _, values = jax.lax.scan(step, None, jnp.arange(batches))
    return values.reshape(batches * size, rows.shape[1], rows.shape[2])[:n]


def _pad_detector(rows: jnp.ndarray, detector: Detector) -> jnp.ndarray:
    """Zero-extend measured rows symmetrically to the filter detector."""
    pad_v, pad_u = (detector.nv - rows.shape[1]) // 2, (detector.nu - rows.shape[2]) // 2
    return jnp.pad(rows, ((0, 0), (pad_v, pad_v), (pad_u, pad_u))) if pad_u or pad_v else rows


def _parallel_filter_batch_size(n_views: int, nv: int, n_fft: int) -> int:
    """Use a conservative 512 MiB FFT-workspace estimate, not a VRAM guarantee."""
    capacity = max(1, (512 * 1024**2) // (16 * nv * n_fft))
    return n_views if capacity >= n_views else 1 << (capacity.bit_length() - 1)


@jax.jit(
    static_argnames=(
        "grid",
        "detector",
        "backend",
        "batch_size",
        "z_mode",
        "separable",
        "cache_filtered",
    )
)
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
    cache_filtered: bool = False,
) -> jnp.ndarray:
    """Bound filter batches; a bounded axial consumer may retain filtered rows."""
    n, nv, nu = projections.shape
    b = min(batch_size, n)
    if cache_filtered and backend == "pallas" and separable and z_mode == "integer":
        from ._fbp_cuda import backproject_filtered_cuda

        filtered = _filter_axial_stack(projections, view_scale, spectrum, detector, b)
        return backproject_filtered_cuda(
            poses, filtered, grid=grid, detector=detector, z_mode=z_mode
        )

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
            z_mode=z_mode,
        )

    return jax.lax.fori_loop(
        0,
        (n + b - 1) // b,
        step,
        jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32),
    )


def _filter_axial_stack(
    projections: jnp.ndarray,
    view_scale: jnp.ndarray,
    spectrum: jnp.ndarray,
    detector: Detector,
    batch_size: int,
) -> jnp.ndarray:
    """Retain a corrected row slab's filtered views, without a whole-stack FFT.

    The row-slab consumer bounds this storage and retries smaller row blocks on
    OOM. A single gather then avoids repeatedly reading and writing its volume.
    """
    n, nv, nu = projections.shape
    batches = (n + batch_size - 1) // batch_size
    extra = batches * batch_size - n
    data = jnp.pad(projections, ((0, extra), (0, 0), (0, 0)))
    scale = jnp.pad(view_scale, ((0, extra),))

    def step(_: None, chunk: jnp.ndarray) -> tuple[None, jnp.ndarray]:
        start = chunk * batch_size
        rows = jax.lax.dynamic_slice(data, (start, 0, 0), (batch_size, nv, nu))
        weights = jax.lax.dynamic_slice(scale, (start,), (batch_size,))
        filtered = fft_filter_rows(_pad_detector(rows, detector) * weights[:, None, None], spectrum)
        return None, filtered

    _, filtered = jax.lax.scan(step, None, jnp.arange(batches))
    return filtered.reshape(batches * batch_size, detector.nv, detector.nu)[:n]


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
        from ._fbp_cuda import backproject_filtered_cuda

        return backproject_filtered_cuda(
            poses, filtered, grid=grid, detector=detector, z_mode=z_mode, accumulate=accum
        )
    return accum + _backproject_voxels_jax(poses, filtered, grid=grid, detector=detector)


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
        accum, poses, filtered, grid=grid, detector=detector, backend=backend, z_mode=z_mode
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
    z_mode: str,
    check_finite: bool,
    cache_filtered: bool,
) -> jnp.ndarray:
    """Run projections already on the device without a host round trip."""
    projections = projections.astype(jnp.float32)
    if check_finite and not bool(_finite_projections(projections)):
        raise ValueError("fbp_host: projections must be finite in FP32")
    device = next(iter(projections.devices()))
    poses, view_scale, params = (
        jax.device_put(np.asarray(a, np.float32), device) for a in (poses, view_scale, params)
    )
    return _run_fbp_streamed(
        poses,
        projections,
        view_scale,
        params,
        jax.device_put(spectrum, device),
        jnp.float32(arc_length),
        grid=grid,
        detector=detector,
        backend=backend,
        batch_size=min(batch_size, projections.shape[0]),
        z_mode=z_mode,
        separable=separable,
        cache_filtered=cache_filtered,
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
    cache_filtered: bool = False,
) -> jnp.ndarray:
    """Stream host projections through the device in fixed-size view batches.

    Only the output volume, two batches and one filtering workspace occupy the
    device. The final batch is padded with zero-weight views so every batch
    reuses one compiled step, and the next batch is read and transferred while
    the current one runs. Device arrays are filtered in place instead.
    """
    if isinstance(projections, jax.Array):
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
            cache_filtered=cache_filtered,
        )
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
            z_mode=z_mode,
        )
        if start + b < n:
            pending = batch(start + b)
            # Prefetch overlaps this step, but do not enqueue another compute
            # step until it finishes: asynchronous submissions can otherwise
            # retain an unbounded number of input batches on the device.
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
    """Run unscaled z-parallel FBP, retaining filtered tails through the volume.

    Parameters
    ----------
    T_all : array-like
        Object-to-world rigid poses of shape ``(views, 4, 4)`` for rotations about z.
    proj : array-like
        Line integrals of shape ``(views, nv, nu)``.
    grid : Grid
        Reconstruction grid with z slices aligned to detector rows.
    detector : Detector
        Canonical detector grid with rows aligned to voxel z slices.
    filter : str
        Frequency filter: "ramp", "shepp-logan" or "hann".

    Returns
    -------
    jax.Array
        Unscaled float32 backprojection of shape ``(nx, ny, nz)``. Angular
        quadrature weights have not been applied.

    Raises
    ------
    ValueError
        If shapes, poses or detector-row alignment are incompatible.
    """
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
    ramp = rfft_filter_array(filter, detector.nu, float(detector.du), jnp.float32)
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
                        self.spectrum,
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
    parallel = _can_use_direct_parallel_fbp(geometry, None)
    if parallel:
        filter_detector = _parallel_filter_detector(grid, detector)
    else:
        filter_detector = _filter_detector(grid, detector, np.asarray(poses), pad_v=not separable)
    spectrum = rfft_filter_array(cfg.filter, filter_detector.nu, float(detector.du), jnp.float32)
    n_fft_v = 1 if separable else _fft_length(filter_detector.nv)
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
        batch=_parallel_filter_batch_size(
            n_views, filter_detector.nv * n_fft_v, 2 * (spectrum.shape[0] - 1)
        ),
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

    Parameters
    ----------
    geometry : Geometry
        Scan geometry, including view angles and any per-view poses.
    grid : Grid
        Reconstruction grid in ``(x, y, z)`` order, with physical voxel spacing.
    detector : Detector
        Detector dimensions and placement in the same length unit as the grid.
    projections : array-like
        Measured line integrals with shape ``(views, detector.nv, detector.nu)``.
    config : FBPConfig or None, optional
        Solver settings. None uses FBPConfig defaults.
    det_grid : tuple of array-like or None, optional
        Explicit detector-coordinate arrays ``(u, v)``, each of shape ``(nv, nu)``.
        None uses the canonical detector grid.

    Returns
    -------
    jax.Array
        Float32 reconstruction of shape ``(nx, ny, nz)``.

    Raises
    ------
    ValueError
        If shapes, geometry or solver settings are incompatible with this method.

    Examples
    --------
    >>> import numpy as np
    >>> import tomojax as tj
    >>> from tomojax.recon import fbp, FBPConfig, project
    >>> grid = tj.Grid(4, 4, 1, 1.0, 1.0, 1.0)
    >>> detector = tj.Detector(4, 1, 1.0, 1.0)
    >>> geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 8, endpoint=False))
    >>> data = project(geometry, np.ones((4, 4, 1), np.float32))
    >>> volume = fbp(geometry, geometry.grid, geometry.detector, np.asarray(data))
    >>> volume.shape
    (4, 4, 1)
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
    cuda = all(
        device.platform == "gpu" and device.client.platform_version.lower().startswith("cuda")
        for device in (T_all.devices() if on_host else proj.devices())
    )
    if cfg.backprojector == "pallas" and not cuda:
        raise ValueError("Pallas FBP requires CUDA arrays")
    plan = _plan(geometry, grid, detector, T_all, proj, cfg, cuda=cuda)
    split = view_split(cfg.devices, n_views)
    whole = range(n_views)
    acc = plan.backprojected(None, whole) if split is None else split.summed(plan.backprojected)
    for _ in progress_iter(range(n_views), total=n_views, desc="FBP: views"):
        pass
    return acc
