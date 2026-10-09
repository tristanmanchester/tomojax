"""Reference JAX projector and backprojector operators."""

# ruff: noqa: ANN001, ANN202

from __future__ import annotations

import functools
import logging
import math
import operator
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.backend_policy import cuda_backend
from tomojax.core.pallas_resolver import resolve_pallas_callable

from .backend_policy import ProjectorBackendInput, normalize_projector_backend
from .compilation_cache import enable_persistent_compilation_cache
from .geometry.base import Detector, Geometry, Grid, ScanGeometry, grid_volume_origin
from .validation import (
    validate_detector,
    validate_detector_grid,
    validate_detector_image,
    validate_grid,
    validate_pose_matrix,
    validate_pose_stack,
    validate_projection_stack,
    validate_volume,
)

# Frame conventions:
# - geometry.pose_for_view(i) must return a 4x4 transform T_world_from_obj that maps
#   object (sample) coordinates into world (lab) coordinates for view i.
# - Rays are defined in the world frame with directions along +y (parallel beam).
# - We compute object_from_world = inv(T_world_from_obj) and sample the volume in the
#   object frame directly. This makes reconstructed volumes live in the object (sample) frame.

LOG = logging.getLogger(__name__)

_JOSEPH_INTEGRATORS = {"joseph": "linear", "joseph_cubic": "cubic"}
RAY_INTEGRATORS = ("sampled", "exact", *_JOSEPH_INTEGRATORS)


def _joseph_operands(
    poses: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    ray_integrator: str,
) -> tuple[jnp.ndarray, str]:
    """Plane coefficients and interpolation for the Joseph integrators.

    An explicit ``det_grid`` must be an affine pixel lattice (offset or rolled).
    """
    from tomojax.core.joseph import plane_coefficients

    coefficients = plane_coefficients(poses, grid, detector, det_grid)
    return coefficients, _JOSEPH_INTEGRATORS[ray_integrator]


enable_persistent_compilation_cache()


def _volume_origin(grid: Grid) -> jnp.ndarray:
    """Return the configured location of voxel (0, 0, 0)'s centre."""
    return jnp.asarray(grid_volume_origin(grid), dtype=jnp.float32)


def _interpolation_support_bounds(
    grid: Grid, vol_origin: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a conservative object-space support box for trilinear sampling.

    ``vol_origin`` denotes the centre of voxel ``(0, 0, 0)``. The projector
    samples voxel centres and relies on trilinear interpolation, so the non-zero
    support extends one voxel before that first centre and one voxel beyond the
    last voxel centre along each axis.
    """
    voxel = jnp.array([grid.vx, grid.vy, grid.vz], dtype=jnp.float32)
    upper = vol_origin + jnp.array(
        [grid.nx * grid.vx, grid.ny * grid.vy, grid.nz * grid.vz],
        dtype=jnp.float32,
    )
    return vol_origin - voxel, upper


@functools.lru_cache(maxsize=8)
def _build_detector_grid_cached(
    nu: int,
    nv: int,
    du: float,
    dv: float,
    cx: float,
    cz: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Build on host as NumPy to avoid capturing JAX tracers in global cache under jit
    u = (np.arange(nu, dtype=np.float32) - (nu / 2.0 - 0.5)) * np.float32(du) + np.float32(cx)
    v = (np.arange(nv, dtype=np.float32) - (nv / 2.0 - 0.5)) * np.float32(dv) + np.float32(cz)
    X = np.tile(u, nv)
    Z = np.repeat(v, nu)
    X.setflags(write=False)
    Z.setflags(write=False)
    return X, Z


def _build_detector_grid(det: Detector) -> tuple[np.ndarray, np.ndarray]:
    return _build_detector_grid_cached(
        int(det.nu),
        int(det.nv),
        float(det.du),
        float(det.dv),
        float(det.center[0]),
        float(det.center[1]),
    )


def _build_detector_grid_device_uncached(det: Detector) -> tuple[jnp.ndarray, jnp.ndarray]:
    nu, nv = int(det.nu), int(det.nv)
    du, dv = jnp.float32(float(det.du)), jnp.float32(float(det.dv))
    cx, cz = jnp.float32(float(det.center[0])), jnp.float32(float(det.center[1]))
    u = (jnp.arange(nu, dtype=jnp.float32) - jnp.float32(nu / 2.0 - 0.5)) * du + cx
    v = (jnp.arange(nv, dtype=jnp.float32) - jnp.float32(nv / 2.0 - 0.5)) * dv + cz
    return jnp.tile(u, nv), jnp.repeat(v, nu)


def get_detector_grid_device(det: Detector) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return detector coordinate grids as device arrays.

    Note: call this outside of any JAX-transformed context (jit/grad/scan) to avoid
    side effects during tracing. Safe to cache at the application level.
    """
    X_np, Z_np = _build_detector_grid(det)
    return jnp.asarray(X_np, dtype=jnp.float32), jnp.asarray(Z_np, dtype=jnp.float32)


def _resolve_gather_target(gather_dtype: str) -> jnp.dtype:
    """Resolve the gather/interpolation dtype for the forward projector."""
    if not isinstance(gather_dtype, str):
        raise ValueError(f"gather_dtype must be a string; got {type(gather_dtype).__name__}")
    gd = gather_dtype.lower()
    if gd == "auto":
        platform = jax.default_backend()
        if platform in ("gpu", "tpu"):
            return jnp.bfloat16 if platform == "tpu" else jnp.float16
        return jnp.float32
    if gd in ("bf16", "bfloat16"):
        return jnp.bfloat16
    if gd in ("fp16", "float16", "half"):
        return jnp.float16
    if gd in ("fp32", "float32", "single"):
        return jnp.float32
    raise ValueError(
        "gather_dtype must be one of 'auto', 'fp32', 'float32', 'single', "
        "'bf16', 'bfloat16', 'fp16', 'float16', or 'half'; "
        f"got {gather_dtype!r}"
    )


def _prepare_volume_for_gather(volume: jnp.ndarray, gather_dtype: str) -> jnp.ndarray:
    target = _resolve_gather_target(gather_dtype)
    vol_cast = volume if volume.dtype == target else volume.astype(target)
    return jnp.ravel(vol_cast, order="C")


def _resolve_detector_grid(
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    if det_grid is None:
        Xr, Zr = _build_detector_grid_device_uncached(detector)
        n_rays = int(detector.nu) * int(detector.nv)
    else:
        Xr, Zr = det_grid
        n_rays = int(Xr.shape[0])
    return Xr, Zr, n_rays


def _resolve_n_steps(grid: Grid, step_size: float, n_steps: int | None) -> int:
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError(f"projector traversal step_size must be finite and > 0; got {step_size!r}")
    if n_steps is not None:
        if isinstance(n_steps, bool):
            raise ValueError(
                f"projector traversal n_steps must be a positive integer; got {n_steps!r}"
            )
        try:
            n_steps_val = operator.index(n_steps)
        except TypeError:
            raise ValueError(
                f"projector traversal n_steps must be a positive integer; got {n_steps!r}"
            ) from None
        if n_steps_val <= 0:
            raise ValueError(
                f"projector traversal n_steps must be a positive integer; got {n_steps!r}"
            )
        return n_steps_val
    support_lengths = (
        float((grid.nx + 1) * grid.vx),
        float((grid.ny + 1) * grid.vy),
        float((grid.nz + 1) * grid.vz),
    )
    max_path_length = math.sqrt(sum(length * length for length in support_lengths))
    return math.ceil(max_path_length / float(step_size))


def _projector_traversal_state(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.float32,
    int,
    int,
]:
    """Return the fixed per-ray traversal state shared by forward and adjoint passes."""
    vol_origin = _volume_origin(grid)
    Xr, Zr, n_rays = _resolve_detector_grid(detector, det_grid)
    if step_size is None:
        step_size = float(grid.vy)
    n_steps_val = _resolve_n_steps(grid, float(step_size), n_steps)

    R = T[:3, :3]
    t = T[:3, 3]
    Rinv = R.T
    # Geometry must retain FP32 accuracy even when users enable reduced
    # precision for large matrix products elsewhere in their application.
    tinv = -jnp.matmul(Rinv, t, precision=jax.lax.Precision.HIGHEST)
    ey_obj = Rinv[:, 1]
    support_lower, support_upper = _interpolation_support_bounds(grid, vol_origin)

    xr = Xr[jnp.newaxis, :]
    zr = Zr[jnp.newaxis, :]
    base = Rinv[:, 0:1] * xr + Rinv[:, 2:3] * zr + tinv[:, None]

    lower = support_lower[:, None]
    upper = support_upper[:, None]
    denom = ey_obj[:, None]
    eps = jnp.float32(1e-8)
    parallel = jnp.abs(denom) < eps
    safe_denom = jnp.where(parallel, jnp.ones_like(denom), denom)
    t1 = (lower - base) / safe_denom
    t2 = (upper - base) / safe_denom
    lo = jnp.minimum(t1, t2)
    hi = jnp.maximum(t1, t2)
    inside = (base >= lower) & (base <= upper)
    inf = jnp.asarray(jnp.inf, dtype=jnp.float32)
    lo = jnp.where(parallel, jnp.where(inside, -inf, inf), lo)
    hi = jnp.where(parallel, jnp.where(inside, inf, -inf), hi)
    y_entry = jnp.max(lo, axis=0)
    y_exit = jnp.min(hi, axis=0)
    path_length = jnp.maximum(jnp.float32(0.0), y_exit - y_entry)
    valid_rays = path_length > 0.0
    y_start = jnp.where(valid_rays, y_entry, jnp.float32(0.0))
    step_size32 = jnp.float32(step_size)
    n_steps_ray = jnp.where(
        valid_rays,
        jnp.ceil(path_length / step_size32).astype(jnp.int32),
        jnp.int32(0),
    )

    q0 = base + y_start[None, :] * ey_obj[:, None]
    dq = (step_size32 * ey_obj)[:, None]
    inv_vx = jnp.float32(1.0 / grid.vx)
    inv_vy = jnp.float32(1.0 / grid.vy)
    inv_vz = jnp.float32(1.0 / grid.vz)
    ix0 = (q0[0] - vol_origin[0]) * inv_vx
    iy0 = (q0[1] - vol_origin[1]) * inv_vy
    iz0 = (q0[2] - vol_origin[2]) * inv_vz
    dix = dq[0] * inv_vx
    diy = dq[1] * inv_vy
    diz = dq[2] * inv_vz
    return ix0, iy0, iz0, dix, diy, diz, n_steps_ray, step_size32, n_steps_val, n_rays


@jax.jit
def _flat_index(ix, iy, iz, _nx, ny, nz):
    return ix * (ny * nz) + iy * nz + iz


@jax.jit
def _trilinear_gather(recon_flat, ix_f, iy_f, iz_f, nx, ny, nz):
    fx = jnp.floor(ix_f).astype(jnp.int32)
    fy = jnp.floor(iy_f).astype(jnp.int32)
    fz = jnp.floor(iz_f).astype(jnp.int32)
    cx, cy, cz = fx + 1, fy + 1, fz + 1

    wx1 = ix_f - fx.astype(jnp.float32)
    wy1 = iy_f - fy.astype(jnp.float32)
    wz1 = iz_f - fz.astype(jnp.float32)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    wz0 = 1.0 - wz1

    def gather(ix, iy, iz):
        inb = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)).astype(
            jnp.float32
        )
        idx = _flat_index(ix, iy, iz, nx, ny, nz)
        val = jnp.take(recon_flat, idx, mode="clip")
        return inb * val

    c000 = gather(fx, fy, fz) * (wx0 * wy0 * wz0)
    c001 = gather(fx, fy, cz) * (wx0 * wy0 * wz1)
    c010 = gather(fx, cy, fz) * (wx0 * wy1 * wz0)
    c011 = gather(fx, cy, cz) * (wx0 * wy1 * wz1)
    c100 = gather(cx, fy, fz) * (wx1 * wy0 * wz0)
    c101 = gather(cx, fy, cz) * (wx1 * wy0 * wz1)
    c110 = gather(cx, cy, fz) * (wx1 * wy1 * wz0)
    c111 = gather(cx, cy, cz) * (wx1 * wy1 * wz1)

    return c000 + c001 + c010 + c011 + c100 + c101 + c110 + c111


@jax.jit
def _trilinear_scatter_add(acc_flat, ray_vals, ix_f, iy_f, iz_f, nx, ny, nz):
    if acc_flat.dtype == jnp.float32:
        # Scatter into the existing accumulator. Transposing gather first builds
        # a dense zero-filled volume at every ray step, then adds that volume.
        # A single sparse update avoids O(volume_size * n_steps) memory traffic.
        fx = jnp.floor(ix_f).astype(jnp.int32)
        fy = jnp.floor(iy_f).astype(jnp.int32)
        fz = jnp.floor(iz_f).astype(jnp.int32)
        ox = jnp.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32)[:, None]
        oy = jnp.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=jnp.int32)[:, None]
        oz = jnp.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=jnp.int32)[:, None]
        ix, iy, iz = fx[None, :] + ox, fy[None, :] + oy, fz[None, :] + oz
        wx, wy, wz = ix_f - fx, iy_f - fy, iz_f - fz
        weight = (
            jnp.where(ox == 0, 1.0 - wx, wx)
            * jnp.where(oy == 0, 1.0 - wy, wy)
            * jnp.where(oz == 0, 1.0 - wz, wz)
        )
        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        idx = jnp.where(inb, ix * (ny * nz) + iy * nz + iz, acc_flat.size)
        return acc_flat.at[idx.ravel()].add((weight * ray_vals).ravel(), mode="drop")
    # Keep the transpose's per-corner casts for the mixed-precision contract.
    scatter = jax.linear_transpose(
        lambda recon: _trilinear_gather(recon, ix_f, iy_f, iz_f, nx, ny, nz),
        acc_flat,
    )
    return acc_flat + scatter(ray_vals)[0]


def _pallas_unsupported_exception_type() -> type[Exception] | None:
    unsupported_exc, _reason = resolve_pallas_callable(
        "PallasProjectorUnsupported",
        missing_reason="pallas_unsupported_exception_missing",
    )
    if isinstance(unsupported_exc, type) and issubclass(unsupported_exc, Exception):
        return unsupported_exc
    return None


def _cone_backend(projector_backend: object) -> str:
    from tomojax.core.cone import use_cuda_cone

    return "cuda" if projector_backend == "pallas" and use_cuda_cone() else "jax"


def _cone_views(
    poses: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    frames: jnp.ndarray | np.ndarray,
    projector_backend: object = "jax",
) -> jnp.ndarray:
    from tomojax.core.cone import cone_project, frame_coefficients

    coeff = frame_coefficients(poses, frames, grid, detector)
    return cone_project(volume, coeff, grid, detector, backend=_cone_backend(projector_backend))


def _cone_backproject_views(
    poses: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    images: jnp.ndarray,
    frames: jnp.ndarray | np.ndarray,
) -> jnp.ndarray:
    from tomojax.core.cone import cone_backproject, frame_coefficients

    coeff = frame_coefficients(poses, frames, grid, detector)
    return cone_backproject(images, coeff, grid, detector)


def _view_frame(geometry: Geometry, detector: Detector, view_index: int) -> np.ndarray | None:
    """View ``view_index``'s cone-beam lab frame, or None for a parallel beam."""
    from tomojax.core.cone import cone_model

    cone = cone_model(geometry, detector)
    if cone is None:
        return None
    return cone.frames(len(cast("ScanGeometry", geometry).angles))[view_index]


def forward_project_view_T(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    use_checkpoint: bool = True,
    unroll: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    projector_backend: ProjectorBackendInput = "jax",
    ray_integrator: str = "sampled",
    frames: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Forward project a single view given pose `T` (4x4, row-major).

    Contract: T is world_from_object for the view. The projector constructs detector
    rays in world coordinates and transforms them into object coordinates using
    inv(T), then performs incremental stepping along the beam direction expressed
    in the object frame. This avoids a matmul per step and keeps gradients clean.
    ``frames``, the view's cone-beam lab frame (see
    :func:`tomojax.core.cone.beam_frame`), selects the cone-beam Joseph
    projector: ``projector_backend`` ``"pallas"`` runs its CUDA kernels,
    ``"jax"`` the differentiable reference.
    """
    if frames is not None:
        return _cone_views(T[None], grid, detector, volume, frames, projector_backend)[0]
    backend = normalize_projector_backend(projector_backend)
    if ray_integrator == "exact":
        if step_size is not None or n_steps is not None:
            raise ValueError("exact ray integration does not accept a step size or sample count")
        from tomojax.core.trilinear import exact_forward

        return exact_forward(
            jnp.asarray(T)[None], grid, detector, volume, backend=backend, det_grid=det_grid
        )[0]
    if ray_integrator in _JOSEPH_INTEGRATORS:
        from tomojax.core.joseph import forward_project_planes

        coeff, interpolation = _joseph_operands(
            jnp.asarray(T)[None], grid, detector, det_grid, ray_integrator
        )
        return forward_project_planes(
            coeff, volume, grid, detector, backend=backend, interpolation=interpolation
        )[0]
    if ray_integrator != "sampled":
        raise ValueError(f"ray_integrator must be one of {RAY_INTEGRATORS}")
    if backend == "pallas":
        pallas_project, fallback_reason = resolve_pallas_callable(
            "forward_project_view_T_pallas",
            missing_reason="pallas_single_view_callable_missing",
        )
        if pallas_project is not None:
            unsupported_exc = _pallas_unsupported_exception_type()
            try:
                options_cls, _options_reason = resolve_pallas_callable(
                    "PallasProjectorOptions",
                    missing_reason="pallas_options_missing",
                )
                options = (
                    options_cls(
                        step_size=step_size,
                        n_steps=n_steps,
                        unroll=unroll,
                        gather_dtype=gather_dtype,
                        det_grid=det_grid,
                    )
                    if options_cls is not None
                    else None
                )
                return pallas_project(
                    T,
                    grid,
                    detector,
                    volume,
                    options=options,
                )
            except Exception as exc:
                if unsupported_exc is None or not isinstance(exc, unsupported_exc):
                    raise
                fallback_reason = f"{type(exc).__name__}: {exc}"
        if fallback_reason is not None:
            LOG.debug(
                "Falling back to JAX single-view projector after Pallas rejection: %s",
                fallback_reason,
            )
    vol = volume
    nx, ny, nz = validate_volume(vol, grid, context="forward_project_view_T", name="volume")
    validate_detector(detector, "forward_project_view_T")
    validate_detector_grid(det_grid, detector, context="forward_project_view_T")
    validate_pose_matrix(T, context="forward_project_view_T")
    recon_flat = _prepare_volume_for_gather(vol, gather_dtype)
    ix0, iy0, iz0, dix, diy, diz, n_steps_ray, step_size32, n_steps, n_rays = (
        _projector_traversal_state(
            T,
            grid,
            detector,
            step_size=step_size,
            n_steps=n_steps,
            det_grid=det_grid,
        )
    )

    def step(carry, step_idx):
        acc, ix, iy, iz = carry
        samp = _trilinear_gather(recon_flat, ix, iy, iz, nx, ny, nz)
        active = (step_idx < n_steps_ray).astype(jnp.float32)
        samp32 = samp.astype(jnp.float32) * active
        return (acc + samp32 * step_size32, ix + dix, iy + diy, iz + diz), None

    scan_step = step if not use_checkpoint else jax.checkpoint(step)
    acc0 = jnp.zeros((n_rays,), dtype=jnp.float32)
    carry_final, _ = jax.lax.scan(
        scan_step,
        (acc0, ix0, iy0, iz0),
        jnp.arange(n_steps, dtype=jnp.int32),
        length=n_steps,
        unroll=unroll or 1,
    )
    acc, _, _, _ = carry_final
    return acc.reshape((detector.nv, detector.nu))


def _backproject_view_accum_T(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    image: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    img = jnp.asarray(image, dtype=jnp.float32)
    validate_grid(grid, "backproject_view_T")
    validate_detector_image(img, detector, context="backproject_view_T", name="image")
    validate_detector_grid(det_grid, detector, context="backproject_view_T")
    validate_pose_matrix(T, context="backproject_view_T")
    traversal = _projector_traversal_state(
        T,
        grid,
        detector,
        step_size=step_size,
        n_steps=n_steps,
        det_grid=det_grid,
    )
    return _backproject_rays(
        img.reshape(-1),
        traversal[:7],
        grid,
        traversal[7],
        traversal[8],
        gather_dtype=gather_dtype,
        unroll=unroll,
    )


def _backproject_rays(
    ray_vals,
    traversal,
    grid,
    step_size32,
    n_steps,
    *,
    gather_dtype,
    unroll,
):
    """Accumulate a collection of rays into a single volume."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    ix0, iy0, iz0, dix, diy, diz, n_steps_ray = traversal

    def step(carry, step_idx):
        acc_flat, ix, iy, iz = carry
        active = (step_idx < n_steps_ray).astype(jnp.float32)
        step_vals = ray_vals * active * step_size32
        acc_flat = _trilinear_scatter_add(acc_flat, step_vals, ix, iy, iz, nx, ny, nz)
        return (acc_flat, ix + dix, iy + diy, iz + diz), None

    acc_dtype = _resolve_gather_target(gather_dtype)
    init = (
        jnp.zeros((nx * ny * nz,), dtype=acc_dtype),
        ix0,
        iy0,
        iz0,
    )
    carry_final, _ = jax.lax.scan(
        step,
        init,
        # Replay the forward coordinates in the same order. Starting at the far
        # endpoint and subtracting steps accumulates different fp32 rounding,
        # so the resulting operator is no longer the discrete transpose.
        jnp.arange(n_steps, dtype=jnp.int32),
        length=n_steps,
        unroll=unroll or 1,
    )
    return carry_final[0].astype(jnp.float32).reshape((nx, ny, nz))


def backproject_view_T(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    image: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ray_integrator: str = "sampled",
    frames: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Backproject one detector image as the explicit adjoint of the configured projector."""
    if frames is not None:
        return _cone_backproject_views(T[None], grid, detector, jnp.asarray(image)[None], frames)
    if ray_integrator == "exact":
        if step_size is not None or n_steps is not None:
            raise ValueError("exact ray integration does not accept a step size or sample count")
        from tomojax.core.trilinear import exact_adjoint

        return exact_adjoint(
            jnp.asarray(T)[None], grid, detector, jnp.asarray(image)[None], det_grid=det_grid
        )
    if ray_integrator in _JOSEPH_INTEGRATORS:
        return sum_backproject_views_T(
            jnp.asarray(T)[None],
            grid,
            detector,
            jnp.asarray(image)[None],
            det_grid=det_grid,
            ray_integrator=ray_integrator,
        )
    if ray_integrator != "sampled":
        raise ValueError(f"ray_integrator must be one of {RAY_INTEGRATORS}")
    return _backproject_view_accum_T(
        T,
        grid,
        detector,
        image,
        step_size=step_size,
        n_steps=n_steps,
        unroll=unroll,
        gather_dtype=gather_dtype,
        det_grid=det_grid,
    )


def sum_backproject_views_T(
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    images: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ray_integrator: str = "sampled",
    frames: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Sum explicit mixed-precision adjoints over a fixed chunk (``frames`` as above)."""
    if frames is not None:
        return _cone_backproject_views(T_all, grid, detector, images, frames)
    if ray_integrator == "exact":
        if step_size is not None or n_steps is not None:
            raise ValueError("exact ray integration does not accept a step size or sample count")
        from tomojax.core.trilinear import exact_adjoint

        return exact_adjoint(T_all, grid, detector, images, det_grid=det_grid)
    if ray_integrator in _JOSEPH_INTEGRATORS:
        from tomojax.core.joseph import sum_backproject_planes

        coeff, interpolation = _joseph_operands(T_all, grid, detector, det_grid, ray_integrator)
        backend = "pallas" if cuda_backend() else "jax"
        return sum_backproject_planes(
            coeff,
            jnp.asarray(images, jnp.float32),
            grid,
            detector,
            backend=backend,
            interpolation=interpolation,
        )
    if ray_integrator != "sampled":
        raise ValueError(f"ray_integrator must be one of {RAY_INTEGRATORS}")
    n_views, _, _ = validate_projection_stack(
        images,
        detector,
        context="sum_backproject_views_T",
    )
    validate_pose_stack(T_all, n_views, context="sum_backproject_views_T")
    validate_detector_grid(det_grid, detector, context="sum_backproject_views_T")
    validate_grid(grid, "sum_backproject_views_T")
    img = jnp.asarray(images, dtype=jnp.float32)

    def backproject_one(T_i: jnp.ndarray, img_i: jnp.ndarray) -> jnp.ndarray:
        return backproject_view_T(
            T_i,
            grid,
            detector,
            img_i,
            step_size=step_size,
            n_steps=n_steps,
            unroll=unroll,
            gather_dtype=gather_dtype,
            det_grid=det_grid,
        )

    if int(n_views) == 1:
        return backproject_one(T_all[0], img[0])
    if _resolve_gather_target(gather_dtype) != jnp.float32:
        # Half-precision transposes round each view's accumulation before the
        # fp32 reduction; combining those accumulators changes that contract.
        return jnp.sum(jax.vmap(backproject_one)(T_all, img), axis=0, dtype=jnp.float32)

    def traversal_for_view(T):
        return _projector_traversal_state(
            T,
            grid,
            detector,
            step_size=step_size,
            n_steps=n_steps,
            det_grid=det_grid,
        )[:7]

    traversal = jax.vmap(traversal_for_view)(T_all)
    ray_shape = traversal[0].shape
    flattened = tuple(jnp.broadcast_to(value, ray_shape).reshape(-1) for value in traversal)
    step = float(grid.vy) if step_size is None else float(step_size)
    return _backproject_rays(
        img.reshape(-1),
        flattened,
        grid,
        jnp.float32(step),
        _resolve_n_steps(grid, step, n_steps),
        gather_dtype=gather_dtype,
        unroll=unroll,
    )


def forward_project_view(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    view_index: int,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    use_checkpoint: bool = True,
    unroll: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ray_integrator: str = "sampled",
    projector_backend: ProjectorBackendInput = "jax",
) -> jnp.ndarray:
    """Wrapper that fetches pose from geometry and calls the pose-aware variant.

    The projector uses the supplied ``Grid``/``Detector`` for the standard
    detector-plane ray model; ``geometry.rays_for_view`` is not consumed.
    Cone-beam geometries use the cone Joseph projector, whose only options are
    the defaults.
    """
    T = jnp.asarray(geometry.pose_for_view(view_index), dtype=jnp.float32)
    frames = _view_frame(geometry, detector, view_index)
    if frames is not None:
        if det_grid is not None or ray_integrator == "exact":
            raise ValueError("cone-beam projection needs the canonical grid and Joseph sampling")
        return _cone_views(T[None], grid, detector, volume, frames)[0]
    return forward_project_view_T(
        T,
        grid,
        detector,
        volume,
        step_size=step_size,
        n_steps=n_steps,
        use_checkpoint=use_checkpoint,
        unroll=unroll,
        gather_dtype=gather_dtype,
        det_grid=det_grid,
        projector_backend=projector_backend,
        ray_integrator=ray_integrator,
    )


def backproject_view(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    image: jnp.ndarray,
    view_index: int,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ray_integrator: str = "sampled",
) -> jnp.ndarray:
    """Wrapper that fetches pose and calls the explicit gather-dtype adjoint.

    The projector uses the supplied ``Grid``/``Detector`` for the standard
    detector-plane ray model; ``geometry.rays_for_view`` is not consumed.
    Cone-beam geometries use the transpose of the cone Joseph projector.
    """
    T = jnp.asarray(geometry.pose_for_view(view_index), dtype=jnp.float32)
    frames = _view_frame(geometry, detector, view_index)
    if frames is not None:
        if det_grid is not None or ray_integrator == "exact":
            raise ValueError("cone-beam projection needs the canonical grid and Joseph sampling")
        return _cone_backproject_views(T[None], grid, detector, jnp.asarray(image)[None], frames)
    return backproject_view_T(
        T,
        grid,
        detector,
        image,
        step_size=step_size,
        n_steps=n_steps,
        unroll=unroll,
        gather_dtype=gather_dtype,
        det_grid=det_grid,
        ray_integrator=ray_integrator,
    )
