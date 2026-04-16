from __future__ import annotations

import math
from typing import Tuple
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np

from .geometry.base import Grid, Detector, Geometry

# Frame conventions:
# - geometry.pose_for_view(i) must return a 4x4 transform T_world_from_obj that maps
#   object (sample) coordinates into world (lab) coordinates for view i.
# - Rays are defined in the world frame with directions along +y (parallel beam).
# - We compute object_from_world = inv(T_world_from_obj) and sample the volume in the
#   object frame directly. This makes reconstructed volumes live in the object (sample) frame.

# Cache detector grids keyed by (nu, nv, du, dv, cx, cz)
_DET_GRID_CACHE: "OrderedDict[Tuple[int, int, float, float, float, float], Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_DET_GRID_CACHE_CAP = 8


def _default_volume_origin(grid: Grid) -> jnp.ndarray:
    """Return the centred default location of voxel (0, 0, 0)'s centre."""
    ox = -((grid.nx / 2.0) - 0.5) * grid.vx
    oy = -((grid.ny / 2.0) - 0.5) * grid.vy
    oz = -((grid.nz / 2.0) - 0.5) * grid.vz
    return jnp.array([ox, oy, oz], dtype=jnp.float32)


def _interpolation_support_bounds(
    grid: Grid, vol_origin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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


def _build_detector_grid(det: Detector) -> Tuple[np.ndarray, np.ndarray]:
    key = (
        int(det.nu),
        int(det.nv),
        float(det.du),
        float(det.dv),
        float(det.det_center[0]),
        float(det.det_center[1]),
    )
    if key in _DET_GRID_CACHE:
        _DET_GRID_CACHE.move_to_end(key)
        return _DET_GRID_CACHE[key]
    nu, nv = int(det.nu), int(det.nv)
    du, dv = float(det.du), float(det.dv)
    cx, cz = float(det.det_center[0]), float(det.det_center[1])
    # Build on host as NumPy to avoid capturing JAX tracers in global cache under jit
    u = (np.arange(nu, dtype=np.float32) - (nu / 2.0 - 0.5)) * np.float32(du) + np.float32(cx)
    v = (np.arange(nv, dtype=np.float32) - (nv / 2.0 - 0.5)) * np.float32(dv) + np.float32(cz)
    X = np.tile(u, nv)
    Z = np.repeat(v, nu)
    _DET_GRID_CACHE[key] = (X, Z)
    if len(_DET_GRID_CACHE) > _DET_GRID_CACHE_CAP:
        _DET_GRID_CACHE.popitem(last=False)
    return X, Z


def get_detector_grid_device(det: Detector) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return detector coordinate grids as device arrays.

    Note: call this outside of any JAX-transformed context (jit/grad/scan) to avoid
    side effects during tracing. Safe to cache at the application level.
    """
    X_np, Z_np = _build_detector_grid(det)
    return jnp.asarray(X_np, dtype=jnp.float32), jnp.asarray(Z_np, dtype=jnp.float32)


def _resolve_gather_target(gather_dtype: str) -> jnp.dtype:
    """Resolve the gather/interpolation dtype for the forward projector."""
    gd = gather_dtype.lower()
    if gd == "auto":
        try:
            platform = jax.devices()[0].platform if jax.devices() else "cpu"
        except Exception:
            platform = "cpu"
        if platform in ("gpu", "tpu"):
            return jnp.bfloat16 if platform == "tpu" else jnp.float16
        return jnp.float32
    if gd in ("bf16", "bfloat16"):
        return jnp.bfloat16
    if gd in ("fp16", "float16", "half"):
        return jnp.float16
    return jnp.float32


def _prepare_volume_for_gather(volume: jnp.ndarray, gather_dtype: str) -> jnp.ndarray:
    target = _resolve_gather_target(gather_dtype)
    vol_cast = volume if volume.dtype == target else volume.astype(target)
    return jnp.ravel(vol_cast, order="C")


def _resolve_detector_grid(
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    if det_grid is None:
        Xr_np, Zr_np = _build_detector_grid(detector)
        Xr = jnp.asarray(Xr_np, dtype=jnp.float32)
        Zr = jnp.asarray(Zr_np, dtype=jnp.float32)
        n_rays = int(Xr_np.shape[0])
    else:
        Xr, Zr = det_grid
        n_rays = int(Xr.shape[0])
    return Xr, Zr, n_rays


def _resolve_n_steps(grid: Grid, step_size: float, n_steps: int | None) -> int:
    if n_steps is not None:
        return int(n_steps)
    support_lengths = (
        float((grid.nx + 1) * grid.vx),
        float((grid.ny + 1) * grid.vy),
        float((grid.nz + 1) * grid.vz),
    )
    max_path_length = math.sqrt(sum(length * length for length in support_lengths))
    return int(math.ceil(max_path_length / float(step_size)))


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
    vol_origin = (
        jnp.asarray(grid.vol_origin, dtype=jnp.float32)
        if grid.vol_origin is not None
        else _default_volume_origin(grid)
    )
    Xr, Zr, n_rays = _resolve_detector_grid(detector, det_grid)
    if step_size is None:
        step_size = float(grid.vy)
    n_steps_val = _resolve_n_steps(grid, float(step_size), n_steps)

    R = T[:3, :3]
    t = T[:3, 3]
    Rinv = R.T
    tinv = -(Rinv @ t)
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
def _flat_index(ix, iy, iz, nx, ny, nz):
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
    scatter = jax.linear_transpose(
        lambda recon: _trilinear_gather(recon, ix_f, iy_f, iz_f, nx, ny, nz),
        acc_flat,
    )
    return acc_flat + scatter(ray_vals)[0]


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
) -> jnp.ndarray:
    """Forward project a single view given pose `T` (4x4, row-major).

    Contract: T is world_from_object for the view. The projector constructs detector
    rays in world coordinates and transforms them into object coordinates using
    inv(T), then performs incremental stepping along the beam direction expressed
    in the object frame. This avoids a matmul per step and keeps gradients clean.
    """
    vol = volume
    if vol.ndim != 3:
        raise ValueError("volume must be 3D array")
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    if vol.shape != (nx, ny, nz):
        raise ValueError(f"Volume must be (nx,ny,nz)={nx, ny, nz}, got {vol.shape}")
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
    if img.ndim != 2:
        raise ValueError("image must be 2D array")
    if img.shape != (int(detector.nv), int(detector.nu)):
        raise ValueError(
            f"Image must be (nv,nu)={(int(detector.nv), int(detector.nu))}, got {img.shape}"
        )
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    ix0, iy0, iz0, dix, diy, diz, n_steps_ray, step_size32, n_steps, _ = _projector_traversal_state(
        T,
        grid,
        detector,
        step_size=step_size,
        n_steps=n_steps,
        det_grid=det_grid,
    )
    ray_vals = img.reshape((-1,))

    def step(carry, step_idx):
        acc_flat, ix, iy, iz = carry
        active = (step_idx < n_steps_ray).astype(jnp.float32)
        step_vals = ray_vals * active * step_size32
        acc_flat = _trilinear_scatter_add(acc_flat, step_vals, ix, iy, iz, nx, ny, nz)
        return (acc_flat, ix - dix, iy - diy, iz - diz), None

    acc_dtype = _resolve_gather_target(gather_dtype)
    last_step = jnp.int32(max(n_steps - 1, 0))
    init = (
        jnp.zeros((nx * ny * nz,), dtype=acc_dtype),
        ix0 + dix * last_step.astype(jnp.float32),
        iy0 + diy * last_step.astype(jnp.float32),
        iz0 + diz * last_step.astype(jnp.float32),
    )
    carry_final, _ = jax.lax.scan(
        step,
        init,
        jnp.arange(n_steps - 1, -1, -1, dtype=jnp.int32),
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
) -> jnp.ndarray:
    """Backproject one detector image as the explicit adjoint of the configured projector."""
    acc = _backproject_view_accum_T(
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
    return acc


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
) -> jnp.ndarray:
    """Sum explicit mixed-precision adjoints over a fixed chunk without stacking volumes."""
    img = jnp.asarray(images, dtype=jnp.float32)

    def body(accum, inputs):
        T_i, img_i = inputs
        bp = backproject_view_T(
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
        return accum + bp, None

    init = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    acc, _ = jax.lax.scan(body, init, (T_all, img))
    return acc


_sum_backproject_views_T = sum_backproject_views_T


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
) -> jnp.ndarray:
    """Wrapper that fetches pose from geometry and calls pose-aware variant."""
    T = jnp.asarray(geometry.pose_for_view(view_index), dtype=jnp.float32)
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
) -> jnp.ndarray:
    """Wrapper that fetches pose and calls the explicit gather-dtype adjoint."""
    T = jnp.asarray(geometry.pose_for_view(view_index), dtype=jnp.float32)
    return backproject_view_T(
        T,
        grid,
        detector,
        image,
        step_size=step_size,
        n_steps=n_steps,
        unroll=unroll,
        gather_dtype=gather_dtype,
    )
