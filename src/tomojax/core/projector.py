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
    ox = -((grid.nx / 2.0) - 0.5) * grid.vx
    oy = -((grid.ny / 2.0) - 0.5) * grid.vy
    oz = -((grid.nz / 2.0) - 0.5) * grid.vz
    return jnp.array([ox, oy, oz], dtype=jnp.float32)


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
        inb = (
            (ix >= 0)
            & (ix < nx)
            & (iy >= 0)
            & (iy < ny)
            & (iz >= 0)
            & (iz < nz)
        ).astype(jnp.float32)
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
        raise ValueError(f"Volume must be (nx,ny,nz)={nx,ny,nz}, got {vol.shape}")
    # Mixed-precision gather option (accumulate in fp32)
    gd = gather_dtype.lower()
    if gd == "auto":
        # Choose mixed precision on accelerators; fp32 on CPU for numerical stability
        try:
            platform = jax.devices()[0].platform if jax.devices() else "cpu"
        except Exception:
            platform = "cpu"
        if platform in ("gpu", "tpu"):
            # Prefer bf16 on TPU, fp16 on GPU
            target = jnp.bfloat16 if platform == "tpu" else jnp.float16
        else:
            target = jnp.float32
    elif gd in ("bf16", "bfloat16"):
        target = jnp.bfloat16
    elif gd in ("fp16", "float16", "half"):
        target = jnp.float16
    else:
        target = jnp.float32
    vol_cast = vol if vol.dtype == target else vol.astype(target)
    recon_flat = jnp.ravel(vol_cast, order="C")

    vol_origin = (
        jnp.asarray(grid.vol_origin, dtype=jnp.float32)
        if grid.vol_origin is not None
        else _default_volume_origin(grid)
    )

    if det_grid is None:
        Xr_np, Zr_np = _build_detector_grid(detector)
        # Convert to device arrays inside the jitted function scope
        Xr = jnp.asarray(Xr_np, dtype=jnp.float32)
        Zr = jnp.asarray(Zr_np, dtype=jnp.float32)
        n_rays = int(Xr_np.shape[0])
    else:
        Xr, Zr = det_grid
        n_rays = int(Xr.shape[0])

    vy = float(grid.vy)
    if step_size is None:
        step_size = vy
    if n_steps is None:
        n_steps = int(math.ceil((ny * vy) / float(step_size)))

    # Inverse pose and incremental stepping set-up
    R = T[:3, :3]
    t = T[:3, 3]
    Rinv = R.T
    tinv = -(Rinv @ t)
    ey_obj = Rinv[:, 1]  # world +y axis mapped into object frame (beam dir in object coords)

    base = Rinv @ jnp.stack([Xr, jnp.zeros_like(Xr), Zr], axis=0) + tinv[:, None]
    y0 = vol_origin[1]
    q0 = base + y0 * ey_obj[:, None]
    dq = (step_size * ey_obj)[:, None]

    # Precompute reciprocal voxel sizes to avoid divides in inner loop
    inv_vx = jnp.float32(1.0 / grid.vx)
    inv_vy = jnp.float32(1.0 / grid.vy)
    inv_vz = jnp.float32(1.0 / grid.vz)

    # Optional incremental index updates to avoid repeated subtracts
    ix0 = (q0[0] - vol_origin[0]) * inv_vx
    iy0 = (q0[1] - vol_origin[1]) * inv_vy
    iz0 = (q0[2] - vol_origin[2]) * inv_vz
    dix = dq[0] * inv_vx
    diy = dq[1] * inv_vy
    diz = dq[2] * inv_vz

    def step(carry, _):
        acc, ix, iy, iz = carry
        samp = _trilinear_gather(recon_flat, ix, iy, iz, nx, ny, nz)
        samp32 = samp.astype(jnp.float32)
        return (acc + samp32 * jnp.float32(step_size), ix + dix, iy + diy, iz + diz), None

    scan_step = step if not use_checkpoint else jax.checkpoint(step)
    acc0 = jnp.zeros((n_rays,), dtype=jnp.float32)
    carry_final, _ = jax.lax.scan(
        scan_step, (acc0, ix0, iy0, iz0), None, length=n_steps, unroll=unroll or 1
    )
    acc, _, _, _ = carry_final
    return acc.reshape((detector.nv, detector.nu))


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
