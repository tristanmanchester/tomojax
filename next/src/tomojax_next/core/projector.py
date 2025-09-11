from __future__ import annotations

import math
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from .geometry.base import Grid, Detector, Geometry


def _default_volume_origin(grid: Grid) -> jnp.ndarray:
    ox = -((grid.nx / 2.0) - 0.5) * grid.vx
    oy = -((grid.ny / 2.0) - 0.5) * grid.vy
    oz = -((grid.nz / 2.0) - 0.5) * grid.vz
    return jnp.array([ox, oy, oz], dtype=jnp.float32)


def _build_detector_grid(det: Detector) -> Tuple[jnp.ndarray, jnp.ndarray]:
    nu, nv = int(det.nu), int(det.nv)
    du, dv = float(det.du), float(det.dv)
    cx, cz = float(det.det_center[0]), float(det.det_center[1])
    u = (jnp.arange(nu, dtype=jnp.float32) - (nu / 2.0 - 0.5)) * jnp.float32(du) + jnp.float32(cx)
    v = (jnp.arange(nv, dtype=jnp.float32) - (nv / 2.0 - 0.5)) * jnp.float32(dv) + jnp.float32(cz)
    X = jnp.tile(u, nv)
    Z = jnp.repeat(v, nu)
    return X, Z


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


@jax.jit
def _apply_Tinv(Tinv, pts):
    # pts: (3, N)
    R = Tinv[:3, :3]
    t = Tinv[:3, 3:4]
    return R @ pts + t


def forward_project_view(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,  # (nz, ny, nx) or (ny, nx, nz)? We use (nx,ny,nz) flattened consistent with existing.
    view_index: int,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    use_checkpoint: bool = True,
) -> jnp.ndarray:
    """Forward project a single view using the provided geometry.

    Current implementation assumes parallel rays along +y in world (as in our
    Parallel and Laminography geometries) and streams integration across the y
    extent of the volume. This will be generalized alongside geometry evolution.
    """
    # Ensure volume flattening convention (nx,ny,nz) -> flatten x-major
    # Accept both (nx, ny, nz) or (ny, nx, nz); require/note shape in docs later
    vol = volume
    if vol.ndim != 3:
        raise ValueError("volume must be 3D array")
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    # Expect volume organized as (nx, ny, nz)
    if vol.shape != (nx, ny, nz):
        # Try (ny, nx, nz) -> transpose to (nx, ny, nz)
        if vol.shape == (ny, nx, nz):
            vol = jnp.transpose(vol, (1, 0, 2))
        else:
            raise ValueError(f"Unexpected volume shape {vol.shape}; expected (nx,ny,nz) or (ny,nx,nz)")
    recon_flat = jnp.ravel(vol.astype(jnp.float32), order="C")

    vol_origin = (
        jnp.asarray(grid.vol_origin, dtype=jnp.float32)
        if grid.vol_origin is not None
        else _default_volume_origin(grid)
    )

    Xr, Zr = _build_detector_grid(detector)
    n_rays = Xr.shape[0]

    vy = float(grid.vy)
    if step_size is None:
        step_size = vy
    if n_steps is None:
        n_steps = int(math.ceil((ny * vy) / float(step_size)))

    # Build per-y samples
    y0 = vol_origin[1]
    ys = y0 + jnp.arange(n_steps, dtype=jnp.float32) * jnp.float32(step_size)

    # Inverse object pose (world -> object) for this view
    T = jnp.asarray(geometry.pose_for_view(view_index), dtype=jnp.float32)
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = jnp.eye(4, dtype=jnp.float32).at[:3, :3].set(R.T).at[:3, 3].set(-(R.T @ t))

    @jax.jit
    def step(acc, y):
        w = jnp.stack([Xr, jnp.full((n_rays,), y, dtype=jnp.float32), Zr], axis=0)  # (3, N)
        q = _apply_Tinv(Tinv, w)  # (3, N) in object frame
        ix = (q[0, :] - vol_origin[0]) / jnp.float32(grid.vx)
        iy = (q[1, :] - vol_origin[1]) / jnp.float32(grid.vy)
        iz = (q[2, :] - vol_origin[2]) / jnp.float32(grid.vz)
        samp = _trilinear_gather(recon_flat, ix, iy, iz, nx, ny, nz)
        return acc + samp * jnp.float32(step_size), None

    scan_step = step if not use_checkpoint else jax.checkpoint(step)
    acc0 = jnp.zeros((n_rays,), dtype=jnp.float32)
    acc, _ = jax.lax.scan(scan_step, acc0, ys)
    return acc.reshape((detector.nv, detector.nu))

