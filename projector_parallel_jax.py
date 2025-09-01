# projector_parallel_jax.py
# Differentiable, memory-efficient parallel-beam projector for 3D CT
# - Parallel-beam geometry (beam along +y in world frame)
# - 5 DOF view parameters per projection: (alpha, beta, phi, dx, dz)
# - Rigid-body convention matches Eq. (1): T x = R_y(β) R_x(α) R_z(φ) x + t
# - Forward integrates along y using trilinear sampling with Δs scaling
# - Differentiable via JAX, streaming integration via lax.scan (+checkpoint)
# - No O(n_rays * n_steps) intermediates; gradients via autodiff
#
# API highlights:
#   forward_project_view(params, recon_flat, grid, det, ...)
#   view_loss_value_and_grad(params, measured, recon_flat, grid, det, ...)
#   batch_forward_project(params_per_view, ...)
#
# Notes:
# - recon_flat is a flattened C-order (x-major) 1D array of shape (nx*ny*nz,)
#   index i = ix * (ny*nz) + iy * nz + iz
# - grid and detector are simple dicts
# - angles in radians; translations in same units as voxel/detector units
#
# Tested with JAX 0.4.x on CPU and GPU.

from __future__ import annotations

from functools import partial
import math
from typing import Dict, Tuple

import jax
import jax.numpy as jnp


# ----------------------------
# Rotation utilities (float32)
# ----------------------------
def rot_x(a: jnp.ndarray) -> jnp.ndarray:
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=jnp.float32
    )


def rot_y(b: jnp.ndarray) -> jnp.ndarray:
    c, s = jnp.cos(b), jnp.sin(b)
    return jnp.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=jnp.float32
    )


def rot_z(p: jnp.ndarray) -> jnp.ndarray:
    c, s = jnp.cos(p), jnp.sin(p)
    return jnp.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32
    )


def compose_R(alpha: jnp.ndarray, beta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    # Eq. (1) convention: R = R_y(β) R_x(α) R_z(φ)
    return rot_y(beta) @ rot_x(alpha) @ rot_z(phi)


# ----------------------------
# Geometry helpers
# ----------------------------
def default_volume_origin(
    nx: int, ny: int, nz: int, vx: float, vy: float, vz: float
) -> jnp.ndarray:
    # Center the volume at world (0,0,0) with voxel centers aligned
    ox = -((nx / 2.0) - 0.5) * vx
    oy = -((ny / 2.0) - 0.5) * vy
    oz = -((nz / 2.0) - 0.5) * vz
    return jnp.array([ox, oy, oz], dtype=jnp.float32)


def default_detector_centers(nu: int, nv: int, du: float, dv: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Detector pixel centers centered at (0, 0) in (x,z)
    u = (jnp.arange(nu, dtype=jnp.float32) - (nu / 2.0 - 0.5)) * du
    v = (jnp.arange(nv, dtype=jnp.float32) - (nv / 2.0 - 0.5)) * dv
    return u, v


def build_detector_grid(
    nu: int, nv: int, du: float, dv: float, det_center_x: float = 0.0, det_center_z: float = 0.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Returns per-ray (x,z) world coordinates for detector pixels (flattened)
    u, v = default_detector_centers(nu, nv, du, dv)
    u = u + jnp.float32(det_center_x)
    v = v + jnp.float32(det_center_z)
    # Flattened grid: X repeats across z, Z repeats across x
    X = jnp.tile(u, nv)          # length = nu * nv
    Z = jnp.repeat(v, nu)        # length = nu * nv
    return X, Z


# ----------------------------
# Trilinear sampling (gather)
# ----------------------------
def _flat_index(ix, iy, iz, nx, ny, nz):
    return ix * (ny * nz) + iy * nz + iz


@partial(
    jax.jit,
    static_argnames=(
        "nx",
        "ny",
        "nz",
    ),
)
def trilinear_gather(
    recon_flat: jnp.ndarray,
    ix_f: jnp.ndarray,
    iy_f: jnp.ndarray,
    iz_f: jnp.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> jnp.ndarray:
    """
    Gather trilinear samples from recon_flat at fractional indices (ix_f, iy_f, iz_f).
    Shapes: ix_f,iy_f,iz_f: (n,), recon_flat: (nx*ny*nz,) -> returns (n,)
    """
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


# ----------------------------
# Forward projector: single view
# ----------------------------
@partial(
    jax.jit,
    static_argnames=(
        "nx",
        "ny",
        "nz",
        "nu",
        "nv",
        "n_steps",
        "use_checkpoint",
    ),
)
def forward_project_view(
    params: jnp.ndarray,  # (alpha, beta, phi, dx, dz)
    recon_flat: jnp.ndarray,  # (nx*ny*nz,) float32 (will stop_gradient)
    nx: int,
    ny: int,
    nz: int,
    vx: float,
    vy: float,
    vz: float,
    nu: int,
    nv: int,
    du: float,
    dv: float,
    vol_origin: jnp.ndarray,  # (3,) world coords of voxel (0,0,0) center
    det_center: jnp.ndarray,  # (2,) world coords (x,z) of detector center
    step_size: float,
    n_steps: int,
    use_checkpoint: bool = True,
) -> jnp.ndarray:
    """
    Parallel-beam forward projection for one view.

    - World frame beam direction: +y.
    - View transform: T x = R_y(β) R_x(α) R_z(φ) x + t, with t = [dx, 0, dz].
    - Integral along y with step_size scaling.

    Returns: projection image (nv, nu) float32.
    """
    # Unpack params (α, β, φ, Δx, Δz)
    alpha, beta, phi, dx, dz = params
    R = compose_R(alpha, beta, phi)
    Rinv = R.T  # orthonormal
    t = jnp.array([dx, 0.0, dz], dtype=jnp.float32)

    # Build detector grid (flattened per-ray (x,z) world coords)
    Xr, Zr = build_detector_grid(nu, nv, du, dv, det_center[0], det_center[1])
    n_rays = Xr.shape[0]

    # Precompute y samples
    y0 = vol_origin[1]
    ys = y0 + step_size * jnp.arange(n_steps, dtype=jnp.float32)

    # Make recon a constant for geometry gradient steps
    recon_c = jax.lax.stop_gradient(recon_flat)

    # Step integrator over y (streaming)
    def step(carry, y):
        # Rays at world positions (x = Xr, y, z = Zr) for all rays
        w = jnp.stack([Xr, jnp.full((n_rays,), y, dtype=jnp.float32), Zr], axis=0)
        # Map world -> original volume coordinates via inverse rigid transform
        q = Rinv @ (w - t[:, None])

        # Convert world coords to fractional voxel indices
        ix = (q[0, :] - vol_origin[0]) / vx
        iy = (q[1, :] - vol_origin[1]) / vy
        iz = (q[2, :] - vol_origin[2]) / vz

        samp = trilinear_gather(recon_c, ix, iy, iz, nx, ny, nz)
        return carry + samp * jnp.float32(step_size), None

    step_fn = step
    if use_checkpoint:
        step_fn = jax.checkpoint(step_fn)

    acc0 = jnp.zeros((n_rays,), dtype=jnp.float32)
    acc, _ = jax.lax.scan(step_fn, acc0, ys)

    return acc.reshape((nv, nu))


# ----------------------------
# Loss and gradient (single view)
# ----------------------------
def view_loss(
    params: jnp.ndarray,
    measured: jnp.ndarray,  # (nv, nu)
    recon_flat: jnp.ndarray,  # (nx*ny*nz,)
    grid: Dict,
    det: Dict,
    step_size: float | None = None,
    n_steps: int | None = None,
    use_checkpoint: bool = True,
) -> jnp.ndarray:
    """
    0.5 * || A(params) x - measured ||^2 for one view.
    grid: {nx,ny,nz,vx,vy,vz, vol_center? or vol_origin?}
    det:  {nu,nv,du,dv, det_center?}
    """
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    vx, vy, vz = float(grid["vx"]), float(grid["vy"]), float(grid["vz"])
    if "vol_origin" in grid:
        vol_origin = jnp.asarray(grid["vol_origin"], dtype=jnp.float32)
    else:
        vc = jnp.asarray(grid.get("vol_center", [0.0, 0.0, 0.0]), dtype=jnp.float32)
        vol_origin = default_volume_origin(nx, ny, nz, vx, vy, vz) + vc

    nu, nv = int(det["nu"]), int(det["nv"])
    du, dv = float(det["du"]), float(det["dv"])
    det_center = jnp.asarray(det.get("det_center", [0.0, 0.0]), dtype=jnp.float32)

    # Default Δs and steps: cover the y-extent of the volume
    if step_size is None:
        step_size = vy
    if n_steps is None:
        # Ensure coverage of y-range of the volume
        n_steps = int(math.ceil((ny * vy) / step_size))

    pred = forward_project_view(
        params=params.astype(jnp.float32),
        recon_flat=recon_flat,
        nx=nx,
        ny=ny,
        nz=nz,
        vx=jnp.float32(vx),
        vy=jnp.float32(vy),
        vz=jnp.float32(vz),
        nu=nu,
        nv=nv,
        du=jnp.float32(du),
        dv=jnp.float32(dv),
        vol_origin=vol_origin,
        det_center=det_center,
        step_size=jnp.float32(step_size),
        n_steps=n_steps,
        use_checkpoint=use_checkpoint,
    )
    resid = (pred - measured).astype(jnp.float32)
    return 0.5 * jnp.vdot(resid, resid).real


# JIT+grad wrapper for single-view geometry gradient
view_loss_value_and_grad = jax.jit(
    jax.value_and_grad(view_loss),
    static_argnames=("grid", "det", "step_size", "n_steps", "use_checkpoint"),
)


# ----------------------------
# Batched helpers (multi-view)
# ----------------------------
def batch_forward_project(
    params_per_view: jnp.ndarray,  # (n_views, 5)
    recon_flat: jnp.ndarray,
    grid: Dict,
    det: Dict,
    step_size: float | None = None,
    n_steps: int | None = None,
    use_checkpoint: bool = True,
) -> jnp.ndarray:
    """
    Forward project multiple views sequentially (low memory) or vmapped (GPU).
    Returns: (n_views, nv, nu)
    """
    nv, nu = int(det["nv"]), int(det["nu"])
    n_views = int(params_per_view.shape[0])

    def one_view(p):
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        vx, vy, vz = float(grid["vx"]), float(grid["vy"]), float(grid["vz"])
        if "vol_origin" in grid:
            vol_origin = jnp.asarray(grid["vol_origin"], dtype=jnp.float32)
        else:
            vc = jnp.asarray(grid.get("vol_center", [0.0, 0.0, 0.0]), dtype=jnp.float32)
            vol_origin = default_volume_origin(nx, ny, nz, vx, vy, vz) + vc

        du, dv = float(det["du"]), float(det["dv"])
        det_center = jnp.asarray(
            det.get("det_center", [0.0, 0.0]), dtype=jnp.float32
        )

        # Defaults
        ss = vy if step_size is None else step_size
        ns = (
            int(math.ceil((ny * vy) / ss))
            if n_steps is None
            else int(n_steps)
        )

        return forward_project_view(
            params=p.astype(jnp.float32),
            recon_flat=recon_flat,
            nx=nx,
            ny=ny,
            nz=nz,
            vx=jnp.float32(vx),
            vy=jnp.float32(vy),
            vz=jnp.float32(vz),
            nu=int(det["nu"]),
            nv=int(det["nv"]),
            du=jnp.float32(du),
            dv=jnp.float32(dv),
            vol_origin=vol_origin,
            det_center=det_center,
            step_size=jnp.float32(ss),
            n_steps=ns,
            use_checkpoint=use_checkpoint,
        )

    # Sequential loop to minimize peak memory on CPU
    outs = []
    for i in range(n_views):
        outs.append(one_view(params_per_view[i]))
    return jnp.stack(outs, axis=0)  # (n_views, nv, nu)


# ----------------------------
# Small utilities
# ----------------------------
def adjoint_test_once(
    recon_flat: jnp.ndarray,
    y_like: jnp.ndarray,
    params: jnp.ndarray,
    grid: Dict,
    det: Dict,
    step_size: float | None = None,
    n_steps: int | None = None,
) -> float:
    """
    Inner-product test <A x, y> vs <x, A^T y> for one view.
    This function uses autodiff VJP to compute J^T v wrt recon, which can be
    memory intensive. Use tiny sizes for sanity checks (e.g., 16^3).
    """
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    vx, vy, vz = float(grid["vx"]), float(grid["vy"]), float(grid["vz"])

    if "vol_origin" in grid:
        vol_origin = jnp.asarray(grid["vol_origin"], dtype=jnp.float32)
    else:
        vc = jnp.asarray(grid.get("vol_center", [0.0, 0.0, 0.0]), dtype=jnp.float32)
        vol_origin = default_volume_origin(nx, ny, nz, vx, vy, vz) + vc

    du, dv = float(det["du"]), float(det["dv"])
    det_center = jnp.asarray(det.get("det_center", [0.0, 0.0]), dtype=jnp.float32)

    if step_size is None:
        step_size = vy
    if n_steps is None:
        n_steps = int(math.ceil((ny * vy) / step_size))

    def fwd_wrt_recon(rflat):
        return forward_project_view(
            params=params.astype(jnp.float32),
            recon_flat=rflat,
            nx=nx,
            ny=ny,
            nz=nz,
            vx=jnp.float32(vx),
            vy=jnp.float32(vy),
            vz=jnp.float32(vz),
            nu=int(det["nu"]),
            nv=int(det["nv"]),
            du=jnp.float32(du),
            dv=jnp.float32(dv),
            vol_origin=vol_origin,
            det_center=det_center,
            step_size=jnp.float32(step_size),
            n_steps=int(n_steps),
            use_checkpoint=True,
        ).ravel()

    Ax = fwd_wrt_recon(recon_flat)
    lhs = jnp.vdot(Ax, y_like.ravel())

    # J^T y via VJP (w.r.t. recon)
    _, vjp_fn = jax.vjp(fwd_wrt_recon, recon_flat)
    ATy = vjp_fn(y_like.ravel().astype(jnp.float32))[0]
    rhs = jnp.vdot(recon_flat, ATy)
    rel_err = float(jnp.abs(lhs - rhs) / (jnp.abs(lhs) + 1e-12))
    return rel_err