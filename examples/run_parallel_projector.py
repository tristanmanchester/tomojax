#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Memory-efficient, differentiable parallel-beam projector demo in JAX
# - Creates a 3D phantom (cubes + spheres)
# - Projects N views over 180 degrees
# - Saves phantom and projections as 3D TIFF stacks
# - Prints stats, timings, and memory usage

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import psutil
import tifffile as tiff


# ----------------------------
# Utilities: memory and timing
# ----------------------------
def get_memory_usage_mb() -> float:
    proc = psutil.Process()
    return proc.memory_info().rss / (1024.0 * 1024.0)


def tic() -> float:
    return time.time()


def toc(t0: float) -> float:
    return time.time() - t0


# ----------------------------
# Phantom builder (random rotated cubes + spheres)
# ----------------------------
import numpy as np
from scipy.ndimage import map_coordinates

def rotation_matrix_3d(angles):
    """Create 3D rotation matrix from Euler angles (in radians)."""
    rx, ry, rz = angles
    
    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix (order: Rz * Ry * Rx)
    return Rz @ Ry @ Rx

def add_rotated_cube(vol, center, size, value, angles, rng):
    """Add a rotated cube to the volume using interpolation."""
    nx, ny, nz = vol.shape
    cx, cy, cz = center
    
    # Create rotation matrix
    R = rotation_matrix_3d(angles)
    R_inv = R.T  # Inverse rotation (transpose for orthogonal matrix)
    
    # Define the bounding box of the rotated cube
    half_size = size / 2.0
    cube_corners = np.array([
        [-half_size, -half_size, -half_size],
        [half_size, -half_size, -half_size],
        [-half_size, half_size, -half_size],
        [half_size, half_size, -half_size],
        [-half_size, -half_size, half_size],
        [half_size, -half_size, half_size],
        [-half_size, half_size, half_size],
        [half_size, half_size, half_size]
    ])
    
    # Rotate corners to find bounding box
    rotated_corners = (R @ cube_corners.T).T
    min_bounds = np.min(rotated_corners, axis=0)
    max_bounds = np.max(rotated_corners, axis=0)
    
    # Calculate the region of interest in the volume
    roi_min = np.floor(np.array([cx, cy, cz]) + min_bounds).astype(int)
    roi_max = np.ceil(np.array([cx, cy, cz]) + max_bounds).astype(int)
    
    # Clamp to volume bounds
    roi_min = np.maximum(roi_min, [0, 0, 0])
    roi_max = np.minimum(roi_max, [nx-1, ny-1, nz-1])
    
    # Create coordinate grids for the ROI
    x_roi = np.arange(roi_min[0], roi_max[0] + 1)
    y_roi = np.arange(roi_min[1], roi_max[1] + 1)
    z_roi = np.arange(roi_min[2], roi_max[2] + 1)
    
    X_roi, Y_roi, Z_roi = np.meshgrid(x_roi, y_roi, z_roi, indexing='ij')
    
    # Translate coordinates to cube center
    X_centered = X_roi - cx
    Y_centered = Y_roi - cy
    Z_centered = Z_roi - cz
    
    # Apply inverse rotation to get coordinates in cube's local space
    coords_global = np.stack([X_centered.ravel(), Y_centered.ravel(), Z_centered.ravel()])
    coords_local = R_inv @ coords_global
    
    # Check which points are inside the cube
    inside_mask = (np.abs(coords_local[0]) <= half_size) & \
                  (np.abs(coords_local[1]) <= half_size) & \
                  (np.abs(coords_local[2]) <= half_size)
    
    # For smooth interpolation, create a soft cube with anti-aliasing
    # Calculate distance from cube boundary for smooth edges
    dist_x = np.maximum(0, np.abs(coords_local[0]) - half_size + 0.5)
    dist_y = np.maximum(0, np.abs(coords_local[1]) - half_size + 0.5)
    dist_z = np.maximum(0, np.abs(coords_local[2]) - half_size + 0.5)
    
    # Smooth falloff using distance to cube boundary
    dist_total = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
    alpha = np.clip(1.0 - dist_total, 0.0, 1.0)  # Smooth transition over 1 voxel
    
    # Apply values to volume
    alpha_reshaped = alpha.reshape(X_roi.shape)
    for i, x in enumerate(x_roi):
        for j, y in enumerate(y_roi):
            for k, z in enumerate(z_roi):
                if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                    vol[x, y, z] = np.maximum(vol[x, y, z], value * alpha_reshaped[i, j, k])

def make_phantom(
    nx: int,
    ny: int,
    nz: int,
    n_cubes: int = 8,
    n_spheres: int = 7,
    min_size: int = 4,
    max_size: int = 64,
    min_value: float = 0.3,
    max_value: float = 1.0,
    background_value: float = 0.0,
    add_noise: bool = False,
    noise_std: float = 0.01,
    seed: int = 42,
    max_rotation_angle: float = np.pi,  # Maximum rotation angle in radians
    use_inscribed_fov: bool = True,     # Constrain objects to inscribed circle
) -> np.ndarray:
    """
    Create a 3D phantom with randomly rotated cubes and spheres.
    
    Parameters:
    -----------
    max_rotation_angle : float
        Maximum rotation angle for cubes in radians (default: π for full rotation)
    use_inscribed_fov : bool
        If True, constrains all objects to fit within the inscribed circle/cylinder
        of the volume (important for tomographic reconstruction)
    """
    vol = np.full((nx, ny, nz), background_value, dtype=np.float32)
    rng = np.random.default_rng(seed)
    
    # Calculate inscribed circle parameters (assuming z is the rotation axis)
    center_x, center_y = nx / 2.0, ny / 2.0
    fov_radius = min(nx, ny) / 2.0 if use_inscribed_fov else float('inf')

    # Rotated Cubes
    for _ in range(n_cubes):
        size = float(rng.uniform(min_size, max_size))
        
        if use_inscribed_fov:
            # For rotated cubes, the worst-case projection in xy-plane is the diagonal
            # When rotated, a cube can have maximum extent of size*sqrt(2) in any plane
            max_xy_extent = size * np.sqrt(2) / 2.0  # Half the diagonal
            max_radius_from_center = fov_radius - max_xy_extent
            
            if max_radius_from_center <= 0:
                continue  # Skip this cube if it's too large to fit
            
            # Generate position within the allowed circular region
            # Use polar coordinates for uniform distribution in circle
            r = rng.uniform(0, max_radius_from_center)
            theta = rng.uniform(0, 2 * np.pi)
            
            cx = center_x + r * np.cos(theta)
            cy = center_y + r * np.sin(theta)
            cz = float(rng.uniform(size/2, nz - size/2))  # Full height available
        else:
            # Original behaviour - rectangular bounds
            margin = size * np.sqrt(3) / 2
            cx = float(rng.uniform(margin, nx - margin))
            cy = float(rng.uniform(margin, ny - margin))
            cz = float(rng.uniform(margin, nz - margin))
        
        val = float(rng.uniform(min_value, max_value))
        
        # Generate random rotation angles for all three axes
        angles = rng.uniform(-max_rotation_angle, max_rotation_angle, 3)
        
        add_rotated_cube(vol, (cx, cy, cz), size, val, angles, rng)

    # Spheres
    X, Y, Z = np.mgrid[0:nx, 0:ny, 0:nz]
    for _ in range(n_spheres):
        radius = float(rng.uniform(min_size / 2.0, max_size / 2.0))
        
        if use_inscribed_fov:
            # For spheres, ensure the entire sphere fits within the FOV circle
            max_radius_from_center = fov_radius - radius
            
            if max_radius_from_center <= 0:
                continue  # Skip this sphere if it's too large to fit
            
            # Generate position within the allowed circular region
            r = rng.uniform(0, max_radius_from_center)
            theta = rng.uniform(0, 2 * np.pi)
            
            cx = center_x + r * np.cos(theta)
            cy = center_y + r * np.sin(theta)
            cz = float(rng.uniform(radius, nz - radius))  # Full height available
        else:
            # Original behaviour - rectangular bounds
            cx = float(rng.uniform(radius, nx - radius))
            cy = float(rng.uniform(radius, ny - radius))
            cz = float(rng.uniform(radius, nz - radius))
        
        val = float(rng.uniform(min_value, max_value))
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
        vol[dist <= radius] = val

    if add_noise and noise_std > 0:
        vol = np.clip(vol + rng.normal(0.0, noise_std, vol.shape).astype(np.float32), 0.0, np.inf)

    return vol.astype(np.float32)

    return vol.astype(np.float32)

def add_fov_outline(vol, line_value=0.5):
    """
    Add a circular outline showing the tomographic field of view.
    Useful for visualization to verify all objects are within bounds.
    """
    nx, ny, nz = vol.shape
    center_x, center_y = nx / 2.0, ny / 2.0
    radius = min(nx, ny) / 2.0
    
    # Create circle outline in all z slices
    Y, X = np.ogrid[:ny, :nx]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Create a thin ring (1-2 pixels wide)
    ring_mask = (np.abs(dist_from_center - radius) < 1.0)
    
    for z in range(nz):
        vol[ring_mask.T, z] = np.maximum(vol[ring_mask.T, z], line_value)
    
    return vol



# ----------------------------
# Rotations and geometry
# ----------------------------
def rot_x(a: jnp.ndarray) -> jnp.ndarray:
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=jnp.float32)


def rot_y(b: jnp.ndarray) -> jnp.ndarray:
    c, s = jnp.cos(b), jnp.sin(b)
    return jnp.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=jnp.float32)


def rot_z(p: jnp.ndarray) -> jnp.ndarray:
    c, s = jnp.cos(p), jnp.sin(p)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32)


def compose_R(alpha: jnp.ndarray, beta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    # Eq. (1) convention: R = R_y(β) R_x(α) R_z(φ)
    return rot_y(beta) @ rot_x(alpha) @ rot_z(phi)


def default_volume_origin(nx: int, ny: int, nz: int, vx: float, vy: float, vz: float) -> jnp.ndarray:
    # Center volume about (0,0,0) with voxel centers aligned
    ox = -((nx / 2.0) - 0.5) * vx
    oy = -((ny / 2.0) - 0.5) * vy
    oz = -((nz / 2.0) - 0.5) * vz
    return jnp.array([ox, oy, oz], dtype=jnp.float32)


def build_detector_grid(
    nu: int, nv: int, du: float, dv: float, det_center_x: float = 0.0, det_center_z: float = 0.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Detector pixel centers in (x,z), flattened (nu*nv,)
    u = (jnp.arange(nu, dtype=jnp.float32) - (nu / 2.0 - 0.5)) * du + jnp.float32(det_center_x)
    v = (jnp.arange(nv, dtype=jnp.float32) - (nv / 2.0 - 0.5)) * dv + jnp.float32(det_center_z)
    X = jnp.tile(u, nv)
    Z = jnp.repeat(v, nu)
    return X, Z


# ----------------------------
# Trilinear gather
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
# Forward projector: single view (scan over y)
# ----------------------------
from functools import partial


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
    recon_flat: jnp.ndarray,  # (nx*ny*nz,)
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
    vol_origin: jnp.ndarray,  # (3,)
    det_center: jnp.ndarray,  # (2,) (x,z)
    step_size: float,
    n_steps: int,
    use_checkpoint: bool = True,
) -> jnp.ndarray:
    alpha, beta, phi, dx, dz = params
    R = compose_R(alpha, beta, phi)
    Rinv = R.T
    t = jnp.array([dx, 0.0, dz], dtype=jnp.float32)

    Xr, Zr = build_detector_grid(nu, nv, du, dv, det_center[0], det_center[1])
    n_rays = Xr.shape[0]

    y0 = vol_origin[1]
    ys = y0 + step_size * jnp.arange(n_steps, dtype=jnp.float32)

    # Stop gradients w.r.t. recon when optimizing geometry
    recon_c = jax.lax.stop_gradient(recon_flat)

    def step(carry, y):
        w = jnp.stack([Xr, jnp.full((n_rays,), y, dtype=jnp.float32), Zr], axis=0)
        q = Rinv @ (w - t[:, None])  # world -> object coords

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
# Main driver
# ----------------------------
def save_tiff3d(filepath: Path, array_3d: np.ndarray) -> dict:
    # Save 3D array as 3D TIFF (pages, H, W)
    # If volume is (nx, ny, nz), convert to (nz, ny, nx) stack
    data = np.asarray(array_3d)
    tiff.imwrite(str(filepath), data.astype(np.float32))
    return {
        "filepath": str(filepath),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel-beam JAX projector demo")
    # Volume/grid
    parser.add_argument("--nx", type=int, default=128, help="Volume size in x")
    parser.add_argument("--ny", type=int, default=128, help="Volume size in y")
    parser.add_argument("--nz", type=int, default=128, help="Volume size in z")
    parser.add_argument("--vx", type=float, default=1.0, help="Voxel size x")
    parser.add_argument("--vy", type=float, default=1.0, help="Voxel size y")
    parser.add_argument("--vz", type=float, default=1.0, help="Voxel size z")
    # Phantom
    parser.add_argument("--n-cubes", type=int, default=8, help="# cubes")
    parser.add_argument("--n-spheres", type=int, default=7, help="# spheres")
    parser.add_argument("--min-size", type=int, default=4, help="Min object size")
    parser.add_argument("--max-size", type=int, default=12, help="Max object size")
    parser.add_argument("--min-value", type=float, default=0.3, help="Min density")
    parser.add_argument("--max-value", type=float, default=1.0, help="Max density")
    parser.add_argument("--add-noise", action="store_true", help="Add Gaussian noise")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Noise std")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Detector
    parser.add_argument("--nu", type=int, default=None, help="Detector width (default: nx)")
    parser.add_argument("--nv", type=int, default=None, help="Detector height (default: nz)")
    parser.add_argument("--du", type=float, default=None, help="Detector pixel size u (default: vx)")
    parser.add_argument("--dv", type=float, default=None, help="Detector pixel size v (default: vz)")
    # Projections
    parser.add_argument("--n-proj", type=int, default=128, help="# projections over 180 deg")
    parser.add_argument("--step-size", type=float, default=None, help="Integration step (default: vy)")
    parser.add_argument("--n-steps", type=int, default=None, help="# steps (default: ceil(ny*vy/step))")
    parser.add_argument("--checkpoint", action="store_true", help="Use remat checkpointing in scan")
    # Output
    parser.add_argument("--output-dir", type=str, default="parallel_proj_test", help="Output folder")

    args = parser.parse_args()

    nx, ny, nz = args.nx, args.ny, args.nz
    vx, vy, vz = float(args.vx), float(args.vy), float(args.vz)
    nu = int(args.nu) if args.nu is not None else nx
    nv = int(args.nv) if args.nv is not None else nz
    du = float(args.du) if args.du is not None else vx
    dv = float(args.dv) if args.dv is not None else vz
    n_proj = int(args.n_proj)
    step_size = float(args.step_size) if args.step_size is not None else vy
    if args.n_steps is None:
        n_steps = int(math.ceil((ny * vy) / step_size))
    else:
        n_steps = int(args.n_steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Parallel-beam JAX Projector Demo ===")
    print(f"Device: {jax.devices()[0]}")
    print(f"Volume: {nx} x {ny} x {nz}  voxels")
    print(f"Detector: {nu} x {nv}  pixels")
    print(f"Projections: {n_proj} over 180 degrees")
    print(f"Step size: {step_size} | Steps: {n_steps}")
    print(f"Output: {out_dir}")
    print(f"Initial RSS memory: {get_memory_usage_mb():.1f} MB")
    print()

    # 1) Build phantom
    t0 = tic()
    phantom = make_phantom(
        nx=nx,
        ny=ny,
        nz=nz,
        n_cubes=args.n_cubes,
        n_spheres=args.n_spheres,
        min_size=args.min_size,
        max_size=args.max_size,
        min_value=args.min_value,
        max_value=args.max_value,
        background_value=0.0,
        add_noise=args.add_noise,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    t_phantom = toc(t0)
    print("Phantom built.")
    print(
        f"  stats: min={phantom.min():.4f} max={phantom.max():.4f} "
        f"mean={phantom.mean():.4f} std={phantom.std():.4f}"
    )
    print(f"  time: {t_phantom:.3f} s | RSS: {get_memory_usage_mb():.1f} MB")

    # Save phantom as 3D TIFF (z-stack: pages=z, H=y, W=x)
    tiff_path = out_dir / "phantom.tiff"
    phantom_stack = np.transpose(phantom, (2, 1, 0))  # (z, y, x) pages
    info_ph = save_tiff3d(tiff_path, phantom_stack)
    print(f"Saved phantom TIFF: {info_ph}")

    # 2) Prepare grid and detector
    grid = {"nx": nx, "ny": ny, "nz": nz, "vx": vx, "vy": vy, "vz": vz}
    vol_origin = default_volume_origin(nx, ny, nz, vx, vy, vz)
    grid["vol_origin"] = vol_origin

    det = {"nu": nu, "nv": nv, "du": du, "dv": dv, "det_center": jnp.array([0.0, 0.0], dtype=jnp.float32)}

    # 3) Forward projection (sequential, low memory)
    recon_flat = jnp.asarray(phantom.ravel())  # float32
    params_template = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    phis = np.linspace(0.0, math.pi, n_proj, endpoint=False, dtype=np.float32)

    # Warm up JIT with first angle
    warm_params = params_template.at[2].set(phis[0])
    _ = forward_project_view(
        params=warm_params,
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
        det_center=det["det_center"],
        step_size=jnp.float32(step_size),
        n_steps=n_steps,
        use_checkpoint=bool(args.checkpoint),
    )
    jax.block_until_ready(_)
    print(f"JIT warmup done. RSS: {get_memory_usage_mb():.1f} MB")

    projections = np.zeros((n_proj, nv, nu), dtype=np.float32)
    t_all = tic()
    peak_mb = get_memory_usage_mb()

    for i, phi in enumerate(phis):
        params = params_template.at[2].set(phi)  # (alpha,beta,phi,dx,dz)
        t0 = tic()
        img = forward_project_view(
            params=params,
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
            det_center=det["det_center"],
            step_size=jnp.float32(step_size),
            n_steps=n_steps,
            use_checkpoint=bool(args.checkpoint),
        )
        img_np = np.asarray(jax.block_until_ready(img))
        projections[i] = img_np
        dt = toc(t0)

        curr_mb = get_memory_usage_mb()
        peak_mb = max(peak_mb, curr_mb)
        if (i + 1) % max(1, n_proj // 16) == 0 or (i + 1) == n_proj:
            print(
                f"  view {i+1:4d}/{n_proj} | phi={phi:.4f} rad | "
                f"time={dt:.3f} s | RSS={curr_mb:.1f} MB"
            )

    total_time = toc(t_all)
    avg_time = total_time / max(1, n_proj)
    print()
    print("Forward projection complete.")
    print(
        f"  projections stats: min={projections.min():.4f} "
        f"max={projections.max():.4f} mean={projections.mean():.4f} "
        f"std={projections.std():.4f}"
    )
    print(
        f"  total={total_time:.3f} s | per-view={avg_time:.3f} s "
        f"| RSS peak increase ~ {peak_mb - get_memory_usage_mb():.1f} MB"
    )
    print(f"  current RSS: {get_memory_usage_mb():.1f} MB")

    # 4) Save projections and metadata
    proj_path = out_dir / "projections.tiff"  # (n_proj, nv, nu) pages=angle
    info_pr = save_tiff3d(proj_path, projections)
    np.save(out_dir / "angles.npy", phis)

    metadata = {
        "grid": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "vol_origin": [float(vol_origin[0]), float(vol_origin[1]), float(vol_origin[2])],
        },
        "detector": {"nu": nu, "nv": nv, "du": du, "dv": dv, "det_center": [0.0, 0.0]},
        "projections": {"n_proj": n_proj, "angles_rad": "angles.npy", "step_size": step_size, "n_steps": n_steps},
        "phantom": {
            "n_cubes": args.n_cubes,
            "n_spheres": args.n_spheres,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "min_value": args.min_value,
            "max_value": args.max_value,
            "add_noise": bool(args.add_noise),
            "noise_std": args.noise_std,
            "seed": args.seed,
        },
        "paths": {"phantom_tiff": str(tiff_path), "projections_tiff": str(proj_path), "angles_npy": str(out_dir / "angles.npy")},
        "timing": {
            "phantom_build_s": t_phantom,
            "total_projection_s": total_time,
            "avg_per_view_s": avg_time,
        },
        "memory": {
            "final_rss_mb": get_memory_usage_mb(),
            "peak_rss_mb": peak_mb,
        },
        "jit_checkpoint": bool(args.checkpoint),
        "device": str(jax.devices()[0]),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("Saved outputs:")
    print(f"  Phantom:      {tiff_path}")
    print(f"  Projections:  {proj_path}")
    print(f"  Angles:       {out_dir / 'angles.npy'}")
    print(f"  Metadata:     {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()