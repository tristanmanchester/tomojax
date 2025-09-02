#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Memory-efficient, differentiable parallel-beam projector with random misalignments
# - Creates a 3D phantom (cubes + spheres)
# - Projects N views over 180 degrees with random rigid-body misalignments
# - Applies physically accurate Poisson noise to projections
# - Saves phantom, clean projections, misaligned projections, and alignment parameters
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

# Add parent directory to path to import projector_parallel_jax
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from projector_parallel_jax import (
    rot_x, rot_y, rot_z, compose_R,
    default_volume_origin, build_detector_grid,
    trilinear_gather, forward_project_view,
    _flat_index
)


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

    return vol.astype(np.float32)


# ----------------------------
# Misalignment parameter generation
# ----------------------------
def generate_misalignment_params(
    n_proj: int,
    max_trans_pixels: float,
    max_rot_degrees: float,
    du: float,
    dv: float,
    seed: int = None
) -> np.ndarray:
    """
    Generate random misalignment parameters within specified bounds.
    
    Parameters:
    -----------
    n_proj : int
        Number of projections
    max_trans_pixels : float
        Maximum translational misalignment in pixels (±)
    max_rot_degrees : float
        Maximum rotational misalignment in degrees (±)
    du, dv : float
        Detector pixel sizes for converting pixels to world units
    seed : int, optional
        Random seed
    
    Returns:
    --------
    params : np.ndarray
        Array of shape (n_proj, 5) with (alpha, beta, phi, dx, dz) parameters
    """
    rng = np.random.default_rng(seed)
    
    # Convert bounds to appropriate units
    max_trans_x = max_trans_pixels * du  # pixels to world units
    max_trans_z = max_trans_pixels * dv  # pixels to world units
    max_rot_rad = np.deg2rad(max_rot_degrees)  # degrees to radians
    
    # Generate random parameters within bounds
    params = np.zeros((n_proj, 5), dtype=np.float32)
    
    # Rotational misalignments (alpha, beta, phi)
    params[:, 0] = rng.uniform(-max_rot_rad, max_rot_rad, n_proj)  # alpha
    params[:, 1] = rng.uniform(-max_rot_rad, max_rot_rad, n_proj)  # beta  
    params[:, 2] = rng.uniform(-max_rot_rad, max_rot_rad, n_proj)  # phi
    
    # Translational misalignments (dx, dz)
    params[:, 3] = rng.uniform(-max_trans_x, max_trans_x, n_proj)  # dx
    params[:, 4] = rng.uniform(-max_trans_z, max_trans_z, n_proj)  # dz
    
    return params


# ----------------------------
# Poisson noise application
# ----------------------------
@jax.jit
def apply_poisson_noise(projection: jnp.ndarray, I0: float, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Apply physically accurate Poisson noise to a projection.
    
    Parameters:
    -----------
    projection : jnp.ndarray
        Clean projection (line integrals)
    I0 : float
        Incident photon count per pixel
    key : jax.random.PRNGKey
        Random key for Poisson sampling
    
    Returns:
    --------
    Noisy projection with Poisson statistics
    """
    # Convert projection to transmitted intensity using Beer-Lambert law
    # projection contains line integrals (μ * length)
    transmitted_intensity = I0 * jnp.exp(-projection)
    
    # Apply Poisson noise (photon counting statistics)
    # Note: JAX's poisson expects lambda parameter (mean count)
    noisy_intensity = jax.random.poisson(key, transmitted_intensity)
    
    # Handle low photon counts more realistically
    # When we detect 0 photons, we know at least the true value was very small
    # Use a minimum that scales with I0 to prevent blow-out
    # This represents the detection limit of our system
    min_detected = jnp.maximum(1.0, I0 * 1e-6)  # At least 1 photon or 0.0001% of I0
    noisy_intensity = jnp.maximum(noisy_intensity, min_detected)
    
    # For very high attenuation, cap the projection value to prevent blow-out
    # This represents the maximum measurable attenuation of the system
    max_projection = -jnp.log(min_detected / I0)
    
    # Convert back to projection space with capping
    noisy_projection = -jnp.log(noisy_intensity / I0)
    noisy_projection = jnp.minimum(noisy_projection, max_projection)
    
    return noisy_projection


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
    parser = argparse.ArgumentParser(description="Parallel-beam JAX projector with random misalignments")
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
    parser.add_argument("--max-size", type=int, default=24, help="Max object size")
    parser.add_argument("--min-value", type=float, default=0.01, help="Min attenuation coefficient")
    parser.add_argument("--max-value", type=float, default=0.1, help="Max attenuation coefficient")
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
    # Misalignment parameters
    parser.add_argument("--max-trans-pixels", type=float, default=2.0, 
                        help="Maximum translational misalignment in pixels (±)")
    parser.add_argument("--max-rot-degrees", type=float, default=1.0,
                        help="Maximum rotational misalignment in degrees (±)")
    # Poisson noise parameters
    parser.add_argument("--add-projection-noise", action="store_true", help="Add Poisson noise to projections")
    parser.add_argument("--incident-photons", type=float, default=10000, 
                        help="Incident photon count I0 per pixel (lower = more noise, typical: 1e4 to 1e6)")
    # Output
    parser.add_argument("--output-dir", type=str, default="parallel_proj_misaligned", help="Output folder")

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

    print("=== Parallel-beam JAX Projector with Random Misalignments ===")
    print(f"Device: {jax.devices()[0]}")
    print(f"Volume: {nx} x {ny} x {nz}  voxels")
    print(f"Detector: {nu} x {nv}  pixels")
    print(f"Projections: {n_proj} over 180 degrees")
    print(f"Step size: {step_size} | Steps: {n_steps}")
    print(f"Misalignments: ±{args.max_trans_pixels} pixels, ±{args.max_rot_degrees}°")
    if args.add_projection_noise:
        print(f"Poisson noise: ON | I0 = {args.incident_photons} photons/pixel")
    else:
        print(f"Poisson noise: OFF")
    print(f"Output: {out_dir}")
    print(f"Initial RSS memory: {get_memory_usage_mb():.1f} MB")
    print()

    # Initialize JAX random key if noise is enabled
    if args.add_projection_noise:
        rng_key = jax.random.PRNGKey(args.seed)

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

    # 3) Generate misalignment parameters
    misalignment_params = generate_misalignment_params(
        n_proj=n_proj,
        max_trans_pixels=args.max_trans_pixels,
        max_rot_degrees=args.max_rot_degrees,
        du=du,
        dv=dv,
        seed=args.seed
    )
    
    print(f"Generated misalignment parameters:")
    print(f"  Translation range: ±{args.max_trans_pixels} pixels = ±{args.max_trans_pixels * du:.3f} world units (x), ±{args.max_trans_pixels * dv:.3f} (z)")
    print(f"  Rotation range: ±{args.max_rot_degrees}° = ±{np.deg2rad(args.max_rot_degrees):.4f} rad")

    # 4) Forward projection (sequential, low memory)
    recon_flat = jnp.asarray(phantom.ravel())  # float32
    perfect_params = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    phis = np.linspace(0.0, math.pi, n_proj, endpoint=False, dtype=np.float32)

    # Warm up JIT with first angle
    warm_params = perfect_params.at[2].set(phis[0])
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

    # Arrays for projections
    projections_clean = np.zeros((n_proj, nv, nu), dtype=np.float32)
    projections_misaligned = np.zeros((n_proj, nv, nu), dtype=np.float32)
    projections_noisy = np.zeros((n_proj, nv, nu), dtype=np.float32) if args.add_projection_noise else None
    
    t_all = tic()
    peak_mb = get_memory_usage_mb()

    for i, phi in enumerate(phis):
        t0 = tic()
        
        # Clean projection (perfect alignment)
        perfect_params_phi = perfect_params.at[2].set(phi)
        img_clean = forward_project_view(
            params=perfect_params_phi,
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
        
        # Misaligned projection
        misaligned_params_phi = jnp.array([
            misalignment_params[i, 0],  # alpha
            misalignment_params[i, 1],  # beta
            phi,  # phi (base angle)
            misalignment_params[i, 3],  # dx
            misalignment_params[i, 4],  # dz
        ], dtype=jnp.float32)
        
        img_misaligned = forward_project_view(
            params=misaligned_params_phi,
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
        
        # Apply Poisson noise if requested (to misaligned projections)
        if args.add_projection_noise:
            rng_key, subkey = jax.random.split(rng_key)
            img_noisy = apply_poisson_noise(img_misaligned, args.incident_photons, subkey)
            img_noisy_np = np.asarray(jax.block_until_ready(img_noisy))
            projections_noisy[i] = img_noisy_np
        
        img_clean_np = np.asarray(jax.block_until_ready(img_clean))
        img_misaligned_np = np.asarray(jax.block_until_ready(img_misaligned))
        projections_clean[i] = img_clean_np
        projections_misaligned[i] = img_misaligned_np
        
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
    
    # Report stats for clean projections
    print(f"Clean projections stats:")
    print(
        f"  min={projections_clean.min():.4f} "
        f"max={projections_clean.max():.4f} mean={projections_clean.mean():.4f} "
        f"std={projections_clean.std():.4f}"
    )
    
    # Report stats for misaligned projections
    print(f"Misaligned projections stats:")
    print(
        f"  min={projections_misaligned.min():.4f} "
        f"max={projections_misaligned.max():.4f} mean={projections_misaligned.mean():.4f} "
        f"std={projections_misaligned.std():.4f}"
    )
    
    # Report stats for noisy projections if applicable
    if args.add_projection_noise:
        print(f"Noisy projections stats:")
        print(
            f"  min={projections_noisy.min():.4f} "
            f"max={projections_noisy.max():.4f} mean={projections_noisy.mean():.4f} "
            f"std={projections_noisy.std():.4f}"
        )
        print(f"  SNR = {projections_noisy.mean() / projections_noisy.std():.2f}")
    
    print(
        f"Timing: total={total_time:.3f} s | per-view={avg_time:.3f} s "
        f"| RSS peak ~ {peak_mb:.1f} MB"
    )
    print(f"Current RSS: {get_memory_usage_mb():.1f} MB")

    # 5) Save projections and metadata
    clean_proj_path = out_dir / "projections_clean.tiff"
    misaligned_proj_path = out_dir / "projections_misaligned.tiff"
    
    info_clean = save_tiff3d(clean_proj_path, projections_clean)
    info_misaligned = save_tiff3d(misaligned_proj_path, projections_misaligned)
    
    # Save noisy projections if generated
    if args.add_projection_noise:
        noisy_proj_path = out_dir / "projections_noisy.tiff"
        info_noisy = save_tiff3d(noisy_proj_path, projections_noisy)
        print(f"Saved noisy projections TIFF: {noisy_proj_path}")
    
    # Save angles and alignment parameters
    np.save(out_dir / "angles.npy", phis)
    np.save(out_dir / "misalignment_params.npy", misalignment_params)
    
    # Save perfect alignment parameters for reference
    perfect_params_array = np.zeros((n_proj, 5), dtype=np.float32)
    perfect_params_array[:, 2] = phis  # Only phi angles, rest are zeros
    np.save(out_dir / "perfect_params.npy", perfect_params_array)

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
            "seed": args.seed,
        },
        "misalignment": {
            "max_trans_pixels": args.max_trans_pixels,
            "max_rot_degrees": args.max_rot_degrees,
            "max_trans_world_x": args.max_trans_pixels * du,
            "max_trans_world_z": args.max_trans_pixels * dv,
            "max_rot_radians": np.deg2rad(args.max_rot_degrees),
            "misalignment_params_file": "misalignment_params.npy",
            "perfect_params_file": "perfect_params.npy",
        },
        "noise": {
            "add_projection_noise": bool(args.add_projection_noise),
            "incident_photons": args.incident_photons if args.add_projection_noise else None,
        },
        "paths": {
            "phantom_tiff": str(tiff_path), 
            "projections_clean_tiff": str(clean_proj_path),
            "projections_misaligned_tiff": str(misaligned_proj_path),
            "projections_noisy_tiff": str(out_dir / "projections_noisy.tiff") if args.add_projection_noise else None,
            "angles_npy": str(out_dir / "angles.npy"),
            "misalignment_params_npy": str(out_dir / "misalignment_params.npy"),
            "perfect_params_npy": str(out_dir / "perfect_params.npy"),
        },
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
    print(f"  Phantom:              {tiff_path}")
    print(f"  Clean projections:    {clean_proj_path}")
    print(f"  Misaligned proj:      {misaligned_proj_path}")
    if args.add_projection_noise:
        print(f"  Noisy projections:    {noisy_proj_path}")
    print(f"  Angles:               {out_dir / 'angles.npy'}")
    print(f"  Misalignment params:  {out_dir / 'misalignment_params.npy'}")
    print(f"  Perfect params:       {out_dir / 'perfect_params.npy'}")
    print(f"  Metadata:             {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()