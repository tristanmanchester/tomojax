#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Joint iterative reconstruction and 3D rigid alignment for laminography
# - Loads real data from a NeXus (HDF5) file
# - Initializes with a fixed sample tilt (laminography) about x- or y-axis
# - Optionally auto-selects tilt sign via a coarse pilot run
# - Runs multi-resolution alternating optimization:
#     FISTA-TV reconstruction + per-view alignment (alpha, beta, dx, dz),
#     with phi fixed by measured stage angles.
#
# Requirements:
#   - h5py, numpy, jax, tifffile, tqdm
#   - Your existing modules in this repo:
#       alignment_utils.py
#       optimization_steps.py
#       projector_parallel_jax.py
#
# Notes:
#   - For laminography, we set an initial constant tilt around x (default)
#     or y, then keep per-view parameters free to refine. Phi is fixed from
#     the file's rotation motor readings.
#   - If --tilt-direction auto, the script runs a short coarse round for
#     +tilt and -tilt and picks the one with lower objective.
#   - Real data needs absorption-domain projections. If your NeXus file is
#     already corrected (k11-67214_corrected.nxs), you may set --neglog 0.
#
# Example:
#   python run_laminography_alignment.py \
#     --input-path /test_data_real/k11-67214_corrected.nxs \
#     --proj-dset /entry/imaging/data \
#     --angle-dset /entry/imaging_sum/smaract_zrot_value_set \
#     --tilt-deg 30 --tilt-axis x --tilt-direction auto \
#     --bin-factors 4 2 1 --outer-iters 12 \
#     --vol-shape 256 256 256 --voxel-size 1 1 1 \
#     --neglog 1 --output-dir alignment_results_lamino
#
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import tifffile as tiff

# Repo-relative imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "examples"))

from alignment_utils import (
    create_resolution_pyramid,
    transfer_alignment_params,
    upsample_volume,
    save_intermediate_results,
)
from optimization_steps import (
    fista_tv_reconstruction,
    optimize_alignment_params,
)


def default_vol_origin(nx: int, ny: int, nz: int, vx: float, vy: float, vz: float):
    ox = -((nx / 2.0) - 0.5) * vx
    oy = -((ny / 2.0) - 0.5) * vy
    oz = -((nz / 2.0) - 0.5) * vz
    return np.array([ox, oy, oz], dtype=np.float32)


def load_nexus_data(
    input_path: str,
    proj_dset: str = "/entry/imaging/data",
    angle_dset: str = "/entry/imaging_sum/smaract_zrot_value_set",
    angle_unit: str = "deg",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read projections and angles from a NeXus (HDF5) file.

    Returns:
      projections: (n_proj, nv, nu) float32
      angles_rad: (n_proj,) float32 in radians
    """
    with h5py.File(input_path, "r") as f:
        if proj_dset not in f:
            raise KeyError(f"Dataset not found: {proj_dset}")
        if angle_dset not in f:
            raise KeyError(f"Dataset not found: {angle_dset}")

        projs = f[proj_dset][...]  # likely (n, v, u)
        angles = f[angle_dset][...]

    # Ensure float32
    projs = np.asarray(projs, dtype=np.float32)

    # Squeeze angles to 1D
    angles = np.asarray(angles).reshape(-1)
    if angle_unit.lower().startswith("deg"):
        angles_rad = np.deg2rad(angles.astype(np.float32))
    else:
        angles_rad = angles.astype(np.float32)

    # Standardize projection shape as (n_proj, nv, nu)
    if projs.ndim == 3:
        n0, n1, n2 = projs.shape
        # Assume first dim is n_proj
        n_proj = n0
        nv, nu = n1, n2
        projections = projs
    elif projs.ndim == 4:
        # If extra singleton dimension, squeeze it
        projections = np.squeeze(projs)
        if projections.ndim != 3:
            raise ValueError(f"Unexpected projections shape: {projs.shape}")
    else:
        raise ValueError(f"Unexpected projections shape: {projs.shape}")

    if len(angles_rad) != projections.shape[0]:
        raise ValueError(
            f"Angle count ({len(angles_rad)}) does not match projections "
            f"({projections.shape[0]})."
        )

    return projections, angles_rad.astype(np.float32)


def neglog_transform(
    I: np.ndarray, eps: float = 1e-6, i0_percentile: float = 99.0
) -> np.ndarray:
    """
    Estimate flatfield I0 per pixel from a high percentile across angles,
    then compute -log(I / I0). Clips negative values to 0.
    """
    I0 = np.percentile(I, i0_percentile, axis=0).astype(np.float32)
    I0 = np.maximum(I0, eps)
    Y = -np.log(np.clip(I, eps, None) / I0)
    Y = np.clip(Y, 0.0, np.percentile(Y, 99.9))
    return Y.astype(np.float32)


def build_grid_and_detector(
    nu: int,
    nv: int,
    vol_shape: Tuple[int, int, int],
    voxel_size: Tuple[float, float, float],
    det_pixel_size: Tuple[float, float],
    det_center: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[Dict, Dict]:
    nx, ny, nz = map(int, vol_shape)
    vx, vy, vz = map(float, voxel_size)
    du, dv = map(float, det_pixel_size)

    grid = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "vol_origin": default_vol_origin(nx, ny, nz, vx, vy, vz),
        "step_size": vy,
        "n_steps": int(math.ceil((ny * vy) / max(vy, 1e-12))),
    }
    det = {
        "nu": int(nu),
        "nv": int(nv),
        "du": du,
        "dv": dv,
        "det_center": np.array(det_center, dtype=np.float32),
    }
    return grid, det


def build_initial_params_with_tilt(
    angles_rad: np.ndarray,
    tilt_deg: float,
    tilt_axis: str = "x",  # "x" or "y"
    tilt_direction: str = "towards",  # "towards" | "away"
    jitter_rot_std_deg: float = 0.05,
    jitter_trans_std: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """
    Create initial per-view params (alpha, beta, phi, dx, dz) with a constant
    laminography tilt around x or y, and phi set from measured angles.
    """
    rng = np.random.default_rng(seed)
    n_proj = len(angles_rad)
    params = np.zeros((n_proj, 5), dtype=np.float32)
    params[:, 2] = angles_rad  # phi

    sign = +1.0 if tilt_direction == "towards" else -1.0
    tilt_rad = np.deg2rad(float(tilt_deg)) * sign

    if tilt_axis.lower() == "x":
        params[:, 0] = tilt_rad  # alpha
        params[:, 1] = 0.0
    elif tilt_axis.lower() == "y":
        params[:, 0] = 0.0
        params[:, 1] = tilt_rad  # beta
    else:
        raise ValueError("tilt_axis must be 'x' or 'y'.")

    # Add tiny random jitter to help avoid zero-gradient plateaus
    params[:, 0] += np.deg2rad(
        rng.normal(0.0, jitter_rot_std_deg, size=n_proj).astype(np.float32)
    )
    params[:, 1] += np.deg2rad(
        rng.normal(0.0, jitter_rot_std_deg, size=n_proj).astype(np.float32)
    )
    params[:, 3] += rng.normal(0.0, jitter_trans_std, size=n_proj).astype(np.float32)
    params[:, 4] += rng.normal(0.0, jitter_trans_std, size=n_proj).astype(np.float32)
    return params


def run_alternating_optimization(
    projections: np.ndarray,
    angles: np.ndarray,
    grid: Dict,
    det: Dict,
    initial_params: Optional[np.ndarray] = None,
    initial_recon: Optional[jnp.ndarray] = None,
    bin_factors: List[int] = [4, 2, 1],
    outer_iters: int = 20,
    recon_iters_schedule: List[int] = [10, 20, 30],
    align_iters_schedule: List[int] = [5, 10, 15],
    lambda_tv: float = 0.005,
    output_dir: str = "alignment_results",
    save_intermediates: bool = True,
) -> Tuple[jnp.ndarray, np.ndarray, Dict]:
    """
    Alternating optimization across multi-resolution pyramid.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pyramid = create_resolution_pyramid(projections, grid, det, bin_factors)
    n_proj = len(angles)

    if initial_params is None:
        rng = np.random.default_rng(42)
        current_params = np.zeros((n_proj, 5), dtype=np.float32)
        current_params[:, 2] = angles
        current_params[:, 0] = rng.normal(0, 0.001, n_proj).astype(np.float32)
        current_params[:, 1] = rng.normal(0, 0.001, n_proj).astype(np.float32)
        current_params[:, 3] = rng.normal(0, 0.1, n_proj).astype(np.float32)
        current_params[:, 4] = rng.normal(0, 0.1, n_proj).astype(np.float32)
    else:
        current_params = np.asarray(initial_params, dtype=np.float32).copy()

    current_recon = None
    if initial_recon is not None:
        current_recon = initial_recon

    results = {"resolution_levels": [], "final_metrics": {}}

    print("\n=== Multi-resolution alignment optimization ===")
    print(f"Resolution levels: {bin_factors}")

    for lvl, (bf, (projs_b, grid_b, det_b)) in enumerate(
        zip(bin_factors, pyramid)
    ):
        print(
            f"\n--- Resolution level {lvl+1}/{len(bin_factors)}: "
            f"{bf}x binning ---"
        )
        print(
            f"Grid: {grid_b['nx']}x{grid_b['ny']}x{grid_b['nz']}, "
            f"Det: {det_b['nv']}x{det_b['nu']}"
        )

        if lvl > 0:
            scale = bin_factors[lvl - 1] / bf
            current_params = transfer_alignment_params(
                current_params, scale, pyramid[lvl - 1][2], det_b
            )

            if current_recon is not None:
                tgt_shape = (grid_b["nx"], grid_b["ny"], grid_b["nz"])
                current_recon = upsample_volume(current_recon, tgt_shape)
        else:
            if current_recon is None:
                nx, ny, nz = grid_b["nx"], grid_b["ny"], grid_b["nz"]
                current_recon = jnp.zeros((nx, ny, nz), dtype=jnp.float32)

        recon_iters = recon_iters_schedule[
            min(lvl, len(recon_iters_schedule) - 1)
        ]
        align_iters = align_iters_schedule[
            min(lvl, len(align_iters_schedule) - 1)
        ]

        lvl_res = {
            "level": lvl,
            "bin_factor": bf,
            "outer_iterations": [],
            "final_reconstruction": None,
            "final_params": None,
        }

        cached_L = None
        L_recompute_interval = 5  # Recompute every n outer iter for stability

        for outer in range(outer_iters):
            print(f"\nOuter iteration {outer+1}/{outer_iters}")

            recompute_L = outer == 0 or (outer % L_recompute_interval == 0)

            # Reconstruction step
            print(f"  Reconstruction step ({recon_iters} FISTA-TV iters)...")
            if recompute_L:
                (
                    current_recon,
                    recon_obj_hist,
                    cached_L,
                ) = fista_tv_reconstruction(
                    projs_b,
                    angles,
                    current_recon,
                    grid_b,
                    det_b,
                    lambda_tv=lambda_tv,
                    max_iters=recon_iters,
                    verbose=False,
                    alignment_params=current_params,
                    precomputed_L=None,
                    step_scale=0.5,
                    power_iters=5,
                )
                print(
                    f"  L = {cached_L:.3e} cached for next "
                    f"{L_recompute_interval-1} iterations"
                )
            else:
                (
                    current_recon,
                    recon_obj_hist,
                    _,
                ) = fista_tv_reconstruction(
                    projs_b,
                    angles,
                    current_recon,
                    grid_b,
                    det_b,
                    lambda_tv=lambda_tv,
                    max_iters=recon_iters,
                    verbose=False,
                    alignment_params=current_params,
                    precomputed_L=cached_L,
                    step_scale=0.5,
                    power_iters=5,
                )

            # Alignment step
            print(f"  Alignment step ({align_iters} iterations)...")
            current_params, align_obj_hist = optimize_alignment_params(
                projs_b,
                angles,
                current_recon.ravel(),
                current_params,
                grid_b,
                det_b,
                max_iters=align_iters,
                rot_learning_rate=0.001,
                trans_learning_rate=0.1,
                verbose=True,
            )

            # Save intermediates
            if save_intermediates:
                save_intermediate_results(
                    output_dir,
                    lvl,
                    outer,
                    current_recon.ravel(),
                    current_params,
                    recon_obj_hist + align_obj_hist,
                    grid_b,
                )

            final_obj = (
                (recon_obj_hist[-1] if recon_obj_hist else 0.0)
                + (align_obj_hist[-1] if align_obj_hist else 0.0)
            )

            lvl_res["outer_iterations"].append(
                {
                    "iteration": outer,
                    "reconstruction_objective": (
                        recon_obj_hist[-1] if recon_obj_hist else 0.0
                    ),
                    "alignment_objective": (
                        align_obj_hist[-1] if align_obj_hist else 0.0
                    ),
                    "combined_objective": final_obj,
                }
            )

            print(f"  Combined objective: {final_obj:.6e}")

            if outer > 3:
                recent = [
                    r["combined_objective"]
                    for r in lvl_res["outer_iterations"][-3:]
                ]
                if max(recent) - min(recent) < 1e-5 * abs(recent[0]):
                    print(f"  Converged at outer iteration {outer+1}")
                    break

        lvl_res["final_reconstruction"] = current_recon
        lvl_res["final_params"] = current_params
        results["resolution_levels"].append(lvl_res)
        print(f"Resolution level {lvl+1} complete.")

    return current_recon, current_params, results


def choose_tilt_direction_pilot(
    projections: np.ndarray,
    angles: np.ndarray,
    grid: Dict,
    det: Dict,
    tilt_deg: float,
    tilt_axis: str,
    bin_factors: List[int],
    recon_iters: int,
    align_iters: int,
    outer_iters: int,
    lambda_tv: float,
    output_dir: str,
) -> str:
    """
    Run a short coarse pilot for both tilt signs and pick the one with
    lower final combined objective.
    """
    coarse_bf = [bin_factors[0]]
    pilot_recon_iters = [recon_iters]
    pilot_align_iters = [align_iters]
    pilot_outer_iters = outer_iters

    def run_for(direction: str) -> float:
        init_params = build_initial_params_with_tilt(
            angles, tilt_deg, tilt_axis, direction
        )
        pilot_dir = os.path.join(output_dir, f"pilot_{direction}")
        recon, params, res = run_alternating_optimization(
            projections,
            angles,
            grid,
            det,
            initial_params=init_params,
            initial_recon=None,
            bin_factors=coarse_bf,
            outer_iters=pilot_outer_iters,
            recon_iters_schedule=pilot_recon_iters,
            align_iters_schedule=pilot_align_iters,
            lambda_tv=lambda_tv,
            output_dir=pilot_dir,
            save_intermediates=False,
        )
        lvl = res["resolution_levels"][-1]
        last_obj = lvl["outer_iterations"][-1]["combined_objective"]
        return float(last_obj)

    print("\n=== Pilot run to choose tilt sign (coarse) ===")
    obj_towards = run_for("towards")
    print(f"Pilot final objective (towards): {obj_towards:.6e}")
    obj_away = run_for("away")
    print(f"Pilot final objective (away):    {obj_away:.6e}")

    best = "towards" if obj_towards <= obj_away else "away"
    print(f"Selected tilt direction: {best}")
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Laminography alignment and reconstruction"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="/test_data_real/k11-67214_corrected.nxs",
        help="NeXus (HDF5) input file",
    )
    parser.add_argument(
        "--proj-dset",
        type=str,
        default="/entry/imaging/data",
        help="Dataset path for projections",
    )
    parser.add_argument(
        "--angle-dset",
        type=str,
        default="/entry/imaging_sum/smaract_zrot_value_set",
        help="Dataset path for angles",
    )
    parser.add_argument(
        "--angle-unit",
        type=str,
        default="deg",
        choices=["deg", "rad"],
        help="Angle units in file",
    )
    parser.add_argument(
        "--neglog",
        type=int,
        default=1,
        help="Apply -log(I/I0) using per-pixel I0 (1=yes, 0=no)",
    )
    parser.add_argument(
        "--tilt-deg",
        type=float,
        default=30.0,
        help="Initial laminography tilt magnitude in degrees",
    )
    parser.add_argument(
        "--tilt-axis",
        type=str,
        default="x",
        choices=["x", "y"],
        help="Axis of initial tilt",
    )
    parser.add_argument(
        "--tilt-direction",
        type=str,
        default="auto",
        choices=["auto", "towards", "away"],
        help="Tilt direction sign: auto runs a coarse pilot to choose",
    )
    parser.add_argument(
        "--bin-factors",
        type=int,
        nargs="+",
        default=[4, 2, 1],
        help="Multi-resolution binning factors",
    )
    parser.add_argument(
        "--outer-iters",
        type=int,
        default=12,
        help="Outer iterations per resolution level",
    )
    parser.add_argument(
        "--lambda-tv",
        type=float,
        default=0.002,
        help="TV regularization weight",
    )
    parser.add_argument(
        "--recon-iters",
        type=int,
        nargs="+",
        default=[15, 25, 35],
        help="FISTA-TV iterations per resolution level",
    )
    parser.add_argument(
        "--align-iters",
        type=int,
        nargs="+",
        default=[8, 15, 20],
        help="Alignment iterations per resolution level",
    )
    parser.add_argument(
        "--vol-shape",
        type=int,
        nargs=3,
        default=None,
        help="Volume shape (nx ny nz). If not set, defaults to (nv, nu, nu)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Voxel size (vx vy vz) in world units",
    )
    parser.add_argument(
        "--det-pixel-size",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        help="Detector pixel size (du dv) in world units",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="alignment_results_lamino",
        help="Output directory",
    )
    parser.add_argument(
        "--auto-tilt-pilot-recon-iters",
        type=int,
        default=8,
        help="FISTA iters for pilot tilt-sign selection",
    )
    parser.add_argument(
        "--auto-tilt-pilot-align-iters",
        type=int,
        default=6,
        help="Align iters for pilot tilt-sign selection",
    )
    parser.add_argument(
        "--auto-tilt-pilot-outer-iters",
        type=int,
        default=3,
        help="Outer iters for pilot tilt-sign selection",
    )

    args = parser.parse_args()

    print("=== Laminography Alignment ===")
    print(f"Input: {args.input_path}")
    print(f"Device: {jax.devices()[0]}")

    # Load data
    projections, angles_rad = load_nexus_data(
        args.input_path, args.proj_dset, args.angle_dset, args.angle_unit
    )
    n_proj, nv, nu = projections.shape
    print(
        f"Loaded projections: {n_proj} frames, each {nv} (v) x {nu} (u)"
    )

    # Absorption-domain transform if requested
    if args.neglog:
        print("Applying -log(I/I0) transform (per-pixel I0 from 99th pct)...")
        projections = neglog_transform(projections)

    # Auto volume shape if not provided
    if args.vol_shape is None:
        # For laminography: square slices (nu x nu) in sample plane, nv slices total
        vol_shape = (nu, nu, nv)
        print(f"Auto volume shape: {vol_shape} (square {nu}x{nu} slices)")
    else:
        vol_shape = tuple(args.vol_shape)

    # Build geometry (world units assumed consistent)
    grid, det = build_grid_and_detector(
        nu,
        nv,
        vol_shape,
        tuple(args.voxel_size),
        tuple(args.det_pixel_size),
        det_center=(0.0, 0.0),
    )

    # Determine tilt direction
    if args.tilt_direction == "auto":
        best_dir = choose_tilt_direction_pilot(
            projections=projections,
            angles=angles_rad,
            grid=grid,
            det=det,
            tilt_deg=args.tilt_deg,
            tilt_axis=args.tilt_axis,
            bin_factors=args.bin_factors,
            recon_iters=args.auto_tilt_pilot_recon_iters,
            align_iters=args.auto_tilt_pilot_align_iters,
            outer_iters=args.auto_tilt_pilot_outer_iters,
            lambda_tv=args.lambda_tv,
            output_dir=args.output_dir,
        )
    else:
        best_dir = args.tilt_direction
        print(f"Tilt direction fixed: {best_dir}")

    # Build initial params with chosen tilt
    init_params = build_initial_params_with_tilt(
        angles_rad,
        tilt_deg=args.tilt_deg,
        tilt_axis=args.tilt_axis,
        tilt_direction=best_dir,
    )

    # Run full multi-resolution alternating optimization
    start = time.time()
    final_recon, final_params, results = run_alternating_optimization(
        projections=projections,
        angles=angles_rad,
        grid=grid,
        det=det,
        initial_params=init_params,
        initial_recon=None,
        bin_factors=args.bin_factors,
        outer_iters=args.outer_iters,
        recon_iters_schedule=args.recon_iters,
        align_iters_schedule=args.align_iters,
        lambda_tv=args.lambda_tv,
        output_dir=args.output_dir,
        save_intermediates=True,
    )
    total_time = time.time() - start

    # Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol_np = np.asarray(final_recon, dtype=np.float32)
    vol_stack = np.transpose(vol_np, (2, 1, 0))  # (z, y, x)
    tiff.imwrite(str(out_dir / "final_reconstruction.tiff"), vol_stack)

    np.save(out_dir / "final_alignment_params.npy", final_params)
    np.save(out_dir / "angles_rad.npy", angles_rad)

    results["total_time_seconds"] = float(total_time)
    results["algorithm_params"] = vars(args)
    results["chosen_tilt_direction"] = best_dir

    def convert(obj):
        if isinstance(obj, (np.ndarray, jnp.ndarray)) or hasattr(
            obj, "__array__"
        ):
            return np.asarray(obj).tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    with open(out_dir / "results.json", "w") as f:
        json.dump(convert(results), f, indent=2)

    print("\n=== Done ===")
    print(f"Total time: {total_time:.2f} s")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()