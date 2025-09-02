#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Joint iterative reconstruction and 3D rigid alignment for X-ray tomography
# Implementation of Pande et al. (2022) algorithm using JAX projector
# Multi-resolution alternating optimization: FISTA-TV reconstruction + per-view alignment

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tifffile as tiff

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'examples'))

# Import components
from alignment_utils import (
    create_resolution_pyramid, transfer_alignment_params,
    bin_volume, upsample_volume, check_convergence,
    save_intermediate_results, compute_alignment_metrics
)
from optimization_steps import fista_tv_reconstruction, optimize_alignment_params


def run_alternating_optimization(
    projections: np.ndarray, 
    angles: np.ndarray,
    grid: Dict, det: Dict,
    bin_factors: List[int] = [4, 2, 1],
    outer_iters: int = 20,
    recon_iters_schedule: List[int] = [10, 20, 30],
    align_iters_schedule: List[int] = [5, 10, 15],
    lambda_tv: float = 0.005,
    output_dir: str = "alignment_results"
) -> Tuple[jnp.ndarray, np.ndarray, Dict]:
    """Main alternating optimization algorithm with multi-resolution."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create resolution pyramid
    pyramid = create_resolution_pyramid(projections, grid, det, bin_factors)
    n_proj = len(angles)
    
    # Initialize alignment parameters (small random values to avoid zero gradients)
    rng = np.random.default_rng(42)
    current_params = np.zeros((n_proj, 5), dtype=np.float32)
    current_params[:, 2] = angles  # Set phi angles
    
    # Add small random initialization for alignment parameters (not phi)
    # Scale to be much smaller than expected true misalignments
    current_params[:, 0] = rng.normal(0, 0.001, n_proj)  # alpha: ~0.06° std
    current_params[:, 1] = rng.normal(0, 0.001, n_proj)  # beta: ~0.06° std  
    current_params[:, 3] = rng.normal(0, 0.1, n_proj)    # dx: 0.1 world units std
    current_params[:, 4] = rng.normal(0, 0.1, n_proj)    # dz: 0.1 world units std
    current_recon = None
    
    results = {"resolution_levels": [], "final_metrics": {}}
    
    print(f"\n=== Multi-resolution alignment optimization ===")
    print(f"Resolution levels: {bin_factors}")
    
    for level, (bin_factor, (projs_binned, grid_binned, det_binned)) in enumerate(zip(bin_factors, pyramid)):
        print(f"\n--- Resolution level {level+1}/{len(bin_factors)}: {bin_factor}x binning ---")
        print(f"Grid: {grid_binned['nx']}x{grid_binned['ny']}x{grid_binned['nz']}")
        
        # Transfer parameters from previous level
        if level > 0:
            scale_factor = bin_factors[level-1] / bin_factor
            current_params = transfer_alignment_params(
                current_params, scale_factor, pyramid[level-1][2], det_binned
            )
            
            # Upsample reconstruction
            target_shape = (grid_binned['nx'], grid_binned['ny'], grid_binned['nz'])
            current_recon = upsample_volume(current_recon, target_shape)
        else:
            # Initialize with zeros at coarsest level
            nx, ny, nz = grid_binned['nx'], grid_binned['ny'], grid_binned['nz']
            current_recon = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
        
        # Get iteration counts for this level
        recon_iters = recon_iters_schedule[min(level, len(recon_iters_schedule)-1)]
        align_iters = align_iters_schedule[min(level, len(align_iters_schedule)-1)]
        
        level_results = {
            "level": level,
            "bin_factor": bin_factor,
            "outer_iterations": [],
            "final_reconstruction": None,
            "final_params": None
        }
        
        # Initialize Lipschitz constant tracking
        cached_L = None
        L_recompute_interval = 3  # Re-estimate L every 3 outer iterations
        
        # Outer loop alternating optimization
        for outer_iter in range(outer_iters):
            print(f"\nOuter iteration {outer_iter+1}/{outer_iters}")
            
            # Determine whether to recompute Lipschitz constant
            should_recompute_L = (outer_iter == 0 or outer_iter % L_recompute_interval == 0)
            
            # 1. Reconstruction step (fix alignment, update reconstruction)
            print(f"  Reconstruction step ({recon_iters} FISTA-TV iterations)...")
            if should_recompute_L:
                # Estimate L and cache it
                current_recon, recon_obj_hist, cached_L = fista_tv_reconstruction(
                    projs_binned, angles, current_recon, grid_binned, det_binned,
                    lambda_tv=lambda_tv, max_iters=recon_iters, verbose=False,
                    alignment_params=current_params, precomputed_L=None
                )
                print(f"  L = {cached_L:.3e} cached for next {L_recompute_interval-1} iterations")
            else:
                # Reuse cached L
                current_recon, recon_obj_hist, _ = fista_tv_reconstruction(
                    projs_binned, angles, current_recon, grid_binned, det_binned,
                    lambda_tv=lambda_tv, max_iters=recon_iters, verbose=False,
                    alignment_params=current_params, precomputed_L=cached_L
                )
            
            # 2. Alignment step (fix reconstruction, update alignment)
            print(f"  Alignment step ({align_iters} iterations)...")
            current_params, align_obj_hist = optimize_alignment_params(
                projs_binned, angles, current_recon.ravel(), 
                current_params, grid_binned, det_binned,
                max_iters=align_iters, 
                rot_learning_rate=0.001,   # 0.001 rad ≈ 0.057° per step
                trans_learning_rate=0.01,  # 0.01 world units per step
                verbose=True  # Enable verbose mode to see parameter evolution
            )
            
            # Save intermediate results
            save_intermediate_results(
                output_dir, level, outer_iter, current_recon.ravel(),
                current_params, recon_obj_hist + align_obj_hist, grid_binned
            )
            
            # Track objectives
            final_obj = (recon_obj_hist[-1] if recon_obj_hist else 0.0) + \
                       (align_obj_hist[-1] if align_obj_hist else 0.0)
            
            level_results["outer_iterations"].append({
                "iteration": outer_iter,
                "reconstruction_objective": recon_obj_hist[-1] if recon_obj_hist else 0.0,
                "alignment_objective": align_obj_hist[-1] if align_obj_hist else 0.0,
                "combined_objective": final_obj
            })
            
            print(f"  Combined objective: {final_obj:.6e}")
            
            # Early stopping check
            if outer_iter > 3:
                recent_objs = [r["combined_objective"] for r in level_results["outer_iterations"][-3:]]
                if max(recent_objs) - min(recent_objs) < 1e-5 * abs(recent_objs[0]):
                    print(f"  Converged at outer iteration {outer_iter+1}")
                    break
        
        level_results["final_reconstruction"] = current_recon
        level_results["final_params"] = current_params
        results["resolution_levels"].append(level_results)
        
        print(f"Resolution level {level+1} complete.")
    
    return current_recon, current_params, results


def main():
    parser = argparse.ArgumentParser(description="Joint iterative reconstruction and 3D rigid alignment")
    parser.add_argument("--input-dir", type=str, default="../misaligned_test",
                        help="Directory with misaligned projections")
    parser.add_argument("--output-dir", type=str, default="alignment_results",
                        help="Output directory")
    parser.add_argument("--bin-factors", type=int, nargs="+", default=[4, 2, 1],
                        help="Multi-resolution binning factors")
    parser.add_argument("--outer-iters", type=int, default=15,
                        help="Outer iterations per resolution level")
    parser.add_argument("--lambda-tv", type=float, default=0.005,
                        help="TV regularization weight")
    parser.add_argument("--recon-iters", type=int, nargs="+", default=[10, 20, 30],
                        help="FISTA-TV iterations per resolution level")
    parser.add_argument("--align-iters", type=int, nargs="+", default=[5, 10, 15],
                        help="Alignment iterations per resolution level")
    
    args = parser.parse_args()
    
    print("=== Joint Iterative Reconstruction and 3D Rigid Alignment ===")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {jax.devices()[0]}")
    
    # Load data
    input_dir = Path(args.input_dir)
    with open(input_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load projections and parameters
    projections_misaligned = tiff.imread(input_dir / "projections_misaligned.tiff")
    angles = np.load(input_dir / "angles.npy").astype(np.float32)
    true_params = np.load(input_dir / "misalignment_params.npy").astype(np.float32)
    true_params[:, 2] = angles  # Set phi angles
    
    print(f"Loaded {projections_misaligned.shape[0]} projections, shape: {projections_misaligned.shape}")
    print(f"True misalignment range: ±{metadata['misalignment']['max_trans_pixels']} pixels, "
          f"±{metadata['misalignment']['max_rot_degrees']}°")
    
    # Setup grid and detector
    grid = metadata["grid"]
    det = metadata["detector"]
    grid["step_size"] = grid["vy"]
    grid["n_steps"] = int(math.ceil((grid["ny"] * grid["vy"]) / grid["step_size"]))
    
    # Run alignment
    start_time = time.time()
    final_recon, final_params, results = run_alternating_optimization(
        projections_misaligned, angles, grid, det,
        bin_factors=args.bin_factors,
        outer_iters=args.outer_iters,
        recon_iters_schedule=args.recon_iters,
        align_iters_schedule=args.align_iters,
        lambda_tv=args.lambda_tv,
        output_dir=args.output_dir
    )
    total_time = time.time() - start_time
    
    # Compute final metrics
    metrics = compute_alignment_metrics(final_params, true_params, det)
    
    print("\n=== Final Results ===")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Alignment errors (RMSE):")
    print(f"  Rotation: {metrics['rot_rmse_deg']:.4f}°")
    print(f"  Translation: {metrics['trans_rmse_pix']:.4f} pixels")
    
    # Save final results
    output_dir = Path(args.output_dir)
    
    # Save final reconstruction
    final_stack = np.transpose(np.asarray(final_recon), (2, 1, 0))
    tiff.imwrite(str(output_dir / "final_reconstruction.tiff"), final_stack.astype(np.float32))
    
    # Save alignment parameters and results
    np.save(output_dir / "final_alignment_params.npy", final_params)
    np.save(output_dir / "true_alignment_params.npy", true_params)
    
    results["final_metrics"] = metrics
    results["total_time_seconds"] = total_time
    results["algorithm_params"] = vars(args)
    
    with open(output_dir / "results.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_arrays(results), f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()