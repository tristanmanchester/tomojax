#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Quick test script for debugging alignment algorithm
# Uses small data sizes and fast settings for rapid iteration

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import tifffile as tiff

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'examples'))

from alignment_utils import compute_alignment_metrics, bin_projections, bin_volume
from run_alignment import run_alternating_optimization


def create_quick_test_data(
    full_projections: np.ndarray,
    angles: np.ndarray,
    true_params: np.ndarray,
    grid: Dict, det: Dict,
    downsample_factor: int = 8,
    n_proj_subset: int = 64
) -> tuple:
    """Create smaller test data for quick debugging."""
    
    # Subsample projections (every nth angle)
    proj_step = max(1, len(angles) // n_proj_subset)
    proj_indices = np.arange(0, len(angles), proj_step)[:n_proj_subset]
    
    small_projections = full_projections[proj_indices]
    small_angles = angles[proj_indices]
    small_true_params = true_params[proj_indices]
    
    # Downsample spatial dimensions
    small_projections = bin_projections(small_projections, downsample_factor)
    
    # Update grid and detector parameters
    small_grid = grid.copy()
    small_grid['nx'] = grid['nx'] // downsample_factor
    small_grid['ny'] = grid['ny'] // downsample_factor
    small_grid['nz'] = grid['nz'] // downsample_factor
    small_grid['vx'] = grid['vx'] * downsample_factor
    small_grid['vy'] = grid['vy'] * downsample_factor
    small_grid['vz'] = grid['vz'] * downsample_factor
    
    small_det = det.copy()
    small_det['nu'] = det['nu'] // downsample_factor
    small_det['nv'] = det['nv'] // downsample_factor
    small_det['du'] = det['du'] * downsample_factor
    small_det['dv'] = det['dv'] * downsample_factor
    
    print(f"Quick test data created:")
    print(f"  Original: {full_projections.shape} projections, {grid['nx']}³ volume")
    print(f"  Quick test: {small_projections.shape} projections, {small_grid['nx']}³ volume")
    print(f"  Speedup factor: ~{(full_projections.shape[0]/small_projections.shape[0]) * (downsample_factor**3):.0f}x")
    
    return small_projections, small_angles, small_true_params, small_grid, small_det


def run_quick_test(data_dir: str = "../misaligned_test") -> Dict:
    """Run quick alignment test with small data for debugging."""
    
    print("=== Quick Alignment Test (for debugging) ===")
    print(f"Device: {jax.devices()[0]}")
    
    # Load full data
    data_path = Path(data_dir)
    
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    projections_full = tiff.imread(data_path / "projections_misaligned.tiff")
    angles_full = np.load(data_path / "angles.npy").astype(np.float32)
    true_params_full = np.load(data_path / "misalignment_params.npy").astype(np.float32)
    true_params_full[:, 2] = angles_full
    
    grid_full = metadata["grid"]
    det_full = metadata["detector"]
    
    print(f"Loaded full dataset: {projections_full.shape}")
    
    # Create quick test data
    projections, angles, true_params, grid, det = create_quick_test_data(
        projections_full, angles_full, true_params_full, grid_full, det_full,
        downsample_factor=8,  # 256³ -> 32³
        n_proj_subset=64      # 512 -> 64 projections
    )
    
    # Setup quick test parameters
    test_config = {
        "bin_factors": [2, 1],           # Only 2 levels for speed
        "outer_iters": 2,                # Very few outer iterations for debugging
        "lambda_tv": 0.01,               # Slightly higher for faster convergence
        "recon_iters_schedule": [3, 5],  # Very few FISTA iterations
        "align_iters_schedule": [5, 8],  # More alignment iterations to see convergence
        "output_dir": "quick_test_results"
    }
    
    print(f"\nQuick test configuration:")
    print(f"  Resolution levels: {test_config['bin_factors']}")
    print(f"  Outer iterations: {test_config['outer_iters']}")
    print(f"  Reconstruction iterations: {test_config['recon_iters_schedule']}")
    print(f"  Alignment iterations: {test_config['align_iters_schedule']}")
    
    # Setup grid parameters
    grid["step_size"] = grid["vy"]
    grid["n_steps"] = int(math.ceil((grid["ny"] * grid["vy"]) / grid["step_size"]))
    
    # Run alignment
    start_time = time.time()
    try:
        final_recon, estimated_params, results = run_alternating_optimization(
            projections, angles, grid, det, **test_config
        )
        total_time = time.time() - start_time
        
        # Compute metrics
        metrics = compute_alignment_metrics(estimated_params, true_params, det)
        
        print(f"\n=== Quick Test Results ===")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Success: Algorithm completed without errors")
        print(f"Alignment errors (RMSE):")
        print(f"  Rotation: {metrics['rot_rmse_deg']:.4f}°")
        print(f"  Translation: {metrics['trans_rmse_pix']:.4f} pixels")
        print(f"  Alpha: {metrics['alpha_rmse_deg']:.4f}°")
        print(f"  Beta: {metrics['beta_rmse_deg']:.4f}°")
        print(f"  dx: {metrics['dx_rmse_pix']:.4f} pixels")
        print(f"  dz: {metrics['dz_rmse_pix']:.4f} pixels")
        
        # Check if results are reasonable
        success = True
        if metrics['rot_rmse_deg'] > 10.0:  # Should improve rotation by some amount
            print("WARNING: Rotation error is quite high")
            success = False
        if metrics['trans_rmse_pix'] > 10.0:  # Should improve translation by some amount  
            print("WARNING: Translation error is quite high")
            success = False
        
        if success:
            print("✓ Quick test passed - algorithm appears to be working")
        else:
            print("⚠ Quick test completed with warnings - check algorithm or parameters")
        
        # Save quick results
        output_dir = Path(test_config["output_dir"])
        np.save(output_dir / "quick_estimated_params.npy", estimated_params)
        np.save(output_dir / "quick_true_params.npy", true_params)
        
        with open(output_dir / "quick_test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nQuick test results saved to: {output_dir}")
        
        return {
            "success": success,
            "metrics": metrics,
            "total_time": total_time,
            "test_config": test_config
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n=== Quick Test FAILED ===")
        print(f"Error after {total_time:.2f} seconds: {e}")
        print(f"This indicates an issue with the algorithm implementation.")
        
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "total_time": total_time,
            "test_config": test_config
        }


def main():
    """Run quick debugging test."""
    
    # Run quick test
    result = run_quick_test()
    
    if result["success"]:
        print(f"\n🎉 Quick test completed successfully in {result['total_time']:.1f} seconds!")
        print("Algorithm is ready for full-scale testing.")
    else:
        print(f"\n❌ Quick test failed after {result['total_time']:.1f} seconds")
        print("Debug the algorithm before running full tests.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())