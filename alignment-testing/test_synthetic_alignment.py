#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test script for synthetic alignment validation
# Runs alignment algorithm and compares results with ground truth

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'examples'))

from alignment_utils import compute_alignment_metrics
from run_alignment import run_alternating_optimization


def load_synthetic_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, Dict]:
    """Load synthetic misaligned data."""
    data_path = Path(data_dir)
    
    # Load metadata
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load projections and parameters
    projections_misaligned = tiff.imread(data_path / "projections_misaligned.tiff")
    angles = np.load(data_path / "angles.npy").astype(np.float32)
    true_params = np.load(data_path / "misalignment_params.npy").astype(np.float32)
    true_params[:, 2] = angles  # Set phi angles
    
    grid = metadata["grid"]
    det = metadata["detector"]
    
    print(f"Loaded synthetic data from {data_dir}")
    print(f"  Volume: {grid['nx']}x{grid['ny']}x{grid['nz']}")
    print(f"  Projections: {projections_misaligned.shape}")
    print(f"  Misalignment: ±{metadata['misalignment']['max_trans_pixels']} pixels, "
          f"±{metadata['misalignment']['max_rot_degrees']}°")
    
    return projections_misaligned, angles, true_params, grid, det


def run_alignment_test(
    projections: np.ndarray,
    angles: np.ndarray,
    true_params: np.ndarray,
    grid: Dict, det: Dict,
    test_name: str = "default",
    output_dir: str = "test_results",
    **alignment_kwargs
) -> Dict:
    """Run alignment algorithm and compute metrics."""
    
    print(f"\n=== Running alignment test: {test_name} ===")
    
    # Setup grid parameters
    grid = grid.copy()
    grid["step_size"] = grid["vy"]
    grid["n_steps"] = int(np.ceil((grid["ny"] * grid["vy"]) / grid["step_size"]))
    
    # Run alignment
    start_time = time.time()
    final_recon, estimated_params, results = run_alternating_optimization(
        projections, angles, grid, det, output_dir=output_dir, **alignment_kwargs
    )
    total_time = time.time() - start_time
    
    # Compute metrics
    metrics = compute_alignment_metrics(estimated_params, true_params, det)
    
    # Add timing and test info
    metrics["total_time_seconds"] = total_time
    metrics["test_name"] = test_name
    
    print(f"\n=== Test Results: {test_name} ===")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Alignment errors (RMSE):")
    print(f"  Rotation: {metrics['rot_rmse_deg']:.4f}°")
    print(f"  Translation: {metrics['trans_rmse_pix']:.4f} pixels")
    print(f"  Max rotation error: {metrics['rot_max_deg']:.4f}°")
    print(f"  Max translation error: {metrics['trans_max_pix']:.4f} pixels")
    
    return {
        "metrics": metrics,
        "final_reconstruction": final_recon,
        "estimated_params": estimated_params,
        "true_params": true_params,
        "results": results,
        "grid": grid,
        "det": det
    }


def plot_alignment_comparison(test_results: Dict, output_dir: str) -> None:
    """Create comparison plots of estimated vs true alignment parameters."""
    
    estimated_params = test_results["estimated_params"]
    true_params = test_results["true_params"]
    metrics = test_results["metrics"]
    
    n_proj = estimated_params.shape[0]
    proj_indices = np.arange(n_proj)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Alignment Parameter Comparison - {metrics['test_name']}", fontsize=14)
    
    # Parameter names and units
    param_info = [
        ("Alpha (°)", 0, np.rad2deg),
        ("Beta (°)", 1, np.rad2deg), 
        ("Phi (°)", 2, np.rad2deg),
        ("dx (pixels)", 3, lambda x: x / test_results["det"]["du"]),
        ("dz (pixels)", 4, lambda x: x / test_results["det"]["dv"])
    ]
    
    for i, (title, param_idx, convert_fn) in enumerate(param_info[:5]):  # Skip phi for now
        if i == 2:  # Skip phi (it's fixed)
            continue
            
        row = i // 3
        col = i % 3
        if i > 2:  # Adjust for skipped phi
            row = (i-1) // 3
            col = (i-1) % 3
        
        ax = axes[row, col]
        
        true_vals = convert_fn(true_params[:, param_idx])
        est_vals = convert_fn(estimated_params[:, param_idx])
        
        ax.plot(proj_indices, true_vals, 'b-', label='True', linewidth=2)
        ax.plot(proj_indices, est_vals, 'r--', label='Estimated', linewidth=2)
        ax.set_xlabel('Projection Index')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add RMSE to title
        if param_idx == 0:
            rmse = metrics['alpha_rmse_deg']
        elif param_idx == 1:
            rmse = metrics['beta_rmse_deg']
        elif param_idx == 3:
            rmse = metrics['dx_rmse_pix']
        elif param_idx == 4:
            rmse = metrics['dz_rmse_pix']
        
        ax.set_title(f"{title} (RMSE: {rmse:.4f})")
    
    # Remove empty subplot
    if len(param_info) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / f"alignment_comparison_{metrics['test_name']}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot: {plot_path}")
    plt.close()


def plot_convergence_history(test_results: Dict, output_dir: str) -> None:
    """Plot convergence history for each resolution level."""
    
    results = test_results["results"]
    metrics = test_results["metrics"]
    
    fig, axes = plt.subplots(1, len(results["resolution_levels"]), figsize=(15, 5))
    if len(results["resolution_levels"]) == 1:
        axes = [axes]
    
    fig.suptitle(f"Convergence History - {metrics['test_name']}", fontsize=14)
    
    for level_idx, level_data in enumerate(results["resolution_levels"]):
        ax = axes[level_idx]
        
        outer_iters = level_data["outer_iterations"]
        if not outer_iters:
            continue
            
        iterations = [r["iteration"] for r in outer_iters]
        recon_objs = [r["reconstruction_objective"] for r in outer_iters]
        align_objs = [r["alignment_objective"] for r in outer_iters]
        combined_objs = [r["combined_objective"] for r in outer_iters]
        
        ax.semilogy(iterations, recon_objs, 'b-', label='Reconstruction', linewidth=2)
        ax.semilogy(iterations, align_objs, 'r-', label='Alignment', linewidth=2)
        ax.semilogy(iterations, combined_objs, 'k--', label='Combined', linewidth=2)
        
        ax.set_xlabel('Outer Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title(f"Level {level_idx+1} ({level_data['bin_factor']}x binning)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / f"convergence_history_{metrics['test_name']}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved convergence plot: {plot_path}")
    plt.close()


def save_test_results(test_results: Dict, output_dir: str) -> None:
    """Save test results to files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    test_name = test_results["metrics"]["test_name"]
    
    # Save metrics as JSON
    metrics_path = output_path / f"metrics_{test_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(test_results["metrics"], f, indent=2)
    
    # Save parameters
    np.save(output_path / f"estimated_params_{test_name}.npy", test_results["estimated_params"])
    np.save(output_path / f"true_params_{test_name}.npy", test_results["true_params"])
    
    print(f"Saved test results to: {output_dir}")


def main():
    """Run synthetic alignment validation tests."""
    
    print("=== Synthetic Alignment Validation ===")
    
    # Test configurations
    test_configs = [
        {
            "name": "fast",
            "bin_factors": [4, 2],
            "outer_iters": 5,
            "lambda_tv": 0.01,
            "recon_iters": [5, 10],
            "align_iters": [3, 5]
        },
        {
            "name": "standard",
            "bin_factors": [4, 2, 1],
            "outer_iters": 10,
            "lambda_tv": 0.005,
            "recon_iters": [10, 15, 20],
            "align_iters": [5, 8, 10]
        }
    ]
    
    # Load synthetic data
    data_dir = "../misaligned_test"
    projections, angles, true_params, grid, det = load_synthetic_data(data_dir)
    
    # Run tests
    all_results = {}
    for config in test_configs:
        test_name = config.pop("name")
        output_dir = f"test_results_{test_name}"
        
        test_result = run_alignment_test(
            projections, angles, true_params, grid, det,
            test_name=test_name, output_dir=output_dir, **config
        )
        
        # Save results and create plots
        save_test_results(test_result, output_dir)
        plot_alignment_comparison(test_result, output_dir)
        plot_convergence_history(test_result, output_dir)
        
        all_results[test_name] = test_result
    
    # Summary comparison
    print("\n=== Test Summary ===")
    for test_name, result in all_results.items():
        metrics = result["metrics"]
        print(f"{test_name:10s}: Rot RMSE = {metrics['rot_rmse_deg']:.4f}°, "
              f"Trans RMSE = {metrics['trans_rmse_pix']:.4f} pix, "
              f"Time = {metrics['total_time_seconds']:.1f}s")
    
    print("\nValidation complete!")


if __name__ == "__main__":
    main()