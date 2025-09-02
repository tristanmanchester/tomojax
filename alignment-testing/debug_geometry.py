#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from alignment_utils import create_resolution_pyramid
from run_quick_test import create_quick_test_data

def debug_geometry():
    """Debug geometric parameter differences between quick test and full test."""
    
    # Load original data
    data_path = Path("../misaligned_test")
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    projections_full = np.random.random((512, 256, 256)).astype(np.float32)  # Dummy data
    angles_full = np.linspace(0, 2*np.pi, 512, endpoint=False).astype(np.float32)
    
    grid_orig = metadata["grid"]
    det_orig = metadata["detector"]
    
    print("=== ORIGINAL PARAMETERS ===")
    print(f"Grid: {grid_orig}")
    print(f"Detector: {det_orig}")
    
    # Quick test parameters  
    print("\n=== QUICK TEST PARAMETERS ===")
    _, _, _, grid_quick, det_quick = create_quick_test_data(
        projections_full, angles_full, np.zeros((512, 5)), 
        grid_orig, det_orig, downsample_factor=8, n_proj_subset=64
    )
    print(f"Grid: {grid_quick}")
    print(f"Detector: {det_quick}")
    
    # Full test parameters (4x binning level)
    print("\n=== FULL TEST PARAMETERS (4x binning) ===")
    pyramid = create_resolution_pyramid(projections_full, grid_orig, det_orig, [4])
    _, grid_full, det_full = pyramid[0]
    print(f"Grid: {grid_full}")  
    print(f"Detector: {det_full}")
    
    # Compare key parameters
    print("\n=== COMPARISON ===")
    print(f"Quick nx,ny,nz: {grid_quick['nx']}, {grid_quick['ny']}, {grid_quick['nz']}")
    print(f"Full  nx,ny,nz: {grid_full['nx']}, {grid_full['ny']}, {grid_full['nz']}")
    
    print(f"Quick vx,vy,vz: {grid_quick['vx']:.6f}, {grid_quick['vy']:.6f}, {grid_quick['vz']:.6f}")
    print(f"Full  vx,vy,vz: {grid_full['vx']:.6f}, {grid_full['vy']:.6f}, {grid_full['vz']:.6f}")
    
    if 'vol_origin' in grid_quick:
        print(f"Quick vol_origin: {grid_quick['vol_origin']}")
    if 'vol_origin' in grid_full:
        print(f"Full  vol_origin: {grid_full['vol_origin']}")
    
    print(f"Quick det nu,nv: {det_quick['nu']}, {det_quick['nv']}")
    print(f"Full  det nu,nv: {det_full['nu']}, {det_full['nv']}")
    
    print(f"Quick det du,dv: {det_quick['du']:.6f}, {det_quick['dv']:.6f}")
    print(f"Full  det du,dv: {det_full['du']:.6f}, {det_full['dv']:.6f}")
    
    if 'det_center' in det_quick:
        print(f"Quick det_center: {det_quick['det_center']}")
    if 'det_center' in det_full:
        print(f"Full  det_center: {det_full['det_center']}")

if __name__ == "__main__":
    debug_geometry()