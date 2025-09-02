#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Alignment utilities for multi-resolution joint reconstruction and alignment
# Based on Pande et al. (2022) "Joint iterative reconstruction and 3D rigid alignment for X-ray tomography"

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.ndimage import zoom
from typing import Dict, Tuple, List, Any
import tifffile as tiff


def bin_projections(projections: np.ndarray, bin_factor: int) -> np.ndarray:
    """
    Bin projections by averaging pixels in bin_factor x bin_factor blocks.
    
    Parameters:
    -----------
    projections : np.ndarray
        Shape (n_proj, nv, nu)
    bin_factor : int
        Binning factor (e.g., 2 means 2x2 binning)
        
    Returns:
    --------
    binned_projections : np.ndarray
        Shape (n_proj, nv//bin_factor, nu//bin_factor)
    """
    if bin_factor == 1:
        return projections
    
    n_proj, nv, nu = projections.shape
    
    # Ensure dimensions are divisible by bin_factor
    nv_new = (nv // bin_factor) * bin_factor
    nu_new = (nu // bin_factor) * bin_factor
    
    # Crop to divisible size if necessary
    proj_cropped = projections[:, :nv_new, :nu_new]
    
    # Reshape and average
    binned = proj_cropped.reshape(
        n_proj, nv_new // bin_factor, bin_factor, nu_new // bin_factor, bin_factor
    ).mean(axis=(2, 4))
    
    return binned.astype(np.float32)


def bin_volume(volume: np.ndarray, bin_factor: int) -> np.ndarray:
    """
    Bin volume by averaging voxels in bin_factor^3 blocks.
    
    Parameters:
    -----------
    volume : np.ndarray
        Shape (nx, ny, nz)
    bin_factor : int
        Binning factor
        
    Returns:
    --------
    binned_volume : np.ndarray
        Shape (nx//bin_factor, ny//bin_factor, nz//bin_factor)
    """
    if bin_factor == 1:
        return volume
    
    nx, ny, nz = volume.shape
    
    # Ensure dimensions are divisible by bin_factor
    nx_new = (nx // bin_factor) * bin_factor
    ny_new = (ny // bin_factor) * bin_factor
    nz_new = (nz // bin_factor) * bin_factor
    
    # Crop to divisible size if necessary
    vol_cropped = volume[:nx_new, :ny_new, :nz_new]
    
    # Reshape and average
    binned = vol_cropped.reshape(
        nx_new // bin_factor, bin_factor,
        ny_new // bin_factor, bin_factor,
        nz_new // bin_factor, bin_factor
    ).mean(axis=(1, 3, 5))
    
    return binned.astype(np.float32)


def upsample_volume(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Upsample volume using trilinear interpolation.
    
    Parameters:
    -----------
    volume : np.ndarray
        Coarse volume
    target_shape : tuple
        Target (nx, ny, nz) shape
        
    Returns:
    --------
    upsampled_volume : np.ndarray
        Upsampled volume with target_shape
    """
    if volume.shape == target_shape:
        return volume
    
    # Calculate zoom factors for each dimension
    zoom_factors = [t / c for t, c in zip(target_shape, volume.shape)]
    
    # Use scipy's zoom for high-quality upsampling
    upsampled = zoom(volume, zoom_factors, order=1)  # Linear interpolation
    
    return upsampled.astype(np.float32)


def create_resolution_pyramid(projections: np.ndarray, 
                            grid: Dict[str, Any], 
                            det: Dict[str, Any],
                            bin_factors: List[int]) -> List[Tuple[np.ndarray, Dict, Dict]]:
    """
    Create multi-resolution pyramid of projections and corresponding grid/detector parameters.
    
    Parameters:
    -----------
    projections : np.ndarray
        Original projections (n_proj, nv, nu)
    grid : dict
        Original grid parameters
    det : dict  
        Original detector parameters
    bin_factors : list
        List of binning factors (e.g., [4, 2, 1])
        
    Returns:
    --------
    pyramid : list
        List of (binned_projections, grid_params, det_params) tuples
    """
    pyramid = []
    
    for bin_factor in bin_factors:
        # Bin projections
        binned_projs = bin_projections(projections, bin_factor)
        
        # Update grid parameters
        grid_binned = grid.copy()
        grid_binned['nx'] = grid['nx'] // bin_factor
        grid_binned['ny'] = grid['ny'] // bin_factor  
        grid_binned['nz'] = grid['nz'] // bin_factor
        grid_binned['vx'] = grid['vx'] * bin_factor
        grid_binned['vy'] = grid['vy'] * bin_factor
        grid_binned['vz'] = grid['vz'] * bin_factor
        
        # Update volume origin if present
        if 'vol_origin' in grid:
            # Keep the same PHYSICAL region, just with coarser sampling
            # Original volume spans: [vol_origin, vol_origin + n*voxel_size]
            # Binned volume should span the same physical region
            vol_origin = np.asarray(grid['vol_origin'])
            grid_binned['vol_origin'] = vol_origin  # Same physical origin!
        
        # Update detector parameters
        det_binned = det.copy()
        det_binned['nu'] = det['nu'] // bin_factor
        det_binned['nv'] = det['nv'] // bin_factor
        det_binned['du'] = det['du'] * bin_factor
        det_binned['dv'] = det['dv'] * bin_factor
        
        # Scale detector center if present
        if 'det_center' in det:
            det_center = np.asarray(det['det_center'])
            det_binned['det_center'] = det_center * bin_factor
        
        pyramid.append((binned_projs, grid_binned, det_binned))
    
    return pyramid


def transfer_alignment_params(params: np.ndarray, 
                            scale_factor: float,
                            old_det: Dict[str, Any],
                            new_det: Dict[str, Any]) -> np.ndarray:
    """
    Transfer alignment parameters between resolution levels.
    
    Parameters:
    -----------
    params : np.ndarray
        Shape (n_proj, 5) with (alpha, beta, phi, dx, dz) parameters
    scale_factor : float
        Scale factor from old to new resolution (e.g., 2.0 for 2x upsampling)
    old_det : dict
        Old detector parameters
    new_det : dict
        New detector parameters
        
    Returns:
    --------
    scaled_params : np.ndarray
        Scaled parameters for new resolution
    """
    scaled_params = params.copy()
    
    # Rotational parameters (alpha, beta, phi) stay the same
    # Only translational parameters (dx, dz) need scaling
    
    # Scale translations by the change in pixel size
    old_du, old_dv = old_det['du'], old_det['dv']
    new_du, new_dv = new_det['du'], new_det['dv']
    
    scaled_params[:, 3] *= (old_du / new_du)  # dx scaling
    scaled_params[:, 4] *= (old_dv / new_dv)  # dz scaling
    
    return scaled_params


def initialize_reconstruction_fbp(projections: np.ndarray,
                                angles: np.ndarray,
                                grid: Dict[str, Any],
                                det: Dict[str, Any]) -> np.ndarray:
    """
    Initialize reconstruction using FBP (simplified Ram-Lak).
    
    Parameters:
    -----------
    projections : np.ndarray
        Projections for reconstruction
    angles : np.ndarray
        Projection angles
    grid : dict
        Grid parameters
    det : dict
        Detector parameters
        
    Returns:
    --------
    fbp_recon : np.ndarray
        FBP reconstruction as flat array (nx*ny*nz,)
    """
    # This is a simplified FBP - for full implementation, 
    # could call the existing FBP reconstruction script
    
    # For now, return zeros as placeholder
    # TODO: Implement proper FBP initialization
    nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
    return np.zeros(nx * ny * nz, dtype=np.float32)


def check_convergence(objective_history: List[float], 
                     param_history: List[np.ndarray],
                     tol_obj: float = 1e-4,
                     tol_param: float = 1e-3,
                     window: int = 3) -> bool:
    """
    Check convergence based on objective and parameter changes.
    
    Parameters:
    -----------
    objective_history : list
        History of objective function values
    param_history : list
        History of alignment parameter arrays
    tol_obj : float
        Tolerance for objective function relative change
    tol_param : float
        Tolerance for parameter relative change
    window : int
        Number of recent iterations to check
        
    Returns:
    --------
    converged : bool
        True if converged
    """
    if len(objective_history) < window + 1:
        return False
    
    # Check objective convergence
    recent_obj = objective_history[-window-1:]
    if len(recent_obj) > 1:
        obj_change = abs(recent_obj[-1] - recent_obj[0]) / (abs(recent_obj[0]) + 1e-12)
        if obj_change < tol_obj:
            return True
    
    # Check parameter convergence
    if len(param_history) >= 2:
        param_change = np.linalg.norm(param_history[-1] - param_history[-2])
        param_norm = np.linalg.norm(param_history[-2]) + 1e-12
        if param_change / param_norm < tol_param:
            return True
    
    return False


def save_intermediate_results(output_dir: str,
                            resolution_level: int,
                            outer_iter: int,
                            reconstruction: np.ndarray,
                            alignment_params: np.ndarray,
                            objective_history: List[float],
                            grid: Dict[str, Any]) -> None:
    """
    Save intermediate reconstruction and alignment results.
    
    Parameters:
    -----------
    output_dir : str
        Output directory path
    resolution_level : int
        Current resolution level
    outer_iter : int
        Current outer iteration
    reconstruction : np.ndarray
        Current reconstruction (flat array)
    alignment_params : np.ndarray
        Current alignment parameters
    objective_history : list
        Objective function history
    grid : dict
        Grid parameters for reshaping
    """
    import os
    from pathlib import Path
    
    # Create output directory
    res_dir = Path(output_dir) / f"resolution_{resolution_level}"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    # Save reconstruction as TIFF
    nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
    recon_vol = reconstruction.reshape(nx, ny, nz)
    recon_stack = np.transpose(recon_vol, (2, 1, 0))  # (z, y, x) for TIFF
    
    recon_path = res_dir / f"reconstruction_iter_{outer_iter:04d}.tiff"
    tiff.imwrite(str(recon_path), recon_stack.astype(np.float32))
    
    # Save alignment parameters
    params_path = res_dir / f"alignment_params_iter_{outer_iter:04d}.npy"
    np.save(params_path, alignment_params)
    
    # Save objective history
    if objective_history:
        obj_path = res_dir / f"objective_history_iter_{outer_iter:04d}.npy"
        np.save(obj_path, np.array(objective_history))


def compute_alignment_metrics(estimated_params: np.ndarray,
                            true_params: np.ndarray,
                            det: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute alignment error metrics.
    
    Parameters:
    -----------
    estimated_params : np.ndarray
        Estimated parameters (n_proj, 5)
    true_params : np.ndarray
        True parameters (n_proj, 5)
    det : dict
        Detector parameters for converting to pixel units
        
    Returns:
    --------
    metrics : dict
        Dictionary of error metrics
    """
    diff = estimated_params - true_params
    
    # Rotational errors (degrees)
    rot_errors_rad = diff[:, :3]  # alpha, beta, phi
    rot_errors_deg = np.rad2deg(rot_errors_rad)
    
    # Translational errors (pixels)
    trans_errors_world = diff[:, 3:]  # dx, dz
    trans_errors_pix = np.array([
        trans_errors_world[:, 0] / det['du'],  # dx in pixels
        trans_errors_world[:, 1] / det['dv']   # dz in pixels
    ]).T
    
    metrics = {
        'rot_rmse_deg': float(np.sqrt(np.mean(rot_errors_deg**2))),
        'rot_max_deg': float(np.max(np.abs(rot_errors_deg))),
        'trans_rmse_pix': float(np.sqrt(np.mean(trans_errors_pix**2))),
        'trans_max_pix': float(np.max(np.abs(trans_errors_pix))),
        'alpha_rmse_deg': float(np.sqrt(np.mean(rot_errors_deg[:, 0]**2))),
        'beta_rmse_deg': float(np.sqrt(np.mean(rot_errors_deg[:, 1]**2))),
        'phi_rmse_deg': float(np.sqrt(np.mean(rot_errors_deg[:, 2]**2))),
        'dx_rmse_pix': float(np.sqrt(np.mean(trans_errors_pix[:, 0]**2))),
        'dz_rmse_pix': float(np.sqrt(np.mean(trans_errors_pix[:, 1]**2))),
    }
    
    return metrics