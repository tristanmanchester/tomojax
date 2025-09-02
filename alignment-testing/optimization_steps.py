#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Optimization step implementations for joint reconstruction and alignment
# FISTA-TV reconstruction and per-view alignment optimization

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'examples'))

# Import FISTA-TV components
from run_fista_tv import (
    prox_tv_cp, data_f_grad_vjp, estimate_L_power
)
from projector_parallel_jax import forward_project_view


def build_aligned_loss_and_grad_scan(
    projections: np.ndarray,  # (n_views, nv, nu)
    alignment_params: np.ndarray,  # (n_views, 5) [alpha, beta, phi, dx, dz]
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
    det_center: jnp.ndarray,  # (2,)
    step_size: float,
    n_steps: int,
):
    """
    Returns a compiled function loss_and_grad(x_flat) -> (fval, grad)
    that computes f(x) = 0.5 * sum_i ||A_i x - y_i||^2 and its grad via
    AD through a lax.scan over views.
    """
    # Device arrays once; keep shapes fixed for the compilation.
    y_stack = jnp.asarray(projections, dtype=jnp.float32)  # (n, nv, nu)
    params_stack = jnp.asarray(alignment_params, dtype=jnp.float32)  # (n, 5)

    # Flatten detector dims for lighter carries
    y_flat = y_stack.reshape(y_stack.shape[0], -1)  # (n, nv*nu)

    def loss_fn(x_flat: jnp.ndarray) -> jnp.ndarray:
        # Accumulate scalar loss over views via scan
        def body(f_acc, inputs):
            p_i, y_i = inputs  # (5,), (nv*nu,)
            pred_i = forward_project_view(
                params=p_i,
                recon_flat=x_flat,
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
                n_steps=int(n_steps),
                use_checkpoint=True,
                stop_grad_recon=False,  # need grad wrt x
            )
            r = pred_i.reshape(-1) - y_i
            l_i = 0.5 * jnp.vdot(r, r).real
            return f_acc + l_i, None

        f_sum, _ = jax.lax.scan(body, jnp.float32(0.0), (params_stack, y_flat))
        return f_sum

    # Single compile; reuse across all FISTA iterations at this level
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    return loss_and_grad


def build_AtA_scan(
    alignment_params: np.ndarray,
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
    vol_origin: jnp.ndarray,
    det_center: jnp.ndarray,
    step_size: float,
    n_steps: int,
):
    """
    Returns a compiled A^T A operator using scan over views.
    Used for power method Lipschitz constant estimation.
    """
    # Zero sinogram -> f(x) = 0.5 * ||A x||^2, grad = A^T A x
    zeros = jnp.zeros((alignment_params.shape[0], nv, nu), dtype=jnp.float32)

    loss_and_grad = build_aligned_loss_and_grad_scan(
        projections=zeros,
        alignment_params=alignment_params,
        nx=nx,
        ny=ny,
        nz=nz,
        vx=vx,
        vy=vy,
        vz=vz,
        nu=nu,
        nv=nv,
        du=du,
        dv=dv,
        vol_origin=vol_origin,
        det_center=det_center,
        step_size=step_size,
        n_steps=n_steps,
    )

    @jax.jit
    def AtA(x_flat: jnp.ndarray) -> jnp.ndarray:
        _, g = loss_and_grad(x_flat)
        return g

    return AtA


def compute_view_loss(
    params: jnp.ndarray,
    measured: jnp.ndarray,
    recon_flat: jnp.ndarray,
    nx: int, ny: int, nz: int,
    vx: float, vy: float, vz: float,
    nu: int, nv: int,
    du: float, dv: float,
    vol_origin: jnp.ndarray,
    det_center: jnp.ndarray,
    step_size: float,
    n_steps: int
) -> jnp.ndarray:
    """Compute view loss without static arguments to avoid JAX hashability issues."""
    
    # Forward project with current parameters
    pred = forward_project_view(
        params=params,
        recon_flat=recon_flat,
        nx=nx, ny=ny, nz=nz,
        vx=jnp.float32(vx), vy=jnp.float32(vy), vz=jnp.float32(vz),
        nu=nu, nv=nv,
        du=jnp.float32(du), dv=jnp.float32(dv),
        vol_origin=vol_origin,
        det_center=det_center,
        step_size=jnp.float32(step_size),
        n_steps=n_steps,
        use_checkpoint=True,
        stop_grad_recon=True
    )
    
    # Compute MSE loss
    resid = (pred - measured).astype(jnp.float32)
    return 0.5 * jnp.vdot(resid, resid).real


def fista_tv_reconstruction(
    projections: np.ndarray,
    angles: np.ndarray,
    initial_recon: jnp.ndarray,
    grid: Dict, det: Dict,
    lambda_tv: float = 0.005,
    max_iters: int = 20,
    tv_iters: int = 20,
    step_scale: float = 0.9,
    power_iters: int = 3,
    verbose: bool = True,
    alignment_params: np.ndarray = None,
    precomputed_L: float = None
) -> Tuple[jnp.ndarray, List[float], float]:
    """Run FISTA-TV reconstruction for given projections and alignment."""
    
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    step_size = float(grid.get("step_size", grid["vy"]))
    n_steps = int(grid.get("n_steps", math.ceil((ny * grid["vy"]) / step_size)))
    
    # Make dictionaries hashable by converting arrays to tuples
    grid_hashable = grid.copy()
    if 'vol_origin' in grid_hashable and isinstance(grid_hashable['vol_origin'], np.ndarray):
        grid_hashable['vol_origin'] = tuple(float(x) for x in grid_hashable['vol_origin'])
    
    det_hashable = det.copy()
    if 'det_center' in det_hashable and isinstance(det_hashable['det_center'], np.ndarray):
        det_hashable['det_center'] = tuple(float(x) for x in det_hashable['det_center'])
    
    # Use alignment-aware functions if parameters provided
    if alignment_params is not None:
        # Extract constants and device arrays for scan-based approach
        vx, vy, vz = float(grid["vx"]), float(grid["vy"]), float(grid["vz"])
        nu, nv = int(det["nu"]), int(det["nv"])
        du, dv = float(det["du"]), float(det["dv"])

        vol_origin = jnp.asarray(grid["vol_origin"], dtype=jnp.float32)
        det_center = jnp.asarray(det.get("det_center", [0.0, 0.0]), dtype=jnp.float32)

        # Use precomputed L or estimate it if not provided
        if precomputed_L is not None:
            L = precomputed_L
            if verbose:
                print(f"  Using precomputed Lipschitz constant L = {L:.3e}")
        else:
            if verbose:
                print(f"  Estimating Lipschitz constant using scan-based AtA ({power_iters} iterations)...")
            L = estimate_L_power_scan(alignment_params, nx, ny, nz, vx, vy, vz,
                                    nu, nv, du, dv, vol_origin, det_center,
                                    step_size, n_steps, n_iter=power_iters, seed=42)
            if verbose:
                print(f"  L = {L:.3e} (estimated)")
        
        alpha = step_scale / L
        if verbose:
            print(f"  alpha = {alpha:.3e}")

        # Build scan-based loss and gradient function once for this level
        if verbose:
            print(f"  Compiling scan-based loss and gradient function for {len(alignment_params)} views...")
        loss_and_grad_scan = build_aligned_loss_and_grad_scan(
            projections=projections,
            alignment_params=alignment_params,
            nx=nx,
            ny=ny,
            nz=nz,
            vx=vx,
            vy=vy,
            vz=vz,
            nu=nu,
            nv=nv,
            du=du,
            dv=dv,
            vol_origin=vol_origin,
            det_center=det_center,
            step_size=step_size,
            n_steps=n_steps,
        )
        if verbose:
            print(f"  Compilation complete, starting FISTA iterations...")
        
        # FISTA variables
        xk = initial_recon.ravel()
        xk_prev = xk
        tk = 1.0
        objective_history = []
        
        for it in range(1, max_iters + 1):
            # Momentum step
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
            beta = (tk - 1.0) / t_next
            zk = xk + jnp.float32(beta) * (xk - xk_prev)
            
            # Gradient step using scan-based loss and gradient
            fval, grad = loss_and_grad_scan(zk)  # single compiled call
            yk = zk - jnp.float32(alpha) * grad
            
            # TV proximal step
            mu = lambda_tv * alpha
            x_next = prox_tv_cp(yk.reshape(nx, ny, nz), mu=mu, n_iters=tv_iters).ravel()
            
            # Update
            xk_prev = xk
            xk = x_next
            tk = t_next
            
            objective_history.append(float(fval))
            
            if verbose and (it % max(1, max_iters // 10) == 0 or it == max_iters):
                grad_norm = float(jnp.linalg.norm(grad))
                print(f"    FISTA iter {it:3d}/{max_iters} | f={fval:.6e} | ||grad||={grad_norm:.3e}")
    else:
        # Use original functions when no alignment parameters provided (backward compatibility)
        # Estimate Lipschitz constant
        if verbose:
            print(f"  Estimating Lipschitz constant ({power_iters} iterations)...")
        L = estimate_L_power(grid_hashable, det_hashable, angles, step_size, n_steps, n_iter=power_iters, seed=42)
        alpha = step_scale / L
        
        if verbose:
            print(f"  L = {L:.3e}, alpha = {alpha:.3e}")
        
        # FISTA variables
        xk = initial_recon.ravel()
        xk_prev = xk
        tk = 1.0
        objective_history = []
        
        for it in range(1, max_iters + 1):
            # Momentum step
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
            beta = (tk - 1.0) / t_next
            zk = xk + jnp.float32(beta) * (xk - xk_prev)
            
            # Gradient step
            fval, grad = data_f_grad_vjp(zk, projections, angles, grid_hashable, det_hashable, step_size, n_steps)
            yk = zk - jnp.float32(alpha) * grad
            
            # TV proximal step
            mu = lambda_tv * alpha
            x_next = prox_tv_cp(yk.reshape(nx, ny, nz), mu=mu, n_iters=tv_iters).ravel()
            
            # Update
            xk_prev = xk
            xk = x_next
            tk = t_next
            
            objective_history.append(float(fval))
            
            if verbose and (it % max(1, max_iters // 10) == 0 or it == max_iters):
                grad_norm = float(jnp.linalg.norm(grad))
                print(f"    FISTA iter {it:3d}/{max_iters} | f={fval:.6e} | ||grad||={grad_norm:.3e}")
    
    return xk.reshape(nx, ny, nz), objective_history, L


def data_f_grad_vjp_aligned(x_flat: jnp.ndarray,
                           projections: np.ndarray,  # (n_proj, nv, nu)
                           angles: np.ndarray,       # (n_proj,)
                           alignment_params: np.ndarray,  # (n_proj, 5)
                           grid, det,
                           step_size: float,
                           n_steps: int,
                           verbose: bool = False,
                           desc: str = "") -> Tuple[float, jnp.ndarray]:
    """Compute data fidelity gradient using current alignment parameters."""
    n_proj = int(projections.shape[0])
    fval = 0.0
    grad = jnp.zeros_like(x_flat)
    
    # Extract parameters to avoid dictionary hashability issues
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    vx, vy, vz = float(grid["vx"]), float(grid["vy"]), float(grid["vz"])
    nu, nv = int(det["nu"]), int(det["nv"])
    du, dv = float(det["du"]), float(det["dv"])
    
    # Convert arrays to JAX arrays
    vol_origin = jnp.asarray(grid["vol_origin"], dtype=jnp.float32)
    det_center = jnp.asarray(det.get("det_center", [0.0, 0.0]), dtype=jnp.float32)
    
    if verbose and desc:
        print(f"  {desc}: Processing {n_proj} projections...")
        import time
        t_start = time.time()

    for i in range(n_proj):
        if verbose and desc and (i % max(1, n_proj // 10) == 0):
            import time
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 0.01)
            eta = (n_proj - i - 1) / max(rate, 0.01)
            print(f"    Progress: {i+1}/{n_proj} ({100*(i+1)/n_proj:.1f}%) | "
                  f"Rate: {rate:.1f} proj/s | ETA: {eta:.1f}s")
        
        # Use current alignment parameters for this view
        params = jnp.asarray(alignment_params[i], dtype=jnp.float32)
        measured = jnp.asarray(projections[i], dtype=jnp.float32).ravel()
        
        # Define projection function
        def proj_fn(vol):
            proj_2d = forward_project_view(
                params=params,
                recon_flat=vol,
                nx=nx, ny=ny, nz=nz,
                vx=jnp.float32(vx), vy=jnp.float32(vy), vz=jnp.float32(vz),
                nu=nu, nv=nv,
                du=jnp.float32(du), dv=jnp.float32(dv),
                vol_origin=vol_origin,
                det_center=det_center,
                step_size=jnp.float32(step_size),
                n_steps=n_steps,
                use_checkpoint=True,
                stop_grad_recon=False
            )
            return proj_2d.ravel()

        pred, vjp_fn = jax.vjp(proj_fn, x_flat)
        resid = pred - measured

        fval += float(0.5 * jnp.vdot(resid, resid).real)
        grad = grad + vjp_fn(resid)[0]
    
    if verbose and desc:
        import time
        total_time = time.time() - t_start
        print(f"  {desc}: Completed in {total_time:.2f}s")

    return fval, grad


def apply_AtA_vjp_aligned(v: jnp.ndarray,
                         angles: np.ndarray,
                         alignment_params: np.ndarray,
                         grid, det,
                         step_size: float,
                         n_steps: int,
                         verbose: bool = False,
                         power_iter: int = 0) -> jnp.ndarray:
    """Compute A^T A v using current alignment parameters (with zero sinogram)."""
    zeros = np.zeros((len(angles), int(det["nv"]), int(det["nu"])), dtype=np.float32)
    desc = f"Power iter {power_iter}" if power_iter > 0 else "A^T A"
    _, g = data_f_grad_vjp_aligned(v, zeros, angles, alignment_params, 
                                  grid, det, step_size, n_steps, 
                                  verbose=verbose, desc=desc)
    return g


def estimate_L_power_scan(
    alignment_params: np.ndarray,
    nx: int, ny: int, nz: int,
    vx: float, vy: float, vz: float,
    nu: int, nv: int,
    du: float, dv: float,
    vol_origin: jnp.ndarray,
    det_center: jnp.ndarray,
    step_size: float, n_steps: int,
    n_iter: int = 3, seed: int = 0
) -> float:
    """Estimate Lipschitz constant using scan-based A^T A operator."""
    rng = np.random.default_rng(seed)
    v = jnp.asarray(rng.standard_normal(nx * ny * nz).astype(np.float32))
    v = v / (jnp.linalg.norm(v) + 1e-8)

    print(f"\n=== Estimating Lipschitz constant L (scan-based power method, {n_iter} iterations) ===")
    print(f"Processing all {len(alignment_params)} views in compiled scan")
    
    # Build compiled A^T A operator once
    AtA = build_AtA_scan(
        alignment_params=alignment_params,
        nx=nx, ny=ny, nz=nz,
        vx=vx, vy=vy, vz=vz,
        nu=nu, nv=nv,
        du=du, dv=dv,
        vol_origin=vol_origin,
        det_center=det_center,
        step_size=step_size,
        n_steps=n_steps,
    )
    
    lam = 0.0
    for k in range(n_iter):
        import time
        t_iter = time.time()
        print(f"\nPower iteration {k+1}/{n_iter}:")
        
        w = AtA(v)  # Single compiled call for A^T A v
        lam = float(jnp.vdot(v, w).real / (jnp.vdot(v, v).real + 1e-8))
        w_norm = jnp.linalg.norm(w)
        v = w / (w_norm + 1e-8)
        
        print(f"  Lambda estimate: {lam:.6e} | ||w||: {float(w_norm):.6e} | Iter time: {time.time() - t_iter:.2f}s")

    print(f"\n=== Power method complete ===")
    return max(lam, 1e-6)


def estimate_L_power_aligned(grid, det, angles: np.ndarray,
                           alignment_params: np.ndarray,
                           step_size: float, n_steps: int,
                           n_iter: int = 3, seed: int = 0) -> float:
    """Estimate Lipschitz constant using current alignment parameters."""
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    rng = np.random.default_rng(seed)
    v = jnp.asarray(rng.standard_normal(nx * ny * nz).astype(np.float32))
    v = v / (jnp.linalg.norm(v) + 1e-8)

    print(f"\n=== Estimating Lipschitz constant L (power method, {n_iter} iterations) ===")
    print(f"Note: Each power iteration requires processing all {len(angles)} projections twice (forward + adjoint)")
    print(f"Using current alignment parameters (not assuming perfect alignment)")
    
    lam = 0.0
    for k in range(n_iter):
        import time
        t_iter = time.time()
        print(f"\nPower iteration {k+1}/{n_iter}:")
        
        w = apply_AtA_vjp_aligned(v, angles, alignment_params, grid, det, 
                                step_size, n_steps, verbose=True, power_iter=k+1)
        lam = float(jnp.vdot(v, w).real / (jnp.vdot(v, v).real + 1e-8))
        w_norm = jnp.linalg.norm(w)
        v = w / (w_norm + 1e-8)
        
        print(f"  Lambda estimate: {lam:.6e} | ||w||: {float(w_norm):.6e} | Iter time: {time.time() - t_iter:.2f}s")

    print(f"\n=== Power method complete ===")
    return max(lam, 1e-6)


def optimize_alignment_params(
    projections: np.ndarray,
    angles: np.ndarray,
    recon_flat: jnp.ndarray,
    initial_params: np.ndarray,
    grid: Dict, det: Dict,
    max_iters: int = 10,
    rot_learning_rate: float = 0.001,  # Smaller for rotations (radians)
    trans_learning_rate: float = 0.01,  # Larger for translations (world units)
    verbose: bool = True
) -> Tuple[np.ndarray, List[float]]:
    """Optimize alignment parameters for all views using gradient descent."""
    
    n_proj = len(angles)
    params = jnp.asarray(initial_params.copy(), dtype=jnp.float32)
    step_size = float(grid.get("step_size", grid["vy"]))
    n_steps = int(grid.get("n_steps", math.ceil((grid["ny"] * grid["vy"]) / step_size)))
    objective_history = []
    
    # Extract parameters to avoid dictionary hashability issues
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    vx, vy, vz = float(grid["vx"]), float(grid["vy"]), float(grid["vz"])
    nu, nv = int(det["nu"]), int(det["nv"])
    du, dv = float(det["du"]), float(det["dv"])
    
    # Convert arrays to JAX arrays
    vol_origin = jnp.asarray(grid["vol_origin"], dtype=jnp.float32)
    det_center = jnp.asarray(det.get("det_center", [0.0, 0.0]), dtype=jnp.float32)
    
    # Pre-convert data to device arrays once per level to reduce host↔device churn
    projections_jnp = jax.device_put(jnp.asarray(projections, dtype=jnp.float32))
    angles_jnp = jax.device_put(jnp.asarray(angles, dtype=jnp.float32))
    
    # Compile a single-view loss+grad function once per resolution level
    # Avoid passing dicts to static_argnames - use closed-over constants instead
    def _single_view_loss(align_only, angle, measured, recon_flat):
        """Single view loss function with geometry constants closed over."""
        full_params = jnp.array(
            [align_only[0], align_only[1], angle, align_only[2], align_only[3]],
            dtype=jnp.float32,
        )
        return compute_view_loss(
            full_params,
            measured,
            recon_flat,
            nx, ny, nz,
            vx, vy, vz,
            nu, nv,
            du, dv,
            vol_origin,
            det_center,
            step_size,
            n_steps,
        )
    
    # JIT compile once for this resolution level - reuse for all views and iterations
    loss_and_grad = jax.jit(jax.value_and_grad(_single_view_loss))
    
    # Create progress bar for alignment iterations
    pbar_desc = f"Alignment ({n_proj} views)"
    pbar = tqdm(range(max_iters), desc=pbar_desc, leave=False, 
                disable=not verbose, ascii=True, ncols=80)
    
    for it in pbar:
        total_loss = 0.0
        
        # Update each view's parameters
        for i in range(n_proj):
            measured = projections_jnp[i]
            angle = angles_jnp[i]
            align_only = jnp.array(
                [params[i, 0], params[i, 1], params[i, 3], params[i, 4]],
                dtype=jnp.float32,
            )
            
            # Use pre-compiled loss+grad function
            loss_val, grad_align = loss_and_grad(align_only, angle, measured, recon_flat)
            
            # Clip gradients to prevent explosion
            grad_clip_norm = 1.0
            grad_norm = jnp.linalg.norm(grad_align)
            grad_align = jnp.where(grad_norm > grad_clip_norm, 
                                 grad_align * (grad_clip_norm / grad_norm), 
                                 grad_align)
            
            # Do a single fused update for this view to avoid 4 separate scatters
            delta = jnp.array(
                [
                    rot_learning_rate * grad_align[0],   # alpha
                    rot_learning_rate * grad_align[1],   # beta
                    0.0,                                 # phi (unchanged)
                    trans_learning_rate * grad_align[2], # dx
                    trans_learning_rate * grad_align[3], # dz
                ],
                dtype=jnp.float32,
            )
            params = params.at[i].add(-delta)
            
            total_loss += float(loss_val)
        
        objective_history.append(total_loss / n_proj)
        
        # Update progress bar with current stats
        rot_params = params[:, [0, 1]]  # alpha, beta
        trans_params = params[:, [3, 4]]  # dx, dz
        
        rot_rms = float(jnp.sqrt(jnp.mean(rot_params**2)))
        trans_rms = float(jnp.sqrt(jnp.mean(trans_params**2)))
        
        pbar.set_postfix({
            'loss': f'{total_loss/n_proj:.2e}',
            'rot': f'{np.rad2deg(rot_rms):.2f}°', 
            'trans': f'{trans_rms:.3f}'
        })
        
        # Warning if parameters are getting too large
        if rot_rms > 0.1:  # > ~6 degrees
            pbar.write(f"      WARNING: Rotation parameters getting large (RMS={np.rad2deg(rot_rms):.1f}°)")
        if trans_rms > 10.0:  # > 10 world units
            pbar.write(f"      WARNING: Translation parameters getting large (RMS={trans_rms:.1f})")
    
    pbar.close()
    
    return np.asarray(params), objective_history