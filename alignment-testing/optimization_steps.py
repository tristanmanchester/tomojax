#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization steps for joint reconstruction and alignment.
Clean implementation with FISTA-TV reconstruction and alignment optimization.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple, Optional
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from projector_parallel_jax import forward_project_view


# ============================================================================
# CORE LOSS AND GRADIENT COMPUTATION
# ============================================================================

def build_aligned_loss_and_grad_scan(
    projections: np.ndarray,  # (n_views, nv, nu)
    alignment_params: np.ndarray,  # (n_views, 5) [alpha, beta, phi, dx, dz]
    nx: int, ny: int, nz: int,
    vx: float, vy: float, vz: float,
    nu: int, nv: int,
    du: float, dv: float,
    vol_origin: jnp.ndarray,
    det_center: jnp.ndarray,
    step_size: float,
    n_steps: int,
):
    """
    Build compiled loss and gradient function using scan over views.
    Returns a function loss_and_grad(x_flat) -> (fval, grad).
    """
    y_stack = jnp.asarray(projections, dtype=jnp.float32)
    params_stack = jnp.asarray(alignment_params, dtype=jnp.float32)
    y_flat = y_stack.reshape(y_stack.shape[0], -1)  # (n, nv*nu)

    def loss_fn(x_flat: jnp.ndarray) -> jnp.ndarray:
        def body(f_acc, inputs):
            p_i, y_i = inputs
            pred_i = forward_project_view(
                params=p_i,
                recon_flat=x_flat,
                nx=nx, ny=ny, nz=nz,
                vx=jnp.float32(vx),
                vy=jnp.float32(vy),
                vz=jnp.float32(vz),
                nu=nu, nv=nv,
                du=jnp.float32(du),
                dv=jnp.float32(dv),
                vol_origin=vol_origin,
                det_center=det_center,
                step_size=jnp.float32(step_size),
                n_steps=int(n_steps),
                use_checkpoint=True,
                stop_grad_recon=False,
            )
            r = pred_i.reshape(-1) - y_i
            l_i = 0.5 * jnp.vdot(r, r).real
            return f_acc + l_i, None

        f_sum, _ = jax.lax.scan(body, jnp.float32(0.0), (params_stack, y_flat))
        return f_sum

    return jax.jit(jax.value_and_grad(loss_fn))


def build_AtA_scan(
    alignment_params: np.ndarray,
    nx: int, ny: int, nz: int,
    vx: float, vy: float, vz: float,
    nu: int, nv: int,
    du: float, dv: float,
    vol_origin: jnp.ndarray,
    det_center: jnp.ndarray,
    step_size: float, n_steps: int,
):
    """Build A^T A operator using scan over views."""
    zeros = jnp.zeros((alignment_params.shape[0], nv, nu), dtype=jnp.float32)
    
    loss_and_grad = build_aligned_loss_and_grad_scan(
        projections=zeros,
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

    @jax.jit
    def AtA(x_flat: jnp.ndarray) -> jnp.ndarray:
        _, g = loss_and_grad(x_flat)
        return g

    return AtA


# ============================================================================
# TV PROXIMAL OPERATOR
# ============================================================================

@jax.jit
def grad3(u: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """3D gradient with zero-padding (forward differences)."""
    gx = jnp.pad(u[1:, :, :] - u[:-1, :, :], ((0, 1), (0, 0), (0, 0)))
    gy = jnp.pad(u[:, 1:, :] - u[:, :-1, :], ((0, 0), (0, 1), (0, 0)))
    gz = jnp.pad(u[:, :, 1:] - u[:, :, :-1], ((0, 0), (0, 0), (0, 1)))
    return gx, gy, gz


@jax.jit
def div3(px: jnp.ndarray, py: jnp.ndarray, pz: jnp.ndarray) -> jnp.ndarray:
    """3D divergence (backward differences)."""
    dx = jnp.concatenate([px[:1, :, :], px[1:, :, :] - px[:-1, :, :]], axis=0)
    dy = jnp.concatenate([py[:, :1, :], py[:, 1:, :] - py[:, :-1, :]], axis=1)
    dz = jnp.concatenate([pz[:, :, :1], pz[:, :, 1:] - pz[:, :, :-1]], axis=2)
    return dx + dy + dz


def prox_tv_chambolle_pock(y: jnp.ndarray, lambda_tv: float, n_iters: int = 20) -> jnp.ndarray:
    """
    Chambolle-Pock TV proximal operator.
    Solves: argmin_x { 0.5 ||x - y||^2 + lambda_tv * TV(x) }
    """
    tau = jnp.float32(0.25)
    sigma = jnp.float32(1.0 / 3.0)
    theta = jnp.float32(1.0)
    
    x = y
    x_bar = x
    px = jnp.zeros_like(y)
    py = jnp.zeros_like(y)
    pz = jnp.zeros_like(y)
    
    for _ in range(n_iters):
        # Dual update
        gx, gy, gz = grad3(x_bar)
        px_new = px + sigma * gx
        py_new = py + sigma * gy
        pz_new = pz + sigma * gz
        
        # Project dual variables onto unit ball
        norm = jnp.maximum(1.0, jnp.sqrt(px_new**2 + py_new**2 + pz_new**2))
        px = px_new / norm
        py = py_new / norm
        pz = pz_new / norm
        
        # Primal update
        div_p = div3(px, py, pz)
        x_new = (x + tau * div_p + (tau / jnp.float32(lambda_tv)) * y) / (1.0 + tau / jnp.float32(lambda_tv))
        
        # Over-relaxation
        x_bar = x_new + theta * (x_new - x)
        x = x_new
    
    return x


# ============================================================================
# LIPSCHITZ CONSTANT ESTIMATION
# ============================================================================

def estimate_lipschitz_scan(
    alignment_params: np.ndarray,
    nx: int, ny: int, nz: int,
    vx: float, vy: float, vz: float,
    nu: int, nv: int,
    du: float, dv: float,
    vol_origin: jnp.ndarray,
    det_center: jnp.ndarray,
    step_size: float, n_steps: int,
    n_iter: int = 3, seed: int = 42
) -> float:
    """Estimate Lipschitz constant using power method with scan-based AtA."""
    rng = np.random.default_rng(seed)
    v = jnp.asarray(rng.standard_normal(nx * ny * nz).astype(np.float32))
    v = v / (jnp.linalg.norm(v) + 1e-8)
    
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
        w = AtA(v)
        lam = float(jnp.vdot(v, w).real / (jnp.vdot(v, v).real + 1e-8))
        w_norm = jnp.linalg.norm(w)
        v = w / (w_norm + 1e-8)
    
    return max(lam, 1e-6)


# ============================================================================
# FISTA-TV RECONSTRUCTION
# ============================================================================

def fista_tv_reconstruction(
    projections: np.ndarray,
    angles: np.ndarray,
    initial_recon: jnp.ndarray,
    grid: Dict, det: Dict,
    lambda_tv: float = 0.005,
    max_iters: int = 20,
    tv_iters: int = 20,
    step_scale: float = 0.9,
    power_iters: int = 5,
    verbose: bool = True,
    alignment_params: Optional[np.ndarray] = None,
    precomputed_L: Optional[float] = None
) -> Tuple[jnp.ndarray, List[float], float]:
    """
    Run FISTA-TV reconstruction for given projections and alignment.
    
    Args:
        projections: (n_views, nv, nu) measured projections
        angles: (n_views,) projection angles
        initial_recon: Initial reconstruction volume
        grid, det: Geometry dictionaries
        lambda_tv: TV regularization weight
        max_iters: FISTA iterations
        tv_iters: TV proximal iterations
        step_scale: Step size scaling factor
        power_iters: Power iterations for Lipschitz estimation
        verbose: Print progress
        alignment_params: Optional (n_views, 5) alignment parameters
        precomputed_L: Optional precomputed Lipschitz constant
        
    Returns:
        Final reconstruction, objective history, Lipschitz constant
    """
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    step_size = float(grid.get("step_size", grid["vy"]))
    n_steps = int(grid.get("n_steps", math.ceil((ny * grid["vy"]) / step_size)))
    
    if alignment_params is not None:
        # Use alignment-aware functions
        vx, vy, vz = float(grid["vx"]), float(grid["vy"]), float(grid["vz"])
        nu, nv = int(det["nu"]), int(det["nv"])
        du, dv = float(det["du"]), float(det["dv"])
        vol_origin = jnp.asarray(grid["vol_origin"], dtype=jnp.float32)
        det_center = jnp.asarray(det.get("det_center", [0.0, 0.0]), dtype=jnp.float32)
        
        # Estimate or use provided Lipschitz constant
        if precomputed_L is not None:
            L = precomputed_L
            if verbose:
                print(f"  Using precomputed Lipschitz constant L = {L:.3e}")
        else:
            if verbose:
                print(f"  Estimating Lipschitz constant ({power_iters} iterations)...")
            L = estimate_lipschitz_scan(
                alignment_params, nx, ny, nz, vx, vy, vz,
                nu, nv, du, dv, vol_origin, det_center,
                step_size, n_steps, n_iter=power_iters, seed=42
            )
            if verbose:
                print(f"  L = {L:.3e}")
        
        alpha = step_scale / L
        
        # Build loss and gradient function
        if verbose:
            print(f"  Compiling loss and gradient function for {len(alignment_params)} views...")
        loss_and_grad = build_aligned_loss_and_grad_scan(
            projections=projections,
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
        
        # FISTA iterations
        xk = initial_recon.ravel()
        xk_prev = xk
        tk = 1.0
        objective_history = []
        
        for it in range(1, max_iters + 1):
            # Momentum step
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
            beta = jnp.float32((tk - 1.0) / t_next)
            zk = xk + beta * (xk - xk_prev)
            
            # Gradient step
            fval, grad = loss_and_grad(zk)
            yk = zk - jnp.float32(alpha) * grad
            
            # TV proximal step
            x_next = prox_tv_chambolle_pock(
                yk.reshape(nx, ny, nz), lambda_tv=lambda_tv, n_iters=tv_iters
            ).ravel()
            
            # Update
            xk_prev = xk
            xk = x_next
            tk = t_next
            
            objective_history.append(float(fval))
            
            if verbose and (it % max(1, max_iters // 10) == 0 or it == max_iters):
                grad_norm = float(jnp.linalg.norm(grad))
                print(f"    FISTA iter {it:3d}/{max_iters} | f={fval:.6e} | ||grad||={grad_norm:.3e}")
    else:
        raise ValueError("alignment_params must be provided")
    
    return xk.reshape(nx, ny, nz), objective_history, L


# ============================================================================
# ALIGNMENT OPTIMIZATION
# ============================================================================

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
    """Compute loss for a single view."""
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
    resid = (pred - measured).astype(jnp.float32)
    return 0.5 * jnp.vdot(resid, resid).real


def optimize_alignment_params(
    projections: np.ndarray,
    angles: np.ndarray,
    recon_flat: jnp.ndarray,
    initial_params: np.ndarray,
    grid: Dict, det: Dict,
    optimizer: str = "adabelief",
    max_iters: int = 100,
    learning_rate: float = 0.01,
    optimize_phi: bool = False,
    rot_scale: float = 0.1,
    trans_scale: float = 1.0,
    verbose: bool = True
) -> Tuple[np.ndarray, List[float]]:
    """
    Optimize alignment parameters using specified optimizer.
    
    Args:
        projections: (n_proj, nv, nu) measured projections
        angles: (n_proj,) projection angles
        recon_flat: Flattened reconstruction volume
        initial_params: (n_proj, 5) initial alignment parameters
        grid, det: Geometry dictionaries
        optimizer: Optimizer name (adabelief, adam, nadam, lbfgs)
        max_iters: Maximum iterations
        learning_rate: Base learning rate
        optimize_phi: Whether to optimize phi angle
        rot_scale: Scale for rotation parameters
        trans_scale: Scale for translation parameters
        verbose: Print progress
        
    Returns:
        Optimized parameters and loss history
    """
    n_proj = len(angles)
    step_size = float(grid.get("step_size", grid["vy"]))
    n_steps = int(grid.get("n_steps", math.ceil((grid["ny"] * grid["vy"]) / step_size)))
    
    # Extract parameters
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    vx, vy, vz = float(grid["vx"]), float(grid["vy"]), float(grid["vz"])
    nu, nv = int(det["nu"]), int(det["nv"])
    du, dv = float(det["du"]), float(det["dv"])
    
    vol_origin = jnp.asarray(grid["vol_origin"], dtype=jnp.float32)
    det_center = jnp.asarray(det.get("det_center", [0.0, 0.0]), dtype=jnp.float32)
    
    # Convert to device arrays
    projections_jnp = jax.device_put(jnp.asarray(projections, dtype=jnp.float32))
    angles_jnp = jax.device_put(jnp.asarray(angles, dtype=jnp.float32))
    
    params = jnp.asarray(initial_params.copy(), dtype=jnp.float32)
    params_flat = params.ravel()
    
    # Create loss function
    @jax.checkpoint
    def compute_loss(p_flat):
        n_views = projections_jnp.shape[0]
        params = p_flat.reshape(n_views, 5)
        
        if not optimize_phi:
            params = params.at[:, 2].set(angles_jnp)
        
        @jax.checkpoint
        def single_view_loss(i):
            return compute_view_loss(
                params[i], projections_jnp[i], recon_flat,
                nx, ny, nz, vx, vy, vz, nu, nv, du, dv,
                vol_origin, det_center, step_size, n_steps
            )
        
        losses = jax.vmap(single_view_loss)(jnp.arange(n_views))
        return jnp.mean(losses)
    
    # Handle L-BFGS separately (different update mechanism)
    if optimizer == "lbfgs":
        opt = optax.lbfgs(
            learning_rate=None,
            memory_size=20,
            scale_init_precond=True,
            linesearch=optax.scale_by_zoom_linesearch(
                max_linesearch_steps=20,
                initial_guess_strategy="one"
            )
        )
        opt_state = opt.init(params_flat)
        
        # L-BFGS requires special handling
        try:
            val_and_grad = optax.value_and_grad_from_state(compute_loss)
        except AttributeError:
            val_and_grad_fn = jax.value_and_grad(compute_loss)
            def val_and_grad(params, state=None):
                return val_and_grad_fn(params)
        
        @jax.jit
        def lbfgs_step(params_flat, state):
            value, grad = val_and_grad(params_flat, state=state)
            updates, state = opt.update(
                grad, state, params_flat,
                value=value, grad=grad, value_fn=compute_loss
            )
            params_flat = optax.apply_updates(params_flat, updates)
            return params_flat, state, value
        
        if verbose:
            print(f"  Running L-BFGS optimization (max_iters={max_iters})...")
        
        objective_history = []
        for it in range(max_iters):
            params_flat, opt_state, loss = lbfgs_step(params_flat, opt_state)
            objective_history.append(float(loss))
            
            if verbose and (it % max(1, max_iters // 10) == 0 or it == max_iters - 1):
                print(f"    L-BFGS iter {it+1}/{max_iters} | loss={loss:.3e}")
        
        return np.asarray(params_flat.reshape(n_proj, 5)), objective_history
    
    # Standard optimizers (Adam, AdaBelief, etc.)
    if optimizer == "adabelief":
        opt = optax.adabelief(learning_rate, b1=0.9, b2=0.999, eps=1e-16, eps_root=1e-16)
    elif optimizer == "adam":
        opt = optax.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8)
    elif optimizer == "nadam":
        opt = optax.nadam(learning_rate, b1=0.9, b2=0.999, eps=1e-8)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    opt_state = opt.init(params_flat)
    objective_history = []
    
    @jax.jit
    def step(params_flat, opt_state):
        loss, grads = jax.value_and_grad(compute_loss)(params_flat)
        
        # Apply scaling
        grads_reshaped = grads.reshape(n_proj, 5)
        grads_reshaped = grads_reshaped.at[:, :3].multiply(rot_scale)
        grads_reshaped = grads_reshaped.at[:, 3:].multiply(trans_scale)
        
        if not optimize_phi:
            grads_reshaped = grads_reshaped.at[:, 2].set(0.0)
        
        grads = grads_reshaped.ravel()
        updates, opt_state = opt.update(grads, opt_state, params_flat)
        params_flat = optax.apply_updates(params_flat, updates)
        return params_flat, opt_state, loss
    
    if verbose:
        print(f"  Running {optimizer.upper()} optimization (max_iters={max_iters})...")
    
    for it in range(max_iters):
        params_flat, opt_state, loss = step(params_flat, opt_state)
        objective_history.append(float(loss))
        
        if verbose and (it % max(1, max_iters // 10) == 0 or it == max_iters - 1):
            print(f"    {optimizer.upper()} iter {it+1}/{max_iters} | loss={loss:.3e}")
        
        # Early stopping
        if it > 10 and len(objective_history) > 5:
            recent_change = abs(objective_history[-1] - objective_history[-5]) / abs(objective_history[-5])
            if recent_change < 1e-6:
                if verbose:
                    print(f"    Converged after {it+1} iterations")
                break
    
    return np.asarray(params_flat.reshape(n_proj, 5)), objective_history


def optimize_alignment_hybrid(
    projections: np.ndarray,
    angles: np.ndarray,
    recon_flat: jnp.ndarray,
    initial_params: np.ndarray,
    grid: Dict,
    det: Dict,
    optimize_phi: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, List[float]]:
    """
    Two-stage hybrid optimization:
    1. AdaBelief for fast convergence (300 iters)
    2. NAdam for refinement (100 iters)
    """
    if verbose:
        print("  Stage 1: AdaBelief optimization...")
    
    params, history1 = optimize_alignment_params(
        projections, angles, recon_flat, initial_params,
        grid, det,
        optimizer="adabelief",
        max_iters=300,
        learning_rate=0.01,
        optimize_phi=optimize_phi,
        rot_scale=0.1,
        trans_scale=1.0,
        verbose=verbose
    )
    
    if verbose:
        print("  Stage 2: NAdam refinement...")
    
    params, history2 = optimize_alignment_params(
        projections, angles, recon_flat, params,
        grid, det,
        optimizer="nadam",
        max_iters=100,
        learning_rate=0.005,
        optimize_phi=optimize_phi,
        rot_scale=0.1,
        trans_scale=1.0,
        verbose=verbose
    )
    
    total_history = history1 + history2
    
    if verbose:
        final_loss = total_history[-1] if total_history else float('nan')
        print(f"  Hybrid optimization complete: final loss = {final_loss:.3e}")
    
    return params, total_history