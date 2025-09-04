#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive optimizer comparison for multi-scale problems
Relevant to tomography alignment: rotations (small) + translations (large)
"""

import time
from typing import Tuple, Dict, List
import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial
import matplotlib.pyplot as plt
from dataclasses import dataclass
from jax.scipy.optimize import minimize as jax_minimize


@dataclass
class OptimizerResult:
    name: str
    x_final: jnp.ndarray
    f_final: float
    grad_norm: float
    iterations: int
    time: float
    history: List[float]
    converged: bool


def make_multiscale_tomography_problem(n_views: int = 50, noise_level: float = 0.1):
    """
    Create a problem that mimics tomography alignment optimization:
    - 5 parameters per view: [alpha, beta, phi, dx, dz]
    - Rotations (alpha, beta, phi) are in radians (small scale)
    - Translations (dx, dz) are in pixels (large scale)
    - Coupled objective with different sensitivities
    """
    dim = n_views * 5
    
    # Ground truth parameters
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)
    
    # Rotations: small values in radians (typically < 0.1 rad ~ 6 degrees)
    rot_truth = jax.random.uniform(keys[0], (n_views, 3), minval=-0.05, maxval=0.05)
    
    # Translations: larger values in pixels (typically < 10 pixels)
    trans_truth = jax.random.uniform(keys[1], (n_views, 2), minval=-5.0, maxval=5.0)
    
    x_star = jnp.concatenate([
        rot_truth.ravel(), 
        trans_truth.ravel()
    ])
    
    # Create scaling to mimic different parameter sensitivities
    # Rotations are more sensitive (small changes = big effect)
    # Translations less sensitive (need larger changes)
    rot_scale = jnp.ones(n_views * 3) * 100.0  # High sensitivity
    trans_scale = jnp.ones(n_views * 2) * 1.0   # Lower sensitivity
    scale = jnp.concatenate([rot_scale, trans_scale])
    
    # Add coupling matrix to simulate view interactions
    # (like how alignment of one view affects others in reconstruction)
    coupling = jax.random.normal(keys[2], (dim, dim)) * 0.01
    coupling = coupling + coupling.T  # Symmetric
    coupling = coupling + jnp.diag(jnp.ones(dim))  # Ensure positive definite
    
    def f(x_flat: jnp.ndarray) -> jnp.ndarray:
        """Coupled multi-scale objective function."""
        # Reshape to separate rotations and translations
        n_rot = n_views * 3
        x_rot = x_flat[:n_rot]
        x_trans = x_flat[n_rot:]
        x = jnp.concatenate([x_rot, x_trans])
        
        # Apply scaling
        dx_scaled = scale * (x - x_star)
        
        # Coupled quadratic term
        quad_term = 0.5 * jnp.dot(dx_scaled, jnp.dot(coupling, dx_scaled))
        
        # Add non-convex term for rotations (simulating projection nonlinearity)
        rot_penalty = noise_level * jnp.sum(jnp.cos(x_rot * 10))
        
        return quad_term + rot_penalty
    
    return f, x_star, dim


def run_gradient_descent(
    x0: jnp.ndarray,
    fun,
    learning_rate: float = 0.001,
    max_iters: int = 10000,
    tol: float = 1.5e01,
    track_history: bool = True
) -> OptimizerResult:
    """Run standard gradient descent."""
    
    @jax.jit
    def step(x):
        val, grad = jax.value_and_grad(fun)(x)
        x_new = x - learning_rate * grad
        return x_new, val, grad
    
    x = x0
    history = []
    t_start = time.time()
    
    for i in range(max_iters):
        x, val, grad = step(x)
        grad_norm = jnp.linalg.norm(grad)
        
        if track_history:
            history.append(float(val))
        
        # Check convergence
        if grad_norm < tol:
            break
        
        # Check for NaN
        if jnp.isnan(val) or jnp.isnan(grad_norm):
            break
    
    t_elapsed = time.time() - t_start
    
    # Final values
    x_final = jax.block_until_ready(x)
    f_final = float(fun(x_final))
    grad_final = jax.grad(fun)(x_final)
    grad_norm_final = float(jnp.linalg.norm(grad_final))
    
    return OptimizerResult(
        name='gd',
        x_final=x_final,
        f_final=f_final,
        grad_norm=grad_norm_final,
        iterations=i + 1,
        time=t_elapsed,
        history=history,
        converged=(grad_norm_final < tol)
    )


def run_jax_bfgs(
    x0: jnp.ndarray,
    fun,
    max_iters: int = 10000,
    tol: float = 1.5e01,
    track_history: bool = True
) -> OptimizerResult:
    """Run JAX's scipy.optimize.minimize with BFGS."""
    
    t_start = time.time()
    
    # JAX minimize expects 1D arrays and will auto-diff the function
    result = jax_minimize(
        fun=fun,
        x0=x0,
        method='BFGS',
        tol=tol,
        options={'maxiter': max_iters}
    )
    
    t_elapsed = time.time() - t_start
    
    # Extract results
    x_final = jax.block_until_ready(result.x)
    f_final = float(result.fun)
    
    # Compute gradient at final point since it's not in the result object
    grad_final = jax.grad(fun)(x_final)
    grad_norm_final = float(jnp.linalg.norm(grad_final))
    
    # For history, we can't easily track it with jax.scipy.optimize.minimize
    history = [f_final] if track_history else []
    
    return OptimizerResult(
        name='jax_bfgs',
        x_final=x_final,
        f_final=f_final,
        grad_norm=grad_norm_final,
        iterations=int(result.nit),
        time=t_elapsed,
        history=history,
        converged=result.success
    )


def create_optimizer(name: str, learning_rate: float = 0.01, 
                     n_views: int = 50, use_scaling: bool = True):
    """Create various Optax optimizers with appropriate configurations."""
    
    # Simpler approach - just use the base optimizer without complex masking
    optimizers = {
        'sgd': lambda: optax.sgd(learning_rate * 0.001),  # Much smaller LR
        'sgd_momentum': lambda: optax.sgd(learning_rate * 0.001, momentum=0.9),
        'sgd_nesterov': lambda: optax.sgd(learning_rate * 0.001, momentum=0.9, nesterov=True),
        
        'adam': lambda: optax.adam(learning_rate),
        'adamw': lambda: optax.adamw(learning_rate, weight_decay=1e-4),
        'nadam': lambda: optax.nadam(learning_rate),
        'radam': lambda: optax.radam(learning_rate),
        'amsgrad': lambda: optax.amsgrad(learning_rate),
        'adamax': lambda: optax.adamax(learning_rate),
        'adabelief': lambda: optax.adabelief(learning_rate),
        
        'lbfgs': lambda: optax.lbfgs(
            learning_rate=1.0,
            memory_size=20,
            scale_init_precond=True,
            linesearch=optax.scale_by_zoom_linesearch(
                max_linesearch_steps=20,
                initial_guess_strategy="one"
            )
        ),
        
        'fromage': lambda: optax.fromage(learning_rate * 0.1, min_norm=1e-6),
        'adagrad': lambda: optax.adagrad(learning_rate),
        'rmsprop': lambda: optax.rmsprop(learning_rate, decay=0.9),
        'lion': lambda: optax.lion(learning_rate * 0.01, b1=0.9, b2=0.99),
        'yogi': lambda: optax.yogi(learning_rate),
        'adadelta': lambda: optax.adadelta(learning_rate),
        'adafactor': lambda: optax.adafactor(learning_rate),
        'lamb': lambda: optax.lamb(learning_rate),
        'lars': lambda: optax.lars(learning_rate * 0.01, weight_decay=1e-4),
        'novograd': lambda: optax.novograd(learning_rate),
    }
    
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizers[name]()


def run_optimizer(
    opt_name: str,
    x0: jnp.ndarray,
    fun,
    max_iters: int = 1000,
    tol: float = 1e-6,
    n_views: int = 50,
    track_history: bool = True,
    verbose: bool = False
) -> OptimizerResult:
    """Run a single optimizer and track results."""
    
    # Special handling for different optimizer types
    if opt_name == 'lbfgs':
        return run_lbfgs_special(opt_name, x0, fun, max_iters, tol, track_history)
    elif opt_name == 'jax_bfgs':
        return run_jax_bfgs(x0, fun, max_iters, tol, track_history)
    elif opt_name == 'gd':
        return run_gradient_descent(x0, fun, learning_rate=0.0001, 
                                   max_iters=max_iters, tol=tol, 
                                   track_history=track_history)
    
    # Standard Optax optimizers
    opt = create_optimizer(opt_name, learning_rate=0.01, n_views=n_views)
    opt_state = opt.init(x0)
    
    # JIT compile the step function
    @jax.jit
    def step(x, opt_state):
        val, grad = jax.value_and_grad(fun)(x)
        updates, opt_state = opt.update(grad, opt_state, x)
        x = optax.apply_updates(x, updates)
        return x, opt_state, val, grad
    
    # Optimization loop
    x = x0
    history = []
    t_start = time.time()
    
    for i in range(max_iters):
        x, opt_state, val, grad = step(x, opt_state)
        grad_norm = jnp.linalg.norm(grad)
        
        if track_history:
            history.append(float(val))
        
        # Check convergence
        if grad_norm < tol:
            break
        
        # Check for NaN
        if jnp.isnan(val) or jnp.isnan(grad_norm):
            if verbose:
                print(f"  {opt_name}: Diverged at iteration {i+1}")
            break
    
    t_elapsed = time.time() - t_start
    
    # Final values
    x_final = jax.block_until_ready(x)
    f_final = float(fun(x_final))
    grad_final = jax.grad(fun)(x_final)
    grad_norm_final = float(jnp.linalg.norm(grad_final))
    
    return OptimizerResult(
        name=opt_name,
        x_final=x_final,
        f_final=f_final,
        grad_norm=grad_norm_final,
        iterations=i + 1,
        time=t_elapsed,
        history=history,
        converged=(grad_norm_final < tol)
    )


def run_lbfgs_special(
    opt_name: str,
    x0: jnp.ndarray,
    fun,
    max_iters: int,
    tol: float,
    track_history: bool
) -> OptimizerResult:
    """Special handling for L-BFGS with while_loop."""
    
    opt = create_optimizer('lbfgs')
    val_and_grad = optax.value_and_grad_from_state(fun)
    
    history = []
    
    def cond(carry):
        x, state = carry
        count = optax.tree.get(state, "count")
        grad = optax.tree.get(state, "grad")
        err = optax.tree.norm(grad)
        return jnp.logical_or(count == 0, 
                             jnp.logical_and(count < max_iters, err >= tol))
    
    def body(carry):
        x, state = carry
        val, grad = val_and_grad(x, state=state)
        updates, state = opt.update(grad, state, x, value=val, grad=grad, value_fn=fun)
        x = optax.apply_updates(x, updates)
        return x, state
    
    @jax.jit
    def run(x0):
        state0 = opt.init(x0)
        x_fin, state_fin = jax.lax.while_loop(cond, body, (x0, state0))
        return x_fin, state_fin
    
    t_start = time.time()
    x_fin, state_fin = run(x0)
    t_elapsed = time.time() - t_start
    
    val_fin = float(fun(x_fin))
    grad_fin = optax.tree.get(state_fin, "grad")
    gnorm_fin = float(optax.tree.norm(grad_fin))
    iters = int(optax.tree.get(state_fin, "count"))
    
    # Can't easily track history with while_loop
    if track_history:
        history = [val_fin]
    
    return OptimizerResult(
        name=opt_name,
        x_final=jax.block_until_ready(x_fin),
        f_final=val_fin,
        grad_norm=gnorm_fin,
        iterations=iters,
        time=t_elapsed,
        history=history,
        converged=(gnorm_fin < tol)
    )

# ... [plot_results function remains the same] ...

def main():
    # Problem setup - mimicking tomography alignment
    n_views = 100  # Number of projection views
    print(f"Setting up multi-scale tomography-like problem...")
    print(f"  Views: {n_views}")
    print(f"  Parameters per view: 5 (3 rotations + 2 translations)")
    print(f"  Total dimension: {n_views * 5}")
    print(f"  Rotations: Small scale (radians)")
    print(f"  Translations: Large scale (pixels)")
    
    # Create problem
    f, x_star, dim = make_multiscale_tomography_problem(n_views, noise_level=0.05)
    
    # Initial point
    key = jax.random.PRNGKey(123)
    x0 = jax.random.normal(key, (dim,), dtype=jnp.float32) * 0.5
    
    # Evaluate initial point
    f0 = float(f(x0))
    grad0 = jax.grad(f)(x0)
    gnorm0 = float(jnp.linalg.norm(grad0))
    
    print(f"\nInitial point:")
    print(f"  f(x0) = {f0:.3e}")
    print(f"  ||grad f(x0)|| = {gnorm0:.3e}")
    
    # Optimizers to test - now including GD and JAX BFGS
    optimizer_names = [
        # Basic methods
        'gd',            # Standard gradient descent
        'jax_bfgs',      # Full BFGS from jax.scipy.optimize
        
        # Essential ones for tomography
        'adam',
        'lbfgs',         # L-BFGS from Optax
        'fromage',       # Good for multi-scale
        
        # Variations of Adam
        'nadam',
        'radam',
        'adamw',
        'amsgrad',
        'adabelief',
        
        # Classic methods
        'sgd',
        'sgd_momentum',
        'sgd_nesterov',
        
        # Adaptive methods
        'rmsprop',
        'adagrad',
        
        # Newer/experimental
        'lion',
        'yogi',
        
        # Heavy-duty (might be slow)
        'lamb',
        'adafactor',
    ]
    
    # Run all optimizers
    print(f"\nTesting {len(optimizer_names)} optimizers...")
    print("="*80)
    
    results = []
    max_iters = 10000
    tol = 1.5e01
    
    for opt_name in optimizer_names:
        print(f"Running {opt_name:15s} ", end='', flush=True)
        try:
            result = run_optimizer(
                opt_name, x0, f, 
                max_iters=max_iters, 
                tol=tol, 
                n_views=n_views,
                track_history=True,
                verbose=False
            )
            results.append(result)
            
            status = "✓ CONVERGED" if result.converged else "✗ MAX_ITERS"
            print(f" {status:12s} | iters: {result.iterations:5d} | "
                  f"time: {result.time:6.2f}s | f: {result.f_final:10.3e} | "
                  f"||g||: {result.grad_norm:.3e}")
        except Exception as e:
            print(f" ✗ FAILED: {str(e)[:50]}")
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY RESULTS")
    print("="*100)
    print(f"{'Optimizer':<15} {'Status':<12} {'Iterations':<12} {'Time (s)':<10} "
          f"{'Final f(x)':<15} {'||grad||':<12}")
    print("-"*100)
    
    # Sort by final function value
    results_sorted = sorted(results, key=lambda r: r.f_final)
    
    for r in results_sorted:
        status = "CONVERGED" if r.converged else "MAX_ITERS"
        print(f"{r.name:<15} {status:<12} {r.iterations:<12d} {r.time:<10.2f} "
              f"{r.f_final:<15.3e} {r.grad_norm:<12.3e}")
    
    # Analysis section with special attention to BFGS comparison
    print("\n" + "="*100)
    print("BFGS COMPARISON")
    print("="*100)
    
    jax_bfgs = next((r for r in results if r.name == 'jax_bfgs'), None)
    lbfgs = next((r for r in results if r.name == 'lbfgs'), None)
    
    if jax_bfgs and lbfgs:
        print(f"{'Method':<15} {'Iterations':<12} {'Time (s)':<10} {'Final f(x)':<15}")
        print(f"{'JAX BFGS (full)':<15} {jax_bfgs.iterations:<12d} {jax_bfgs.time:<10.2f} {jax_bfgs.f_final:<15.3e}")
        print(f"{'L-BFGS (m=20)':<15} {lbfgs.iterations:<12d} {lbfgs.time:<10.2f} {lbfgs.f_final:<15.3e}")
        
        if jax_bfgs.iterations < lbfgs.iterations:
            print(f"\n✓ Full BFGS converged {lbfgs.iterations - jax_bfgs.iterations} iterations faster!")
        elif lbfgs.iterations < jax_bfgs.iterations:
            print(f"\n✓ L-BFGS converged {jax_bfgs.iterations - lbfgs.iterations} iterations faster!")
    
    # Rest of analysis...
    print("\n" + "="*100)
    print("ANALYSIS FOR TOMOGRAPHY ALIGNMENT")
    print("="*100)
    
    converged = [r for r in results if r.converged]
    if converged:
        fastest = min(converged, key=lambda r: r.time)
        fewest = min(converged, key=lambda r: r.iterations)
        best = min(converged, key=lambda r: r.f_final)
        
        print(f"✓ Converged: {len(converged)}/{len(results)}")
        print(f"⚡ Fastest: {fastest.name} ({fastest.time:.2f}s)")
        print(f"📊 Fewest iterations: {fewest.name} ({fewest.iterations} iters)")
        print(f"🎯 Best objective: {best.name} (f={best.f_final:.3e})")
    
    return results


if __name__ == "__main__":
    results = main()