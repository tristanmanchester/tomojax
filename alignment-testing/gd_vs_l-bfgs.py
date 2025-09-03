#!/usr/bin/env python3
# Compare simple Gradient Descent vs L-BFGS on a large quadratic in JAX

import time
from typing import Tuple

import jax
import jax.numpy as jnp
import optax


def make_quadratic(dim: int, cond: float = 1e4):
    """
    Build a diagonal quadratic:
      f(x) = 0.5 * sum_i d_i * (x_i - x*_i)^2
    where d_i span [1, cond].
    """
    # Diagonal spectrum increasing -> ill-conditioned
    d = jnp.logspace(0.0, jnp.log10(cond), num=dim, dtype=jnp.float32)
    x_star = jnp.ones((dim,), dtype=jnp.float32)

    def f(x: jnp.ndarray) -> jnp.ndarray:
        r = x - x_star
        return jnp.float32(0.5) * jnp.sum(d * (r * r))

    return f


def run_gd(
    x0: jnp.ndarray,
    fun,
    lr: float,
    max_iters: int,
    tol: float = 1e-6,
) -> Tuple[jnp.ndarray, float, float, int]:
    opt = optax.sgd(learning_rate=lr)
    state = opt.init(x0)

    @jax.jit
    def step(x, state):
        val, grad = jax.value_and_grad(fun)(x)
        updates, state = opt.update(grad, state, x)
        x = optax.apply_updates(x, updates)
        return x, state, val, jnp.linalg.norm(grad)

    x = x0
    val = jnp.nan
    gnorm = jnp.nan
    
    # Add convergence checking and early stopping
    for i in range(max_iters):
        x, state, val, gnorm = step(x, state)
        
        # Check for NaN (divergence)
        if jnp.isnan(val) or jnp.isnan(gnorm):
            print(f"GD diverged at iteration {i+1}")
            return x, float('nan'), float('nan'), i+1
            
        # Check convergence
        if float(gnorm) < tol:
            x, val, gnorm = jax.block_until_ready(x), float(val), float(gnorm)
            return x, val, gnorm, i+1
    
    # Materialize final result
    x, val, gnorm = jax.block_until_ready(x), float(val), float(gnorm)
    return x, val, gnorm, max_iters


def run_lbfgs(
    x0: jnp.ndarray,
    fun,
    max_iters: int,
    tol: float,
    memory_size: int = 15,
    max_linesearch_steps: int = 50,
) -> Tuple[jnp.ndarray, float, float, int]:
    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps, initial_guess_strategy="one"
    )
    opt = optax.lbfgs(
        memory_size=memory_size, scale_init_precond=True, linesearch=linesearch
    )
    val_and_grad = optax.value_and_grad_from_state(fun)

    def cond(carry):
        x, state = carry
        count = optax.tree.get(state, "count")
        grad = optax.tree.get(state, "grad")
        err = optax.tree.norm(grad)
        return jnp.logical_or(count == 0, jnp.logical_and(count < max_iters, err >= tol))

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

    x_fin, state_fin = run(x0)
    val = float(fun(x_fin))
    gnorm = float(optax.tree.norm(optax.tree.get(state_fin, "grad")))
    iters = int(optax.tree.get(state_fin, "count"))
    x_fin = jax.block_until_ready(x_fin)
    return x_fin, val, gnorm, iters


def main():
    # Use smaller problem for clearer comparison
    dim = 10000  # still large enough to be meaningful
    cond = 1e4
    f = make_quadratic(dim=dim, cond=cond)
    tol = 1e-6

    # Init
    x0 = jax.random.normal(jax.random.PRNGKey(0), (dim,), dtype=jnp.float32)
    
    # Compute initial function value and optimal value
    f0 = float(f(x0))
    f_opt = 0.0  # minimum is at x_star = ones, f(x_star) = 0
    
    print(f"Problem setup:")
    print(f"  Dimension: {dim:,}")
    print(f"  Condition number: {cond:.0e}")
    print(f"  Initial f(x0): {f0:.3e}")
    print(f"  Target tolerance: {tol:.0e}")

    # Warmup JIT
    print("\nWarming up JIT compilation...")
    val, grad = jax.jit(jax.value_and_grad(f))(x0)
    val.block_until_ready()
    grad.block_until_ready()
    
    # Calculate appropriate learning rate for GD
    # For quadratic f(x) = 0.5 * x^T A x, optimal lr ≈ 2/(λ_min + λ_max)
    # With our diagonal spectrum [1, cond], this gives lr ≈ 2/(1 + cond)
    lr_stable = 2.0 / (1.0 + cond)  # Conservative stable learning rate
    lr_aggressive = 1.0 / cond       # More aggressive but still stable
    
    print(f"\nTesting gradient descent with different learning rates...")
    
    # Test conservative GD
    print(f"\nRunning GD (conservative lr={lr_stable:.2e})...")
    t0 = time.time()
    x_gd1, val_gd1, gnorm_gd1, iters_gd1 = run_gd(
        x0, f, lr=lr_stable, max_iters=100000, tol=tol
    )
    t_gd1 = time.time() - t0
    
    # Test aggressive GD 
    print(f"Running GD (aggressive lr={lr_aggressive:.2e})...")
    t0 = time.time()
    x_gd2, val_gd2, gnorm_gd2, iters_gd2 = run_gd(
        x0, f, lr=lr_aggressive, max_iters=100000, tol=tol
    )
    t_gd2 = time.time() - t0

    # Run L-BFGS
    print(f"Running L-BFGS...")
    t0 = time.time()
    x_lb, val_lb, gnorm_lb, iters_lb = run_lbfgs(
        x0, f, max_iters=2000, tol=tol, memory_size=15, max_linesearch_steps=50
    )
    t_lb = time.time() - t0

    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Method':<20} {'Iters':<8} {'Time':<8} {'Final f(x)':<12} {'||grad||':<12} {'Status':<15}")
    print("-"*80)
    
    # Check convergence status
    def status(gnorm, tol):
        if gnorm != gnorm:  # NaN check
            return "DIVERGED"
        elif gnorm < tol:
            return "CONVERGED"
        else:
            return "MAX_ITERS"
    
    print(f"{'GD (conservative)':<20} {iters_gd1:<8d} {t_gd1:<8.2f} {val_gd1:<12.3e} {gnorm_gd1:<12.3e} {status(gnorm_gd1, tol):<15}")
    print(f"{'GD (aggressive)':<20} {iters_gd2:<8d} {t_gd2:<8.2f} {val_gd2:<12.3e} {gnorm_gd2:<12.3e} {status(gnorm_gd2, tol):<15}")
    print(f"{'L-BFGS':<20} {iters_lb:<8d} {t_lb:<8.2f} {val_lb:<12.3e} {gnorm_lb:<12.3e} {status(gnorm_lb, tol):<15}")
    
    # Show efficiency metrics
    print("\nEfficiency Analysis:")
    converged_methods = []
    if status(gnorm_gd1, tol) == "CONVERGED":
        converged_methods.append(("GD (conservative)", iters_gd1, t_gd1))
    if status(gnorm_gd2, tol) == "CONVERGED":
        converged_methods.append(("GD (aggressive)", iters_gd2, t_gd2))
    if status(gnorm_lb, tol) == "CONVERGED":
        converged_methods.append(("L-BFGS", iters_lb, t_lb))
    
    if converged_methods:
        fastest = min(converged_methods, key=lambda x: x[2])
        fewest_iters = min(converged_methods, key=lambda x: x[1])
        print(f"  Fastest method: {fastest[0]} ({fastest[2]:.2f}s)")
        print(f"  Fewest iterations: {fewest_iters[0]} ({fewest_iters[1]} iters)")
    else:
        print(f"  No methods converged to tolerance {tol:.0e}")


if __name__ == "__main__":
    main()