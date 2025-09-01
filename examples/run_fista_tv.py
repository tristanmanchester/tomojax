#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FISTA-TV reconstruction using JAX projector on saved projections

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Tuple
import os

import jax
import jax.numpy as jnp
import numpy as np
import psutil
import tifffile as tiff

# Add parent directory to path to import projector_parallel_jax
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from projector_parallel_jax import forward_project_view


def get_memory_usage_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def tic() -> float:
    return time.time()


def toc(t0: float) -> float:
    return time.time() - t0


# ---------- 3D TV prox (Chambolle-Pock ROF) ----------
def grad3(u: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    gx = jnp.pad(u[1:, :, :] - u[:-1, :, :], ((0, 1), (0, 0), (0, 0)))
    gy = jnp.pad(u[:, 1:, :] - u[:, :-1, :], ((0, 0), (0, 1), (0, 0)))
    gz = jnp.pad(u[:, :, 1:] - u[:, :, :-1], ((0, 0), (0, 0), (0, 1)))
    return gx, gy, gz


def div3(px: jnp.ndarray, py: jnp.ndarray, pz: jnp.ndarray) -> jnp.ndarray:
    dx = jnp.concatenate([px[:1, :, :], px[1:, :, :] - px[:-1, :, :]], axis=0)
    dy = jnp.concatenate([py[:, :1, :], py[:, 1:, :] - py[:, :-1, :]], axis=1)
    dz = jnp.concatenate([pz[:, :, :1], pz[:, :, 1:] - pz[:, :, :-1]], axis=2)
    return dx + dy + dz


def prox_tv_cp(y: jnp.ndarray, mu: float, n_iters: int = 20) -> jnp.ndarray:
    tau = jnp.float32(0.25)
    sigma = jnp.float32(1.0 / 3.0)
    theta = jnp.float32(1.0)

    x = y
    x_bar = x
    px = jnp.zeros_like(y)
    py = jnp.zeros_like(y)
    pz = jnp.zeros_like(y)

    for _ in range(n_iters):
        gx, gy, gz = grad3(x_bar)
        px_new = px + sigma * gx
        py_new = py + sigma * gy
        pz_new = pz + sigma * gz
        norm = jnp.maximum(1.0, jnp.sqrt(px_new * px_new + py_new * py_new + pz_new * pz_new))
        px = px_new / norm
        py = py_new / norm
        pz = pz_new / norm

        div_p = div3(px, py, pz)
        x_new = (x + tau * div_p + (tau / jnp.float32(mu)) * y) / (1.0 + tau / jnp.float32(mu))

        x_bar = x_new + theta * (x_new - x)
        x = x_new

    return x


# ---------- Forward helper (one view, flattened) ----------
def forward_one_view_flat(x_flat: jnp.ndarray,
                          params: jnp.ndarray,
                          grid, det,
                          step_size: float, n_steps: int,
                          stop_grad_recon: bool,
                          use_checkpoint: bool) -> jnp.ndarray:
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    vx, vy, vz = float(grid["vx"]), float(grid["vy"]), float(grid["vz"])
    nu, nv = int(det["nu"]), int(det["nv"])
    du, dv = float(det["du"]), float(det["dv"])
    vol_origin = jnp.asarray(grid["vol_origin"], dtype=jnp.float32)
    det_center = jnp.asarray(det.get("det_center", [0.0, 0.0]), dtype=jnp.float32)

    out = forward_project_view(
        params=params,
        recon_flat=x_flat,
        nx=nx, ny=ny, nz=nz,
        vx=jnp.float32(vx), vy=jnp.float32(vy), vz=jnp.float32(vz),
        nu=nu, nv=nv,
        du=jnp.float32(du), dv=jnp.float32(dv),
        vol_origin=vol_origin,
        det_center=det_center,
        step_size=jnp.float32(step_size),
        n_steps=int(n_steps),
        use_checkpoint=use_checkpoint,
        stop_grad_recon=stop_grad_recon,
    )
    return out.ravel()


# ---------- Data term and gradient via VJP ----------
def data_f_grad_vjp(x_flat: jnp.ndarray,
                    projections: np.ndarray,  # (n_proj, nv, nu)
                    angles: np.ndarray,       # (n_proj,)
                    grid, det,
                    step_size: float,
                    n_steps: int,
                    verbose: bool = False,
                    desc: str = "") -> Tuple[float, jnp.ndarray]:
    n_proj = int(projections.shape[0])
    fval = 0.0
    grad = jnp.zeros_like(x_flat)
    
    if verbose and desc:
        print(f"  {desc}: Processing {n_proj} projections...")
        t_start = time.time()

    for i in range(n_proj):
        if verbose and desc and (i % max(1, n_proj // 10) == 0):
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 0.01)
            eta = (n_proj - i - 1) / max(rate, 0.01)
            print(f"    Progress: {i+1}/{n_proj} ({100*(i+1)/n_proj:.1f}%) | "
                  f"Rate: {rate:.1f} proj/s | ETA: {eta:.1f}s")
        
        phi = float(angles[i])
        params = jnp.array([0.0, 0.0, phi, 0.0, 0.0], dtype=jnp.float32)

        def proj_fn(vol):
            return forward_one_view_flat(
                vol, params, grid, det, step_size, n_steps,
                stop_grad_recon=False, use_checkpoint=False
            )

        pred, vjp_fn = jax.vjp(proj_fn, x_flat)
        meas = jnp.asarray(projections[i].ravel(), dtype=jnp.float32)
        resid = pred - meas

        fval += float(0.5 * jnp.vdot(resid, resid).real)
        grad = grad + vjp_fn(resid)[0]
    
    if verbose and desc:
        total_time = time.time() - t_start
        print(f"  {desc}: Completed in {total_time:.2f}s")

    return fval, grad


def apply_AtA_vjp(v: jnp.ndarray,
                  angles: np.ndarray,
                  grid, det,
                  step_size: float,
                  n_steps: int,
                  verbose: bool = False,
                  power_iter: int = 0) -> jnp.ndarray:
    """Compute A^T A v as grad of 0.5 ||A v||^2 (with zero sinogram)."""
    zeros = np.zeros((len(angles), int(det["nv"]), int(det["nu"])), dtype=np.float32)
    desc = f"Power iter {power_iter}" if power_iter > 0 else "A^T A"
    _, g = data_f_grad_vjp(v, zeros, angles, grid, det, step_size, n_steps, 
                           verbose=verbose, desc=desc)
    return g


def estimate_L_power(grid, det, angles: np.ndarray,
                     step_size: float, n_steps: int,
                     n_iter: int = 3, seed: int = 0) -> float:
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    rng = np.random.default_rng(seed)
    v = jnp.asarray(rng.standard_normal(nx * ny * nz).astype(np.float32))
    v = v / (jnp.linalg.norm(v) + 1e-8)

    print(f"\n=== Estimating Lipschitz constant L (power method, {n_iter} iterations) ===")
    print(f"Note: Each power iteration requires processing all {len(angles)} projections twice (forward + adjoint)")
    
    lam = 0.0
    for k in range(n_iter):
        t_iter = time.time()
        print(f"\nPower iteration {k+1}/{n_iter}:")
        
        w = apply_AtA_vjp(v, angles, grid, det, step_size, n_steps, 
                         verbose=True, power_iter=k+1)
        lam = float(jnp.vdot(v, w).real / (jnp.vdot(v, v).real + 1e-8))
        v = w / (jnp.linalg.norm(w) + 1e-8)
        
        print(f"  Lambda estimate: {lam:.6e} | Iter time: {time.time() - t_iter:.2f}s")

    print(f"\n=== Power method complete ===")
    return max(lam, 1e-6)


def main():
    parser = argparse.ArgumentParser(description="FISTA-TV reconstruction with JAX projector (VJP-based)")
    parser.add_argument("--input-dir", type=str, default="parallel_proj_test",
                        help="Directory with projections.tiff, angles.npy, metadata.json")
    parser.add_argument("--output-dir", type=str, default="fista_tv_recon",
                        help="Output directory")
    parser.add_argument("--lambda-tv", type=float, default=0.01,
                        help="TV regularization weight (lambda)")
    parser.add_argument("--max-iters", type=int, default=30,
                        help="FISTA iterations")
    parser.add_argument("--tv-iters", type=int, default=20,
                        help="Inner iterations for TV prox per FISTA step")
    parser.add_argument("--angle-step", type=int, default=1,
                        help="Use every k-th angle to speed up (k=1 uses all)")
    parser.add_argument("--power-iters", type=int, default=3,
                        help="Power iterations to estimate L")
    parser.add_argument("--step-scale", type=float, default=0.9,
                        help="Safety scale for step size: alpha = step_scale / L")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== FISTA-TV Reconstruction (JAX Projector, VJP) ===")
    print(f"Device: {jax.devices()[0]}")
    print(f"Input:  {in_dir}")
    print(f"Output: {out_dir}")
    print(f"Initial RSS: {get_memory_usage_mb():.1f} MB")

    # Load metadata, projections, angles
    print("\n--- Loading data ---")
    with open(in_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    grid = meta["grid"]
    det = meta["detector"]

    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    vx, vy, vz = float(grid["vx"]), float(grid["vy"]), float(grid["vz"])
    nu, nv = int(det["nu"]), int(det["nv"])
    du, dv = float(det["du"]), float(det["dv"])

    print("Loading projections...")
    projections = tiff.imread(in_dir / "projections.tiff")
    angles = np.load(in_dir / "angles.npy").astype(np.float32)
    print(f"Loaded projections: {projections.shape}, angles: {angles.shape}")

    # Optional angle subsampling
    if args.angle_step > 1:
        print(f"Subsampling: using every {args.angle_step}-th angle")
        projections = projections[:: args.angle_step, :, :]
        angles = angles[:: args.angle_step]

    n_proj = int(projections.shape[0])

    # Step parameters consistent with projector
    if "projections" in meta and "step_size" in meta["projections"]:
        step_size = float(meta["projections"]["step_size"])
        n_steps = int(meta["projections"]["n_steps"])
    else:
        step_size = vy
        n_steps = int(math.ceil((ny * vy) / step_size))

    print(f"\n--- Configuration ---")
    print(f"Grid: {nx}x{ny}x{nz} vox | Detector: {nu}x{nv} pix")
    print(f"Projections: {n_proj} | step_size={step_size} | n_steps={n_steps}")
    print(f"lambda_TV={args.lambda_tv}, iters={args.max_iters}, TV iters={args.tv_iters}")

    # Warm-up JIT on a single forward call (no checkpoint, no stop_grad)
    print("\n--- JIT warm-up ---")
    print("Compiling forward projection...")
    x0 = jnp.zeros((nx * ny * nz,), dtype=jnp.float32)
    params0 = jnp.array([0.0, 0.0, float(angles[0]), 0.0, 0.0], dtype=jnp.float32)
    t_jit = time.time()
    _ = forward_one_view_flat(
        x0, params0, grid, det, step_size, n_steps,
        stop_grad_recon=False, use_checkpoint=False
    )
    jax.block_until_ready(_)
    print(f"JIT warmup done in {time.time() - t_jit:.2f}s. RSS: {get_memory_usage_mb():.1f} MB")

    # Lipschitz estimate L ~ ||A||^2 and step size alpha
    t0 = tic()
    L = estimate_L_power(grid, det, angles, step_size, n_steps, n_iter=args.power_iters)
    alpha = args.step_scale / L
    print(f"\nFinal: L ≈ {L:.3e} | alpha = {alpha:.3e} | Total time: {toc(t0):.2f}s")

    # FISTA variables
    print("\n=== Starting FISTA iterations ===")
    xk = jnp.zeros((nx * ny * nz,), dtype=jnp.float32)
    xk_prev = xk
    tk = 1.0

    t_all = tic()
    for it in range(1, args.max_iters + 1):
        t_iter = time.time()
        print(f"\n--- FISTA iteration {it}/{args.max_iters} ---")
        
        # Momentum step
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
        beta = (tk - 1.0) / t_next
        zk = xk + jnp.float32(beta) * (xk - xk_prev)
        
        # Gradient step
        print(f"Computing gradient (processing {n_proj} projections)...")
        t_grad = time.time()
        fval, grad = data_f_grad_vjp(zk, projections, angles, grid, det, step_size, n_steps)
        print(f"  Gradient computed in {time.time() - t_grad:.2f}s")
        
        yk = zk - jnp.float32(alpha) * grad

        # TV proximal step
        print(f"Applying TV prox ({args.tv_iters} iterations)...")
        t_tv = time.time()
        mu = args.lambda_tv * alpha
        x_next = prox_tv_cp(yk.reshape(nx, ny, nz), mu=mu, n_iters=args.tv_iters).ravel()
        print(f"  TV prox done in {time.time() - t_tv:.2f}s")

        xk_prev = xk
        xk = x_next
        tk = t_next

        iter_time = time.time() - t_iter
        print(
            f"iter {it:3d} | f={fval:.6e} | ||grad||={float(jnp.linalg.norm(grad)):.3e} "
            f"| iter_time={iter_time:.2f}s | RSS={get_memory_usage_mb():.1f} MB"
        )

    t_recon = toc(t_all)
    print(f"\n=== FISTA complete ===")
    print(f"Total time: {t_recon:.2f}s | Per-iteration: {t_recon/max(1,args.max_iters):.2f}s")

    # Save reconstruction (z, y, x)
    print("\n--- Saving results ---")
    x_vol = np.asarray(xk).reshape(nx, ny, nz)
    out_stack = np.transpose(x_vol, (2, 1, 0)).astype(np.float32)
    out_path = out_dir / "reconstruction_fista_tv.tiff"
    tif_kwargs = {}
    tiff.imwrite(str(out_path), out_stack, **tif_kwargs)
    print(f"Saved reconstruction: {out_path}")

    meta_out = {
        "method": "FISTA-TV (VJP-based)",
        "lambda_tv": args.lambda_tv,
        "max_iters": args.max_iters,
        "tv_iters": args.tv_iters,
        "angle_step": args.angle_step,
        "power_iters": args.power_iters,
        "step_scale": args.step_scale,
        "L_est": L,
        "alpha": alpha,
        "grid": grid,
        "detector": det,
        "n_proj": n_proj,
        "step_size": step_size,
        "n_steps": n_steps,
        "timing": {"total_seconds": t_recon},
        "device": str(jax.devices()[0]),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)
    print("Saved metadata.json")

    print("\n=== All done! ===")


if __name__ == "__main__":
    main()