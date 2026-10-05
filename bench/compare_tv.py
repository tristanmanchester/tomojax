#!/usr/bin/env python3
"""Compare TV-regularised reconstruction: TomoJAX FISTA-TV against TIGRE FISTA and ASD-POCS.

Usage: ``python bench/compare_tv.py SIZE tomojax|tigre``. A structured parallel
phantom (analytic line integrals, 180 views) gets 3% Gaussian noise. Each
library runs 50 iterations over a grid of its own regularisation weights; the
two weight conventions differ, so compare each library's best result. TomoJAX
timings start from device-resident data (warm); TIGRE's API includes host
transfers. Override TIGRE's grids with TIGRE_LAMBDAS / TIGRE_ALPHAS.
"""

import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_projectors import make_case
import numpy as np

size, views, iters = int(sys.argv[1]), 180, 50
case = make_case(size, views, "parallel", phantom="structured")
rng = np.random.default_rng(1)
data = case.analytic + (
    0.03 * np.sqrt(np.mean(case.analytic**2)) * rng.standard_normal(case.analytic.shape)
).astype(np.float32)
truth = case.volume


def err(x: object) -> float:
    """Full-volume relative L2 error against the sampled truth."""
    return float(np.linalg.norm(np.asarray(x, np.float64) - truth) / np.linalg.norm(truth))


results = []

which = sys.argv[2]
if which == "tomojax":
    import jax.numpy as jnp

    from tomojax.geometry import ParallelGeometry
    from tomojax.recon import CGLSConfig, FistaConfig, cgls, fista_tv

    geo = ParallelGeometry(case.grid, case.detector, case.angles_deg)
    p = jnp.asarray(data)
    for lam in [0.3, 1, 3, 10, 30]:
        cfg = FistaConfig(iters=iters, lambda_tv=lam, positivity=True)
        x, _ = fista_tv(geo, case.grid, case.detector, p, config=cfg)
        x.block_until_ready()
        t = time.perf_counter()
        x, _ = fista_tv(geo, case.grid, case.detector, p, config=cfg)
        x.block_until_ready()
        dt = time.perf_counter() - t
        results.append(("tomojax fista_tv", lam, err(x), dt))
    for it in [5, 10, 20]:
        x, _ = cgls(geo, case.grid, case.detector, p, config=CGLSConfig(iters=it, rtol=0))
        x.block_until_ready()
        t = time.perf_counter()
        x, _ = cgls(geo, case.grid, case.detector, p, config=CGLSConfig(iters=it, rtol=0))
        x.block_until_ready()
        dt = time.perf_counter() - t
        results.append(("tomojax cgls (no TV)", it, err(x), dt))
else:
    import tigre
    from tigre.algorithms import asd_pocs, fista

    grid, detector = case.grid, case.detector
    geo = tigre.geometry(mode="parallel", nVoxel=np.asarray([grid.nz, grid.ny, grid.nx]))
    geo.dVoxel = np.asarray([grid.vz, grid.vy, grid.vx])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.nDetector = np.asarray([detector.nv, detector.nu])
    geo.dDetector = np.asarray([detector.dv, detector.du])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.accuracy = 1.0
    angles = np.deg2rad(-90.0 - case.angles_deg).astype(np.float32)
    for lam in [
        float(v) for v in os.environ.get("TIGRE_LAMBDAS", "0.003,0.01,0.03,0.1,0.3").split(",")
    ]:
        t = time.perf_counter()
        x = fista(data.copy(), geo, angles, iters, tviter=20, tvlambda=lam, verbose=False)
        dt = time.perf_counter() - t
        results.append(("tigre fista", lam, err(np.ascontiguousarray(x.transpose(2, 1, 0))), dt))
    for alpha in [float(v) for v in os.environ.get("TIGRE_ALPHAS", "0.002,0.02").split(",")]:
        t = time.perf_counter()
        x = asd_pocs(data.copy(), geo, angles, iters, alpha=alpha, verbose=False)
        dt = time.perf_counter() - t
        results.append(
            ("tigre asd_pocs", alpha, err(np.ascontiguousarray(x.transpose(2, 1, 0))), dt)
        )
for name, param, e, dt in results:
    print(f"{size} {name:22s} param {param:<6} relL2 {e:.4f}  {dt:.2f}s", flush=True)
