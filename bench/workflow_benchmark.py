#!/usr/bin/env python3
"""Synchronized public reconstruction and alignment workflow benchmarks.

Measures complete calls, including validation, geometry setup, norm estimation,
JIT compilation, optimizer bookkeeping, and diagnostics. Analytic projections
come from asymmetric Gaussian ellipsoids. Alignment starts from perturbed poses;
its zero-pose truth is only approximate for the discretized ray model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from compare_projectors import environment, make_case
import jax
import jax.numpy as jnp
import numpy as np

from tomojax.alignment import AlignConfig
from tomojax.alignment.api import L2LossSpec, align
from tomojax.geometry import LaminographyGeometry, ParallelGeometry
from tomojax.recon import FistaConfig, SPDHGConfig, fista_tv, spdhg_tv


def run_case(size: int, kind: str, method: str, args: argparse.Namespace) -> dict[str, Any]:
    """Time the public call and record its numerical and execution diagnostics."""
    case = make_case(size, args.views, kind)
    grid, detector = case.grid, case.detector
    geometry = (
        LaminographyGeometry(grid, detector, case.angles, tilt_deg=30)
        if kind == "lamino"
        else ParallelGeometry(grid, detector, case.angles)
    )
    data = jnp.asarray(case.analytic)
    n_views = len(case.angles)
    lipschitz = 1.5 * n_views * max(grid.nx, grid.ny, grid.nz)
    if method == "fista":
        config = FistaConfig(
            iterations=args.recon_iters,
            regulariser="huber_tv",
            lipschitz=None if args.auto_norm else lipschitz,
            views_per_batch=args.batch,
            gather_dtype="fp32",
            nonnegative=True,
        )

        def call() -> tuple:
            return fista_tv(geometry, grid, detector, data, config=config)
    elif method == "spdhg":
        config = SPDHGConfig(iterations=args.spdhg_iters, views_per_batch=args.batch, seed=31)

        def call() -> tuple:
            return spdhg_tv(geometry, grid, detector, data, config=config)
    else:
        optimizer = method.removeprefix("align_")
        config = AlignConfig(
            outer_iterations=args.outer_iters,
            iterations=args.inner_iters,
            lipschitz=lipschitz,
            projector_backend="jax",
            gather_dtype="fp32",
            views_per_batch=args.batch,
            loss=L2LossSpec(),
            early_stop=False,
            gauge_policy="anchor_mean",
            opt_method=optimizer,
            lbfgs_maxiter=args.lbfgs_iters,
        )
        phase = np.linspace(0.0, 2 * np.pi, n_views, endpoint=False)
        initial = np.zeros((n_views, 5), dtype=np.float32)
        initial[:, 2] = 0.003 * np.sin(phase)
        initial[:, 3] = 0.2 * detector.du * np.sin(phase)
        initial[:, 4] = 0.2 * detector.dv * np.cos(phase)
        initial = jnp.asarray(initial)

        def call() -> tuple:
            return align(geometry, grid, detector, data, config=config, init_pose_params=initial)

    runs = []
    for repeat in range(args.repeats + 1):
        start = time.perf_counter()
        output = call()
        jax.block_until_ready(output[0])
        elapsed_ms = (time.perf_counter() - start) * 1000
        volume = np.asarray(output[0])
        nrmse = float(np.linalg.norm(volume - case.volume) / np.linalg.norm(case.volume))
        info = output[-1]
        row = {"repeat": repeat, "wall_ms": elapsed_ms, "volume_nrmse": nrmse, "info": info}
        if method.startswith("align_"):
            params = np.asarray(output[1])
            row["pose_rotation_rms_rad"] = float(np.sqrt(np.mean(params[:, :3] ** 2)))
            row["pose_translation_rms_world"] = float(np.sqrt(np.mean(params[:, 3:] ** 2)))
            row["finite_pose"] = bool(np.isfinite(params).all())
        workflow_ok = not any(
            stat.get("reconstruction_failed", False) for stat in info.get("outer_stats", [])
        )
        row["passed"] = bool(
            workflow_ok
            and np.isfinite(volume).all()
            and nrmse <= args.max_nrmse
            and row.get("finite_pose", True)
        )
        runs.append(row)
        print(
            f"{case.name} {method} repeat={repeat} {elapsed_ms:.2f} ms NRMSE={nrmse:.5f}",
            flush=True,
        )
    return {
        "case": case.name,
        "method": method,
        "grid": grid.to_dict(),
        "detector": detector.to_dict(),
        "config": asdict(config),
        "input": "analytic_gaussian_line_integrals",
        "timing_scope": "public_API_with_resident_input_and_Python_diagnostics",
        "cold_ms": runs[0]["wall_ms"],
        "warm_median_ms": float(np.median([run["wall_ms"] for run in runs[1:]])),
        "runs": runs,
        "passed": all(run["passed"] for run in runs),
    }


def main() -> int:
    """Run every requested case and fail on errors or failed quality gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64])
    parser.add_argument(
        "--geometries",
        nargs="+",
        choices=["parallel", "lamino", "anisotropic"],
        default=["parallel", "lamino", "anisotropic"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["fista", "spdhg", "align_gn", "align_gd", "align_lbfgs"],
        default=["fista", "spdhg", "align_gn"],
    )
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--recon-iters", type=int, default=10)
    parser.add_argument("--spdhg-iters", type=int, default=20)
    parser.add_argument("--inner-iters", type=int, default=4)
    parser.add_argument("--outer-iters", type=int, default=3)
    parser.add_argument("--lbfgs-iters", type=int, default=5)
    parser.add_argument("--auto-norm", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-nrmse", type=float, default=0.6)
    parser.add_argument("--output", type=Path, default=Path("bench/results/workflows.json"))
    args = parser.parse_args()
    if (
        min(
            *args.sizes,
            args.views,
            args.batch,
            args.repeats,
            args.recon_iters,
            args.spdhg_iters,
            args.inner_iters,
            args.outer_iters,
            args.lbfgs_iters,
        )
        <= 0
    ):
        parser.error("sizes, iteration counts, views, batch, and repeats must be positive")
    if not np.isfinite(args.max_nrmse) or args.max_nrmse <= 0:
        parser.error("max-nrmse must be finite and positive")
    records = []
    for size in args.sizes:
        for kind in args.geometries:
            for method in args.methods:
                try:
                    records.append(run_case(size, kind, method, args))
                except Exception as exc:
                    records.append(
                        {
                            "case": f"{kind}-{size}-{args.views}",
                            "method": method,
                            "passed": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(records[-1], flush=True)
    payload = {
        "environment": environment(),
        "arguments": {**vars(args), "output": str(args.output)},
        "records": records,
        "passed": all(record["passed"] for record in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
