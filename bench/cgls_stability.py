#!/usr/bin/env python3
"""Stress FP32 CGLS against FP64 dense least-squares solutions, without timing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compare_reconstructions import environment
import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import Detector, Grid, LaminographyGeometry
from tomojax.recon import CGLSConfig, cgls


def problem() -> tuple[Grid, Detector, LaminographyGeometry, np.ndarray]:
    """Assemble the forward matrix independently of the iterative algorithm."""
    grid = Grid(3, 2, 2, 0.8, 1.1, 1.3)
    detector = Detector(5, 4, 0.7, 1.2, (0.17, -0.23))
    geometry = LaminographyGeometry(grid, detector, [13, 49, 83, 122, 161], tilt_deg=23)
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(5)])

    def project(flat: jax.Array) -> jax.Array:
        return jax.vmap(
            lambda t: forward_project_view_T(t, grid, detector, flat.reshape((3, 2, 2)))
        )(poses).ravel()

    matrix = np.asarray(jax.jacfwd(project)(jnp.zeros(12))).astype(np.float64)
    return grid, detector, geometry, matrix


def run(backends: list[str], seeds: int) -> list[dict[str, Any]]:
    """Retain every signed-data solve, including breakdowns and tolerance misses."""
    grid, detector, geometry, matrix = problem()
    records = []
    for backend in backends:
        for seed in range(seeds):
            amplitude = 10.0 ** (3 * (seed % 3 - 1))
            data = (amplitude * np.random.default_rng(9100 + seed).normal(size=(5, 4, 5))).astype(
                np.float32
            )
            for damping in (0.0, 0.1, 0.7):
                expected = np.linalg.lstsq(
                    np.concatenate([matrix, damping * np.eye(12)]),
                    np.concatenate([data.ravel(), np.zeros(12)]),
                    rcond=None,
                )[0]
                actual, info = cgls(
                    geometry,
                    grid,
                    detector,
                    data,
                    config=CGLSConfig(
                        iterations=100,
                        rtol=1e-7,
                        damping=damping,
                        views_per_batch=3,
                        projector_backend=backend,
                    ),
                )
                x = np.asarray(actual).ravel()
                error = float(np.linalg.norm(x - expected) / np.linalg.norm(expected))
                gradient = matrix.T @ (data.ravel() - matrix @ x) - damping**2 * x
                stationarity = float(
                    np.linalg.norm(gradient) / np.linalg.norm(matrix.T @ data.ravel())
                )
                finite = bool(np.isfinite(error) and np.isfinite(stationarity))
                records.append(
                    {
                        "backend": backend,
                        "seed": 9100 + seed,
                        "damping": damping,
                        "amplitude": amplitude,
                        "relative_error": error if finite else None,
                        "normalized_stationarity": stationarity if finite else None,
                        "info": info,
                        "accepted": finite
                        and error < 2e-4
                        and stationarity < 2e-5
                        and info["termination"] != "numerical_breakdown",
                    }
                )
    return records


def main() -> int:
    """Write the declared numerical stress experiment and its full results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", nargs="+", choices=["jax", "pallas"], default=["jax", "pallas"]
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("bench/results/cgls-stability.json"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("seeds must be positive")
    if "pallas" in args.backends and jax.default_backend() != "gpu":
        parser.error("Pallas requires CUDA; use --backends jax on CPU")
    result = {
        "environment": environment(),
        "suite": "cgls-dense-stability-v1",
        "reference": "FP32 forward Jacobian promoted to FP64; augmented dense numpy.linalg.lstsq",
        "limitations": (
            "Small matrices test solver stability, not independent physical "
            "projection accuracy or large-system conditioning."
        ),
        "relative_error_limit": 2e-4,
        "normalized_stationarity_limit": 2e-5,
        "records": run(args.backends, args.seeds),
    }
    result["passed"] = all(record["accepted"] for record in result["records"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(result["records"]),
                "passed": result["passed"],
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
