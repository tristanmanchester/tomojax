#!/usr/bin/env python3
"""Measure FBP accuracy and synchronized kernel/API timing on a Gaussian phantom.

Run ``uv run --extra cuda12 python bench/reconstruction_benchmark.py``. Analytic
sinograms avoid evaluating reconstruction with data from its own forward model.
Compilation, warmups, and device transfer are excluded from steady-state timings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from compare_projectors import environment, errors, jax_sync, measure
import jax
import jax.numpy as jnp
import numpy as np

from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import FBPConfig, fbp
from tomojax.recon.fbp import _run_fbp_streamed
from tomojax.recon.filters import rfft_filter_array

if TYPE_CHECKING:
    from collections.abc import Callable


def run_case(size: int, n_views: int, repeats: int) -> dict:
    """Time matching filtered backprojections and check physical amplitudes."""
    grid = Grid(size, size, size, 1.0, 1.0, 1.0)
    # The diagonal field of view covers the corners of the output volume.
    nu = int(np.ceil(np.sqrt(2.0) * size))
    detector = Detector(nu, size, 1.0, 1.0)
    geometry = ParallelGeometry(grid, detector, np.linspace(0, 180, n_views, endpoint=False))
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(n_views)], dtype=jnp.float32)
    sigma = size / 8.0
    u = np.arange(nu) - (nu - 1) / 2
    x = np.arange(size) - (size - 1) / 2
    profile = np.sqrt(2 * np.pi) * sigma * np.exp(-u * u / (2 * sigma * sigma))
    axial = np.exp(-x * x / (2 * sigma * sigma))
    projection = (axial[:, None] * profile).astype(np.float32)
    projections = jax.device_put(np.broadcast_to(projection, (n_views, size, nu)).copy())
    truth = np.exp(
        -(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
        / (2 * sigma * sigma)
    )
    ramp = rfft_filter_array("ramp", nu, 1.0, jnp.float32)
    jax.block_until_ready((poses, projections, ramp))
    ones, unused = jnp.ones((n_views,), jnp.float32), jnp.zeros((n_views, 6), jnp.float32)

    def kernel_for(backend: str) -> Callable[[jax.Array], jax.Array]:
        return lambda y: _run_fbp_streamed(
            poses,
            y,
            ones,
            unused,
            ramp,
            jnp.float32(0),
            grid=grid,
            detector=detector,
            backend=backend,
            batch_size=n_views,
            z_integer=backend == "pallas",
            separable=True,
        )

    kernels = {"jax": kernel_for("jax")}
    if jax.default_backend() == "gpu":
        kernels["pallas"] = kernel_for("pallas")
    records = []
    reference = None
    for backend, kernel in kernels.items():
        output, timing = measure(lambda kernel=kernel: kernel(projections), jax_sync, repeats)
        host = np.asarray(output) * np.pi / n_views
        if reference is None:
            reference = host
        records.append(
            {
                "backend": backend,
                "scope": "filter_and_backproject",
                **timing,
                "analytic_error": errors(host, truth),
                "reference_error": errors(host, reference),
            }
        )
        output, timing = measure(
            lambda backend=backend: fbp(
                geometry, grid, detector, projections, config=FBPConfig(backprojector=backend)
            ),
            jax_sync,
            repeats,
        )
        records.append(
            {
                "backend": backend,
                "scope": "public_api",
                **timing,
                "analytic_error": errors(np.asarray(output), truth),
            }
        )
    return {
        "size": size,
        "views": n_views,
        "grid": grid.to_dict(),
        "detector": detector.to_dict(),
        "records": records,
    }


def main() -> int:
    """Write a reproducible timing and accuracy report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="32,64,128,256")
    parser.add_argument("--views", type=int, default=180)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--max-relative-error", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path("bench/results/fbp.json"))
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    if min(sizes) < 8 or args.views < 1 or args.repeats < 1:
        parser.error("sizes must be >= 8; views and repeats must be positive")
    if not np.isfinite(args.max_relative_error) or args.max_relative_error <= 0:
        parser.error("max-relative-error must be finite and positive")
    payload = {
        "environment": environment(),
        "cases": [],
        "failed": False,
        "max_relative_error": args.max_relative_error,
        "max_reference_error": 5e-5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        record = run_case(size, args.views, args.repeats)
        for result in record["records"]:
            result["accuracy_passed"] = bool(
                result["analytic_error"]["relative_l2"] <= args.max_relative_error
                and result.get("reference_error", {}).get("relative_l2", 0.0) <= 5e-5
            )
            payload["failed"] |= not result["accuracy_passed"]
        payload["cases"].append(record)
        print(json.dumps(record), flush=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        jax.clear_caches()
    return int(payload["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
