#!/usr/bin/env python3
"""Compare CGLS operator costs at identical geometry, batch size and accuracy data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from compare_projectors import environment, make_case
import jax
import jax.numpy as jnp
import numpy as np

from tomojax.recon.cgls import _operators


def run_case(size: int, views: int, kind: str, batch: int, repeats: int) -> dict:
    """Alternate resident forward/adjoint calls; keep geometry preparation in the jit."""
    case = make_case(size, views, kind)
    poses, volume, data = (
        jnp.asarray(case.poses),
        jnp.asarray(case.volume),
        jnp.asarray(case.analytic),
    )
    operations = []
    for model in ("ray", "joseph"):
        for index, name, argument in [(0, "forward", volume), (1, "adjoint", data)]:
            function = jax.jit(
                lambda t, x, index=index, model=model: _operators(
                    t, case.grid, case.detector, None, "pallas", batch, model
                )[index](x)
            )
            operations.append((f"{model}_{name}", function, argument))
    outputs, cold = {}, {}
    for name, function, argument in operations:
        start = time.perf_counter()
        outputs[name] = np.asarray(function(poses, argument))
        cold[name] = (time.perf_counter() - start) * 1000
    samples = {name: [] for name, _, _ in operations}
    for iteration in range(repeats):
        for name, function, argument in operations if iteration % 2 else operations[::-1]:
            start = time.perf_counter()
            function(poses, argument).block_until_ready()
            samples[name].append((time.perf_counter() - start) * 1000)
    medians = {name: float(np.median(timings)) for name, timings in samples.items()}
    return {
        "case": case.name,
        "grid": case.grid.to_dict(),
        "detector": case.detector.to_dict(),
        "views_per_batch": batch,
        "median_ms": medians,
        "samples_ms": samples,
        "cold_compile_execute_download_ms": cold,
        "analytic_relative_l2": {
            model: float(
                np.linalg.norm(outputs[f"{model}_forward"] - case.analytic)
                / np.linalg.norm(case.analytic)
            )
            for model in ("ray", "joseph")
        },
        "pair_speedup": (medians["ray_forward"] + medians["ray_adjoint"])
        / (medians["joseph_forward"] + medians["joseph_adjoint"]),
    }


def main() -> int:
    """Retain every case and sample, including model changes and regressions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--views", type=int, default=180)
    parser.add_argument("--batch", type=int, default=180)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--geometries",
        nargs="+",
        choices=["parallel", "lamino", "anisotropic"],
        default=["parallel", "lamino", "anisotropic"],
    )
    parser.add_argument("--output", type=Path, default=Path("bench/results/operator-models.json"))
    args = parser.parse_args()
    if min(args.sizes) < 8 or min(args.views, args.batch, args.repeats) < 1:
        parser.error("sizes must be >=8 and views/batch/repeats must be positive")
    if jax.default_backend() != "gpu":
        parser.error("Pallas comparisons require a CUDA GPU")
    payload = {
        "environment": environment(),
        "scope": (
            "synchronized resident CGLS operators with dynamic poses, "
            "including per-call plane coefficient preparation"
        ),
        "limitation": (
            "Different integration models; compare analytic errors as well as speed. "
            "This is not complete reconstruction or process memory evidence."
        ),
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        for kind in args.geometries:
            record = run_case(size, args.views, kind, args.batch, args.repeats)
            payload["records"].append(record)
            args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            print(record["case"], record["median_ms"], record["analytic_relative_l2"], flush=True)
            jax.clear_caches()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
