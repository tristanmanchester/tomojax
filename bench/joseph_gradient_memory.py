#!/usr/bin/env python3
"""Sample whole-process GPU memory for isolated Joseph derivative calls."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from compare_reconstructions import environment, isolated_run, write_result


def worker(args: argparse.Namespace) -> int:
    """Keep each method's allocator, compilation and derivatives in its own process."""
    from compare_projectors import make_case
    import jax
    import jax.numpy as jnp
    from joseph_derivatives import measure
    import numpy as np

    from tomojax.forward import joseph_l2_value_and_grad, project_joseph

    case = make_case(args.size, args.views, args.geometry)
    volume, poses, target = map(jnp.asarray, (case.volume, case.poses, case.analytic))
    if args.method == "fused_cuda":

        def fn(x: jax.Array, t: jax.Array, y: jax.Array) -> tuple:
            return joseph_l2_value_and_grad(x, t, y, case.grid, case.detector, backend="pallas")

    else:

        def loss(x: jax.Array, t: jax.Array, y: jax.Array) -> jax.Array:
            prediction = project_joseph(x, t, case.grid, case.detector, backend="jax")
            return 0.5 * jnp.sum((prediction - y) ** 2)

        fn = jax.value_and_grad(loss, argnums=(0, 1))
    result, measurements = measure(fn, (volume, poses, target), args.repeats)
    finite = all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(result))
    write_result(
        args.output,
        {
            "case": case.name,
            "method": args.method,
            "status": "finite" if finite else "nonfinite",
            "loss": float(result[0]),
            "volume_gradient_norm": float(jnp.linalg.norm(result[1][0])),
            "pose_gradient_norm": float(jnp.linalg.norm(result[1][1])),
            "component_measurements": measurements,
        },
    )
    return 0 if finite else 1


def main() -> int:
    """Run serial workers and retain sampled accounting, including failed executions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("bench/results/joseph-memory.json"))
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, default=64, help=argparse.SUPPRESS)
    parser.add_argument(
        "--geometry",
        choices=["parallel", "anisotropic", "lamino"],
        default="parallel",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--method",
        choices=["fused_cuda", "jax_reference"],
        default="fused_cuda",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if min(args.views, args.repeats, args.size) < 1:
        parser.error("dimensions/repeats must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new record path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args)

    directory = args.output.parent / f"{args.output.stem}-workers"
    directory.mkdir(exist_ok=False)
    payload = {
        "suite": "joseph-gradient-memory-v1",
        "environment": environment(),
        "views": args.views,
        "repeats": args.repeats,
        "scope": (
            "isolated process including imports, preparation, compilation and repeated derivatives"
        ),
        "accuracy_scope": (
            "finite-output memory probe; derivative accuracy established by the separate "
            "comparison and tests"
        ),
        "complete": False,
        "records": [],
    }
    write_result(args.output, payload)
    for size in (64, 256):
        for geometry in ("parallel", "anisotropic", "lamino"):
            methods = ("jax_reference", "fused_cuda") if size == 64 else ("fused_cuda",)
            for method in methods:
                output = directory / f"{geometry}-{size}-{method}.json"
                command = [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--size",
                    str(size),
                    "--views",
                    str(args.views),
                    "--geometry",
                    geometry,
                    "--method",
                    method,
                    "--repeats",
                    str(args.repeats),
                    "--output",
                    str(output),
                ]
                row = isolated_run(command, output, timeout=300)
                payload["records"].append(row)
                write_result(args.output, payload)
                print(
                    geometry,
                    size,
                    method,
                    row.get("status"),
                    row.get("sampled_process_peak_mib"),
                    flush=True,
                )
    payload["complete"] = True
    payload["passed"] = all(
        row.get("status") == "finite" and row.get("sampled_process_peak_mib") is not None
        for row in payload["records"]
    )
    write_result(args.output, payload)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
