#!/usr/bin/env python3
"""Measure explicit adjoint accuracy against autodiff along long oblique rays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_projectors import environment, errors, jax_sync, make_case, measure
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from tomojax.core.projector import (
    backproject_view_T,
    forward_project_view_T,
    sum_backproject_views_T,
)
from tomojax.geometry import Detector, Grid


def run_case(size: int, repeats: int) -> dict:
    """Check the Euclidean transpose without fitting scale or changing geometry."""
    rng = np.random.default_rng(19)
    grid = Grid(size, size, 5, 0.8, 1.1, 1.3)
    detector = Detector(size, 7, 0.9, 1.1, (0.23, -0.41))
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Rotation.from_euler("xyz", [17, 11, 43], degrees=True).as_matrix()
    pose[:3, 3] = [0.13, -0.31, 0.53]
    pose = jnp.asarray(pose)
    volume = jnp.asarray(rng.normal(size=(size, size, 5)), dtype=jnp.float32)
    data = jnp.asarray(rng.normal(size=(7, size)), dtype=jnp.float32)
    forward = jax.jit(
        lambda x: forward_project_view_T(
            pose, grid, detector, x, step_size=0.3, use_checkpoint=True
        )
    )
    adjoint = jax.jit(lambda y: backproject_view_T(pose, grid, detector, y, step_size=0.3))
    reference = jax.jit(lambda y: jax.vjp(forward, volume)[1](y)[0])(data)
    projected = forward(volume)
    jax.block_until_ready((reference, projected))
    output, timing = measure(lambda: adjoint(data), jax_sync, repeats)
    return {
        "size": size,
        "grid": grid.to_dict(),
        "detector": detector.to_dict(),
        "pose": np.asarray(pose).tolist(),
        "step_size": 0.3,
        **timing,
        "vjp_error": errors(np.asarray(output), np.asarray(reference)),
        "dot_relative_error": float(
            abs(jnp.vdot(projected, data) - jnp.vdot(volume, output))
            / (jnp.linalg.norm(projected) * jnp.linalg.norm(data))
        ),
    }


def run_stack_case(size: int, views: int, kind: str, repeats: int, backend: str = "jax") -> dict:
    """Compare shared accumulation with a sum of independent view adjoints."""
    case = make_case(size, views, kind)
    poses = jnp.asarray(case.poses)
    data = jnp.asarray(
        np.random.default_rng(37).normal(size=case.analytic.shape), dtype=jnp.float32
    )
    backproject_one, backproject_stack = backproject_view_T, sum_backproject_views_T
    if backend == "pallas":
        from tomojax.core.pallas.api import (
            backproject_view_T_pallas,
            sum_backproject_views_T_pallas,
        )

        backproject_one, backproject_stack = (
            backproject_view_T_pallas,
            sum_backproject_views_T_pallas,
        )
    independent = jax.jit(
        lambda y: jnp.sum(
            jax.vmap(lambda t, image: backproject_one(t, case.grid, case.detector, image))(
                poses, y
            ),
            axis=0,
        )
    )
    shared = jax.jit(lambda y: backproject_stack(poses, case.grid, case.detector, y))
    outputs = []
    variants = {}
    for name, function in [("independent_views", independent), ("shared_volume", shared)]:
        output, timing = measure(lambda function=function: function(data), jax_sync, repeats)
        memory = function.lower(data).compile().memory_analysis()
        variants[name] = {
            **timing,
            "compiler_temporary_bytes": None if memory is None else memory.temp_size_in_bytes,
        }
        outputs.append(np.asarray(output))
    return {
        "case": case.name,
        "backend": backend,
        "grid": case.grid.to_dict(),
        "detector": case.detector.to_dict(),
        "views": views,
        "variants": variants,
        "sum_error": errors(outputs[1], outputs[0]),
        "memory_scope": "XLA compiler temporary buffers, not process peak VRAM",
    }


def main() -> int:
    """Write timings and return failure if the explicit adjoint loses accuracy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="16,64,128")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--stack", action="store_true", help="compare batch accumulation and temporary storage"
    )
    parser.add_argument("--batches", default="4,16")
    parser.add_argument("--backend", choices=["jax", "pallas"], default="jax")
    parser.add_argument(
        "--geometries",
        nargs="+",
        choices=["parallel", "lamino", "anisotropic"],
        default=["parallel", "lamino", "anisotropic"],
    )
    parser.add_argument("--max-relative-error", type=float, default=2e-5)
    parser.add_argument("--output", type=Path, default=Path("bench/results/adjoint.json"))
    args = parser.parse_args()
    if args.backend == "pallas" and not args.stack:
        parser.error("--backend pallas currently requires --stack")
    sizes = [int(size) for size in args.sizes.split(",")]
    batches = [int(batch) for batch in args.batches.split(",")]
    if min(sizes) < 1 or min(batches) < 1 or args.repeats < 1:
        parser.error("sizes and repeats must be positive")
    if not np.isfinite(args.max_relative_error) or args.max_relative_error <= 0:
        parser.error("max-relative-error must be finite and positive")
    payload = {
        "environment": environment(),
        "cases": [],
        "failed": False,
        "max_relative_error": args.max_relative_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = (
        ((size, views, kind) for size in sizes for views in batches for kind in args.geometries)
        if args.stack
        else ((size, 1, "oblique") for size in sizes)
    )
    for size, views, kind in cases:
        record = (
            run_stack_case(size, views, kind, args.repeats, args.backend)
            if args.stack
            else run_case(size, args.repeats)
        )
        error = record["sum_error" if args.stack else "vjp_error"]["relative_l2"]
        record["accuracy_passed"] = bool(error <= args.max_relative_error)
        payload["failed"] |= not record["accuracy_passed"]
        payload["cases"].append(record)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
        jax.clear_caches()
    return int(payload["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
