#!/usr/bin/env python3
"""Resident-input Joseph projection/derivative timings with explicit accuracy gates.

This is a component benchmark, not a complete reconstruction/alignment workflow.
Compiler temporary-storage estimates are reported separately from input/output
bytes; they are not measurements of peak process VRAM. Each case uses analytic
Gaussian data from the existing physical-geometry projector fixtures.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Literal

from compare_projectors import make_case
from compare_reconstructions import environment, write_result
import jax
import jax.numpy as jnp
import numpy as np

from tomojax.forward import joseph_l2_value_and_grad, project_joseph

if TYPE_CHECKING:
    from collections.abc import Callable


def measure(function: Callable, args: tuple, repeats: int) -> tuple[Any, dict]:
    """Separate lowering/compilation, first execution and synchronized repetitions."""
    started = time.perf_counter()
    compiled = jax.jit(function).lower(*args).compile()
    compile_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    result = compiled(*args)
    jax.block_until_ready(result)
    first_ms = (time.perf_counter() - started) * 1000
    # Very short kernels can finish all seven samples before the laptop GPU
    # leaves its idle clock state after compilation. Warm every method for the
    # same minimum duration; record this untimed work instead of selecting slow
    # forward samples that would artificially improve the derivative ratio.
    started = time.perf_counter()
    warmup_calls = 0
    while time.perf_counter() - started < 0.2:
        jax.block_until_ready(compiled(*args))
        warmup_calls += 1
    warmup_ms = (time.perf_counter() - started) * 1000
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = compiled(*args)
        jax.block_until_ready(result)
        samples.append((time.perf_counter() - started) * 1000)
    memory = compiled.memory_analysis()
    return result, {
        "compile_ms": compile_ms,
        "first_execution_ms": first_ms,
        "warmup_calls": warmup_calls,
        "warmup_ms": warmup_ms,
        "samples_ms": samples,
        "median_ms": float(np.median(samples)),
        "compiler_bytes": {
            name: getattr(memory, name)
            for name in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
            )
        },
    }


def relative_error(actual: Any, expected: Any) -> float:
    """Compute relative L2 error in host FP64, including scalar loss comparisons."""
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(
        np.linalg.norm((actual - expected).ravel()) / max(np.linalg.norm(expected.ravel()), 1e-30)
    )


def run_case(
    size: int,
    views: int,
    kind: str,
    repeats: int,
    interpolation: Literal["linear", "cubic"] = "linear",
) -> dict:
    """Check full gradients at small sizes and directional derivatives at every size."""
    case = make_case(size, views, kind)
    g, d = case.grid, case.detector
    volume, poses, target = (jnp.asarray(a) for a in (case.volume, case.poses, case.analytic))

    def loss(
        v: jax.Array, t: jax.Array, y: jax.Array, backend: Literal["jax", "pallas"]
    ) -> jax.Array:
        return 0.5 * jnp.sum(
            (project_joseph(v, t, g, d, backend=backend, interpolation=interpolation) - y) ** 2
        )

    _, fp = measure(
        lambda v, t: project_joseph(v, t, g, d, backend="pallas", interpolation=interpolation),
        (volume, poses),
        repeats,
    )
    general, general_timing = measure(
        jax.value_and_grad(lambda v, t, y: loss(v, t, y, "pallas"), argnums=(0, 1)),
        (volume, poses, target),
        repeats,
    )
    fused, fused_timing = measure(
        lambda v, t, y: joseph_l2_value_and_grad(
            v, t, y, g, d, backend="pallas", interpolation=interpolation
        ),
        (volume, poses, target),
        repeats,
    )
    row = {
        "case": case.name,
        "grid": g.to_dict(),
        "detector": d.to_dict(),
        "methods": {"forward": fp, "general_reverse": general_timing, "fused_l2": fused_timing},
        "general_reverse_to_forward": general_timing["median_ms"] / fp["median_ms"],
        "fused_l2_to_forward": fused_timing["median_ms"] / fp["median_ms"],
        "fused_vs_general_relative_errors": [
            relative_error(a, b)
            for a, b in zip(jax.tree.leaves(fused), jax.tree.leaves(general), strict=True)
        ],
        "finite": all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(fused)),
    }
    reference_limit = 64 if interpolation == "linear" else 32
    if size <= reference_limit:
        reference, timing = measure(
            jax.value_and_grad(lambda v, t, y: loss(v, t, y, "jax"), argnums=(0, 1)),
            (volume, poses, target),
            repeats,
        )
        row["methods"]["jax_reverse_reference"] = timing
        row["fused_vs_jax_relative_errors"] = [
            relative_error(a, b)
            for a, b in zip(jax.tree.leaves(fused), jax.tree.leaves(reference), strict=True)
        ]
        del reference
    else:
        row["full_reverse_reference"] = (
            f"not run above {reference_limit}; directional JAX JVP checked instead"
        )

    rng = np.random.default_rng(9827)
    # A directional derivative through the ordinary JAX reference needs no
    # reverse-mode tape, and checks large shapes without requiring its huge VJP.
    dv = jnp.asarray(rng.uniform(-1, 1, size=volume.shape), jnp.float32)
    dt = jnp.asarray(rng.uniform(-1, 1, size=poses.shape), jnp.float32).at[:, 3, :].set(0)
    rotation_direction = dt[:, :3, :3]
    skew = 0.5 * (rotation_direction - rotation_direction.transpose(0, 2, 1))
    dt = dt.at[:, :3, :3].set(skew @ poses[:, :3, :3])
    _, direction = jax.jit(
        lambda v, t, dv, dt: jax.jvp(lambda v, t: loss(v, t, target, "jax"), (v, t), (dv, dt))
    )(volume, poses, dv, dt)
    gv, gt = fused[1]
    dot = float(jnp.vdot(gv, dv) + jnp.vdot(gt, dt))
    scale = float(
        jnp.linalg.norm(gv) * jnp.linalg.norm(dv) + jnp.linalg.norm(gt) * jnp.linalg.norm(dt)
    )
    row["directional_check"] = {
        "jax_jvp": float(direction),
        "gradient_dot_direction": dot,
        "absolute_error": abs(dot - float(direction)),
        "norm_scaled_error": abs(dot - float(direction)) / max(scale, 1e-30),
    }
    row["passed"] = (
        row["finite"]
        and max(row["fused_vs_general_relative_errors"]) < 2e-4
        and max(row.get("fused_vs_jax_relative_errors", [0])) < 2e-4
        and row["directional_check"]["norm_scaled_error"] < 2e-5
    )
    return row


def main() -> int:
    """Retain every completed case and reject accidental replacement of records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--interpolation", choices=["linear", "cubic"], default="linear")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--geometries",
        nargs="+",
        choices=["parallel", "anisotropic", "lamino"],
        default=["parallel", "anisotropic", "lamino"],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("bench/results/joseph-derivatives.json")
    )
    args = parser.parse_args()
    if min(args.sizes) < 8 or args.views < 1 or args.repeats < 1:
        parser.error("sizes must be >=8 and views/repeats must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new record path")
    if jax.default_backend() != "gpu":
        parser.error("this comparison requires CUDA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": "joseph-derivatives-v3",
        "environment": environment(),
        "parameters": {
            "sizes": args.sizes,
            "interpolation": args.interpolation,
            "views": args.views,
            "geometries": args.geometries,
            "repeats": args.repeats,
        },
        "scope": (
            "resident inputs; geometry preparation and synchronized outputs included; "
            "imports and host transfers excluded"
        ),
        "memory_scope": "XLA compiler estimates only; not peak process GPU memory",
        "complete": False,
        "cases": [],
    }
    write_result(args.output, payload)
    for size in args.sizes:
        for kind in args.geometries:
            row = run_case(size, args.views, kind, args.repeats, args.interpolation)
            payload["cases"].append(row)
            write_result(args.output, payload)
            print(
                row["case"],
                "fused/FP",
                row["fused_l2_to_forward"],
                "passed",
                row["passed"],
                flush=True,
            )
            jax.clear_caches()
            gc.collect()
    payload["complete"] = True
    payload["passed"] = all(row["passed"] for row in payload["cases"])
    write_result(args.output, payload)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
