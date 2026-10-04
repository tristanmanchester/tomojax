#!/usr/bin/env python3
"""Compare streamed Joseph pose normal equations with an explicit Jacobian.

Resident inputs, synchronized component outputs and geometry preparation are
included. This is not a successful joint-alignment workflow measurement.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from compare_projectors import make_case
from compare_reconstructions import environment, write_result
import jax
import jax.numpy as jnp
from joseph_derivatives import measure, relative_error
import numpy as np
from pose_recovery import poses_jax

from tomojax.forward import joseph_pose_normal_equations, project_joseph


def run_case(size: int, views: int, kind: str, interpolation: str, repeats: int) -> dict:
    """Compare all output blocks and an independent JAX directional computation."""
    case = make_case(size, views, kind)
    g, d = case.grid, case.detector
    volume, nominal, target = (jnp.asarray(a) for a in (case.volume, case.poses, case.analytic))
    parameters = jnp.asarray(np.random.default_rng(404).uniform(-0.2, 0.2, (views, 5)), jnp.float32)

    def pose(p: jax.Array, t: jax.Array) -> jax.Array:
        return poses_jax(p, t, d)

    def project(p: jax.Array, x: jax.Array) -> jax.Array:
        return project_joseph(
            x, jax.vmap(pose)(p, nominal), g, d, backend="pallas", interpolation=interpolation
        )

    def explicit(p: jax.Array, x: jax.Array, y: jax.Array) -> tuple:
        def probe(delta: jax.Array) -> tuple:
            prediction = project(p + delta, x)
            return prediction, prediction

        jacobian, prediction = jax.jacfwd(probe, has_aux=True)(jnp.zeros(p.shape[1]))
        jacobian = jacobian.reshape(views, -1, p.shape[1])
        residual = prediction - y
        flat = residual.reshape(views, -1)
        gradient = jnp.einsum("nmp,nm->np", jacobian, flat, precision=jax.lax.Precision.HIGHEST)
        matrix = jnp.einsum("nmp,nmq->npq", jacobian, jacobian, precision=jax.lax.Precision.HIGHEST)
        return 0.5 * jnp.sum(flat**2, axis=1), gradient, matrix, residual

    def fused(p: jax.Array, x: jax.Array, y: jax.Array) -> tuple:
        matrices = jax.vmap(pose)(p, nominal)
        directions = jax.vmap(jax.jacfwd(pose))(p, nominal)
        return joseph_pose_normal_equations(
            x, matrices, directions, y, g, d, backend="pallas", interpolation=interpolation
        )

    _, fp_timing = measure(project, (parameters, volume), repeats)
    reference, explicit_timing = measure(explicit, (parameters, volume, target), repeats)
    actual, fused_timing = measure(fused, (parameters, volume, target), repeats)
    errors = [relative_error(a, b) for a, b in zip(actual, reference, strict=True)]
    direction = jnp.asarray(
        np.random.default_rng(943).uniform(-1, 1, parameters.shape), jnp.float32
    )

    def ordinary(p: jax.Array) -> jax.Array:
        return project_joseph(
            volume, jax.vmap(pose)(p, nominal), g, d, backend="jax", interpolation=interpolation
        )

    pred, tangent = jax.jit(lambda p, dp: jax.jvp(ordinary, (p,), (dp,)))(parameters, direction)
    residual = np.asarray(pred, np.float64) - np.asarray(target, np.float64)
    tangent = np.asarray(tangent, np.float64)
    dp = np.asarray(direction, np.float64)
    gradient, hessian = np.asarray(actual[1], np.float64), np.asarray(actual[2], np.float64)
    directional_errors = {
        "gradient": relative_error(
            np.sum(gradient * dp, axis=1), np.sum(residual * tangent, axis=(1, 2))
        ),
        "quadratic_form": relative_error(
            np.einsum("ni,nij,nj->n", dp, hessian, dp), np.sum(tangent**2, axis=(1, 2))
        ),
        "residual": relative_error(actual[3], residual),
    }
    finite = all(np.isfinite(np.asarray(a)).all() for a in actual)
    return {
        "case": case.name,
        "grid": g.to_dict(),
        "detector": d.to_dict(),
        "methods": {
            "forward": fp_timing,
            "explicit_jacobian": explicit_timing,
            "fused_normals": fused_timing,
        },
        "fused_to_forward": fused_timing["median_ms"] / fp_timing["median_ms"],
        "explicit_to_fused": explicit_timing["median_ms"] / fused_timing["median_ms"],
        "fused_vs_explicit_relative_errors": errors,
        "jax_directional_relative_errors": directional_errors,
        "finite": finite,
        "passed": finite and max(errors) < 2e-4 and max(directional_errors.values()) < 2e-4,
    }


def main() -> int:
    """Write complete case records atomically without overwriting earlier runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--interpolation", choices=["linear", "cubic"], default="cubic")
    parser.add_argument(
        "--geometries",
        nargs="+",
        choices=["parallel", "anisotropic", "lamino"],
        default=["parallel", "anisotropic", "lamino"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 8 or min(args.views, args.repeats) < 1:
        parser.error("sizes must be >=8 and views/repeats must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new record path")
    if jax.default_backend() != "gpu":
        parser.error("this comparison requires CUDA")
    parameters = vars(args).copy()
    parameters.pop("output")
    payload = {
        "suite": "joseph-pose-normal-equations-v1",
        "environment": environment(),
        "parameters": parameters,
        "complete": False,
        "cases": [],
        "scope": (
            "resident inputs; geometry preparation and synchronized outputs included; "
            "imports, host transfers and recovery iterations excluded"
        ),
        "memory_scope": "compiler estimates, not process peak VRAM; both methods return residuals",
    }
    write_result(args.output, payload)
    for n in args.sizes:
        for kind in args.geometries:
            row = run_case(n, args.views, kind, args.interpolation, args.repeats)
            payload["cases"].append(row)
            write_result(args.output, payload)
            print(
                row["case"],
                "explicit/fused",
                row["explicit_to_fused"],
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
