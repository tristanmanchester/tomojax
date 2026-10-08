#!/usr/bin/env python3
"""Known-volume pose recovery from independent Gaussian line integrals.

This is a prerequisite diagnostic, not a joint reconstruction/alignment benchmark.
Measurements integrate continuous Gaussians independently of either projector.
A physical volume margin includes their otherwise omitted tails. It is sampled
from the same continuous object, not filled by a reconstruction or by zeros.
Errors and failed targets are retained; no fitted amplitude or gauge correction
is applied. Per-view rotations and translations are independently identifiable
for this asymmetric known object, but can remain very sensitive to noise.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time
import traceback
from typing import Any

from compare_projectors import _sample_volume, make_geometry
from compare_reconstructions import environment, write_result
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from tomojax.alignment.api import se3_from_pose_params
from tomojax.forward import joseph_pose_normal_equations, project_joseph
from tomojax.geometry import Detector, Grid, grid_volume_origin

GAUSSIANS = (
    (1.0, (-0.10, 0.08, -0.04), (0.08, 0.10, 0.09)),
    (0.65, (0.13, -0.09, 0.10), (0.065, 0.07, 0.06)),
)
RICH_GAUSSIANS = (
    (1.0, (-0.19, -0.15, -0.17), (0.035, 0.041, 0.032)),
    (0.8, (0.17, -0.16, -0.12), (0.040, 0.033, 0.037)),
    (1.2, (-0.13, 0.18, -0.10), (0.032, 0.038, 0.042)),
    (0.65, (0.18, 0.13, -0.18), (0.041, 0.035, 0.030)),
    (0.9, (-0.18, -0.09, 0.17), (0.038, 0.031, 0.041)),
    (1.1, (0.14, -0.18, 0.19), (0.032, 0.041, 0.036)),
    (0.75, (-0.09, 0.17, 0.16), (0.043, 0.032, 0.035)),
    (1.25, (0.19, 0.14, 0.10), (0.030, 0.039, 0.033)),
    (0.6, (0.01, -0.01, -0.02), (0.040, 0.037, 0.031)),
)
PHANTOMS = {"two-gaussian": GAUSSIANS, "nine-gaussian": RICH_GAUSSIANS}


def poses_jax(parameters: jax.Array, nominal: jax.Array, detector: Detector) -> jax.Array:
    """Right-compose rotations; translate in detector/world x,z coordinates."""
    scale = jnp.array([np.pi / 180] * 3 + [detector.du, detector.dv], jnp.float32)
    delta = se3_from_pose_params(parameters * scale)
    result = jnp.matmul(nominal, delta, precision=jax.lax.Precision.HIGHEST)
    return result.at[:3, 3].set(nominal[:3, 3] + delta[:3, 3])


def poses_numpy(parameters: np.ndarray, nominal: np.ndarray, detector: Detector) -> np.ndarray:
    """Independent FP64 physical poses for data and error measurements."""
    result = nominal.astype(np.float64).copy()
    for i, param in enumerate(parameters):
        alpha, beta, phi = np.deg2rad(param[:3].astype(np.float64))
        rotation = (
            Rotation.from_euler("y", beta).as_matrix()
            @ Rotation.from_euler("x", alpha).as_matrix()
            @ Rotation.from_euler("z", phi).as_matrix()
        )
        result[i, :3, :3] = nominal[i, :3, :3].astype(np.float64) @ rotation
        result[i, :3, 3] = nominal[i, :3, 3] + [param[3] * detector.du, 0, param[4] * detector.dv]
    return result


def integrals(
    poses: np.ndarray, extent: np.ndarray, detector: Detector, gaussians: tuple = GAUSSIANS
) -> np.ndarray:
    """Analytic FP64 line integrals with bounded view workspace and FP32 output."""
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du + detector.center[0]
    v = (np.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv + detector.center[1]
    uu, vv = np.meshgrid(u, v)
    camera = np.stack([uu, np.zeros_like(uu), vv], axis=-1)
    result = np.empty((len(poses), detector.nv, detector.nu), dtype=np.float32)
    batch_size = max(1, min(32, 1_048_576 // (detector.nv * detector.nu)))
    for start in range(0, len(poses), batch_size):
        stop = min(start + batch_size, len(poses))
        result[start:stop] = _integrals_batch(poses[start:stop], extent, camera, gaussians)
    return result


def _integrals_batch(
    poses: np.ndarray, extent: np.ndarray, camera: np.ndarray, gaussians: tuple
) -> np.ndarray:
    """Keep the original FP64 arithmetic and component order for each ray."""
    base = np.einsum("vui,nij->nvuj", camera, poses[:, :3, :3])
    base -= np.einsum("ni,nij->nj", poses[:, :3, 3], poses[:, :3, :3])[:, None, None, :]
    direction = poses[:, 1, :3]
    result = np.zeros(base.shape[:-1])
    for amplitude, center, sigma in gaussians:
        inverse_variance = 1 / (extent * sigma) ** 2
        diff = base - extent * center
        a = np.sum(direction**2 * inverse_variance, axis=-1)[:, None, None]
        b = np.sum(diff * direction[:, None, None] * inverse_variance, axis=-1)
        c = np.sum(diff**2 * inverse_variance, axis=-1)
        result += amplitude * np.sqrt(2 * np.pi / a) * np.exp(-0.5 * (c - b * b / a))
    return result.astype(np.float32)


def pose_errors(estimate: np.ndarray, truth: np.ndarray, nominal: np.ndarray, d: Detector) -> dict:
    """Use FP64 rotation vectors; FP32 arccos(trace) loses sub-.02 degree errors."""
    actual, expected = (poses_numpy(p, nominal, d) for p in (estimate, truth))
    relative = actual[:, :3, :3] @ expected[:, :3, :3].transpose(0, 2, 1)
    angle = np.rad2deg(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(), axis=-1))
    shift = estimate[:, 3:].astype(np.float64) - truth[:, 3:]
    distance = np.linalg.norm(shift, axis=1)
    accepted = (angle <= 0.01) & (distance <= 0.05)
    return {
        "rotation_rmse_deg": float(np.sqrt(np.mean(angle**2))),
        "rotation_max_deg": float(np.max(angle)),
        "translation_component_rmse_px": float(np.sqrt(np.mean(shift**2))),
        "translation_vector_max_px": float(np.max(distance)),
        "rotation_errors_deg": angle.tolist(),
        "translation_errors_px": distance.tolist(),
        "accepted_views": int(np.count_nonzero(accepted)),
        "view_success_fraction": float(np.mean(accepted)),
    }


def fixture(
    size: int,
    views: int,
    kind: str,
    seed: int,
    noise: float,
    margin: float,
    *,
    phantom: str = "two-gaussian",
    initialization: str = "perturbed-truth",
) -> dict:
    """Fixed phantom and draws; margin changes only the reconstruction grid."""
    g, d, nominal, _ = make_geometry(size, views, kind)
    gaussians = PHANTOMS[phantom]
    if initialization not in {"perturbed-truth", "nominal"}:
        raise ValueError("initialization must be 'perturbed-truth' or 'nominal'")
    extent = np.array([g.nx * g.vx, g.ny * g.vy, g.nz * g.vz])
    pad = round(size * margin)
    g = replace(g, nx=g.nx + 2 * pad, ny=g.ny + 2 * pad, nz=g.nz + 2 * pad)
    volume = _sample_volume(
        (g.nx, g.ny, g.nz),
        np.array([g.vx, g.vy, g.vz]),
        np.array(grid_volume_origin(g)),
        [(a, extent * c, 1 / (extent * s) ** 2) for a, c, s in gaussians],
        [],
    )
    rng = np.random.default_rng(seed)
    angle, shift = (0.4, 2) if initialization == "perturbed-truth" else (3, 10)
    truth = np.concatenate(
        [rng.uniform(-angle, angle, (views, 3)), rng.uniform(-shift, shift, (views, 2))], axis=1
    ).astype(np.float32)
    clean = integrals(poses_numpy(truth, nominal, d), extent, d, gaussians)
    sigma = noise * np.sqrt(np.mean(np.square(clean, dtype=np.float64)))
    data = np.empty_like(clean)
    batch_size = max(1, min(32, 1_048_576 // (d.nv * d.nu)))
    for start in range(0, views, batch_size):
        stop = min(start + batch_size, views)
        chunk = clean[start:stop]
        data[start:stop] = chunk + rng.normal(0, sigma, chunk.shape).astype(np.float32)
    initial = truth + np.concatenate(
        [rng.uniform(-3, 3, (views, 3)), rng.uniform(-10, 10, (views, 2))], axis=1
    ).astype(np.float32)
    if initialization == "nominal":
        initial = np.zeros_like(truth)
    return {
        "grid": g,
        "detector": d,
        "volume": volume,
        "nominal": nominal,
        "truth": truth,
        "data": data,
        "initial": initial,
        "noise_sigma": float(sigma),
        "extent": extent,
    }


def workflow_functions(
    g: Grid,
    d: Detector,
    backend: str,
    interpolation: str,
    normal_method: str = "fused",
    line_search: str = "streamed",
) -> tuple:
    """Compile reusable per-view damped Gauss-Newton and discrete line search."""
    if normal_method not in {"explicit", "fused"}:
        raise ValueError("normal_method must be 'explicit' or 'fused'")
    if line_search not in {"stacked", "streamed"}:
        raise ValueError("line_search must be 'stacked' or 'streamed'")

    def one(parameters: jax.Array, nominal: jax.Array, volume: jax.Array) -> jax.Array:
        return project_joseph(
            volume,
            poses_jax(parameters, nominal, d)[None],
            g,
            d,
            backend=backend,
            interpolation=interpolation,
        )[0]

    predict = jax.jit(jax.vmap(one, in_axes=(0, 0, None)))

    def normal_explicit(
        parameters: jax.Array, nominal: jax.Array, volume: jax.Array, target: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # A shared probe direction extracts each view's local columns without
        # constructing the mostly-zero Jacobian between different views.
        def projected(delta: jax.Array) -> tuple[jax.Array, jax.Array]:
            matrices = jax.vmap(lambda p, t: poses_jax(p, t, d))(parameters + delta, nominal)
            prediction = project_joseph(
                volume, matrices, g, d, backend=backend, interpolation=interpolation
            )
            return prediction, prediction

        jacobian, prediction = jax.jacfwd(projected, has_aux=True)(jnp.zeros(parameters.shape[1]))
        jacobian = jacobian.reshape(parameters.shape[0], -1, parameters.shape[1])
        residual = prediction - target
        flat = residual.reshape(parameters.shape[0], -1)
        gradient = jnp.einsum("nmp,nm->np", jacobian, flat, precision=jax.lax.Precision.HIGHEST)
        h = jnp.einsum("nmp,nmq->npq", jacobian, jacobian, precision=jax.lax.Precision.HIGHEST)
        return solve(gradient, h), residual

    def solve(gradient: jax.Array, h: jax.Array) -> jax.Array:
        scale = jnp.maximum(jnp.diagonal(h, axis1=-2, axis2=-1), 1e-6)
        system = h + 0.003 * jnp.eye(gradient.shape[1])[None] * scale[:, None, :]
        step = jnp.linalg.solve(system, -gradient[..., None])[..., 0]
        limit = jnp.array([2.0, 2.0, 2.0, 4.0, 4.0])
        return jnp.clip(step, -limit, limit)

    def normal_fused(
        parameters: jax.Array, nominal: jax.Array, volume: jax.Array, target: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        def pose(p: jax.Array, t: jax.Array) -> jax.Array:
            return poses_jax(p, t, d)

        matrices = jax.vmap(pose)(parameters, nominal)
        directions = jax.vmap(jax.jacfwd(pose))(parameters, nominal)
        _, gradient, h, residual = joseph_pose_normal_equations(
            volume, matrices, directions, target, g, d, backend=backend, interpolation=interpolation
        )
        return solve(gradient, h), residual

    normal = jax.jit(normal_fused if normal_method == "fused" else normal_explicit)
    return predict, normal, _line_search_function(predict, line_search)


def _line_search_function(predict: Any, line_search: str) -> Any:
    """Evaluate the same four candidate steps, retaining only the selected state."""
    scales = jnp.array([1.0, 0.5, 0.25, 0.125])

    @jax.jit
    def accept_stacked(
        parameters: jax.Array,
        step: jax.Array,
        nominal: jax.Array,
        volume: jax.Array,
        target: jax.Array,
        residual: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        candidates = parameters[None] + scales[:, None, None] * step[None]
        projected = jax.vmap(lambda p: predict(p, nominal, volume))(candidates)
        candidate_residual = projected - target[None]
        difference = candidate_residual - residual[None]
        # Evaluate the change directly; avoid subtracting two large noisy SSEs.
        change = jnp.sum(difference * (residual[None] + 0.5 * difference), axis=(2, 3))
        change = jnp.concatenate([change, jnp.zeros((1, parameters.shape[0]))])
        index = jnp.argmin(change, axis=0)
        all_candidates = jnp.concatenate([candidates, parameters[None]])
        all_residuals = jnp.concatenate([candidate_residual, residual[None]])
        result = all_candidates[index, jnp.arange(parameters.shape[0])]
        loss = 0.5 * jnp.sum(all_residuals[index, jnp.arange(parameters.shape[0])] ** 2)
        return result, loss

    @jax.jit
    def accept_streamed(
        parameters: jax.Array,
        step: jax.Array,
        nominal: jax.Array,
        volume: jax.Array,
        target: jax.Array,
        residual: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        def candidate(i: jax.Array, best: tuple) -> tuple:
            best_change, best_parameters, best_loss = best
            proposed = parameters + scales[i] * step
            new_residual = predict(proposed, nominal, volume) - target
            difference = new_residual - residual
            change = jnp.sum(difference * (residual + 0.5 * difference), axis=(1, 2))
            loss = 0.5 * jnp.sum(new_residual**2, axis=(1, 2))
            # Match argmin's first-candidate tie and NaN behavior. Only retain
            # parameters and per-view scalars, not four candidate sinograms.
            take = (change < best_change) | (jnp.isnan(change) & ~jnp.isnan(best_change))
            return (
                jnp.where(take, change, best_change),
                jnp.where(take[:, None], proposed, best_parameters),
                jnp.where(take, loss, best_loss),
            )

        initial_loss = 0.5 * jnp.sum(residual**2, axis=(1, 2))
        change, selected, loss = jax.lax.fori_loop(
            0,
            4,
            candidate,
            (jnp.full((parameters.shape[0],), jnp.inf), parameters, initial_loss),
        )
        # The zero step is last; earlier equal changes win.
        reject = change > 0
        return (
            jnp.where(reject[:, None], parameters, selected),
            jnp.sum(jnp.where(reject, initial_loss, loss)),
        )

    return accept_streamed if line_search == "streamed" else accept_stacked


def recover(inputs: dict, functions: tuple, iters: int) -> tuple[np.ndarray, dict]:
    """Time input transfers, correlation, compilation, iterations and output transfer."""
    started = time.perf_counter()
    predict, normal, accept = functions
    x, nominal, target, parameters = (
        jnp.asarray(inputs[k]) for k in ("volume", "nominal", "data", "initial")
    )
    d = inputs["detector"]
    prediction = np.asarray(predict(parameters, nominal, x))
    cross = np.fft.rfft2(inputs["data"]) * np.conj(np.fft.rfft2(prediction))
    correlation = np.fft.irfft2(cross, s=prediction.shape[1:])
    peaks = np.argmax(correlation.reshape(len(nominal), -1), axis=1)
    dv, du = peaks // d.nu, peaks % d.nu
    dv, du = np.where(dv > d.nv // 2, dv - d.nv, dv), np.where(du > d.nu // 2, du - d.nu, du)
    parameters = parameters.at[:, 3].add(jnp.asarray(du)).at[:, 4].add(jnp.asarray(dv))
    history = []
    termination = "iteration_limit"
    for iteration in range(iters):
        step, residual = normal(parameters, nominal, x, target)
        candidate, loss = accept(parameters, step, nominal, x, target, residual)
        max_step, loss_host = jax.device_get((jnp.max(jnp.abs(candidate - parameters)), loss))
        parameters = candidate
        history.append(
            {"iteration": iteration + 1, "loss": float(loss_host), "max_step": float(max_step)}
        )
        if not np.isfinite([max_step, loss_host]).all():
            termination = "nonfinite"
            break
        if max_step < 1e-5:
            termination = "step_limit"
            break
    result = np.asarray(parameters)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return result, {"elapsed_ms": elapsed_ms, "termination": termination, "iterations": history}


def run_case(size: int, kind: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    """Keep cold and all repeated estimates, timings and per-view errors."""
    inputs = fixture(
        size,
        args.views,
        kind,
        seed,
        args.noise,
        args.margin,
        phantom=args.phantom,
        initialization=args.initialization,
    )
    g, d = inputs["grid"], inputs["detector"]
    functions = workflow_functions(
        g, d, args.backend, args.interpolation, args.normal_method, args.line_search
    )
    runs = []
    for repeat in range(args.repeats + 1):
        estimate, record = recover(inputs, functions, args.iters)
        record.update(
            kind="cold" if repeat == 0 else "warm",
            errors=pose_errors(estimate, inputs["truth"], inputs["nominal"], d),
            estimated_parameters=estimate.tolist(),
        )
        runs.append(record)
    return {
        "size": size,
        "geometry": kind,
        "seed": seed,
        "grid": g.to_dict(),
        "detector": d.to_dict(),
        "truth_parameters": inputs["truth"].tolist(),
        "initial_parameters": inputs["initial"].tolist(),
        "noise_sigma": inputs["noise_sigma"],
        "phantom_extent": inputs["extent"].tolist(),
        "runs": runs,
        "numerical_success": all(r["termination"] != "nonfinite" for r in runs),
        "pose_target_passed": all(r["errors"]["view_success_fraction"] >= 0.99 for r in runs),
    }


def main() -> int:
    """Run the declared cases without overwriting earlier records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--views", type=int, default=30)
    parser.add_argument(
        "--geometries",
        nargs="+",
        choices=["parallel", "anisotropic", "lamino"],
        default=["parallel", "lamino"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[9345])
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backend", choices=["jax", "pallas"], default="pallas")
    parser.add_argument("--interpolation", choices=["linear", "cubic"], default="cubic")
    parser.add_argument("--normal-method", choices=["explicit", "fused"], default="fused")
    parser.add_argument("--line-search", choices=["stacked", "streamed"], default="streamed")
    parser.add_argument("--phantom", choices=list(PHANTOMS), default="two-gaussian")
    parser.add_argument(
        "--initialization", choices=["perturbed-truth", "nominal"], default="perturbed-truth"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new record path")
    if (
        min(args.sizes) < 16
        or args.views < 5
        or min(args.iters, args.repeats) < 1
        or not np.isfinite([args.noise, args.margin]).all()
        or min(args.noise, args.margin) < 0
    ):
        parser.error("invalid size, views, budget, noise or margin")
    if args.backend == "pallas" and jax.default_backend() != "gpu":
        parser.error("Pallas requires CUDA")
    parameters = vars(args).copy()
    parameters.pop("output")
    payload = {
        "suite": "known-volume-pose-recovery-v3",
        "environment": environment(),
        "parameters": parameters,
        "gaussians": PHANTOMS[args.phantom],
        "known_volume": True,
        "joint_alignment": False,
        "scope": (
            "cold and repeated recovery including transfers, FFT correlation and output; "
            "fixture creation and error diagnostics excluded"
        ),
        "translation_frame": (
            "detector/world x,z; rotations right-composed; "
            "not legacy object-frame translation parameters"
        ),
        "success_definition": (
            "at least 99% of views with rotation vector norm <=0.01 degrees "
            "and translation vector norm <=0.05 native pixels, in every recorded run"
        ),
        "cases": [],
        "complete": False,
    }
    write_result(args.output, payload)
    for n in args.sizes:
        for kind in args.geometries:
            for seed in args.seeds:
                try:
                    row = run_case(n, kind, seed, args)
                except Exception as error:
                    row = {
                        "size": n,
                        "geometry": kind,
                        "seed": seed,
                        "numerical_success": False,
                        "pose_target_passed": False,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "runs": [],
                    }
                payload["cases"].append(row)
                write_result(args.output, payload)
                print(
                    n,
                    kind,
                    seed,
                    "target",
                    row["pose_target_passed"],
                    "rotation RMSE",
                    row["runs"][0]["errors"]["rotation_rmse_deg"] if row["runs"] else row["error"],
                    "cold ms",
                    row["runs"][0]["elapsed_ms"] if row["runs"] else None,
                    flush=True,
                )
                jax.clear_caches()
    payload["complete"] = True
    payload["pose_target_passed"] = all(r["pose_target_passed"] for r in payload["cases"])
    write_result(args.output, payload)
    return 0 if payload["pose_target_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
