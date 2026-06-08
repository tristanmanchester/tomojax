#!/usr/bin/env python3
"""Stochastic benchmark for TomoJAX projection loss and explicit gradients.

The suite is intentionally distribution-based: shapes are drawn from fixed buckets
to keep JAX compilation meaningful, while geometry values, phantoms, supports,
weights, and targets vary by seed. Results are written as JSON plus compact slice
PNGs for visual inspection.
"""

# ruff: noqa: D103, PERF401

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry import Detector, Grid
from tomojax.core.projector import sum_backproject_views_T
from tomojax.datasets import random_cubes_spheres
from tomojax.geometry import detector_grid_from_calibration
from tomojax.recon.fista_tv_core import (
    FistaCoreConfig,
    _project_stack,
    _projection_loss_and_explicit_grad,
    fista_tv_core_arrays,
    projection_loss_arrays,
)

VariantName = Literal["jax_jax", "pallas_jax", "pallas_pallas"]


@dataclass(frozen=True)
class CaseSpec:
    """A benchmark case sampled from one anti-overfit family."""

    case_id: str
    family: str
    volume_shape: tuple[int, int, int]
    detector_shape: tuple[int, int]
    n_views: int
    views_per_batch: int
    detector_roll_deg: float
    detector_center: tuple[float, float]
    voxel_size: tuple[float, float, float]
    use_support: bool
    use_weights: bool
    pallas_eligible: bool
    fista_iters: int
    phantom_profile: str


@dataclass(frozen=True)
class CaseData:
    """Materialized arrays for one benchmark case."""

    spec: CaseSpec
    grid: Grid
    detector: Detector
    t_all: jnp.ndarray
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None
    volume: jnp.ndarray
    truth_volume: jnp.ndarray
    target: jnp.ndarray
    weights: jnp.ndarray
    support: jnp.ndarray | None
    direction: jnp.ndarray


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.suite}-seed{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    specs = sample_specs(
        args.suite,
        rng,
        args.cases,
        volume_shape_override=args.volume_shape,
        detector_shape_override=args.detector_shape,
        n_views_override=args.n_views,
        views_per_batch_override=args.views_per_batch,
    )
    variants = resolve_variants(args.variants, args.suite)
    records: list[dict[str, Any]] = []
    slice_paths: list[str] = []

    for index, spec in enumerate(specs):
        case_rng = np.random.default_rng(args.seed * 1009 + index * 9176 + 17)
        data = materialize_case(spec, case_rng)
        case_records, case_slice_paths = run_case(
            data,
            variants=variants,
            run_dir=run_dir,
            repeats=args.repeats,
            warmups=args.warmups,
            fista_iters=args.fista_iters,
            fista_l=args.fista_l,
            fista_repeats=args.fista_repeats,
            fista_warmups=args.fista_warmups,
            skip_loss_grad=args.skip_loss_grad,
            emit_slices=args.emit_slices and len(slice_paths) < args.max_slice_images,
        )
        records.extend(case_records)
        slice_paths.extend(case_slice_paths)

    summary = summarize_records(records)
    payload = {
        "run_id": run_id,
        "suite": args.suite,
        "seed": args.seed,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": environment_payload(),
        "summary": summary,
        "records": records,
        "slice_paths": slice_paths,
    }
    json_path = run_dir / "results.json"
    md_path = run_dir / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(payload))
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["smoke", "public", "holdout", "accelerator", "gpu-target"],
        default="smoke",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cases", type=int, default=0, help="Override number of sampled cases.")
    parser.add_argument(
        "--volume-shape",
        default="",
        help="Override sampled volume shape as nx,ny,nz for one custom case.",
    )
    parser.add_argument(
        "--detector-shape",
        default="",
        help="Override sampled detector shape as nv,nu for one custom case.",
    )
    parser.add_argument("--n-views", type=int, default=0, help="Override number of views.")
    parser.add_argument("--views-per-batch", type=int, default=0, help="Override view batch size.")
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument(
        "--variants",
        default="auto",
        help="Comma list: jax_jax,pallas_jax,pallas_pallas or auto.",
    )
    parser.add_argument("--output-dir", default="bench/profiles")
    parser.add_argument(
        "--fista-iters",
        type=int,
        default=0,
        help="Override macro FISTA iterations for visual/debug runs.",
    )
    parser.add_argument(
        "--fista-l",
        type=float,
        default=0.0,
        help="Override macro FISTA Lipschitz constant for visual/debug runs.",
    )
    parser.add_argument(
        "--fista-repeats",
        type=int,
        default=-1,
        help="Override macro FISTA timed repeats. Defaults to one repeat for gpu-target.",
    )
    parser.add_argument(
        "--fista-warmups",
        type=int,
        default=-1,
        help="Override macro FISTA warmups. Defaults to zero warmups for gpu-target.",
    )
    parser.add_argument(
        "--skip-loss-grad",
        action="store_true",
        help="Skip loss/gradient timing for expensive visual/debug runs.",
    )
    parser.add_argument("--emit-slices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-slice-images", type=int, default=3)
    args = parser.parse_args()
    if args.repeats <= 0:
        args.repeats = 7 if args.suite == "gpu-target" else 3
    if args.warmups <= 0:
        args.warmups = 2 if args.suite == "gpu-target" else 1
    if args.suite == "gpu-target" and args.fista_l <= 0.0:
        args.fista_l = 20000.0
    if args.fista_repeats < 0:
        args.fista_repeats = 1 if args.suite == "gpu-target" else args.repeats
    if args.fista_warmups < 0:
        args.fista_warmups = 0 if args.suite == "gpu-target" else args.warmups
    return args


def sample_specs(
    suite: str,
    rng: np.random.Generator,
    requested_cases: int,
    *,
    volume_shape_override: str = "",
    detector_shape_override: str = "",
    n_views_override: int = 0,
    views_per_batch_override: int = 0,
) -> list[CaseSpec]:
    if volume_shape_override or detector_shape_override or n_views_override:
        return [
            make_custom_spec(
                rng,
                volume_shape=parse_shape(volume_shape_override, 3, default=(64, 64, 64)),
                detector_shape=parse_shape(detector_shape_override, 2, default=(64, 64)),
                n_views=int(n_views_override) if int(n_views_override) > 0 else 32,
                views_per_batch=int(views_per_batch_override)
                if int(views_per_batch_override) > 0
                else 4,
            )
        ]
    if suite == "gpu-target":
        return make_gpu_target_specs(rng)
    default_counts = {"smoke": 3, "public": 12, "holdout": 16, "accelerator": 8}
    count = requested_cases if requested_cases > 0 else default_counts[suite]
    families = family_pool(suite)
    specs = []
    for i in range(count):
        family = families[i % len(families)]
        specs.append(make_spec(family, i, rng, suite))
    rng.shuffle(specs)
    return specs


def parse_shape(raw: str, ndim: int, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if not raw:
        return default
    values = tuple(int(part) for part in raw.replace("x", ",").split(",") if part.strip())
    if len(values) != ndim or any(value <= 0 for value in values):
        raise ValueError(f"expected {ndim} positive dimensions, got {raw!r}")
    return values


def make_custom_spec(
    rng: np.random.Generator,
    *,
    volume_shape: tuple[int, ...],
    detector_shape: tuple[int, ...],
    n_views: int,
    views_per_batch: int,
) -> CaseSpec:
    return CaseSpec(
        case_id=f"custom_{volume_shape[0]}x{volume_shape[1]}x{volume_shape[2]}_det{detector_shape[0]}x{detector_shape[1]}_views{n_views}",
        family="custom",
        volume_shape=tuple(int(v) for v in volume_shape),  # type: ignore[arg-type]
        detector_shape=tuple(int(v) for v in detector_shape),  # type: ignore[arg-type]
        n_views=int(n_views),
        views_per_batch=max(1, min(int(n_views), int(views_per_batch))),
        detector_roll_deg=0.0,
        detector_center=(float(rng.uniform(-0.25, 0.25)), float(rng.uniform(-0.25, 0.25))),
        voxel_size=(1.0, 1.0, 1.0),
        use_support=False,
        use_weights=False,
        pallas_eligible=True,
        fista_iters=3,
        phantom_profile="default",
    )


def make_gpu_target_specs(rng: np.random.Generator) -> list[CaseSpec]:
    cases = [
        ("gpu_target_64_det64_views32", (64, 64, 64), (64, 64), 32, 4, 25),
        ("gpu_target_64_det96_views48", (64, 64, 64), (96, 96), 48, 4, 25),
        ("gpu_target_80x64x64_det80x64_views48", (80, 64, 64), (80, 64), 48, 4, 25),
    ]
    specs = []
    for case_id, volume_shape, detector_shape, n_views, views_per_batch, fista_iters in cases:
        specs.append(
            CaseSpec(
                case_id=case_id,
                family="gpu_target",
                volume_shape=volume_shape,
                detector_shape=detector_shape,
                n_views=n_views,
                views_per_batch=views_per_batch,
                detector_roll_deg=0.0,
                detector_center=(
                    float(rng.uniform(-0.25, 0.25)),
                    float(rng.uniform(-0.25, 0.25)),
                ),
                voxel_size=(1.0, 1.0, 1.0),
                use_support=False,
                use_weights=False,
                pallas_eligible=True,
                fista_iters=fista_iters,
                phantom_profile="gpu_target_double_plus30",
            )
        )
    return specs


def family_pool(suite: str) -> list[str]:
    if suite == "smoke":
        return ["canonical_small", "jittered_small", "rect_detector"]
    if suite == "accelerator":
        return ["canonical_medium", "weighted_medium", "support_medium", "rect_detector"]
    return [
        "canonical_small",
        "jittered_small",
        "rect_detector",
        "odd_shape",
        "weighted_medium",
        "support_medium",
    ]


def make_spec(
    family: str,
    index: int,
    rng: np.random.Generator,
    suite: str,
) -> CaseSpec:
    small_shapes = [(8, 8, 8), (9, 8, 7), (10, 8, 9)]
    medium_shapes = [(16, 16, 16), (18, 16, 14), (20, 18, 16)]
    if suite == "accelerator":
        medium_shapes = [(32, 32, 32), (40, 32, 28), (48, 32, 40)]
    if family in {"canonical_medium", "weighted_medium", "support_medium"}:
        volume_shape = medium_shapes[int(rng.integers(0, len(medium_shapes)))]
        n_views = int(rng.choice([8, 11, 16, 23]))
        detector_shape = tuple(int(v) for v in rng.choice([(16, 16), (18, 22), (24, 20)]))
    elif family == "odd_shape":
        volume_shape = tuple(int(v) for v in rng.choice([(7, 9, 11), (11, 9, 7), (13, 10, 9)]))
        detector_shape = tuple(int(v) for v in rng.choice([(9, 11), (13, 9), (11, 15)]))
        n_views = int(rng.choice([5, 7, 9]))
    else:
        volume_shape = small_shapes[int(rng.integers(0, len(small_shapes)))]
        detector_shape = tuple(int(v) for v in rng.choice([(8, 8), (8, 10), (10, 8)]))
        n_views = int(rng.choice([4, 5, 7]))

    pallas_eligible = family in {
        "canonical_small",
        "canonical_medium",
        "weighted_medium",
        "support_medium",
    }
    detector_roll_deg = 0.0 if pallas_eligible else float(rng.uniform(-7.5, 7.5))
    detector_center = (
        float(rng.uniform(-0.4, 0.4)),
        float(rng.uniform(-0.4, 0.4)),
    )
    voxel_size = (
        (1.0, 1.0, 1.0)
        if pallas_eligible
        else tuple(float(v) for v in rng.uniform(0.85, 1.25, size=3))
    )
    views_per_batch_choices = [1, 2, 3, 4]
    views_per_batch = min(n_views, int(rng.choice(views_per_batch_choices)))
    use_support = family in {"support_medium", "odd_shape"} or bool(rng.random() < 0.25)
    use_weights = family in {"weighted_medium", "jittered_small"} or bool(rng.random() < 0.35)
    return CaseSpec(
        case_id=f"{family}_{index:03d}",
        family=family,
        volume_shape=volume_shape,
        detector_shape=detector_shape,
        n_views=n_views,
        views_per_batch=views_per_batch,
        detector_roll_deg=detector_roll_deg,
        detector_center=detector_center,
        voxel_size=voxel_size,
        use_support=use_support,
        use_weights=use_weights,
        pallas_eligible=pallas_eligible,
        fista_iters=2 if suite == "smoke" else 3,
        phantom_profile="default",
    )


def materialize_case(spec: CaseSpec, rng: np.random.Generator) -> CaseData:
    grid = Grid(
        nx=spec.volume_shape[0],
        ny=spec.volume_shape[1],
        nz=spec.volume_shape[2],
        vx=spec.voxel_size[0],
        vy=spec.voxel_size[1],
        vz=spec.voxel_size[2],
    )
    detector = Detector(
        nu=spec.detector_shape[1],
        nv=spec.detector_shape[0],
        du=1.0,
        dv=1.0,
        det_center=spec.detector_center,
    )
    theta = jittered_angles(spec.n_views, rng)
    dx = rng.normal(0.0, 0.18, size=spec.n_views).astype(np.float32)
    dz = rng.normal(0.0, 0.18, size=spec.n_views).astype(np.float32)
    alpha = rng.normal(0.0, 0.018, size=spec.n_views).astype(np.float32)
    beta = rng.normal(0.0, 0.018, size=spec.n_views).astype(np.float32)
    t_all = jnp.asarray(pose_stack(theta, dx, dz, alpha, beta), dtype=jnp.float32)
    det_grid = None
    if abs(spec.detector_roll_deg) > 1e-8:
        det_grid = detector_grid_from_calibration(
            detector,
            detector_roll_deg=spec.detector_roll_deg,
        )

    volume_np = random_phantom(spec.volume_shape, rng, profile=spec.phantom_profile)
    target_np = random_phantom(spec.volume_shape, rng, profile=spec.phantom_profile)
    support_np = random_support(spec.volume_shape, rng) if spec.use_support else None
    if support_np is not None:
        volume_np = volume_np * support_np
        target_np = target_np * support_np
    volume = jnp.asarray(volume_np, dtype=jnp.float32)
    truth = jnp.asarray(target_np, dtype=jnp.float32)
    support = None if support_np is None else jnp.asarray(support_np, dtype=jnp.float32)
    direction = jnp.asarray(normalized_direction(spec.volume_shape, rng), dtype=jnp.float32)
    target = _project_stack(
        T_all=t_all,
        grid=grid,
        detector=detector,
        volume=truth,
        det_grid=det_grid,
        checkpoint_projector=False,
        projector_unroll=1,
        gather_dtype="fp32",
        views_per_batch=spec.views_per_batch,
        forward_projector="jax",
    )
    noise_scale = 0.01 * jnp.maximum(jnp.std(target), jnp.asarray(1e-3, dtype=jnp.float32))
    noise = jnp.asarray(rng.normal(size=tuple(target.shape)).astype(np.float32)) * noise_scale
    target = (target + noise).astype(jnp.float32)
    weights_np = np.ones((spec.n_views,), dtype=np.float32)
    if spec.use_weights:
        phase = rng.uniform(0.0, 2.0 * math.pi)
        weight_phase = np.arange(spec.n_views) * 0.7 + phase
        weights_np = (0.3 + 0.7 * (0.5 + 0.5 * np.sin(weight_phase))).astype(np.float32)
        weights_np *= rng.uniform(0.75, 1.25, size=spec.n_views).astype(np.float32)
    return CaseData(
        spec=spec,
        grid=grid,
        detector=detector,
        t_all=t_all,
        det_grid=det_grid,
        volume=volume,
        truth_volume=truth,
        target=target,
        weights=jnp.asarray(weights_np, dtype=jnp.float32),
        support=support,
        direction=direction,
    )


def jittered_angles(n_views: int, rng: np.random.Generator) -> np.ndarray:
    base = np.linspace(0.0, np.pi, n_views, endpoint=False, dtype=np.float32)
    jitter = rng.normal(0.0, 0.012, size=n_views).astype(np.float32)
    start = np.float32(rng.uniform(-0.08, 0.08))
    angles = base + jitter + start
    rng.shuffle(angles)
    return angles.astype(np.float32)


def pose_stack(
    theta: np.ndarray,
    dx: np.ndarray,
    dz: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    poses = np.zeros((theta.shape[0], 4, 4), dtype=np.float32)
    poses[:, 3, 3] = 1.0
    for i, angle in enumerate(theta):
        rot = rot_z(float(angle)) @ rot_y(float(beta[i])) @ rot_x(float(alpha[i]))
        poses[i, :3, :3] = rot
        poses[i, 0, 3] = -float(dx[i])
        poses[i, 2, 3] = -float(dz[i])
    return poses


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


GPU_TARGET_PHANTOM_PRESETS = (
    # Selected from the +30% double-density candidate grid: N+, P+, T+, W+.
    (4, 5, 4, 12, 0.22),
    (5, 5, 4, 12, 0.24),
    (5, 5, 5, 13, 0.28),
    (4, 6, 4, 12, 0.18),
)


def random_phantom(
    shape: tuple[int, int, int],
    rng: np.random.Generator,
    *,
    profile: str = "default",
) -> np.ndarray:
    if profile == "gpu_target_double_plus30":
        return random_gpu_target_phantom(shape, rng)

    nx, ny, nz = shape
    min_dim = min(shape)
    min_size = max(2, round(min_dim * 0.18))
    max_size = max(min_size + 1, round(min_dim * 0.55))
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    volume = random_cubes_spheres(
        nx,
        ny,
        nz,
        n_cubes=int(rng.integers(3, 8)),
        n_spheres=int(rng.integers(3, 8)),
        min_size=min_size,
        max_size=max_size,
        min_value=0.15,
        max_value=1.0,
        max_rot_degrees=float(rng.uniform(60.0, 180.0)),
        use_inscribed_fov=min_dim >= 14,
        radial_exponent=float(rng.uniform(0.55, 1.2)),
        seed=seed,
    ).astype(np.float32)
    vmax = float(np.max(volume))
    if vmax <= 0.0:
        volume = random_cubes_spheres(
            nx,
            ny,
            nz,
            n_cubes=2,
            n_spheres=2,
            min_size=2,
            max_size=max(3, min_dim // 2),
            min_value=0.25,
            max_value=1.0,
            max_rot_degrees=120.0,
            use_inscribed_fov=False,
            seed=seed + 1,
        ).astype(np.float32)
        vmax = float(np.max(volume))
    if vmax > 0.0:
        volume = volume / vmax
    return volume.astype(np.float32)


def random_gpu_target_phantom(
    shape: tuple[int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    nx, ny, nz = shape
    scale = min(shape) / 64.0
    last_volume = np.zeros(shape, dtype=np.float32)
    for _ in range(12):
        n_cubes, n_spheres, min_size_base, max_size_base, radial_exponent = (
            GPU_TARGET_PHANTOM_PRESETS[int(rng.integers(0, len(GPU_TARGET_PHANTOM_PRESETS)))]
        )
        min_size = max(2, round(min_size_base * scale))
        max_size = max(min_size + 1, round(max_size_base * scale))
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        volume = random_cubes_spheres(
            nx,
            ny,
            nz,
            n_cubes=n_cubes,
            n_spheres=n_spheres,
            min_size=min_size,
            max_size=max_size,
            min_value=0.25,
            max_value=1.0,
            max_rot_degrees=160.0,
            use_inscribed_fov=True,
            radial_exponent=radial_exponent,
            seed=seed,
        ).astype(np.float32)
        volume = normalize_phantom(volume)
        last_volume = volume
        if phantom_is_readable(volume):
            return volume
    return last_volume


def normalize_phantom(volume: np.ndarray) -> np.ndarray:
    vmax = float(np.max(volume))
    if vmax > 0.0:
        volume = volume / vmax
    return volume.astype(np.float32)


def phantom_is_readable(volume: np.ndarray) -> bool:
    occupied = volume > 0.02
    best_slice_occupancy = float(np.max(np.mean(occupied, axis=(0, 1))))
    mip_occupancy = float(np.mean(np.max(occupied, axis=2)))
    return 0.008 <= best_slice_occupancy <= 0.22 and 0.035 <= mip_occupancy <= 0.35


def random_support(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    nx, ny, nz = shape
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)[:, None, None]
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)[None, :, None]
    z = np.linspace(-1.0, 1.0, nz, dtype=np.float32)[None, None, :]
    radii = rng.uniform(0.62, 0.98, size=3).astype(np.float32)
    support = ((x / radii[0]) ** 2 + (y / radii[1]) ** 2 + (z / radii[2]) ** 2) <= 1.0
    return support.astype(np.float32)


def normalized_direction(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    arr = rng.normal(size=shape).astype(np.float32)
    arr -= np.mean(arr, dtype=np.float32)
    norm = np.linalg.norm(arr.reshape(-1))
    return (arr / max(float(norm), 1e-6)).astype(np.float32)


def resolve_variants(raw: str, suite: str) -> list[VariantName]:
    if raw != "auto":
        return [v.strip() for v in raw.split(",") if v.strip()]  # type: ignore[list-item]
    if suite in {"accelerator", "gpu-target"} and jax.default_backend() != "cpu":
        return ["jax_jax", "pallas_jax", "pallas_pallas"]
    return ["jax_jax"]


def run_case(
    data: CaseData,
    *,
    variants: list[VariantName],
    run_dir: Path,
    repeats: int,
    warmups: int,
    fista_iters: int,
    fista_l: float,
    fista_repeats: int,
    fista_warmups: int,
    skip_loss_grad: bool,
    emit_slices: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    if skip_loss_grad:
        baseline_grad = jnp.zeros_like(data.volume)
        records: list[dict[str, Any]] = []
    else:
        baseline_loss, baseline_grad, baseline_extra = run_loss_grad_variant(
            data,
            variant="jax_jax",
            repeats=repeats,
            warmups=warmups,
        )
        records = [baseline_extra]
    slice_paths: list[str] = []
    if not skip_loss_grad:
        adjoint_error = adjoint_consistency_error(data)
        direction_error = directional_derivative_error(data, baseline_grad)
        records[0]["adjoint_rel_error"] = adjoint_error
        records[0]["directional_derivative_rel_error"] = direction_error
        records[0]["passed"] = bool(
            records[0]["passed"] and adjoint_error < 2e-3 and direction_error < 5e-2
        )

        for variant in variants:
            if variant == "jax_jax":
                continue
            pallas_unavailable = not data.spec.pallas_eligible or jax.default_backend() == "cpu"
            if variant.startswith("pallas") and pallas_unavailable:
                records.append(skip_record(data, variant, "pallas_ineligible_or_cpu_backend"))
                continue
            try:
                loss, grad, record = run_loss_grad_variant(
                    data,
                    variant=variant,
                    repeats=repeats,
                    warmups=warmups,
                )
                record.update(correctness_record(loss, grad, baseline_loss, baseline_grad))
            except Exception as exc:
                record = skip_record(data, variant, f"{type(exc).__name__}: {exc}")
            records.append(record)

    try:
        fista_record, fista_volume = run_fista_macro(
            data,
            repeats=fista_repeats,
            warmups=fista_warmups,
            iters_override=fista_iters,
            l_override=fista_l,
        )
        records.append(fista_record)
    except Exception as exc:
        records.append(skip_record(data, "fista_jax_jax", f"{type(exc).__name__}: {exc}"))
        fista_volume = data.volume

    if emit_slices:
        slice_path = run_dir / f"slices_{sanitize(data.spec.case_id)}.png"
        write_slice_png(data, baseline_grad, fista_volume, slice_path)
        if slice_path.exists():
            slice_paths.append(str(slice_path))
    return records, slice_paths


def run_loss_grad_variant(
    data: CaseData,
    *,
    variant: VariantName,
    repeats: int,
    warmups: int,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, Any]]:
    forward, back = variant.split("_", maxsplit=1)
    cfg = FistaCoreConfig(
        iters=1,
        lambda_tv=0.0,
        checkpoint_projector=False,
        projector_unroll=1,
        gather_dtype="fp32",
        views_per_batch=data.spec.views_per_batch,
        support=data.support,
        forward_projector=forward,
        backprojector=back,
        pallas_tile_shape=(8, 8),
        pallas_num_warps=1,
        compute_iteration_loss=True,
        compute_final_data_loss=True,
        compute_final_regulariser_value=False,
    )

    def loss_grad(
        volume: jnp.ndarray,
        target: jnp.ndarray,
        weights: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        masked = volume if cfg.support is None else volume * cfg.support
        loss, grad = _projection_loss_and_explicit_grad(
            T_all=data.t_all,
            grid=data.grid,
            detector=data.detector,
            volume=masked,
            det_grid=data.det_grid,
            projections=target,
            weights=jnp.sqrt(jnp.maximum(weights, jnp.float32(0.0)))[:, None, None],
            checkpoint_projector=cfg.checkpoint_projector,
            projector_unroll=cfg.projector_unroll,
            gather_dtype=cfg.gather_dtype,
            views_per_batch=cfg.views_per_batch,
            forward_projector=cfg.forward_projector,
            backprojector=cfg.backprojector,
            pallas_tile_shape=cfg.pallas_tile_shape,
            pallas_num_warps=cfg.pallas_num_warps,
            compute_loss=True,
        )
        if cfg.support is not None:
            grad = grad * cfg.support
        return loss, grad

    compiled = jax.jit(loss_grad)
    t0 = time.perf_counter()
    loss, grad = compiled(data.volume, data.target, data.weights)
    block_tree((loss, grad))
    compile_ms = elapsed_ms(t0)
    for _ in range(max(0, warmups)):
        loss, grad = compiled(data.volume, data.target, data.weights)
        block_tree((loss, grad))
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        loss, grad = compiled(data.volume, data.target, data.weights)
        block_tree((loss, grad))
        times.append(elapsed_ms(t0))
    finite = bool(jnp.all(jnp.isfinite(loss)) & jnp.all(jnp.isfinite(grad)))
    record = base_record(data, variant)
    record.update(
        {
            "kind": "loss_grad",
            "compile_execute_ms": compile_ms,
            "loss": float(loss),
            "grad_l2": float(jnp.linalg.norm(grad)),
            "passed": finite,
        }
        | timing_record(times)
    )
    return loss, grad, record


def run_fista_macro(
    data: CaseData,
    *,
    repeats: int,
    warmups: int,
    iters_override: int = 0,
    l_override: float = 0.0,
) -> tuple[dict[str, Any], jnp.ndarray]:
    cfg = FistaCoreConfig(
        iters=int(iters_override) if int(iters_override) > 0 else data.spec.fista_iters,
        lambda_tv=0.002,
        L=float(l_override) if float(l_override) > 0.0 else 250.0,
        positivity=True,
        checkpoint_projector=False,
        projector_unroll=1,
        gather_dtype="fp32",
        views_per_batch=data.spec.views_per_batch,
        support=data.support,
        forward_projector="jax",
        backprojector="jax",
        compute_iteration_loss=True,
        compute_final_data_loss=True,
        compute_final_regulariser_value=True,
    )
    x0 = jnp.zeros_like(data.volume)

    def run(volume0: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        result = fista_tv_core_arrays(
            x0=volume0,
            T_all=data.t_all,
            det_grid=data.det_grid,
            projections=data.target,
            grid=data.grid,
            detector=data.detector,
            cfg=cfg,
            view_weights=data.weights,
        )
        return result.x, result.loss, result.data_loss, result.regulariser_value

    compiled = jax.jit(run)
    t0 = time.perf_counter()
    x_result, loss_history, data_loss, regulariser_value = compiled(x0)
    block_tree((x_result, loss_history, data_loss, regulariser_value))
    compile_ms = elapsed_ms(t0)
    for _ in range(max(0, warmups)):
        x_result, loss_history, data_loss, regulariser_value = compiled(x0)
        block_tree((x_result, loss_history, data_loss, regulariser_value))
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        x_result, loss_history, data_loss, regulariser_value = compiled(x0)
        block_tree((x_result, loss_history, data_loss, regulariser_value))
        times.append(elapsed_ms(t0))
    record = base_record(data, "fista_jax_jax")
    finite = bool(
        jnp.all(jnp.isfinite(x_result))
        & jnp.all(jnp.isfinite(loss_history))
        & jnp.all(jnp.isfinite(data_loss))
    )
    loss_start = float(loss_history[0]) if int(loss_history.shape[0]) > 0 else 0.0
    loss_final = float(loss_history[-1]) if int(loss_history.shape[0]) > 0 else 0.0
    loss_drop_ratio = loss_final / max(loss_start, 1e-6) if loss_start > 0.0 else 0.0
    loss_drop_threshold = 0.8 if data.spec.family == "gpu_target" else 1.05
    record.update(
        {
            "kind": "fista_macro",
            "compile_execute_ms": compile_ms,
            "loss_history": [float(v) for v in list(loss_history)],
            "loss_drop_ratio": loss_drop_ratio,
            "loss_drop_threshold": loss_drop_threshold,
            "data_loss": float(data_loss),
            "regulariser_value": float(regulariser_value),
            "passed": finite
            and float(jnp.min(x_result)) >= -1e-6
            and loss_drop_ratio <= loss_drop_threshold,
        }
        | timing_record(times)
    )
    return record, x_result


def directional_derivative_error(data: CaseData, grad: jnp.ndarray) -> float:
    cfg = FistaCoreConfig(
        iters=1,
        lambda_tv=0.0,
        checkpoint_projector=False,
        projector_unroll=1,
        gather_dtype="fp32",
        views_per_batch=data.spec.views_per_batch,
        support=data.support,
    )

    def loss_fn(volume: jnp.ndarray) -> jnp.ndarray:
        return projection_loss_arrays(
            T_all=data.t_all,
            grid=data.grid,
            detector=data.detector,
            volume=volume,
            det_grid=data.det_grid,
            projections=data.target,
            cfg=cfg,
            view_weights=data.weights,
        )

    compiled = jax.jit(loss_fn)
    analytic = jnp.vdot(grad, data.direction).real
    errors = []
    for eps_value in (1.0, 0.3, 0.1, 0.03, 0.01):
        eps = jnp.asarray(eps_value, dtype=jnp.float32)
        plus = compiled(data.volume + eps * data.direction)
        minus = compiled(data.volume - eps * data.direction)
        block_tree((plus, minus))
        fd = (plus - minus) / (2.0 * eps)
        denom = jnp.maximum(
            jnp.maximum(jnp.abs(fd), jnp.abs(analytic)),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        errors.append(float(jnp.abs(fd - analytic) / denom))
    return min(errors)


def adjoint_consistency_error(data: CaseData) -> float:
    probe_volume = data.direction
    y = data.target - jnp.mean(data.target)
    ax = _project_stack(
        T_all=data.t_all,
        grid=data.grid,
        detector=data.detector,
        volume=probe_volume,
        det_grid=data.det_grid,
        checkpoint_projector=False,
        projector_unroll=1,
        gather_dtype="fp32",
        views_per_batch=data.spec.views_per_batch,
        forward_projector="jax",
    )
    aty = sum_backproject_views_T(
        data.t_all,
        data.grid,
        data.detector,
        y,
        unroll=1,
        gather_dtype="fp32",
        det_grid=data.det_grid,
    )
    lhs = jnp.vdot(ax, y).real
    rhs = jnp.vdot(probe_volume, aty).real
    denom = jnp.maximum(
        jnp.maximum(jnp.abs(lhs), jnp.abs(rhs)),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    block_tree((lhs, rhs))
    return float(jnp.abs(lhs - rhs) / denom)


def correctness_record(
    loss: jnp.ndarray,
    grad: jnp.ndarray,
    baseline_loss: jnp.ndarray,
    baseline_grad: jnp.ndarray,
) -> dict[str, Any]:
    loss_abs_error = float(jnp.abs(loss - baseline_loss))
    loss_rel_error = float(loss_abs_error / jnp.maximum(jnp.abs(baseline_loss), 1.0))
    grad_abs_error = float(jnp.linalg.norm(grad - baseline_grad))
    grad_rel_error = float(grad_abs_error / jnp.maximum(jnp.linalg.norm(baseline_grad), 1.0))
    return {
        "loss_abs_error": loss_abs_error,
        "loss_rel_error": loss_rel_error,
        "grad_abs_error": grad_abs_error,
        "grad_rel_error": grad_rel_error,
        "passed": bool(loss_rel_error < 2e-3 and grad_rel_error < 5e-2),
    }


def base_record(data: CaseData, variant: str) -> dict[str, Any]:
    spec = data.spec
    return {
        "case_id": spec.case_id,
        "family": spec.family,
        "variant": variant,
        "volume_shape": list(spec.volume_shape),
        "detector_shape": list(spec.detector_shape),
        "n_views": spec.n_views,
        "views_per_batch": spec.views_per_batch,
        "pallas_eligible": spec.pallas_eligible,
        "use_support": spec.use_support,
        "use_weights": spec.use_weights,
        "detector_roll_deg": spec.detector_roll_deg,
        "detector_center": list(spec.detector_center),
        "voxel_size": list(spec.voxel_size),
        "phantom_profile": spec.phantom_profile,
    }


def skip_record(data: CaseData, variant: str, reason: str) -> dict[str, Any]:
    record = base_record(data, variant)
    record.update(
        {
            "kind": "skipped",
            "passed": False,
            "skip_reason": reason,
        }
    )
    return record


def block_tree(value: Any) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def timing_record(times: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    return {
        "steady_ms": times,
        "steady_ms_p50": statistics.median(times),
        "steady_ms_mean": mean,
        "steady_ms_stdev": stdev,
        "steady_ms_cv": stdev / mean if mean > 0.0 else 0.0,
        "steady_ms_min": min(times),
        "steady_ms_p90": percentile(times, 90.0),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    timed = [r for r in records if "steady_ms_p50" in r]
    passed = [r for r in records if r.get("passed")]
    by_variant: dict[str, dict[str, Any]] = {}
    for variant in sorted({str(r["variant"]) for r in records}):
        rows = [r for r in records if r["variant"] == variant and "steady_ms_p50" in r]
        by_variant[variant] = {
            "count": len(rows),
            "passed": sum(1 for r in rows if r.get("passed")),
            "median_p50_ms": statistics.median([float(r["steady_ms_p50"]) for r in rows])
            if rows
            else None,
            "median_cv": statistics.median([float(r["steady_ms_cv"]) for r in rows])
            if rows
            else None,
            "median_compile_execute_ms": statistics.median(
                [float(r["compile_execute_ms"]) for r in rows]
            )
            if rows
            else None,
        }
    return {
        "record_count": len(records),
        "timed_record_count": len(timed),
        "passed_count": len(passed),
        "failed_count": len(records) - len(passed),
        "by_variant": by_variant,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# TomoJAX FISTA Projection Benchmark: {payload['run_id']}",
        "",
        f"- suite: `{payload['suite']}`",
        f"- seed: `{payload['seed']}`",
        f"- backend: `{payload['environment']['jax_default_backend']}`",
        f"- devices: `{payload['environment']['jax_devices']}`",
        "",
        "## Summary",
        "",
        "| variant | timed | passed | median p50 ms | median cv | median compile+execute ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, row in payload["summary"]["by_variant"].items():
        p50 = "n/a" if row["median_p50_ms"] is None else f"{row['median_p50_ms']:.3f}"
        compile_ms = (
            "n/a"
            if row["median_compile_execute_ms"] is None
            else f"{row['median_compile_execute_ms']:.3f}"
        )
        cv = "n/a" if row["median_cv"] is None else f"{100.0 * row['median_cv']:.1f}%"
        lines.append(
            f"| `{variant}` | {row['count']} | {row['passed']} | {p50} | {cv} | {compile_ms} |"
        )
    lines.extend(["", "## Slowest Timed Records", ""])
    timed = [r for r in payload["records"] if "steady_ms_p50" in r]
    timed.sort(key=lambda r: float(r["steady_ms_p50"]), reverse=True)
    lines.extend(
        [
            "| case | kind | variant | p50 ms | passed |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for row in timed[:10]:
        lines.append(
            f"| `{row['case_id']}` | `{row['kind']}` | `{row['variant']}` | "
            f"{float(row['steady_ms_p50']):.3f} | `{row['passed']}` |"
        )
    if payload["slice_paths"]:
        lines.extend(["", "## Slices", ""])
        for path in payload["slice_paths"]:
            lines.append(f"![{Path(path).name}]({Path(path).name})")
    return "\n".join(lines) + "\n"


def environment_payload() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }


def write_slice_png(
    data: CaseData,
    grad: jnp.ndarray,
    fista_volume: jnp.ndarray,
    path: Path,
) -> None:
    target_view = most_informative_projection_index(data.target)
    z_slice = most_informative_z_slice(data.truth_volume)
    fista_pred = _project_stack(
        T_all=data.t_all,
        grid=data.grid,
        detector=data.detector,
        volume=fista_volume,
        det_grid=data.det_grid,
        checkpoint_projector=False,
        projector_unroll=1,
        gather_dtype="fp32",
        views_per_batch=data.spec.views_per_batch,
        forward_projector="jax",
    )
    fista_residual = fista_pred - data.target
    panels = [
        (f"truth z={z_slice}", np.asarray(data.truth_volume[:, :, z_slice])),
        (f"fista z={z_slice}", np.asarray(fista_volume[:, :, z_slice])),
        (f"candidate grad z={z_slice}", np.asarray(grad[:, :, z_slice])),
        (f"target view={target_view}", np.asarray(data.target[target_view])),
        (f"fista pred view={target_view}", np.asarray(fista_pred[target_view])),
        (f"fista residual view={target_view}", np.asarray(fista_residual[target_view])),
    ]
    try:
        import matplotlib.pyplot as plt
    except Exception:
        write_slice_png_pillow(panels, path, title=data.spec.case_id)
        return
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    for axis, (title, image) in zip(axes.flat, panels, strict=True):
        axis.imshow(image.T if image.ndim == 2 else image, cmap="viridis", origin="lower")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(data.spec.case_id)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_slice_png_pillow(
    panels: list[tuple[str, np.ndarray]],
    path: Path,
    *,
    title: str,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return
    panel_w, panel_h = 220, 190
    label_h = 24
    title_h = 28
    canvas = Image.new("RGB", (panel_w * 3, title_h + panel_h * 2), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    for idx, (label, array) in enumerate(panels):
        row, col = divmod(idx, 3)
        x0 = col * panel_w
        y0 = title_h + row * panel_h
        draw.text((x0 + 6, y0 + 5), label, fill=(0, 0, 0), font=font)
        image = normalize_image(array)
        tile = Image.fromarray(image, mode="L").resize(
            (panel_w, panel_h - label_h),
            resample=Image.Resampling.BILINEAR,
        )
        canvas.paste(Image.merge("RGB", (tile, tile, tile)), (x0, y0 + label_h))
    canvas.save(path)


def most_informative_z_slice(volume: jnp.ndarray) -> int:
    array = np.asarray(volume)
    energy = np.sum(np.abs(array), axis=(0, 1))
    return int(np.argmax(energy))


def most_informative_projection_index(projections: jnp.ndarray) -> int:
    array = np.asarray(projections)
    energy = np.sum(np.abs(array), axis=(1, 2))
    return int(np.argmax(energy))


def normalize_image(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.asarray(np.round(scaled * 255.0), dtype=np.uint8)


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


if __name__ == "__main__":
    main()
