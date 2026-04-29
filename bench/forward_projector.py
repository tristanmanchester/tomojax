#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from tomojax.bench.forward_projector import (
    ForwardProjectorBenchmarkConfig,
    PRESET_NAMES,
    SUITE_NAMES,
    preset_config,
    run_forward_sinogram_suite,
    run_forward_projector_benchmark,
    run_forward_projector_suite,
    write_benchmark_json,
)


def _parse_tile_shape(value: str) -> tuple[int, int]:
    separator = "x" if "x" in value else ","
    parts = value.lower().split(separator)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected TILE_VxTILE_U, for example 8x8")
    try:
        tile_v, tile_u = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tile shape values must be integers") from exc
    if tile_v <= 0 or tile_u <= 0:
        raise argparse.ArgumentTypeError("tile shape values must be positive")
    return tile_v, tile_u


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark single-view forward projection.")
    parser.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        default="smoke",
        help="Named benchmark size. Ignored when --suite is provided.",
    )
    parser.add_argument(
        "--suite",
        choices=SUITE_NAMES,
        help="Run a named benchmark suite instead of one preset.",
    )
    parser.add_argument("--out", help="Optional metrics JSON output path.")
    parser.add_argument("--nx", type=int, help="Override volume x size.")
    parser.add_argument("--ny", type=int, help="Override volume y size.")
    parser.add_argument("--nz", type=int, help="Override volume z size.")
    parser.add_argument("--nu", type=int, help="Override detector u size.")
    parser.add_argument("--nv", type=int, help="Override detector v size.")
    parser.add_argument("--n-steps", type=int, help="Override projector traversal steps.")
    parser.add_argument("--step-size", type=float, help="Override projector traversal step size.")
    parser.add_argument("--warm-runs", type=int, help="Number of warm repeats per backend.")
    parser.add_argument("--seed", type=int, help="Deterministic fixture seed.")
    parser.add_argument("--gather-dtype", default=None, help="Projector gather dtype.")
    parser.add_argument("--unroll", type=int, help="JAX scan unroll.")
    parser.add_argument(
        "--pallas-tile-shape",
        type=_parse_tile_shape,
        help="Requested Pallas detector tile shape as TILE_VxTILE_U, for example 8x8.",
    )
    parser.add_argument("--pallas-num-warps", type=int, help="Requested Pallas num_warps.")
    parser.add_argument("--pallas-kernel-variant", help="Requested Pallas kernel variant.")
    parser.add_argument("--pallas-layout-variant", help="Requested Pallas layout variant.")
    parser.add_argument("--pallas-state-mode", help="Requested Pallas state mode.")
    parser.add_argument(
        "--jax-only",
        action="store_true",
        help="Only run the JAX baseline; skip requested-Pallas fallback/provenance.",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> ForwardProjectorBenchmarkConfig:
    config = preset_config(args.preset)
    updates = {
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
        "nu": args.nu,
        "nv": args.nv,
        "n_steps": args.n_steps,
        "step_size": args.step_size,
        "warm_runs": args.warm_runs,
        "seed": args.seed,
        "gather_dtype": args.gather_dtype,
        "unroll": args.unroll,
        "pallas_tile_shape": args.pallas_tile_shape,
        "pallas_num_warps": args.pallas_num_warps,
        "pallas_kernel_variant": args.pallas_kernel_variant,
        "pallas_layout_variant": args.pallas_layout_variant,
        "pallas_state_mode": args.pallas_state_mode,
    }
    concrete_updates = {key: value for key, value in updates.items() if value is not None}
    if args.jax_only:
        concrete_updates["include_pallas"] = False
    return replace(config, **concrete_updates)


def _suite_overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    dimension_overrides = {
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
        "nu": args.nu,
        "nv": args.nv,
        "n_steps": args.n_steps,
        "step_size": args.step_size,
    }
    if any(value is not None for value in dimension_overrides.values()):
        raise SystemExit("--suite does not accept shape/traversal overrides; use --preset instead")
    updates = {
        "warm_runs": args.warm_runs,
        "seed": args.seed,
        "gather_dtype": args.gather_dtype,
        "unroll": args.unroll,
        "pallas_tile_shape": args.pallas_tile_shape,
        "pallas_num_warps": args.pallas_num_warps,
        "pallas_kernel_variant": args.pallas_kernel_variant,
        "pallas_layout_variant": args.pallas_layout_variant,
        "pallas_state_mode": args.pallas_state_mode,
    }
    concrete_updates: dict[str, object] = {
        key: value for key, value in updates.items() if value is not None
    }
    if args.jax_only:
        concrete_updates["include_pallas"] = False
    return concrete_updates


def main() -> None:
    args = _parse_args()
    metrics = (
        run_forward_sinogram_suite(args.suite, overrides=_suite_overrides_from_args(args))
        if args.suite == "sinogram"
        else run_forward_projector_suite(args.suite, overrides=_suite_overrides_from_args(args))
        if args.suite
        else run_forward_projector_benchmark(_config_from_args(args))
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.out:
        out_path = write_benchmark_json(metrics, Path(args.out))
        print(f"Wrote {out_path}")
    print(text)


if __name__ == "__main__":
    main()
