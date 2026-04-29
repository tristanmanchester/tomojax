#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from tomojax.bench.forward_residual import (
    ForwardResidualBenchmarkConfig,
    RESIDUAL_SUITE_NAMES,
    run_forward_residual_benchmark,
    run_forward_residual_suite,
    write_benchmark_json,
)
from tomojax.bench.forward_projector import PRESET_NAMES, preset_config


def _parse_tile_shape(value: str) -> tuple[int, int]:
    separator = "x" if "x" in value else ","
    parts = value.lower().split(separator)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected TILE_VxTILE_U, for example 8x16")
    try:
        tile_v, tile_u = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tile shape values must be integers") from exc
    if tile_v <= 0 or tile_u <= 0:
        raise argparse.ArgumentTypeError("tile shape values must be positive")
    return tile_v, tile_u


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fused forward residual objectives.")
    parser.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        default="smoke",
        help="Named single-view-size base. Ignored when --suite is provided.",
    )
    parser.add_argument(
        "--suite",
        choices=RESIDUAL_SUITE_NAMES,
        help="Run a named residual suite instead of one preset.",
    )
    parser.add_argument("--out", help="Optional metrics JSON output path.")
    parser.add_argument("--n-views", type=int, help="Override number of views for preset mode.")
    parser.add_argument("--warm-runs", type=int, help="Number of warm repeats per mode.")
    parser.add_argument("--seed", type=int, help="Deterministic fixture seed.")
    parser.add_argument("--target-delta", type=float, help="Maximum synthetic target perturbation.")
    parser.add_argument("--gather-dtype", default=None, help="Projector gather dtype.")
    parser.add_argument("--unroll", type=int, help="JAX scan unroll.")
    parser.add_argument("--pallas-tile-shape", type=_parse_tile_shape)
    parser.add_argument("--pallas-num-warps", type=int)
    parser.add_argument("--pallas-kernel-variant")
    parser.add_argument("--pallas-layout-variant")
    parser.add_argument("--pallas-state-mode")
    parser.add_argument(
        "--jax-only",
        action="store_true",
        help="Only run the JAX baseline; skip requested-Pallas fallback/provenance.",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> ForwardResidualBenchmarkConfig:
    base = preset_config(args.preset)
    config = ForwardResidualBenchmarkConfig(
        nx=base.nx,
        ny=base.ny,
        nz=base.nz,
        nu=base.nu,
        nv=base.nv,
        n_views=args.n_views if args.n_views is not None else 30,
        n_steps=base.n_steps,
        step_size=base.step_size,
        seed=base.seed,
        warm_runs=base.warm_runs,
        gather_dtype=base.gather_dtype,
        unroll=base.unroll,
        use_checkpoint=base.use_checkpoint,
        include_pallas=base.include_pallas,
        pallas_tile_shape=base.pallas_tile_shape,
        pallas_num_warps=base.pallas_num_warps,
        pallas_kernel_variant=base.pallas_kernel_variant,
        pallas_layout_variant=base.pallas_layout_variant,
        pallas_state_mode=base.pallas_state_mode,
    )
    updates = {
        "warm_runs": args.warm_runs,
        "seed": args.seed,
        "target_delta": args.target_delta,
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
    updates = {
        "warm_runs": args.warm_runs,
        "seed": args.seed,
        "target_delta": args.target_delta,
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
    return concrete_updates


def main() -> None:
    args = _parse_args()
    metrics = (
        run_forward_residual_suite(args.suite, overrides=_suite_overrides_from_args(args))
        if args.suite
        else run_forward_residual_benchmark(_config_from_args(args))
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.out:
        out_path = write_benchmark_json(metrics, Path(args.out))
        print(f"Wrote {out_path}")
    print(text)


if __name__ == "__main__":
    main()
