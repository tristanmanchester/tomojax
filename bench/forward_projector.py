#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from tomojax.bench.forward_projector import (
    ForwardProjectorBenchmarkConfig,
    preset_config,
    run_forward_projector_benchmark,
    write_benchmark_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark single-view forward projection.")
    parser.add_argument(
        "--preset",
        choices=["tiny", "smoke", "profile-128"],
        default="smoke",
        help="Named benchmark size.",
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
    }
    concrete_updates = {key: value for key, value in updates.items() if value is not None}
    if args.jax_only:
        concrete_updates["include_pallas"] = False
    return replace(config, **concrete_updates)


def main() -> None:
    args = _parse_args()
    metrics = run_forward_projector_benchmark(_config_from_args(args))
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.out:
        out_path = write_benchmark_json(metrics, Path(args.out))
        print(f"Wrote {out_path}")
    print(text)


if __name__ == "__main__":
    main()
