"""CLI for the Phase 7 deterministic auto-alignment smoke pipeline."""
# pyright: reportAny=false

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from tomojax.align.api import (
    AlternatingAlignmentSolver,
    AlternatingSmokeConfig,
    ContinuationScheduleName,
    reference_continuation_schedule,
)
from tomojax.datasets import generate_synthetic_dataset, synthetic128_spec

if TYPE_CHECKING:
    from collections.abc import Sequence

_PROFILE_CHOICES = ("smoke32", "lightning", "balanced", "reference")
_SYNTHETIC_SIZE_CHOICES = (32, 128)
SyntheticSize = Literal[32, 128]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic Phase 7 align=auto smoke pipeline and write "
            "final volume, geometry, and verification artifacts."
        )
    )
    _ = parser.add_argument(
        "--out-dir",
        required=True,
        help="Run directory for smoke artifacts.",
    )
    _ = parser.add_argument(
        "--profile",
        choices=_PROFILE_CHOICES,
        default="smoke32",
        help="Continuation profile for the deterministic smoke run.",
    )
    _ = parser.add_argument("--seed", type=int, default=17, help="Synthetic phantom seed.")
    _ = parser.add_argument(
        "--size",
        type=int,
        choices=_SYNTHETIC_SIZE_CHOICES,
        default=32,
        help="Synthetic cubic volume size.",
    )
    _ = parser.add_argument("--views", type=int, default=4, help="Number of synthetic views.")
    _ = parser.add_argument(
        "--synthetic-dataset",
        help="Optional synthetic128 benchmark spec name to generate and record for this run.",
    )
    _ = parser.add_argument(
        "--dataset-out-dir",
        help="Directory for generated synthetic benchmark artifacts. Defaults under --out-dir.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic Phase 7 auto-alignment smoke command."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    profile = cast("ContinuationScheduleName", args.profile)
    size = cast("SyntheticSize", int(args.size))
    out_dir = Path(args.out_dir)
    dataset_name = None if args.synthetic_dataset is None else str(args.synthetic_dataset)
    dataset_dir: Path | None = None
    if dataset_name is not None:
        _ = synthetic128_spec(dataset_name)
        dataset_root = Path(args.dataset_out_dir) if args.dataset_out_dir else out_dir / "datasets"
        dataset_paths = generate_synthetic_dataset(
            dataset_name,
            dataset_root,
            size=size,
            clean=True,
            views=int(args.views),
        )
        dataset_dir = dataset_paths.dataset_dir
    solver = AlternatingAlignmentSolver(
        AlternatingSmokeConfig(
            seed=int(args.seed),
            size=size,
            n_views=int(args.views),
            schedule=reference_continuation_schedule(profile),
            synthetic_dataset_name=dataset_name,
            synthetic_dataset_artifact_dir=dataset_dir,
        )
    )
    result = solver.run_smoke(out_dir)
    if dataset_dir is not None:
        print(f"synthetic_dataset: {dataset_dir}")
    print(f"verification: {result.artifacts['verification_json']}")
    print(f"geometry: {result.artifacts['geometry_final_json']}")
    print(f"volume: {result.artifacts['final_volume_npy']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
