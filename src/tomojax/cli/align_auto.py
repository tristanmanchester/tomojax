"""CLI for the Phase 7 deterministic auto-alignment smoke pipeline."""
# pyright: reportAny=false

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tomojax.align.api import (
    AlternatingAlignmentSolver,
    AlternatingSmokeConfig,
    ContinuationScheduleName,
    reference_continuation_schedule,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_PROFILE_CHOICES = ("smoke32", "lightning", "balanced", "reference")


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
    _ = parser.add_argument("--size", type=int, default=32, help="Synthetic cubic volume size.")
    _ = parser.add_argument("--views", type=int, default=4, help="Number of synthetic views.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic Phase 7 auto-alignment smoke command."""
    args = _build_parser().parse_args(argv)
    profile = cast("ContinuationScheduleName", args.profile)
    solver = AlternatingAlignmentSolver(
        AlternatingSmokeConfig(
            seed=int(args.seed),
            size=int(args.size),
            n_views=int(args.views),
            schedule=reference_continuation_schedule(profile),
        )
    )
    result = solver.run_smoke(Path(args.out_dir))
    print(f"verification: {result.artifacts['verification_json']}")
    print(f"geometry: {result.artifacts['geometry_final_json']}")
    print(f"volume: {result.artifacts['final_volume_npy']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
