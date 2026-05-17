"""Developer-facing alignment diagnostic runners.

This module owns synthetic alignment benchmark harnesses and keeps them behind
the benchmark/developer surface instead of the production alignment facade.
"""

from __future__ import annotations

# check-public-imports: allow-private
from tomojax.align._alternating import (
    AlternatingAlignmentSolver,
    AlternatingLevelSummary,
    AlternatingSmokeConfig,
    AlternatingSmokeResult,
    run_alternating_solver_smoke,
)
from tomojax.bench._alignment_smoke_core import AlignmentSmokeReport, run_alignment_smoke

__all__ = [
    "AlignmentSmokeReport",
    "AlternatingAlignmentSolver",
    "AlternatingLevelSummary",
    "AlternatingSmokeConfig",
    "AlternatingSmokeResult",
    "run_alignment_smoke",
    "run_alternating_solver_smoke",
]
