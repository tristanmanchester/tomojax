"""Developer-facing alignment smoke diagnostics.

This module is intentionally separate from :mod:`tomojax.align.api`: the
objects here run synthetic smoke/benchmark harnesses and are useful for
verification, but they are not part of the production alignment facade.
"""

from __future__ import annotations

from tomojax.align._alternating import (
    AlternatingAlignmentSolver,
    AlternatingLevelSummary,
    AlternatingSmokeConfig,
    AlternatingSmokeResult,
    run_alternating_solver_smoke,
)
from tomojax.align._smoke import AlignmentSmokeReport, run_alignment_smoke

__all__ = [
    "AlignmentSmokeReport",
    "AlternatingAlignmentSolver",
    "AlternatingLevelSummary",
    "AlternatingSmokeConfig",
    "AlternatingSmokeResult",
    "run_alignment_smoke",
    "run_alternating_solver_smoke",
]
