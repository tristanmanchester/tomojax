"""Public-compatible facade for the v2 alternating alignment smoke runner."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tomojax.align._alternating_orchestration import _run_alternating_solver_smoke_impl
from tomojax.align._alternating_types import (
    AlternatingLevelSummary,
    AlternatingSmokeConfig,
    AlternatingSmokeResult,
    GeometryUpdateSolver,
    GeometryUpdateVolumeSource,
    PreviewInitialization,
    PreviewReconstructionMaskSource,
    PreviewResidualFilterMode,
    PreviewVolumeSupport,
    StoppedPreviewPolicy,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class AlternatingAlignmentSolver:
    """Phase 7 alternating alignment solver entrypoint."""

    config: AlternatingSmokeConfig = field(default_factory=AlternatingSmokeConfig)

    def run_smoke(self, output_dir: str | Path) -> AlternatingSmokeResult:
        """Run the deterministic 32^3 smoke profile."""
        return _run_alternating_solver_smoke_impl(output_dir, config=self.config)


def run_alternating_solver_smoke(
    output_dir: str | Path,
    *,
    config: AlternatingSmokeConfig | None = None,
) -> AlternatingSmokeResult:
    """Run the smallest deterministic v2 alternating solver smoke slice."""
    return AlternatingAlignmentSolver(config=config or AlternatingSmokeConfig()).run_smoke(
        output_dir
    )


__all__ = [
    "AlternatingAlignmentSolver",
    "AlternatingLevelSummary",
    "AlternatingSmokeConfig",
    "AlternatingSmokeResult",
    "GeometryUpdateSolver",
    "GeometryUpdateVolumeSource",
    "PreviewInitialization",
    "PreviewReconstructionMaskSource",
    "PreviewResidualFilterMode",
    "PreviewVolumeSupport",
    "StoppedPreviewPolicy",
    "run_alternating_solver_smoke",
]
