"""Schur normal-equation report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence


class _DiagnosticReport(Protocol):
    def to_dict(self) -> dict[str, object]: ...


class _JointSchurResultReport(Protocol):
    @property
    def initial_loss(self) -> float: ...

    @property
    def final_loss(self) -> float: ...

    @property
    def iterations(self) -> int: ...

    @property
    def active_setup_parameters(self) -> Sequence[str]: ...

    @property
    def active_pose_dofs(self) -> Sequence[str]: ...

    @property
    def frozen_parameters(self) -> Sequence[str]: ...

    @property
    def diagnostics(self) -> _DiagnosticReport: ...

    @property
    def iteration_diagnostics(self) -> Sequence[_DiagnosticReport]: ...


def joint_schur_normal_eq_summary(result: _JointSchurResultReport) -> dict[str, object]:
    """Return the JSON-serializable Schur normal-equation summary artifact."""
    return {
        "solver": "joint_schur_lm_reference",
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "iterations": result.iterations,
        "active_setup_parameters": list(result.active_setup_parameters),
        "active_pose_dofs": list(result.active_pose_dofs),
        "frozen_parameters": list(result.frozen_parameters),
        "diagnostics": result.diagnostics.to_dict(),
        "iteration_diagnostics": [
            diagnostics.to_dict() for diagnostics in result.iteration_diagnostics
        ],
    }


def write_joint_schur_normal_eq_summary(
    result: _JointSchurResultReport,
    path: str | Path,
) -> Path:
    """Write the Schur normal-equation summary artifact as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = output_path.write_text(
        json.dumps(joint_schur_normal_eq_summary(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
