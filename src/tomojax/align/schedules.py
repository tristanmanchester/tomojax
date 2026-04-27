from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from .dof_specs import ActiveParameterView
from .objectives import ObjectiveKind


OptimizerKind = Literal["lbfgs", "adam", "gn", "validation_lm"]
GaugePolicy = Literal["reject", "anchor_mean", "prior_required", "diagnose_only"]


@dataclass(frozen=True, slots=True)
class AlignmentStage:
    name: str
    active_dofs: tuple[str, ...]
    objective_kind: ObjectiveKind
    optimizer: OptimizerKind
    gauge_policy: GaugePolicy = "reject"
    maxiter: int = 12
    early_stop: bool = True

    def active_view(self) -> ActiveParameterView:
        return ActiveParameterView.from_dofs(self.active_dofs)


@dataclass(frozen=True, slots=True)
class AlignmentSchedule:
    name: str
    stages: tuple[AlignmentStage, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> "AlignmentSchedule":
        if not self.stages:
            raise ValueError("alignment schedule must contain at least one stage")
        for stage in self.stages:
            if not stage.active_dofs:
                raise ValueError(f"alignment stage {stage.name!r} has no active DOFs")
            stage.active_view()
            if stage.name == "expert_coupled" and stage.gauge_policy == "reject":
                raise ValueError("expert_coupled stages must declare an explicit GaugePolicy")
            if (
                stage.objective_kind in {"bilevel_cv", "all_data_bilevel"}
                and stage.optimizer == "gn"
            ):
                raise ValueError(
                    f"{stage.objective_kind} stages must use validation_lm, lbfgs, or adam "
                    "unless a residual contract is explicitly provided for gn"
                )
        return self

    @property
    def active_dofs(self) -> tuple[str, ...]:
        seen: list[str] = []
        for stage in self.stages:
            for dof in stage.active_dofs:
                if dof not in seen:
                    seen.append(dof)
        return tuple(seen)


def schedule_preset(
    name: str,
    *,
    active_dofs: Iterable[str] | None = None,
    gauge_policy: GaugePolicy | None = None,
) -> AlignmentSchedule:
    key = str(name).strip().lower().replace("-", "_")
    if key == "pose_only":
        return AlignmentSchedule(
            name=key,
            stages=(
                AlignmentStage(
                    name="pose_only",
                    active_dofs=("alpha", "beta", "phi", "dx", "dz"),
                    objective_kind="fixed_volume",
                    optimizer="lbfgs",
                ),
            ),
        ).validate()
    if key == "cor":
        return AlignmentSchedule(
            name=key,
            stages=(
                AlignmentStage(
                    name="cor",
                    active_dofs=("det_u_px",),
                    objective_kind="bilevel_cv",
                    optimizer="validation_lm",
                ),
            ),
        ).validate()
    if key == "detector_center_2d":
        return AlignmentSchedule(
            name=key,
            stages=(
                AlignmentStage(
                    name="detector_center_2d",
                    active_dofs=("det_u_px", "det_v_px"),
                    objective_kind="bilevel_cv",
                    optimizer="validation_lm",
                    gauge_policy="prior_required",
                ),
            ),
        ).validate()
    if key == "detector_roll":
        return AlignmentSchedule(
            name=key,
            stages=(
                AlignmentStage(
                    name="detector_roll",
                    active_dofs=("detector_roll_deg",),
                    objective_kind="bilevel_cv",
                    optimizer="validation_lm",
                ),
            ),
        ).validate()
    if key == "axis_direction":
        return AlignmentSchedule(
            name=key,
            stages=(
                AlignmentStage(
                    name="axis_direction",
                    active_dofs=("axis_rot_x_deg", "axis_rot_y_deg"),
                    objective_kind="bilevel_cv",
                    optimizer="validation_lm",
                    gauge_policy="diagnose_only",
                ),
            ),
        ).validate()
    if key == "lamino_tilt":
        return AlignmentSchedule(
            name=key,
            stages=(
                AlignmentStage(
                    name="lamino_tilt",
                    active_dofs=("axis_rot_x_deg",),
                    objective_kind="bilevel_cv",
                    optimizer="validation_lm",
                    gauge_policy="diagnose_only",
                ),
            ),
        ).validate()
    if key == "setup_safe":
        return AlignmentSchedule(
            name=key,
            stages=(
                AlignmentStage("cor", ("det_u_px",), "bilevel_cv", "validation_lm"),
                AlignmentStage(
                    "detector_roll",
                    ("detector_roll_deg",),
                    "bilevel_cv",
                    "validation_lm",
                ),
                AlignmentStage(
                    "axis_direction",
                    ("axis_rot_x_deg", "axis_rot_y_deg"),
                    "bilevel_cv",
                    "validation_lm",
                    gauge_policy="diagnose_only",
                ),
                AlignmentStage(
                    "pose_polish",
                    ("alpha", "beta", "phi", "dx", "dz"),
                    "fixed_volume",
                    "gn",
                    gauge_policy="anchor_mean",
                ),
            ),
        ).validate()
    if key == "expert_coupled":
        dofs = tuple(active_dofs or ())
        if not dofs:
            raise ValueError("expert_coupled requires explicit active_dofs")
        return AlignmentSchedule(
            name=key,
            stages=(
                AlignmentStage(
                    name="expert_coupled",
                    active_dofs=dofs,
                    objective_kind="bilevel_cv",
                    optimizer="validation_lm",
                    gauge_policy=gauge_policy or "prior_required",
                ),
            ),
        ).validate()
    raise ValueError(f"Unknown alignment schedule preset {name!r}")
