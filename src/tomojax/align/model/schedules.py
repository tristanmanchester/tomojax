from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, cast

from .dof_specs import ActiveParameterView
from .diagnostics import GaugeDecision, GaugePolicyError, validate_active_gauge_policy
from .dofs import (
    DOF_NAMES,
    GEOMETRY_DOF_NAMES,
    ScopedAlignmentDofs,
    normalize_alignment_dofs,
    resolve_scoped_alignment_dofs,
)


ObjectiveKind = Literal["fixed_volume", "bilevel_cv", "all_data_bilevel"]
OptimizerKind = Literal["lbfgs", "adam", "gd", "gn", "validation_lm"]
GaugePolicy = Literal["reject", "anchor_mean", "prior_required", "diagnose_only"]
ScheduleSource = Literal["preset", "direct", "default", "expert"]


PUBLIC_SCHEDULE_PRESETS = (
    "pose_only",
    "pose_phi_only",
    "pose_dx_dz_after_phi",
    "cor",
    "detector_roll",
    "axis_direction",
    "lamino_tilt",
    "setup_safe",
)


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

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "stages": [
                {
                    "name": stage.name,
                    "active_dofs": list(stage.active_dofs),
                    "objective_kind": stage.objective_kind,
                    "optimizer_kind": stage.optimizer,
                    "gauge_policy": stage.gauge_policy,
                    "maxiter": int(stage.maxiter),
                    "early_stop": bool(stage.early_stop),
                }
                for stage in self.stages
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ResolvedAlignmentStage:
    index: int
    name: str
    active_dofs: tuple[str, ...]
    active_pose_dofs: tuple[str, ...]
    active_geometry_dofs: tuple[str, ...]
    objective_kind: ObjectiveKind
    optimizer_kind: OptimizerKind
    gauge_policy: GaugePolicy
    gauge_decision: GaugeDecision
    maxiter: int
    early_stop: bool

    @property
    def scoped_dofs(self) -> ScopedAlignmentDofs:
        return ScopedAlignmentDofs(
            active_pose_dofs=self.active_pose_dofs,
            active_geometry_dofs=self.active_geometry_dofs,
            frozen_pose_dofs=(),
            frozen_geometry_dofs=(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_index": int(self.index),
            "stage_name": self.name,
            "active_dofs": list(self.active_dofs),
            "active_pose_dofs": list(self.active_pose_dofs),
            "active_geometry_dofs": list(self.active_geometry_dofs),
            "objective_kind": self.objective_kind,
            "optimizer_kind": self.optimizer_kind,
            "gauge_policy": self.gauge_policy,
            "gauge_decision": self.gauge_decision.to_dict(),
            "maxiter": int(self.maxiter),
            "early_stop": bool(self.early_stop),
        }


@dataclass(frozen=True, slots=True)
class ResolvedAlignmentSchedule:
    name: str
    source: ScheduleSource
    stages: tuple[ResolvedAlignmentStage, ...]
    requested_active_dofs: tuple[str, ...]
    frozen_dofs: tuple[str, ...]
    gauge_decision: GaugeDecision
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def active_dofs(self) -> tuple[str, ...]:
        seen: list[str] = []
        for stage in self.stages:
            for dof in stage.active_dofs:
                if dof not in seen:
                    seen.append(dof)
        return tuple(seen)

    @property
    def active_pose_dofs(self) -> tuple[str, ...]:
        active = set(self.active_dofs)
        return tuple(name for name in DOF_NAMES if name in active)

    @property
    def active_geometry_dofs(self) -> tuple[str, ...]:
        active = set(self.active_dofs)
        return tuple(name for name in GEOMETRY_DOF_NAMES if name in active)

    @property
    def pose_mask(self) -> tuple[bool, bool, bool, bool, bool]:
        active = set(self.active_pose_dofs)
        return tuple(name in active for name in DOF_NAMES)  # type: ignore[return-value]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "active_dofs": list(self.active_dofs),
            "active_pose_dofs": list(self.active_pose_dofs),
            "active_geometry_dofs": list(self.active_geometry_dofs),
            "requested_active_dofs": list(self.requested_active_dofs),
            "frozen_dofs": list(self.frozen_dofs),
            "gauge_decision": self.gauge_decision.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
            "metadata": dict(self.metadata),
        }


_PUBLIC_PRESET_STAGES: dict[str, tuple[AlignmentStage, ...]] = {
    "pose_only": (
        AlignmentStage(
            name="pose_only",
            active_dofs=("alpha", "beta", "phi", "dx", "dz"),
            objective_kind="fixed_volume",
            optimizer="gn",
        ),
    ),
    "pose_phi_only": (
        AlignmentStage(
            name="pose_phi_only",
            active_dofs=("phi",),
            objective_kind="fixed_volume",
            optimizer="gn",
            gauge_policy="anchor_mean",
        ),
    ),
    "pose_dx_dz_after_phi": (
        AlignmentStage(
            name="pose_dx_dz_after_phi",
            active_dofs=("dx", "dz"),
            objective_kind="fixed_volume",
            optimizer="gn",
            gauge_policy="anchor_mean",
        ),
    ),
    "cor": (
        AlignmentStage(
            name="cor",
            active_dofs=("det_u_px",),
            objective_kind="bilevel_cv",
            optimizer="validation_lm",
        ),
    ),
    "detector_roll": (
        AlignmentStage(
            name="detector_roll",
            active_dofs=("detector_roll_deg",),
            objective_kind="bilevel_cv",
            optimizer="validation_lm",
        ),
    ),
    "axis_direction": (
        AlignmentStage(
            name="axis_direction",
            active_dofs=("axis_rot_x_deg", "axis_rot_y_deg"),
            objective_kind="bilevel_cv",
            optimizer="validation_lm",
            gauge_policy="diagnose_only",
        ),
    ),
    "lamino_tilt": (
        AlignmentStage(
            name="lamino_tilt",
            active_dofs=("axis_rot_x_deg",),
            objective_kind="bilevel_cv",
            optimizer="validation_lm",
            gauge_policy="diagnose_only",
        ),
    ),
    "setup_safe": (
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
}


def schedule_preset(
    name: str,
    *,
    active_dofs: Iterable[str] | None = None,
    gauge_policy: GaugePolicy | None = None,
) -> AlignmentSchedule:
    key = str(name).strip().lower().replace("-", "_")
    if key == "detector_center_2d":
        raise ValueError(
            "Unknown alignment schedule preset 'detector_center_2d'. "
            "Use low-level optimise_dofs=('det_u_px', 'det_v_px') only for expert diagnostics."
        )
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
    stages = _PUBLIC_PRESET_STAGES.get(key)
    if stages is not None:
        return AlignmentSchedule(name=key, stages=stages).validate()
    raise ValueError(f"Unknown alignment schedule preset {name!r}")


def resolve_alignment_schedule(
    *,
    schedule: str | AlignmentSchedule | None = None,
    optimise_dofs: str | Iterable[str] | None = None,
    freeze_dofs: str | Iterable[str] | None = None,
    geometry_dofs: str | Iterable[str] | None = None,
    geometry: object | None = None,
    gauge_policy: GaugePolicy = "reject",
    gauge_priors: Mapping[str, object] | None = None,
    opt_method: str = "gn",
    outer_iters: int | None = None,
    early_stop: bool | None = None,
) -> ResolvedAlignmentSchedule:
    """Resolve public schedule/DOF inputs into executable staged alignment work."""
    frozen = normalize_alignment_dofs(
        freeze_dofs,
        option_name="freeze_dofs",
        geometry=geometry,
    )
    if schedule is not None and (optimise_dofs is not None or geometry_dofs):
        raise ValueError("schedule and explicit optimise_dofs/geometry_dofs are mutually exclusive")

    source: ScheduleSource
    if isinstance(schedule, AlignmentSchedule):
        base_schedule = schedule.validate()
        source = "preset" if base_schedule.name in PUBLIC_SCHEDULE_PRESETS else "expert"
    elif schedule is not None:
        base_schedule = schedule_preset(str(schedule))
        source = "preset"
    else:
        scoped = resolve_scoped_alignment_dofs(
            optimise_dofs=optimise_dofs,
            freeze_dofs=frozen,
            geometry_dofs=geometry_dofs,
            geometry=geometry,
        )
        if optimise_dofs is None:
            source = "default"
            name = "default_pose"
        else:
            source = "direct"
            name = "direct"
        if scoped.active_geometry_dofs and scoped.active_pose_dofs:
            # Direct mixed setup/pose requests are only an expert surface, and
            # still execute as explicit setup then pose stages so product setup
            # discovery remains validation-LM and pose remains fixed-volume.
            if gauge_policy == "reject":
                raise GaugePolicyError(
                    "Direct mixed setup+pose DOFs require an explicit expert gauge_policy"
                )
            validate_active_gauge_policy(
                scoped.active_dofs,
                policy=gauge_policy,
                priors=gauge_priors,
            )
            stages = (
                AlignmentStage(
                    name="direct_setup",
                    active_dofs=scoped.active_geometry_dofs,
                    objective_kind="bilevel_cv",
                    optimizer="validation_lm",
                    gauge_policy=gauge_policy,
                ),
                AlignmentStage(
                    name="direct_pose",
                    active_dofs=scoped.active_pose_dofs,
                    objective_kind="fixed_volume",
                    optimizer=_normalize_pose_optimizer(opt_method),
                    gauge_policy=gauge_policy,
                ),
            )
        elif scoped.active_geometry_dofs:
            stages = (
                AlignmentStage(
                    name="direct_setup",
                    active_dofs=scoped.active_geometry_dofs,
                    objective_kind="bilevel_cv",
                    optimizer="validation_lm",
                    gauge_policy=gauge_policy,
                ),
            )
        else:
            stages = (
                AlignmentStage(
                    name="direct_pose",
                    active_dofs=scoped.active_pose_dofs,
                    objective_kind="fixed_volume",
                    optimizer=_normalize_pose_optimizer(opt_method),
                    gauge_policy=gauge_policy,
                ),
            )
        base_schedule = AlignmentSchedule(name=name, stages=stages).validate()

    requested = normalize_alignment_dofs(
        base_schedule.active_dofs,
        option_name="schedule active_dofs",
        geometry=geometry,
    )
    top_level_decision = validate_active_gauge_policy(
        requested,
        policy=gauge_policy if source == "direct" else "diagnose_only",
        priors=gauge_priors,
    )

    resolved_stages: list[ResolvedAlignmentStage] = []
    frozen_set = set(frozen)
    for stage in base_schedule.stages:
        active = tuple(
            name
            for name in normalize_alignment_dofs(
                stage.active_dofs,
                option_name=f"schedule stage {stage.name} active_dofs",
                geometry=geometry,
            )
            if name not in frozen_set
        )
        if not active:
            continue
        scoped_stage = resolve_scoped_alignment_dofs(
            optimise_dofs=active,
            freeze_dofs=(),
            geometry=geometry,
        )
        decision = validate_active_gauge_policy(
            active,
            policy=stage.gauge_policy,
            priors=gauge_priors,
        )
        maxiter = int(outer_iters) if outer_iters is not None else int(stage.maxiter)
        if maxiter < 0:
            raise ValueError("alignment stage maxiter must be >= 0")
        resolved_stages.append(
            ResolvedAlignmentStage(
                index=len(resolved_stages),
                name=stage.name,
                active_dofs=scoped_stage.active_dofs,
                active_pose_dofs=scoped_stage.active_pose_dofs,
                active_geometry_dofs=scoped_stage.active_geometry_dofs,
                objective_kind=stage.objective_kind,
                optimizer_kind=stage.optimizer,
                gauge_policy=stage.gauge_policy,
                gauge_decision=decision,
                maxiter=maxiter,
                early_stop=bool(stage.early_stop if early_stop is None else early_stop),
            )
        )
    if not resolved_stages:
        raise ValueError(
            "No active alignment stages remain after applying freeze_dofs; "
            "at least one schedule stage must have an active DOF"
        )

    return ResolvedAlignmentSchedule(
        name=base_schedule.name,
        source=source,
        stages=tuple(resolved_stages),
        requested_active_dofs=requested,
        frozen_dofs=frozen,
        gauge_decision=top_level_decision,
        metadata=dict(base_schedule.metadata),
    )


def _normalize_pose_optimizer(value: str) -> OptimizerKind:
    key = str(value).strip().lower().replace("-", "_")
    if key in {"lbfgsb", "l_bfgs", "l_bfgs_b"}:
        key = "lbfgs"
    if key not in {"gd", "gn", "lbfgs"}:
        raise ValueError("pose optimizer must be one of 'gd', 'gn', or 'lbfgs'")
    return cast(OptimizerKind, key)
