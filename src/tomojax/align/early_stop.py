from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal, Mapping, TypeAlias, cast

from ._observer import OuterStat


EarlyStopProfile: TypeAlias = Literal["compute_saving", "robust", "off"]
EarlyStopStageKind: TypeAlias = Literal["setup", "pose"]


@dataclass(frozen=True, slots=True)
class EarlyStopPolicy:
    """Resolved thresholds for one alignment early-stop profile."""

    profile: EarlyStopProfile = "compute_saving"
    enabled: bool = True
    rel_impr_threshold: float = 1e-3
    patience: int = 2
    min_outer_iters: int = 2
    setup_step_threshold: float = 2.5e-3
    pose_rot_step_threshold: float = 1e-6
    pose_trans_step_threshold: float = 1e-4
    tiny_scale_threshold: float = 1e-6
    max_condition_number: float = 1e12
    stop_on_rejected: bool = True
    stop_on_unhealthy_optimizer: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EarlyStopEvidence:
    """Normalized evidence from one setup or pose outer iteration."""

    stage_kind: EarlyStopStageKind
    active_dofs: tuple[str, ...] = ()
    rel_impr: float | None = None
    accepted_rel_impr: float | None = None
    loss_before: float | None = None
    loss_after: float | None = None
    accepted: bool | None = None
    step_norm: float | None = None
    rot_step: float | None = None
    trans_step: float | None = None
    selected_scale: float | None = None
    condition_number: float | None = None
    loss_drift_from_prev_after: float | None = None


@dataclass(frozen=True, slots=True)
class EarlyStopState:
    """Serializable state carried across outer iterations within one stage."""

    gain_streak: int = 0
    step_streak: int = 0
    rejected_streak: int = 0
    unhealthy_streak: int = 0
    nonfinite_streak: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "EarlyStopState":
        if not value:
            return cls()
        return cls(
            gain_streak=_coerce_int(value.get("gain_streak")),
            step_streak=_coerce_int(value.get("step_streak")),
            rejected_streak=_coerce_int(value.get("rejected_streak")),
            unhealthy_streak=_coerce_int(value.get("unhealthy_streak")),
            nonfinite_streak=_coerce_int(value.get("nonfinite_streak")),
        )

    @classmethod
    def from_resume(
        cls,
        value: Mapping[str, object] | None,
        *,
        legacy_gain_streak: int = 0,
    ) -> "EarlyStopState":
        """Restore early-stop state, preserving pre-structured checkpoint streaks."""

        if value:
            return cls.from_mapping(value)
        return cls(gain_streak=max(0, int(legacy_gain_streak)))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EarlyStopDecision:
    """Decision and updated state for one early-stop evaluation."""

    should_stop: bool
    reason: str
    state: EarlyStopState
    policy: EarlyStopPolicy
    evidence: EarlyStopEvidence
    gain_is_small: bool | None
    step_is_small: bool | None
    optimizer_is_unhealthy: bool
    loss_is_nonfinite: bool

    def telemetry(self) -> dict[str, object]:
        evidence = self.evidence
        return {
            "early_stop_profile": self.policy.profile,
            "early_stop_enabled": bool(self.policy.enabled),
            "early_stop_decision": "stop" if self.should_stop else "continue",
            "early_stop_reason": self.reason,
            "early_stop_gain_streak": int(self.state.gain_streak),
            "early_stop_step_streak": int(self.state.step_streak),
            "early_stop_rejected_streak": int(self.state.rejected_streak),
            "early_stop_unhealthy_streak": int(self.state.unhealthy_streak),
            "early_stop_nonfinite_streak": int(self.state.nonfinite_streak),
            "early_stop_rel_threshold": float(self.policy.rel_impr_threshold),
            "early_stop_patience": int(self.policy.patience),
            "early_stop_min_outer_iters": int(self.policy.min_outer_iters),
            "early_stop_gain_is_small": self.gain_is_small,
            "early_stop_step_is_small": self.step_is_small,
            "early_stop_optimizer_unhealthy": bool(self.optimizer_is_unhealthy),
            "early_stop_loss_nonfinite": bool(self.loss_is_nonfinite),
            "accepted_rel_impr": evidence.accepted_rel_impr,
            "loss_drift_from_prev_after": evidence.loss_drift_from_prev_after,
        }


def normalize_early_stop_profile(value: object) -> EarlyStopProfile:
    raw = str(value if value is not None else "compute_saving").strip().lower().replace("-", "_")
    if raw in {"default", "compute", "compute_saving", "compute_saver"}:
        return "compute_saving"
    if raw in {"robust", "conservative"}:
        return "robust"
    if raw in {"off", "disabled", "none", "false"}:
        return "off"
    raise ValueError("early_stop_profile must be one of 'compute_saving', 'robust', or 'off'")


def resolve_early_stop_policy(
    *,
    enabled: bool,
    profile: object = "compute_saving",
    rel_impr_threshold: float = 1e-3,
    patience: int = 2,
) -> EarlyStopPolicy:
    normalized = normalize_early_stop_profile(profile)
    base_patience = max(1, int(patience))
    rel = max(0.0, float(rel_impr_threshold))
    if (not enabled) or normalized == "off":
        return EarlyStopPolicy(
            profile=normalized,
            enabled=False,
            rel_impr_threshold=rel,
            patience=base_patience,
        )
    if normalized == "robust":
        return EarlyStopPolicy(
            profile="robust",
            enabled=True,
            rel_impr_threshold=rel,
            patience=max(base_patience, 4),
            min_outer_iters=max(2, min(3, max(base_patience, 4))),
            setup_step_threshold=1e-3,
            pose_rot_step_threshold=5e-7,
            pose_trans_step_threshold=5e-5,
            tiny_scale_threshold=1e-8,
            max_condition_number=1e14,
        )
    return EarlyStopPolicy(
        profile="compute_saving",
        enabled=True,
        rel_impr_threshold=rel,
        patience=base_patience,
        min_outer_iters=min(2, base_patience),
        setup_step_threshold=2.5e-3,
        pose_rot_step_threshold=1e-6,
        pose_trans_step_threshold=1e-4,
        tiny_scale_threshold=1e-6,
        max_condition_number=1e12,
    )


def evaluate_early_stop(
    *,
    evidence: EarlyStopEvidence,
    policy: EarlyStopPolicy,
    state: EarlyStopState | Mapping[str, object] | None,
    outer_idx: int,
) -> EarlyStopDecision:
    current = state if isinstance(state, EarlyStopState) else EarlyStopState.from_mapping(state)
    if not policy.enabled:
        return _decision(
            False,
            "disabled",
            EarlyStopState(),
            policy,
            evidence,
            gain_is_small=None,
            step_is_small=None,
            optimizer_is_unhealthy=False,
            loss_is_nonfinite=False,
        )

    gain_value = evidence.accepted_rel_impr
    if gain_value is None:
        gain_value = evidence.rel_impr
    loss_is_nonfinite = _is_nonfinite(gain_value) or _is_nonfinite(evidence.loss_after)
    gain_is_small = None if gain_value is None or loss_is_nonfinite else gain_value < policy.rel_impr_threshold
    step_is_small = _step_is_small(evidence, policy)
    rejected = evidence.accepted is False
    optimizer_is_unhealthy = _optimizer_is_unhealthy(evidence, policy)

    next_state = EarlyStopState(
        gain_streak=(current.gain_streak + 1 if gain_is_small is True else 0),
        step_streak=(current.step_streak + 1 if step_is_small is True else 0),
        rejected_streak=(current.rejected_streak + 1 if rejected else 0),
        unhealthy_streak=(current.unhealthy_streak + 1 if optimizer_is_unhealthy else 0),
        nonfinite_streak=(current.nonfinite_streak + 1 if loss_is_nonfinite else 0),
    )

    can_stop = int(outer_idx) >= int(policy.min_outer_iters)
    reason = "warmup" if not can_stop else "continue"
    should_stop = False
    if can_stop and next_state.nonfinite_streak >= policy.patience:
        should_stop = True
        reason = "nonfinite_loss_or_gain"
    elif (
        can_stop
        and policy.stop_on_rejected
        and next_state.rejected_streak >= policy.patience
    ):
        should_stop = True
        reason = "rejected_updates"
    elif (
        can_stop
        and policy.stop_on_unhealthy_optimizer
        and next_state.unhealthy_streak >= policy.patience
    ):
        should_stop = True
        reason = "unhealthy_optimizer"
    elif (
        can_stop
        and next_state.gain_streak >= policy.patience
        and next_state.step_streak >= policy.patience
    ):
        should_stop = True
        reason = "small_gain_and_step"

    return _decision(
        should_stop,
        reason,
        next_state,
        policy,
        evidence,
        gain_is_small=gain_is_small,
        step_is_small=step_is_small,
        optimizer_is_unhealthy=optimizer_is_unhealthy,
        loss_is_nonfinite=loss_is_nonfinite,
    )


def setup_evidence_from_stat(
    stat: Mapping[str, object],
    *,
    active_dofs: tuple[str, ...] = (),
    prev_loss_after: float | None = None,
) -> EarlyStopEvidence:
    loss_before = _coerce_float(stat.get("geometry_loss_before"))
    loss_after = _coerce_float(stat.get("geometry_loss_after"))
    accepted = _coerce_bool(stat.get("geometry_accepted"))
    accepted_rel = _relative_improvement(loss_before, loss_after) if accepted else 0.0
    drift = None
    if prev_loss_after is not None and loss_after is not None and _finite(prev_loss_after):
        drift = (float(prev_loss_after) - float(loss_after)) / max(abs(float(prev_loss_after)), 1e-12)
    return EarlyStopEvidence(
        stage_kind="setup",
        active_dofs=active_dofs,
        rel_impr=accepted_rel,
        accepted_rel_impr=accepted_rel,
        loss_before=loss_before,
        loss_after=loss_after,
        accepted=accepted,
        step_norm=_coerce_float(
            stat.get("geometry_step_norm", stat.get("step_norm_whitened"))
        ),
        selected_scale=_coerce_float(stat.get("optimizer_selected_scale")),
        condition_number=_coerce_float(stat.get("optimizer_condition_number")),
        loss_drift_from_prev_after=drift,
    )


def pose_evidence_from_stat(
    stat: Mapping[str, object],
    *,
    active_dofs: tuple[str, ...] = (),
) -> EarlyStopEvidence:
    accepted = _pose_accepted(stat)
    return EarlyStopEvidence(
        stage_kind="pose",
        active_dofs=active_dofs,
        rel_impr=_coerce_float(stat.get("rel_impr")),
        loss_before=_coerce_float(stat.get("loss_before")),
        loss_after=_coerce_float(stat.get("loss_after")),
        accepted=accepted,
        rot_step=_first_float(stat, ("rot_rms", "rot_mean")),
        trans_step=_first_float(stat, ("trans_rms", "trans_mean")),
    )


def annotate_stat_with_early_stop(stat: OuterStat, decision: EarlyStopDecision) -> None:
    stat.update(cast(dict[str, object], decision.telemetry()))


def _decision(
    should_stop: bool,
    reason: str,
    state: EarlyStopState,
    policy: EarlyStopPolicy,
    evidence: EarlyStopEvidence,
    *,
    gain_is_small: bool | None,
    step_is_small: bool | None,
    optimizer_is_unhealthy: bool,
    loss_is_nonfinite: bool,
) -> EarlyStopDecision:
    return EarlyStopDecision(
        should_stop=bool(should_stop),
        reason=str(reason),
        state=state,
        policy=policy,
        evidence=evidence,
        gain_is_small=gain_is_small,
        step_is_small=step_is_small,
        optimizer_is_unhealthy=bool(optimizer_is_unhealthy),
        loss_is_nonfinite=bool(loss_is_nonfinite),
    )


def _step_is_small(evidence: EarlyStopEvidence, policy: EarlyStopPolicy) -> bool | None:
    if evidence.stage_kind == "setup":
        if evidence.step_norm is None or not _finite(evidence.step_norm):
            return None
        return abs(float(evidence.step_norm)) <= policy.setup_step_threshold

    active = set(evidence.active_dofs)
    checks: list[bool] = []
    if active.intersection({"alpha", "beta", "phi"}):
        if evidence.rot_step is not None and _finite(evidence.rot_step):
            checks.append(abs(float(evidence.rot_step)) <= policy.pose_rot_step_threshold)
    if active.intersection({"dx", "dz"}):
        if evidence.trans_step is not None and _finite(evidence.trans_step):
            checks.append(abs(float(evidence.trans_step)) <= policy.pose_trans_step_threshold)
    if not active:
        if evidence.rot_step is not None and _finite(evidence.rot_step):
            checks.append(abs(float(evidence.rot_step)) <= policy.pose_rot_step_threshold)
        if evidence.trans_step is not None and _finite(evidence.trans_step):
            checks.append(abs(float(evidence.trans_step)) <= policy.pose_trans_step_threshold)
    if not checks:
        return True if evidence.accepted is False else None
    return all(checks)


def _optimizer_is_unhealthy(evidence: EarlyStopEvidence, policy: EarlyStopPolicy) -> bool:
    if evidence.selected_scale is not None and _finite(evidence.selected_scale):
        if abs(float(evidence.selected_scale)) <= policy.tiny_scale_threshold:
            return True
    if evidence.condition_number is not None:
        if (not _finite(evidence.condition_number)) or (
            float(evidence.condition_number) >= policy.max_condition_number
        ):
            return True
    return False


def _pose_accepted(stat: Mapping[str, object]) -> bool | None:
    if "optimizer_accepted" in stat:
        return _coerce_bool(stat.get("optimizer_accepted"))
    if "lbfgs_accepted" in stat:
        return _coerce_bool(stat.get("lbfgs_accepted"))
    loss_before = _coerce_float(stat.get("loss_before"))
    loss_after = _coerce_float(stat.get("loss_after"))
    if loss_before is None or loss_after is None:
        return None
    return bool(loss_after <= loss_before)


def _relative_improvement(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or not (_finite(before) and _finite(after)):
        return None
    return (float(before) - float(after)) / max(abs(float(before)), 1e-12)


def _first_float(stat: Mapping[str, object], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _coerce_float(stat.get(key))
        if value is not None:
            return value
    return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None


def _coerce_int(value: object) -> int:
    try:
        return max(0, int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return True
    if raw in {"0", "false", "no", "n"}:
        return False
    return None


def _is_nonfinite(value: float | None) -> bool:
    return value is not None and not _finite(value)


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


__all__ = [
    "EarlyStopDecision",
    "EarlyStopEvidence",
    "EarlyStopPolicy",
    "EarlyStopProfile",
    "EarlyStopState",
    "annotate_stat_with_early_stop",
    "evaluate_early_stop",
    "normalize_early_stop_profile",
    "pose_evidence_from_stat",
    "resolve_early_stop_policy",
    "setup_evidence_from_stat",
]
