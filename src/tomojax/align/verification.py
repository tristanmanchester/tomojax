from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping

import numpy as np


VerificationStatus = Literal[
    "accepted",
    "accepted_with_warning",
    "escalate_to_refine",
    "fallback_to_tortoise",
    "unsupported",
    "failed",
]


@dataclass(frozen=True, slots=True)
class AlignmentVerificationResult:
    """Structured acceptance state for an alignment run or benchmark case."""

    status: VerificationStatus
    passed: bool
    reasons: tuple[str, ...] = ()
    metrics: dict[str, object] = field(default_factory=dict)
    recommended_next: str | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        data["metrics"] = dict(self.metrics)
        return data


def verify_alignment_metrics(
    *,
    loss_before: float | None = None,
    loss_after: float | None = None,
    misaligned_mse: float | None = None,
    aligned_mse: float | None = None,
    backend_supported: bool = True,
    final_quality_required: bool = False,
    model_sufficient: bool = True,
) -> AlignmentVerificationResult:
    """Classify common alignment metrics into a workflow acceptance status."""

    metrics: dict[str, object] = {
        "loss_before": loss_before,
        "loss_after": loss_after,
        "misaligned_mse": misaligned_mse,
        "aligned_mse": aligned_mse,
        "backend_supported": bool(backend_supported),
        "final_quality_required": bool(final_quality_required),
        "model_sufficient": bool(model_sufficient),
    }
    reasons: list[str] = []
    if not backend_supported:
        return AlignmentVerificationResult(
            status="unsupported",
            passed=False,
            reasons=("requested fast path was unsupported",),
            metrics=metrics,
            recommended_next="fallback_to_tortoise",
        )
    if not model_sufficient:
        return AlignmentVerificationResult(
            status="escalate_to_refine",
            passed=False,
            reasons=("reduced motion model left structured residuals",),
            metrics=metrics,
            recommended_next="expand_motion_model",
        )
    if loss_before is not None and loss_after is not None:
        if not (_is_finite(loss_before) and _is_finite(loss_after)):
            return AlignmentVerificationResult(
                status="failed",
                passed=False,
                reasons=("loss metrics were non-finite",),
                metrics=metrics,
                recommended_next="fallback_to_tortoise",
            )
        if float(loss_after) >= float(loss_before):
            reasons.append("alignment loss did not decrease")
    if final_quality_required:
        if aligned_mse is None or misaligned_mse is None:
            reasons.append("final quality was required but not available")
        elif not (_is_finite(aligned_mse) and _is_finite(misaligned_mse)):
            reasons.append("final quality metrics were non-finite")
        elif float(aligned_mse) >= float(misaligned_mse):
            reasons.append("aligned final reconstruction did not improve MSE")
    if not reasons:
        return AlignmentVerificationResult(status="accepted", passed=True, metrics=metrics)
    return AlignmentVerificationResult(
        status="accepted_with_warning" if not final_quality_required else "fallback_to_tortoise",
        passed=not final_quality_required,
        reasons=tuple(reasons),
        metrics=metrics,
        recommended_next="run_tortoise_verification" if final_quality_required else "inspect_result",
    )


def verification_from_manifest(payload: Mapping[str, object]) -> AlignmentVerificationResult:
    """Build a verification result from a manifest-like mapping when fields exist."""

    loss = payload.get("loss")
    quality = payload.get("quality")
    pose_recovery = payload.get("pose_recovery")
    loss_before = loss_after = misaligned_mse = aligned_mse = None
    model_sufficient = True
    if isinstance(loss, Mapping):
        loss_before = _float_or_none(loss.get("initial"))
        loss_after = _float_or_none(loss.get("final"))
    if isinstance(quality, Mapping):
        final_quality_required = bool(quality.get("final_quality_required", False))
        misaligned = quality.get("misaligned_recon_vs_truth")
        aligned = quality.get("aligned_recon_vs_truth")
        if isinstance(misaligned, Mapping):
            misaligned_mse = _float_or_none(misaligned.get("mse"))
        if isinstance(aligned, Mapping):
            aligned_mse = _float_or_none(aligned.get("mse"))
    else:
        final_quality_required = False
    if isinstance(pose_recovery, Mapping):
        rot = _float_or_none(pose_recovery.get("rot_rmse_deg"))
        rot_initial = _float_or_none(pose_recovery.get("initial_rot_rmse_deg"))
        trans = _float_or_none(pose_recovery.get("trans_rmse_px"))
        trans_initial = _float_or_none(pose_recovery.get("initial_trans_rmse_px"))
        rot_worse = (
            rot is not None
            and rot_initial is not None
            and rot > max(rot_initial * 1.25, rot_initial + 1.0)
        )
        trans_worse = (
            trans is not None
            and trans_initial is not None
            and trans > max(trans_initial * 1.25, trans_initial + 0.5)
        )
        model_sufficient = not (rot_worse and trans_worse)
    return verify_alignment_metrics(
        loss_before=loss_before,
        loss_after=loss_after,
        misaligned_mse=misaligned_mse,
        aligned_mse=aligned_mse,
        final_quality_required=final_quality_required,
        model_sufficient=model_sufficient,
    )


def _float_or_none(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return out if np.isfinite(out) else None


def _is_finite(value: float) -> bool:
    return bool(np.isfinite(float(value)))


__all__ = [
    "AlignmentVerificationResult",
    "VerificationStatus",
    "verification_from_manifest",
    "verify_alignment_metrics",
]
