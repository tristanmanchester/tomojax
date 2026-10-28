"""Gauge-policy and conditioning diagnostics for alignment DOFs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


GaugePolicy = Literal["reject", "anchor_mean", "prior_required", "diagnose_only"]


@dataclass(frozen=True, slots=True)
class GaugeDecision:
    """Resolved decision for active gauge-coupled alignment DOFs."""

    policy: GaugePolicy
    active_dofs: tuple[str, ...]
    conflicts: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        """Return a compact status label for the gauge decision."""
        if self.conflicts and self.policy == "reject":
            return "rejected"
        if self.conflicts:
            return "allowed_with_gauge_policy"
        return "ok"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-native decision summary."""
        return {
            "policy": self.policy,
            "active_dofs": list(self.active_dofs),
            "conflicts": list(self.conflicts),
            "warnings": list(self.warnings),
            "status": self.status,
        }


class GaugePolicyError(ValueError):
    """Raised when requested DOFs violate the selected gauge policy."""


_COUPLED_RULES: tuple[tuple[str, frozenset[str], str], ...] = (
    (
        "det_u_px_pose_translation",
        frozenset({"det_u_px", "dx"}),
        "det_u_px is gauge-coupled with per-view dx",
    ),
    (
        "det_u_px_pose_depth_translation",
        frozenset({"det_u_px", "dz"}),
        "det_u_px is gauge-coupled with per-view dz",
    ),
    (
        "det_v_px_pose_translation",
        frozenset({"det_v_px", "dx"}),
        "det_v_px is gauge-coupled with per-view dx",
    ),
    (
        "det_v_px_pose_depth_translation",
        frozenset({"det_v_px", "dz"}),
        "det_v_px is gauge-coupled with per-view dz",
    ),
    (
        "detector_roll_phi_mean",
        frozenset({"detector_roll_deg", "phi"}),
        "detector_roll_deg is gauge-coupled with mean in-plane pose phi",
    ),
    (
        "axis_rot_x_alpha_mean",
        frozenset({"axis_rot_x_deg", "alpha"}),
        "axis_rot_x_deg is gauge-coupled with mean pose alpha",
    ),
    (
        "axis_rot_y_beta_mean",
        frozenset({"axis_rot_y_deg", "beta"}),
        "axis_rot_y_deg is gauge-coupled with mean pose beta",
    ),
)


def validate_active_gauge_policy(
    active_dofs: Sequence[str],
    *,
    policy: GaugePolicy = "reject",
    priors: Mapping[str, object] | None = None,
) -> GaugeDecision:
    """Validate active DOFs against known gauge-coupling rules."""
    active = tuple(str(name) for name in active_dofs)
    active_set = set(active)
    conflicts = tuple(code for code, names, _msg in _COUPLED_RULES if names <= active_set)
    if conflicts and policy == "reject":
        raise GaugePolicyError(
            "Gauge-coupled alignment DOFs are underdetermined under policy='reject': "
            + ", ".join(conflicts)
        )
    if conflicts and policy == "prior_required":
        priors = {} if priors is None else dict(priors)
        missing = tuple(name for name in active if name not in priors)
        if missing:
            raise GaugePolicyError(
                "GaugePolicy 'prior_required' requires priors for active coupled DOFs; "
                f"missing: {', '.join(missing)}"
            )
    warnings = conflicts if conflicts and policy in {"anchor_mean", "diagnose_only"} else ()
    return GaugeDecision(policy=policy, active_dofs=active, conflicts=conflicts, warnings=warnings)
