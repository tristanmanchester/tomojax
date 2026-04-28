from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import jax.numpy as jnp
import numpy as np


GaugePolicy = Literal["reject", "anchor_mean", "prior_required", "diagnose_only"]


@dataclass(frozen=True, slots=True)
class GaugeDecision:
    policy: GaugePolicy
    active_dofs: tuple[str, ...]
    conflicts: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        if self.conflicts and self.policy == "reject":
            return "rejected"
        if self.conflicts:
            return "allowed_with_gauge_policy"
        return "ok"

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "active_dofs": list(self.active_dofs),
            "conflicts": list(self.conflicts),
            "warnings": list(self.warnings),
            "status": self.status,
        }


class GaugePolicyError(ValueError):
    pass


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


def conditioning_diagnostics(
    sensitivity: jnp.ndarray,
    *,
    dof_names: Sequence[str],
    rtol: float = 1e-4,
) -> dict[str, object]:
    mat = jnp.asarray(sensitivity, dtype=jnp.float32)
    if mat.ndim != 2:
        raise ValueError("conditioning sensitivity must be a 2-D matrix")
    _u, s, vh = jnp.linalg.svd(mat, full_matrices=False)
    s_np = np.asarray(s)
    if s_np.size == 0:
        threshold = 0.0
    else:
        threshold = float(np.max(s_np)) * float(rtol)
    weak = np.where(s_np <= threshold)[0]
    near_null: list[dict[str, object]] = []
    names = tuple(str(name) for name in dof_names)
    for idx in weak:
        vec = np.asarray(vh[int(idx)])
        terms = [
            {"dof": names[col], "weight": float(weight)}
            for col, weight in enumerate(vec)
            if col < len(names) and abs(float(weight)) > 1e-3
        ]
        near_null.append({"singular_value": float(s_np[int(idx)]), "terms": terms})
    return {
        "singular_values": [float(v) for v in s_np],
        "near_null_vectors": near_null,
        "status": "weak" if near_null else "ok",
    }
