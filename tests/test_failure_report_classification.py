from __future__ import annotations

# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
from typing import cast

import jax.numpy as jnp

# check-public-imports: allow-private
from tomojax.align._alternating_verification import _failure_report_payload
from tomojax.align.api import AlternatingLevelSummary
from tomojax.geometry import GeometryState


def test_failure_report_warns_for_structured_nuisance_residual() -> None:
    geometry = GeometryState.zeros(4)
    final_volume = jnp.zeros((8, 8, 8), dtype=jnp.float32)
    observed = jnp.zeros((4, 8, 8), dtype=jnp.float32)
    observed = observed.at[:, :, 3].set(-1.0)
    observed = observed.at[:, :, 4].set(1.0)
    mask = jnp.ones_like(observed, dtype=jnp.float32)

    report = _failure_report_payload(
        final_volume=final_volume,
        final_geometry=geometry,
        observed=observed,
        mask=mask,
        summaries=(
            AlternatingLevelSummary(
                level_factor=4,
                role="preview",
                reconstruction_iterations=1,
                geometry_updates=1,
                executed_geometry_updates=1,
                residual_filter_kinds=("raw",),
                loss_before=2.0,
                loss_after=1.0,
                loss_nonincreasing=True,
                finite_loss=True,
                residual_sigma_estimated=1.0,
                residual_sigma_effective=1.0,
                prior_strength=0.0,
                heldout_residual_before=None,
                heldout_residual_after=None,
                heldout_residual_passed=None,
                gauge_stable=True,
                parameter_update_norm=0.0,
                parameter_update_small=True,
                verified=True,
                skipped_geometry=False,
                skipped_level=False,
                early_exit_reason=None,
            ),
        ),
        verification={
            "summary": {
                "projection_residual_improved": True,
                "gauge_constraints_satisfied": True,
                "backend_provenance_complete": True,
            }
        },
    )

    assert report["status"] == "warning"
    gates = cast("list[dict[str, object]]", report["gates"])
    gates_by_name = {str(gate["name"]): gate for gate in gates}
    assert gates_by_name["nuisance_residual_structure"]["passed"] is False
    warnings = cast("list[dict[str, object]]", report["warnings"])
    assert [warning["class"] for warning in warnings] == ["nuisance_unmodelled"]
