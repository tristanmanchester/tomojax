from __future__ import annotations

# pyright: reportPrivateUsage=false
from typing import cast

# check-public-imports: allow-private
from tomojax.align._alternating_verification import _observability_report_payload

# check-public-imports: allow-private
from tomojax.align._joint_schur_lm import JointSchurDiagnostics, JointSchurLMResult
from tomojax.geometry import GeometryState, canonicalize_geometry_gauges


def test_det_v_observability_uses_active_schur_slot_without_theta_scale() -> None:
    geometry = GeometryState.zeros(1)
    diagnostics = JointSchurDiagnostics(
        schur_condition=1.0,
        setup_update_norm=0.0,
        pose_update_norm=0.0,
        dense_step_difference_norm=0.0,
        schur_eigenvalues=(1.0,),
        setup_correlation_matrix=((1.0, 0.25), (0.25, 1.0)),
        accepted=True,
        actual_reduction=0.5,
    )
    result = JointSchurLMResult(
        geometry=geometry,
        canonicalized_geometry=canonicalize_geometry_gauges(geometry),
        initial_loss=1.0,
        final_loss=0.5,
        iterations=1,
        active_setup_parameters=("det_u_px", "det_v_px"),
        active_pose_dofs=(),
        frozen_parameters=("theta_scale",),
        diagnostics=diagnostics,
        iteration_diagnostics=(diagnostics,),
    )

    payload = _observability_report_payload(result)
    policy = cast("dict[str, object]", payload["weak_dof_policy"])
    decisions = cast("dict[str, dict[str, object]]", policy["decisions"])
    det_v = decisions["det_v_px"]
    evidence = cast("dict[str, object]", det_v["evidence"])
    correlation = cast("dict[str, object]", evidence["correlation"])

    assert det_v["active"] is True
    assert correlation["parameter_index"] == 1
    assert evidence["accepted_step"] is True
    assert evidence["accepted_step_passed"] is True
    assert "accepted_step" not in cast("list[str]", evidence["missing_evidence"])
    assert decisions["theta_scale"]["decision"] == "keep_frozen"


def test_observability_reports_active_setup_and_pose_dofs() -> None:
    geometry = GeometryState.zeros(2)
    diagnostics = JointSchurDiagnostics(
        schur_condition=2.0,
        setup_update_norm=0.1,
        pose_update_norm=0.2,
        dense_step_difference_norm=0.0,
        schur_eigenvalues=(0.5, 2.0),
        accepted=True,
        actual_reduction=0.25,
    )
    result = JointSchurLMResult(
        geometry=geometry,
        canonicalized_geometry=canonicalize_geometry_gauges(geometry),
        initial_loss=1.0,
        final_loss=0.75,
        iterations=1,
        active_setup_parameters=("theta_offset_rad", "det_u_px", "detector_roll_rad"),
        active_pose_dofs=("phi_residual_rad", "dx_px", "dz_px"),
        frozen_parameters=("det_v_px", "theta_scale"),
        diagnostics=diagnostics,
        iteration_diagnostics=(diagnostics,),
    )

    payload = _observability_report_payload(result)
    dofs = cast("dict[str, dict[str, dict[str, object]]]", payload["dofs"])

    assert dofs["setup"]["theta_offset_rad"]["active"] is True
    assert dofs["setup"]["theta_offset_rad"]["status"] == "evaluated"
    assert dofs["setup"]["detector_roll_rad"]["active"] is True
    assert dofs["setup"]["detector_roll_rad"]["status"] == "evaluated"
    assert dofs["setup"]["axis_rot_x_rad"]["active"] is False
    assert dofs["setup"]["axis_rot_x_rad"]["status"] == "frozen"
    assert dofs["pose"]["phi_residual_rad"]["active"] is True
    assert dofs["pose"]["phi_residual_rad"]["status"] == "gauge_canonicalised"
    assert dofs["pose"]["dx_px"]["active"] is True
    assert dofs["pose"]["dx_px"]["status"] == "gauge_canonicalised"
    assert dofs["pose"]["dz_px"]["active"] is True
    assert dofs["pose"]["dz_px"]["status"] == "gauge_canonicalised"
    assert dofs["pose"]["alpha_rad"]["active"] is False
    assert dofs["pose"]["alpha_rad"]["status"] == "frozen"
