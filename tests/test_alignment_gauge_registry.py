from __future__ import annotations

import jax.numpy as jnp
import pytest

from tomojax.align.diagnostics import (
    GaugePolicyError,
    conditioning_diagnostics,
    validate_active_gauge_policy,
)


def test_detector_center_with_pose_translation_rejects_by_default():
    with pytest.raises(GaugePolicyError, match="det_u_px_pose_translation"):
        validate_active_gauge_policy(("det_u_px", "dx"))


def test_cor_with_pose_frozen_passes_and_records_metadata():
    decision = validate_active_gauge_policy(("det_u_px",))

    assert decision.status == "ok"
    assert decision.to_dict()["policy"] == "reject"
    assert decision.to_dict()["conflicts"] == []


def test_expert_coupled_requires_priors_when_policy_requires_priors():
    with pytest.raises(GaugePolicyError, match="missing"):
        validate_active_gauge_policy(("det_u_px", "dx"), policy="prior_required")

    decision = validate_active_gauge_policy(
        ("det_u_px", "dx"),
        policy="prior_required",
        priors={"det_u_px": {"sigma": 1.0}, "dx": {"sigma": 0.1}},
    )

    assert decision.status == "allowed_with_gauge_policy"
    assert decision.conflicts == ("det_u_px_pose_translation",)


def test_anchor_policy_allows_coupled_set_with_warning():
    decision = validate_active_gauge_policy(("detector_roll_deg", "phi"), policy="anchor_mean")

    assert decision.status == "allowed_with_gauge_policy"
    assert decision.warnings == ("detector_roll_phi_mean",)


def test_conditioning_diagnostics_identifies_near_null_vector():
    sensitivity = jnp.asarray(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=jnp.float32,
    )

    diagnostics = conditioning_diagnostics(sensitivity, dof_names=("det_u_px", "dx"))

    assert diagnostics["status"] == "weak"
    assert diagnostics["near_null_vectors"]
    terms = diagnostics["near_null_vectors"][0]["terms"]
    assert {term["dof"] for term in terms} == {"det_u_px", "dx"}
