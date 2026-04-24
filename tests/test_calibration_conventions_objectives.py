from __future__ import annotations

import json

import pytest

from tomojax.calibration import CandidateScore, ConventionAudit, MetricSpec, ObjectiveCard


def test_convention_audit_records_ambiguity_without_claiming_correction():
    audit = ConventionAudit(flip_u=None, flip_v=True, theta_sign=-1, confidence=0.2)

    payload = audit.to_dict()

    json.dumps(payload, allow_nan=False)
    assert payload["flip_v"] is True
    assert payload["theta_sign"] == -1
    assert payload["ambiguous"] is True
    assert payload["correction_applied"] is False


def test_convention_audit_validates_confidence_and_theta_sign():
    with pytest.raises(ValueError, match="confidence"):
        ConventionAudit(confidence=1.5)

    with pytest.raises(ValueError, match="theta_sign"):
        ConventionAudit(theta_sign=0)  # type: ignore[arg-type]


def test_objective_card_requires_direction_and_serializes_candidates():
    objective = ObjectiveCard(
        primary_metric=MetricSpec(name="heldout_projection_mse", direction="minimize"),
        secondary_metrics=(MetricSpec(name="slice_focus", direction="maximize", domain="image"),),
        validation_split={"views": [0, 4, 8]},
        top_candidates=(
            CandidateScore(
                parameters={"det_u_px": -4.0},
                score=0.1,
                rank=1,
                uncertainty={"sigma_px": 0.2},
                artifacts={"contact_sheet": "candidate_01.png"},
            ),
        ),
    )

    payload = objective.to_dict()

    json.dumps(payload, allow_nan=False)
    assert payload["primary_metric"]["name"] == "heldout_projection_mse"
    assert payload["secondary_metrics"][0]["domain"] == "image"
    assert payload["top_candidates"][0]["parameters"]["det_u_px"] == pytest.approx(-4.0)


def test_metric_spec_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        MetricSpec(name="bad", direction="smaller")  # type: ignore[arg-type]
