from __future__ import annotations

from tomojax.align.verification import verification_from_manifest, verify_alignment_metrics


def test_verify_alignment_metrics_accepts_improved_loss_and_quality():
    result = verify_alignment_metrics(
        loss_before=10.0,
        loss_after=2.0,
        misaligned_mse=0.5,
        aligned_mse=0.1,
        final_quality_required=True,
    )

    assert result.status == "accepted"
    assert result.passed is True
    assert result.to_dict()["metrics"]["loss_after"] == 2.0


def test_verify_alignment_metrics_escalates_reduced_model_failure():
    result = verify_alignment_metrics(
        loss_before=10.0,
        loss_after=2.0,
        model_sufficient=False,
    )

    assert result.status == "escalate_to_refine"
    assert result.passed is False
    assert result.recommended_next == "expand_motion_model"


def test_verify_alignment_metrics_marks_unsupported_fast_path():
    result = verify_alignment_metrics(backend_supported=False)

    assert result.status == "unsupported"
    assert result.passed is False
    assert result.recommended_next == "fallback_to_tortoise"


def test_verify_alignment_metrics_requires_final_quality_when_requested():
    result = verify_alignment_metrics(
        loss_before=10.0,
        loss_after=2.0,
        misaligned_mse=0.1,
        aligned_mse=0.2,
        final_quality_required=True,
    )

    assert result.status == "fallback_to_tortoise"
    assert result.passed is False
    assert "MSE" in result.reasons[0]


def test_verification_from_manifest_reads_alignment_smoke_shape():
    result = verification_from_manifest(
        {
            "loss": {"initial": 4.0, "final": 1.0},
            "quality": {
                "misaligned_recon_vs_truth": {"mse": 0.4},
                "aligned_recon_vs_truth": {"mse": 0.2},
            },
        }
    )

    assert result.status == "accepted"
    assert result.passed is True


def test_verification_from_manifest_does_not_treat_working_recon_as_final_quality():
    result = verification_from_manifest(
        {
            "loss": {"initial": 4.0, "final": 1.0},
            "quality": {
                "misaligned_recon_vs_truth": {"mse": 0.1},
                "aligned_recon_vs_truth": {"mse": 0.2},
            },
        }
    )

    assert result.status == "accepted"
    assert result.passed is True


def test_verification_from_manifest_honours_explicit_final_quality_requirement():
    result = verification_from_manifest(
        {
            "loss": {"initial": 4.0, "final": 1.0},
            "quality": {
                "final_quality_required": True,
                "misaligned_recon_vs_truth": {"mse": 0.1},
                "aligned_recon_vs_truth": {"mse": 0.2},
            },
        }
    )

    assert result.status == "fallback_to_tortoise"
    assert result.passed is False


def test_verification_from_manifest_rejects_worse_synthetic_pose_recovery():
    result = verification_from_manifest(
        {
            "loss": {"initial": 10.0, "final": 1.0},
            "pose_recovery": {
                "initial_rot_rmse_deg": 1.0,
                "rot_rmse_deg": 12.0,
                "initial_trans_rmse_px": 0.5,
                "trans_rmse_px": 4.0,
            },
        }
    )

    assert result.status == "escalate_to_refine"
    assert result.passed is False


def test_verification_from_manifest_accepts_single_axis_pose_tradeoff():
    result = verification_from_manifest(
        {
            "loss": {"initial": 10.0, "final": 1.0},
            "pose_recovery": {
                "initial_rot_rmse_deg": 1.0,
                "rot_rmse_deg": 3.0,
                "initial_trans_rmse_px": 1.0,
                "trans_rmse_px": 0.2,
            },
        }
    )

    assert result.status == "accepted"
    assert result.passed is True
