from dataclasses import replace

import numpy as np
import pytest

from tests.test_alignment_model_contracts import _common_line_gaussians
import tomojax as tj
from tomojax.alignment.api import AlignConfig, AlignmentSchedule, AlignmentStage
from tomojax.geometry import Grid, RotationAxisGeometry


def _fixture():
    data, angles, det, axis, center, basis, truth = _common_line_gaussians(
        (65, 81), 2.0, views=160, turns=2
    )
    det = replace(det, center=(center * det.du, det.center[1]))
    grid = Grid(80, 80, 24, 0.8, 0.8, 1.1)
    scan = tj.Scan(data, RotationAxisGeometry(grid, det, angles, axis))
    schedule = AlignmentSchedule(
        "spectral_motion_only",
        (
            AlignmentStage(
                "spectral_motion",
                ("dx", "dz"),
                "projection_consistency",
                "spectral",
                gauge_policy="diagnose_only",
                stage_role="setup",
                differentiability="not_required",
            ),
        ),
    )
    cfg = AlignConfig(
        schedule=schedule,
        pose_translation_frame="detector",
        seed_translations=True,
        gn_coupling="joint",
        iterations=1,
        outer_iterations=1,
        tv_weight=0,
        backend="jax",
    )
    return scan, cfg, basis, truth


@pytest.mark.parametrize("factor", [1, 2])
def test_public_spectral_motion_updates_only_physical_translations(monkeypatch, factor):
    scan, cfg, basis, truth = _fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError("a projection-only stage must not run broad alignment or reconstruct")

    monkeypatch.setattr(
        "tomojax.alignment._stages._stage_runners.seeded_translation_params", forbidden
    )
    monkeypatch.setattr(
        "tomojax.alignment._stages._reconstruction_stage._run_reconstruction_step", forbidden
    )
    result = tj.align(scan, mode="pose", levels=(factor,), config=cfg)
    poses = np.asarray(result.poses)
    error = (poses[:, 3:5] - truth).ravel()
    gauge = basis.reshape(-1, 3)
    error -= gauge @ np.linalg.lstsq(gauge, error, rcond=None)[0]
    assert np.sqrt(np.mean(error**2)) < 0.1
    np.testing.assert_array_equal(poses[:, [0, 1, 2, 5]], 0)
    np.testing.assert_array_equal(np.asarray(result.volume), 0)
    assert result.info["loss"] == []
    stat = result.info["outer_stats"][0]
    assert stat["motion_seed_status"] == "unvalidated_seed"
    assert stat["motion_validation_performed"] is False
    assert stat["num_train_reconstructions"] == 0
    assert result.info["schedule_stages"][0]["objective_kind"] == "projection_consistency"


def test_public_spectral_motion_preserves_constraints_and_existing_poses():
    scan, cfg, _, _ = _fixture()
    for restricted, work, reason in [
        (replace(cfg, bounds={"dx": (-0.01, 0.01), "dz": (-0.01, 0.01)}), scan, "explicit_bounds"),
        (replace(cfg, freeze=("dx",)), scan, "requires_both_detector_translations"),
        (cfg, tj.Scan(np.zeros_like(scan.projections), scan.geometry), "no_signal"),
    ]:
        result = tj.align(work, mode="pose", levels=(1,), config=restricted)
        np.testing.assert_array_equal(np.asarray(result.poses), 0)
        stat = result.info["outer_stats"][0]
        assert stat["motion_seed_status"] == "seed_declined"
        assert stat["motion_seed_reason"] == reason
    first = tj.align(scan, mode="pose", levels=(1,), config=cfg)
    again = tj.align(first.scan, mode="pose", levels=(1,), config=cfg)
    np.testing.assert_allclose(again.poses, first.poses, atol=2e-5)
    assert again.info["outer_stats"][0]["motion_seed_reason"] == "unsupported_geometry"
    invalid = AlignmentStage(
        "mixed", ("dx", "det_u_px"), "projection_consistency", "spectral", stage_role="setup"
    )
    with pytest.raises(ValueError, match="separate detector"):
        AlignmentSchedule("mixed", (invalid,)).validate()
