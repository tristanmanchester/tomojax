from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from tomojax.calibration.detector_grid import detector_grid_from_detector_roll
from tomojax.calibration.roll import DetectorRollCalibrationConfig, calibrate_detector_roll
from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.projector import forward_project_view


def _asymmetric_phantom(size: int) -> jnp.ndarray:
    vol = np.zeros((size, size, size), dtype=np.float32)
    vol[2 : size // 2 + 1, 2 : size - 2, 2:4] = 1.0
    vol[size // 2 : size - 2, 2:4, size // 2 : size - 1] = 0.6
    vol[3:5, size // 2 : size - 1, size // 2 : size - 2] = 0.8
    return jnp.asarray(vol)


def _simulate_parallel_with_detector_roll(
    *,
    size: int,
    n_views: int,
    detector_roll_deg: float,
) -> tuple[dict[str, object], Grid, Detector, jnp.ndarray]:
    grid = Grid(size, size, size, 1.0, 1.0, 1.0)
    detector = Detector(size, size, 1.0, 1.0, det_center=(0.0, 0.0))
    thetas = np.linspace(0.0, 180.0, n_views, endpoint=False, dtype=np.float32)
    geometry = ParallelGeometry(grid, detector, thetas)
    volume = _asymmetric_phantom(size)
    true_grid = detector_grid_from_detector_roll(
        detector,
        detector_roll_deg=detector_roll_deg,
    )
    projections = jnp.stack(
        [
            forward_project_view(
                geometry,
                grid,
                detector,
                volume,
                i,
                gather_dtype="fp32",
                det_grid=true_grid,
            )
            for i in range(n_views)
        ],
        axis=0,
    )
    return (
        {
            "grid": grid.to_dict(),
            "detector": detector.to_dict(),
            "thetas_deg": thetas,
            "geometry_type": "parallel",
        },
        grid,
        detector,
        projections,
    )


def test_detector_roll_config_validates_gn_options():
    with pytest.raises(ValueError, match="outer_iters"):
        DetectorRollCalibrationConfig(outer_iters=0)

    with pytest.raises(ValueError, match="max_step_deg"):
        DetectorRollCalibrationConfig(max_step_deg=0.0)

    with pytest.raises(ValueError, match="heldout_stride"):
        DetectorRollCalibrationConfig(heldout_stride=1)


def test_parallel_detector_roll_calibration_records_manifest_and_improves_loss():
    geometry_inputs, grid, detector, projections = _simulate_parallel_with_detector_roll(
        size=10,
        n_views=20,
        detector_roll_deg=2.5,
    )

    result = calibrate_detector_roll(
        geometry_inputs,
        grid=grid,
        detector=detector,
        projections=projections,
        config=DetectorRollCalibrationConfig(
            outer_iters=4,
            gn_damping=1e-3,
            max_step_deg=1.0,
            heldout_stride=4,
            gather_dtype="fp32",
            checkpoint_projector=False,
        ),
    )

    assert result.iterations
    assert result.final_volume.shape == (10, 10, 10)
    assert result.calibration_state.detector[0].name == "detector_roll_deg"
    assert result.calibration_state.detector[0].status == "estimated"
    assert result.manifest["calibrated_geometry"]["detector_roll_deg"] == pytest.approx(
        result.detector_roll_deg
    )
    assert result.objective_card.primary_metric.name == "heldout_projection_nmse"
    assert result.iterations[-1].loss_after < result.iterations[0].loss_before
    assert any(iteration.accepted for iteration in result.iterations)
