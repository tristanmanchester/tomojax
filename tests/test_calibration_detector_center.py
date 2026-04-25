from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from tomojax.calibration.center import (
    DetectorCenterCalibrationConfig,
    calibrate_detector_center,
    detector_with_center_offset,
)
from tomojax.core.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.core.projector import forward_project_view


def _block_phantom(size: int) -> jnp.ndarray:
    vol = np.zeros((size, size, size), dtype=np.float32)
    vol[size // 4 : size // 2, size // 3 : size - 3, size // 3 : size // 2] = 1.0
    vol[size - 5 : size - 2, size // 3 : size // 2 + 1, 2:5] = 0.7
    return jnp.asarray(vol)


def _simulate_parallel_with_detector_center(
    *,
    size: int,
    n_views: int,
    det_u_px: float,
) -> tuple[dict[str, object], Grid, Detector, jnp.ndarray]:
    grid = Grid(size, size, size, 1.0, 1.0, 1.0)
    detector_nominal = Detector(size, size, 1.0, 1.0, det_center=(0.0, 0.0))
    detector_true = detector_with_center_offset(detector_nominal, det_u_px=det_u_px)
    thetas = np.linspace(0.0, 180.0, n_views, endpoint=False, dtype=np.float32)
    geometry = ParallelGeometry(grid, detector_true, thetas)
    volume = _block_phantom(size)
    projections = jnp.stack(
        [
            forward_project_view(geometry, grid, detector_true, volume, i, gather_dtype="fp32")
            for i in range(n_views)
        ],
        axis=0,
    )
    return (
        {
            "grid": grid.to_dict(),
            "detector": detector_nominal.to_dict(),
            "thetas_deg": thetas,
            "geometry_type": "parallel",
        },
        grid,
        detector_nominal,
        projections,
    )


def _simulate_lamino_with_detector_center(
    *,
    size: int,
    n_views: int,
    det_u_px: float,
) -> tuple[dict[str, object], Grid, Detector, jnp.ndarray]:
    grid = Grid(size, size, size, 1.0, 1.0, 1.0)
    detector_nominal = Detector(size, size, 1.0, 1.0, det_center=(0.0, 0.0))
    detector_true = detector_with_center_offset(detector_nominal, det_u_px=det_u_px)
    thetas = np.linspace(0.0, 360.0, n_views, endpoint=False, dtype=np.float32)
    geometry = LaminographyGeometry(
        grid,
        detector_true,
        thetas,
        tilt_deg=34.0,
        tilt_about="x",
    )
    volume = _block_phantom(size)
    projections = jnp.stack(
        [
            forward_project_view(geometry, grid, detector_true, volume, i, gather_dtype="fp32")
            for i in range(n_views)
        ],
        axis=0,
    )
    return (
        {
            "grid": grid.to_dict(),
            "detector": detector_nominal.to_dict(),
            "thetas_deg": thetas,
            "geometry_type": "lamino",
            "tilt_deg": 34.0,
            "tilt_about": "x",
        },
        grid,
        detector_nominal,
        projections,
    )


def test_detector_with_center_offset_converts_native_pixels_to_physical_center():
    detector = Detector(nu=8, nv=6, du=0.5, dv=0.25, det_center=(1.0, -1.0))

    shifted = detector_with_center_offset(detector, det_u_px=-4.0, det_v_px=8.0)

    assert shifted.det_center == pytest.approx((-1.0, 1.0))
    assert detector.det_center == pytest.approx((1.0, -1.0))


def test_detector_center_config_validates_gn_options():
    with pytest.raises(ValueError, match="active_detector_dofs"):
        DetectorCenterCalibrationConfig(active_detector_dofs=())

    with pytest.raises(ValueError, match="Unknown detector-centre DOFs"):
        DetectorCenterCalibrationConfig(active_detector_dofs=("world_dx",))

    with pytest.raises(ValueError, match="outer_iters"):
        DetectorCenterCalibrationConfig(outer_iters=0)

    with pytest.raises(ValueError, match="max_step_px"):
        DetectorCenterCalibrationConfig(max_step_px=0.0)

    with pytest.raises(ValueError, match="heldout_stride"):
        DetectorCenterCalibrationConfig(heldout_stride=1)


def test_parallel_detector_center_calibration_recovers_hidden_det_u_px():
    geometry_inputs, grid, detector, projections = _simulate_parallel_with_detector_center(
        size=12,
        n_views=24,
        det_u_px=3.0,
    )

    result = calibrate_detector_center(
        geometry_inputs,
        grid=grid,
        detector=detector,
        projections=projections,
        config=DetectorCenterCalibrationConfig(
            outer_iters=6,
            gn_damping=1e-3,
            max_step_px=2.0,
            heldout_stride=4,
            gather_dtype="fp32",
        ),
    )

    assert result.best_det_u_px == pytest.approx(3.0, abs=0.75)
    assert result.calibrated_detector.det_center[0] == pytest.approx(3.0, abs=0.75)
    assert result.calibration_state.detector[0].status == "estimated"
    assert result.calibration_state.detector[1].status == "frozen"
    assert result.objective_card.primary_metric.name == "heldout_projection_nmse"
    assert result.iterations
    assert any(iteration.accepted for iteration in result.iterations)
    assert result.iterations[-1].loss_after < result.iterations[0].loss_before


def test_lamino_detector_center_calibration_recovers_hidden_det_u_px():
    geometry_inputs, grid, detector, projections = _simulate_lamino_with_detector_center(
        size=12,
        n_views=24,
        det_u_px=-3.0,
    )

    result = calibrate_detector_center(
        geometry_inputs,
        grid=grid,
        detector=detector,
        projections=projections,
        config=DetectorCenterCalibrationConfig(
            outer_iters=6,
            gn_damping=1e-3,
            max_step_px=2.0,
            heldout_stride=4,
            gather_dtype="fp32",
        ),
    )

    assert result.best_det_u_px == pytest.approx(-3.0, abs=1.0)
    assert result.calibrated_detector.det_center[0] == pytest.approx(-3.0, abs=1.0)
    assert result.manifest["calibration_state"]["detector"][0]["gauge"] == (
        "detector_ray_grid_center"
    )
    assert result.manifest["extra"]["iterations"]


def test_detector_center_calibration_falls_back_to_insample_metric_for_tiny_data():
    geometry_inputs, grid, detector, projections = _simulate_parallel_with_detector_center(
        size=8,
        n_views=1,
        det_u_px=1.0,
    )

    result = calibrate_detector_center(
        geometry_inputs,
        grid=grid,
        detector=detector,
        projections=projections,
        config=DetectorCenterCalibrationConfig(
            outer_iters=1,
            heldout_stride=2,
            gather_dtype="fp32",
        ),
    )

    assert result.objective_card.primary_metric.name == "insample_projection_nmse"
