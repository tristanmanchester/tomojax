from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from tomojax.calibration.axis import AxisDirectionCalibrationConfig, calibrate_axis_direction
from tomojax.core.geometry import Detector, Grid, LaminographyGeometry
from tomojax.core.projector import forward_project_view


def _block_phantom(size: int) -> jnp.ndarray:
    vol = np.zeros((size, size, size), dtype=np.float32)
    vol[2 : size // 2 + 1, 2 : size - 1, 2:4] = 1.0
    vol[size // 2 : size - 1, 2:4, size // 2 : size - 1] = 0.7
    return jnp.asarray(vol)


def test_axis_direction_config_validates_gn_options():
    with pytest.raises(ValueError, match="active_axis_dofs"):
        AxisDirectionCalibrationConfig(active_axis_dofs=())

    with pytest.raises(ValueError, match="Unknown axis-direction DOFs"):
        AxisDirectionCalibrationConfig(active_axis_dofs=("dx",))

    with pytest.raises(ValueError, match="outer_iters"):
        AxisDirectionCalibrationConfig(outer_iters=0)

    with pytest.raises(ValueError, match="max_step_deg"):
        AxisDirectionCalibrationConfig(max_step_deg=0.0)

    with pytest.raises(ValueError, match="heldout_stride"):
        AxisDirectionCalibrationConfig(heldout_stride=1)


def test_lamino_axis_direction_calibration_smoke_records_axis_manifest():
    size = 8
    n_views = 12
    grid = Grid(size, size, size, 1.0, 1.0, 1.0)
    detector = Detector(size, size, 1.0, 1.0)
    thetas = np.linspace(0.0, 360.0, n_views, endpoint=False, dtype=np.float32)
    true_geometry = LaminographyGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=thetas,
        tilt_deg=34.4,
        tilt_about="x",
    )
    volume = _block_phantom(size)
    projections = jnp.stack(
        [
            forward_project_view(true_geometry, grid, detector, volume, i, gather_dtype="fp32")
            for i in range(n_views)
        ],
        axis=0,
    )
    geometry_inputs = {
        "grid": grid.to_dict(),
        "detector": detector.to_dict(),
        "thetas_deg": thetas,
        "geometry_type": "lamino",
        "tilt_deg": 30.0,
        "tilt_about": "x",
    }

    result = calibrate_axis_direction(
        geometry_inputs,
        grid=grid,
        detector=detector,
        projections=projections,
        config=AxisDirectionCalibrationConfig(
            active_axis_dofs=("axis_rot_x_deg",),
            outer_iters=1,
            heldout_stride=4,
            max_step_deg=2.0,
            gather_dtype="fp32",
            checkpoint_projector=False,
        ),
    )

    assert result.iterations
    assert result.final_volume.shape == (size, size, size)
    assert result.calibration_state.scan[0].name == "axis_rot_x_deg"
    assert result.calibration_state.scan[0].status == "estimated"
    assert result.calibration_state.scan[1].name == "axis_rot_y_deg"
    assert result.calibration_state.scan[1].status == "frozen"
    assert result.calibration_state.scan[2].name == "axis_unit_lab"
    assert result.calibration_state.scan[2].status == "derived"
    assert result.objective_card.primary_metric.name == "heldout_projection_nmse"
    assert result.manifest["calibrated_geometry"]["gauge"] == "rotation_axis_direction"
    assert result.manifest["extra"]["config"]["active_axis_dofs"] == ["axis_rot_x_deg"]
    assert result.manifest["extra"]["iterations"]
