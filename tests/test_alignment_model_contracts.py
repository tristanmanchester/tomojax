from __future__ import annotations

import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.align._geometry.geometry_applier import setup_axis_unit

# check-public-imports: allow-private
from tomojax.align._geometry.geometry_blocks import _theta_span_from_geometry

# check-public-imports: allow-private
from tomojax.align._model.schedules import AlignmentSchedule, AlignmentStage

# check-public-imports: allow-private
from tomojax.align._model.state import SetupGeometryState
from tomojax.core.geometry import Detector, Grid, ParallelGeometry


@pytest.mark.parametrize("optimizer", ["adam", "validation_lm"])
def test_fixed_volume_stages_reject_unsupported_optimizers(optimizer: str) -> None:
    schedule = AlignmentSchedule(
        name="bad",
        stages=(
            AlignmentStage(
                name="bad",
                active_dofs=("dx",),
                objective_kind="fixed_volume",
                optimizer=optimizer,  # type: ignore[arg-type]
            ),
        ),
    )

    with pytest.raises(ValueError, match="fixed_volume.*'gd', 'gn', or 'lbfgs'"):
        schedule.validate()


def test_axis_unit_lab_matches_projection_axis_unit_with_tilt() -> None:
    setup = SetupGeometryState.from_degrees(tilt_deg=5.0)

    np.testing.assert_allclose(setup.axis_unit_lab(), setup_axis_unit(setup), rtol=1e-6)


def test_theta_span_uses_sampled_geometric_coverage() -> None:
    geometry = ParallelGeometry(
        grid=Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0),
        detector=Detector(nu=4, nv=4, du=1.0, dv=1.0),
        thetas_deg=np.linspace(0.0, 270.0, 12, endpoint=False, dtype=np.float32),
    )

    assert _theta_span_from_geometry(geometry) == pytest.approx(247.5)
