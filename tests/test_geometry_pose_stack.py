"""Built-in bulk poses must retain the scalar geometry and subclass contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.core.geometry.views import stack_view_poses
from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry


@pytest.mark.parametrize("tilt", [0.0, -20.0, 30.0, 180.0])
@pytest.mark.parametrize("axis", ["x", "z"])
def test_laminography_bulk_poses_match_scalar(tilt, axis):
    grid = Grid(5, 4, 3, 1.0, 1.0, 1.0)
    detector = Detector(7, 5, 1.0, 1.0)
    geometry = LaminographyGeometry(
        grid, detector, [-180, 0, 1e-7, 12.3, 137.2, 359.9], tilt_deg=tilt, tilt_about=axis
    )
    for count in (1, 6):
        expected = np.asarray([geometry.pose_for_view(i) for i in range(count)], dtype=np.float32)
        np.testing.assert_allclose(
            stack_view_poses(geometry, count), expected, rtol=2e-6, atol=1e-7
        )


def test_parallel_subclass_pose_override_and_traced_parameters_are_preserved():
    grid = Grid(3, 2, 2, 1.0, 1.0, 1.0)
    detector = Detector(4, 3, 1.0, 1.0)

    class ShiftedParallel(ParallelGeometry):
        def __init__(self, shift):
            super().__init__(grid, detector, [13.0, 78.0])
            self.shift = shift

        def pose_for_view(self, i):
            return jnp.asarray(super().pose_for_view(i)).at[0, 3].set(self.shift)

    def poses(shift):
        return stack_view_poses(ShiftedParallel(shift), 2)

    actual = jax.jit(poses)(0.7)
    np.testing.assert_allclose(actual[:, 0, 3], 0.7)
    derivative = jax.jacfwd(poses)(0.7)
    np.testing.assert_array_equal(derivative[:, 0, 3], np.ones(2))
