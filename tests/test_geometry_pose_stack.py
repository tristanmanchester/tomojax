"""Built-in bulk poses must retain the scalar geometry and subclass contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# check-public-imports: allow-private
from tomojax.core.geometry.views import stack_view_poses
from tomojax.forward import project_joseph
from tomojax.geometry import (
    Detector,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
    RotationAxisGeometry,
)


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


def test_geometry_wrappers_survive_pickling():
    import pickle

    from tomojax.io import build_geometry_from_dataset_metadata

    meta = {
        "detector": Detector(4, 4, 1.0, 1.0).to_dict(),
        "grid": Grid(4, 4, 4, 1.0, 1.0, 1.0).to_dict(),
        "angles": np.asarray([0.0, 90.0], np.float32),
        "geometry_type": "parallel",
        "align_params": np.zeros((2, 5), np.float32),
        "detector_roll_deg": 0.5,
    }
    _, _, wrapped = build_geometry_from_dataset_metadata(meta, poses=True)
    again = pickle.loads(pickle.dumps(wrapped))  # as for a worker process
    assert type(again) is type(wrapped) and list(np.asarray(again.angles)) == [0.0, 90.0]


@pytest.mark.parametrize("kind", ["parallel", "lamino", "axis"])
@pytest.mark.parametrize("metadata_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("output_dtype", [np.float32, np.float64])
def test_geometry_poses_keep_the_precision_of_degree_metadata(kind, metadata_dtype, output_dtype):
    angles = np.asarray([12.3, 89.97, 90, 90.03, 359.9, 450.03, -540], metadata_dtype)
    grid, detector = Grid(5, 4, 3, 1, 1, 1), Detector(7, 5, 1, 1)
    geometry = (
        ParallelGeometry(grid, detector, angles)
        if kind == "parallel"
        else LaminographyGeometry(grid, detector, angles, tilt_deg=0)
        if kind == "lamino"
        else RotationAxisGeometry(grid, detector, angles, (0, 0, 1))
    )
    # The independent rotation implementation sees the represented degrees,
    # not radians that were already rounded to FP32.
    reference = np.broadcast_to(np.eye(4), (len(angles), 4, 4)).copy()
    reference[:, :3, :3] = Rotation.from_euler(
        "z", angles.astype(np.float64)[:, None], degrees=True
    ).as_matrix()
    scalar = np.asarray([geometry.pose_for_view(i) for i in range(len(angles))])
    np.testing.assert_allclose(scalar, reference, atol=2e-14, rtol=2e-14)
    with jax.enable_x64():
        for count in (1, len(angles)):
            actual = np.asarray(stack_view_poses(geometry, count, dtype=output_dtype))
            assert actual.dtype == np.dtype(output_dtype)
            tolerance = 5e-8 if output_dtype is np.float32 else 2e-14
            np.testing.assert_allclose(actual, reference[:count], atol=tolerance, rtol=tolerance)


@pytest.mark.numerical
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_off_axis_parallel_projections_use_the_represented_angles(backend):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("CUDA required")
    grid = Grid(48, 45, 3, 0.8, 1.1, 1.3, vol_center=(180, -130, 0.2))
    detector = Detector(640, 5, 0.7, 1.1, (0.13, -0.27))
    angles = np.array([89.97, 90, 90.03, 137.2, 359.9, 450.03, -540], np.float64)
    geometry = ParallelGeometry(grid, detector, angles)
    index = np.indices((grid.nx, grid.ny, grid.nz), dtype=np.float64)
    centre = np.array([21.3, 17.4, 1.1])[:, None, None, None]
    volume = jnp.asarray(np.exp(-0.5 * np.sum(((index - centre) / 4) ** 2, axis=0)), jnp.float32)
    reference = np.broadcast_to(np.eye(4), (len(angles), 4, 4)).copy()
    reference[:, :3, :3] = Rotation.from_euler("z", angles[:, None], degrees=True).as_matrix()
    expected = project_joseph(
        volume, jnp.asarray(reference, jnp.float32), grid, detector, backend=backend
    )
    actual = project_joseph(
        volume, stack_view_poses(geometry, len(angles)), grid, detector, backend=backend
    )
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=2e-6)
