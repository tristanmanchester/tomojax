"""Joseph plane sampling selected through the ray-integrator option."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.core.projector import (
    backproject_view_T,
    forward_project_view_T,
    sum_backproject_views_T,
)
from tomojax.forward import project_joseph
from tomojax.geometry import Detector, Grid, LaminographyGeometry


def scan() -> tuple[Grid, Detector, jnp.ndarray]:
    grid = Grid(7, 6, 5, 0.8, 1.1, 1.3)
    detector = Detector(9, 6, 0.9, 1.2, (0.17, -0.21))
    geometry = LaminographyGeometry(grid, detector, [11.0, 64.0, 157.0], tilt_deg=25)
    return grid, detector, jnp.asarray([geometry.pose_for_view(i) for i in range(3)])


@pytest.mark.parametrize(
    ("integrator", "interpolation"), [("joseph", "linear"), ("joseph_cubic", "cubic")]
)
def test_joseph_integrators_match_the_plane_projector_and_its_transpose(
    integrator: str, interpolation: str
) -> None:
    grid, detector, poses = scan()
    rng = np.random.default_rng(5)
    volume = jnp.asarray(rng.normal(size=(7, 6, 5)), jnp.float32)
    images = jnp.asarray(rng.normal(size=(3, 6, 9)), jnp.float32)
    expected = project_joseph(volume, poses, grid, detector, interpolation=interpolation)
    actual = jnp.stack(
        [
            forward_project_view_T(t, grid, detector, volume, ray_integrator=integrator)
            for t in poses
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    adjoint = sum_backproject_views_T(poses, grid, detector, images, ray_integrator=integrator)
    single = sum(
        backproject_view_T(t, grid, detector, image, ray_integrator=integrator)
        for t, image in zip(poses, images, strict=True)
    )
    np.testing.assert_allclose(single, adjoint, rtol=1e-5, atol=1e-5)
    # <A x, y> = <x, A^T y>
    np.testing.assert_allclose(
        float(jnp.vdot(expected, images)), float(jnp.vdot(volume, adjoint)), rtol=1e-4
    )


def test_joseph_integrators_accept_affine_detector_grids() -> None:
    from tomojax.core.projector import get_detector_grid_device

    grid, detector, poses = scan()
    volume = jnp.asarray(np.random.default_rng(9).normal(size=(7, 6, 5)), jnp.float32)
    shifted = Detector(9, 6, 0.9, 1.2, (0.17 + 0.35, -0.21 - 0.4))
    # A lattice offset from the nominal detector equals a detector with that centre.
    x, z = get_detector_grid_device(detector)
    for det_grid, reference in [((x, z), detector), ((x + 0.35, z - 0.4), shifted)]:
        actual = forward_project_view_T(
            poses[1], grid, detector, volume, det_grid=det_grid, ray_integrator="joseph"
        )
        expected = forward_project_view_T(
            poses[1], grid, reference, volume, ray_integrator="joseph"
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
