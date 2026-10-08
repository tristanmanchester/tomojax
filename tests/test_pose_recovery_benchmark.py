"""Independent fixture and subpixel/angle reporting checks for the pose suite."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from compare_projectors import make_case, make_geometry
from pose_recovery import RICH_GAUSSIANS, fixture, integrals, pose_errors, poses_numpy


@pytest.mark.parametrize("kind", ["parallel", "anisotropic", "lamino"])
def test_streamed_integrals_match_independent_quadrature_across_view_boundaries(kind):
    from scipy.integrate import quad

    grid, detector, nominal, _ = make_geometry(32, 37, kind)
    extent = np.array([grid.nx * grid.vx, grid.ny * grid.vy, grid.nz * grid.vz])
    rng = np.random.default_rng(731)
    perturbation = rng.uniform(-1, 1, (37, 5)) * [3, 3, 3, 10, 10]
    poses = poses_numpy(perturbation, nominal, detector)[::-1]
    actual = integrals(poses, extent, detector, RICH_GAUSSIANS)
    # Compare rays on both sides of a view boundary, including the partial batch.
    for view, row, col in [(0, 10, 12), (31, 8, 20), (32, 11, 14), (36, 7, 13)]:
        camera = np.array(
            [
                (col - (detector.nu - 1) / 2) * detector.du + detector.center[0],
                0,
                (row - (detector.nv - 1) / 2) * detector.dv + detector.center[1],
            ]
        )
        pose = poses[view]

        def density(distance, camera=camera, pose=pose):
            point = (camera + np.array([0, distance, 0]) - pose[:3, 3]) @ pose[:3, :3]
            return sum(
                amplitude
                * np.exp(-0.5 * np.sum(((point - extent * center) / (extent * sigma)) ** 2))
                for amplitude, center, sigma in RICH_GAUSSIANS
            )

        expected, _ = quad(density, -np.inf, np.inf, epsabs=1e-11, epsrel=1e-11)
        np.testing.assert_allclose(actual[view, row, col], expected, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("kind", ["parallel", "anisotropic", "lamino"])
def test_margin_retains_phantom_scale_draws_and_measured_rays(kind):
    n = 32
    small = fixture(n, 9, kind, 817, 0.005, 0)
    large = fixture(n, 9, kind, 817, 0.005, 0.25)
    for key in ("truth", "initial", "data", "extent"):
        np.testing.assert_array_equal(small[key], large[key])
    np.testing.assert_array_equal(small["volume"], large["volume"][8:-8, 8:-8, 8:-8])
    assert small["detector"] == large["detector"]
    # Added samples contain the continuous object's tails, not synthetic zeros.
    assert np.count_nonzero(large["volume"][:8]) > 0
    case = make_case(n, 9, kind)
    np.testing.assert_array_equal(small["volume"], case.volume)
    exact = integrals(case.poses.astype(np.float64), small["extent"], case.detector)
    assert np.linalg.norm(exact - case.analytic) < 1e-6 * np.linalg.norm(exact)


def test_rotation_error_retains_angles_below_fp32_arccos_resolution():
    case = fixture(32, 9, "lamino", 432, 0, 0)
    truth = np.zeros((9, 5))
    estimate = truth.copy()
    estimate[:, 0] = 0.005
    estimate[:, 3:] = [0.02, -0.03]
    errors = pose_errors(estimate, truth, case["nominal"], case["detector"])
    np.testing.assert_allclose(errors["rotation_errors_deg"], 0.005, rtol=1e-6)
    assert errors["accepted_views"] == 9
    estimate[0, 0] = 0.02
    estimate[1, 3:] = [0.04, 0.04]  # Vector error exceeds .05 although each component does not.
    errors = pose_errors(estimate, truth, case["nominal"], case["detector"])
    assert errors["accepted_views"] == 7


def test_rich_fixture_is_reproducible_with_nominal_initialization():
    options = dict(phantom="nine-gaussian", initialization="nominal")
    small = fixture(32, 9, "lamino", 123, 0.005, 0, **options)
    large = fixture(32, 9, "lamino", 123, 0.005, 0.25, **options)
    for key in ("truth", "data", "initial"):
        np.testing.assert_array_equal(small[key], large[key])
    np.testing.assert_array_equal(small["initial"], np.zeros((9, 5)))
    assert np.max(np.abs(small["truth"][:, :3])) <= 3
    assert np.max(np.abs(small["truth"][:, 3:])) <= 10
    np.testing.assert_array_equal(small["volume"], large["volume"][8:-8, 8:-8, 8:-8])
    assert small["noise_sigma"] > 0


@pytest.mark.gpu
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_fused_pose_update_matches_explicit_jacobian_and_reuses_residual(interpolation):
    import jax
    import jax.numpy as jnp
    from pose_recovery import workflow_functions

    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    case = fixture(32, 9, "lamino", 123, 0.005, 0.25)
    g, d = case["grid"], case["detector"]
    x, nominal, y = (jnp.asarray(case[k]) for k in ("volume", "nominal", "data"))
    p = jnp.asarray(case["truth"]) + 0.1
    predict, normal_explicit, accept = workflow_functions(g, d, "pallas", interpolation, "explicit")
    _, normal_fused, _ = workflow_functions(g, d, "pallas", interpolation, "fused")
    expected, reference_residual = normal_explicit(p, nominal, x, y)
    actual, residual = normal_fused(p, nominal, x, y)
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-4)
    np.testing.assert_allclose(residual, reference_residual, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(residual, predict(p, nominal, x) - y, rtol=1e-5, atol=1e-5)
    q, loss = accept(p, actual, nominal, x, y, residual)
    evaluated = 0.5 * np.sum(np.asarray(predict(q, nominal, x) - y, np.float64) ** 2)
    assert evaluated <= 0.5 * np.sum(np.asarray(residual, np.float64) ** 2) * (1 + 1e-6)
    np.testing.assert_allclose(loss, evaluated, rtol=2e-6)


@pytest.mark.parametrize("backend", ["jax", "pallas"])
def test_streamed_line_search_matches_stacked_candidates_and_ties(backend):
    import jax
    import jax.numpy as jnp
    from pose_recovery import workflow_functions

    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    case = fixture(16, 5, "lamino", 325, 0.005, 0.25)
    g, d = case["grid"], case["detector"]
    x, nominal, y = (jnp.asarray(case[k]) for k in ("volume", "nominal", "data"))
    p = jnp.asarray(case["truth"]) + 0.1
    predict, normal, streamed = workflow_functions(g, d, backend, "cubic")
    _, _, stacked = workflow_functions(g, d, backend, "cubic", line_search="stacked")
    step, residual = normal(p, nominal, x, y)
    # Include overshoot, descent, ascent and zero steps in separate views.
    step *= jnp.array([1.0, 10.0, -1.0, 0.0, 2.0])[:, None]
    expected, expected_loss = stacked(p, step, nominal, x, y, residual)
    actual, loss = streamed(p, step, nominal, x, y, residual)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(loss, expected_loss, rtol=2e-6)
    evaluated = 0.5 * np.sum(np.asarray(predict(actual, nominal, x) - y, np.float64) ** 2)
    np.testing.assert_allclose(loss, evaluated, rtol=2e-6)
    # A zero object makes every candidate equal even with a nonzero step.
    # The earliest candidate is the full step, including ties with the zero step.
    zero = jnp.zeros_like(x)
    all_equal, loss = streamed(p, step, nominal, zero, y, -y)
    np.testing.assert_allclose(all_equal, p + step, rtol=0, atol=0)
    np.testing.assert_allclose(loss, 0.5 * np.sum(np.asarray(y, np.float64) ** 2), rtol=2e-6)
