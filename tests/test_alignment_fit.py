"""Independent contracts of active blocks, global geometry and their exact border."""

from __future__ import annotations

from bench.fit_alignment import density, line_integrals, make_case, scan_rays, score
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# check-public-imports: allow-private
from tomojax.alignment._joint._parameters import Parameters
from tomojax.alignment._joint._pose import MotionPrior  # check-public-imports: allow-private
from tomojax.geometry import Detector


@pytest.mark.parametrize("cone", [False, True])
@pytest.mark.parametrize(
    "fit", ["centre", "translations", "rotations", ("translations", "tilt", "roll")]
)
def test_held_parameters_have_no_state_and_are_exactly_zero_in_the_output(fit, cone):
    parameters = Parameters(60, fit, cone=cone, lever=17)
    state = np.linspace(-0.3, 0.7, parameters.size)
    scales = np.array([0.01, 0.02, 0.03, 0.7, 1.1, 0.9])[: parameters.full_dofs]
    table, calibration = parameters.physical(state, scales)
    selected = (fit,) if isinstance(fit, str) else fit
    expected_local = (3 if "rotations" in selected else 0) + (
        (3 if cone else 2) if "translations" in selected else 0
    )
    expected_shared = ((2 if cone else 1) if "centre" in selected else 0) + sum(
        name in selected for name in ("tilt", "roll")
    )
    assert parameters.size == 60 * expected_local + expected_shared
    assert set(calibration) == set(selected) - {"translations", "rotations"}
    if "rotations" not in selected:
        np.testing.assert_array_equal(table[:, :3], 0)
    if "translations" not in selected:
        np.testing.assert_array_equal(table[:, 3:], 0)
    if not cone:
        np.testing.assert_array_equal(table[:, 5], 0)


def test_shared_columns_broadcast_once_and_scores_sum_over_views():
    parameters = Parameters(60, ("translations", "centre", "tilt", "roll"), lever=12)
    rng = np.random.default_rng(29)
    with jax.enable_x64(True):
        state = jnp.asarray(rng.normal(size=parameters.size))
        dual = jnp.asarray(rng.normal(size=(60, parameters.k)))
        rows = parameters.expand(state)
        np.testing.assert_array_equal(rows[:, -3:], np.tile(np.asarray(state[-3:]), (60, 1)))
        np.testing.assert_allclose(
            jnp.vdot(rows, dual), jnp.vdot(state, parameters.reduce_score(dual)), rtol=2e-14
        )


@pytest.mark.parametrize(
    ("fit", "cone"),
    [
        ("centre", False),
        (("translations", "centre", "tilt", "roll"), False),
        (("translations", "rotations", "centre", "tilt", "roll"), True),
    ],
)
def test_shared_pose_schur_factor_matches_a_dense_constrained_normal_system(fit, cone):
    n = 9
    parameters = Parameters(n, fit, cone=cone, lever=17)
    rng = np.random.default_rng(294)
    with jax.enable_x64(True):
        gauge = rng.normal(size=(n * parameters.full_dofs, 2))
        prior = MotionPrior(
            n,
            gauge,
            knot_spacing=(5,) * parameters.full_dofs,
            delta=0.03,
            parameters=parameters,
            rotation_scales=np.full(parameters.full_dofs, 0.05),
            noise_sigma=0.02,
        )
        a = rng.normal(size=(n, parameters.k + 2, parameters.k))
        blocks = np.einsum("nij,nik->njk", a, a) + np.eye(parameters.k)
        weights = rng.uniform(0.1, 0.7, n - 1)
        rhs = rng.normal(size=parameters.size)

        # Independent dense assembly: every view has its own local indices,
        # while all views contribute to the same global rows and columns.
        hessian = np.zeros((parameters.size, parameters.size))
        shared = list(range(parameters.local_size, parameters.size))
        for view in range(n):
            local = list(range(view * parameters.nlocal, (view + 1) * parameters.nlocal))
            hessian[np.ix_(local + shared, local + shared)] += blocks[view]
        if parameters.nlocal:
            d = np.kron(np.diff(np.eye(n), axis=0), np.diag(prior.translation_mask))
            laplacian = d.T @ np.diag(np.repeat(weights, parameters.nlocal)) @ d
            u = np.asarray(prior.U)
            q = np.eye(parameters.local_size) - u @ u.T
            hessian[: parameters.local_size, : parameters.local_size] += (
                q @ laplacian @ q + np.diag(prior.rotation_precision)
            )
        constraints = np.pad(np.asarray(prior.G), ((0, parameters.nglobal), (0, 0)))
        nc = constraints.shape[1]
        kkt = np.block([[hessian, constraints], [constraints.T, np.zeros((nc, nc))]])
        expected = np.linalg.solve(kkt, np.r_[rhs, np.zeros(nc)])[: parameters.size]
        actual = prior.solve(
            prior.factor(jnp.asarray(blocks), jnp.asarray(weights)), jnp.asarray(rhs)
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)


def test_centre_and_translation_share_one_identifiable_parallel_offset():
    parameters = Parameters(60, ("centre", "translations"))
    with jax.enable_x64(True):
        prior = MotionPrior(
            60,
            np.zeros((60 * 5, 0)),
            knot_spacing=(10,) * 5,
            delta=0.02,
            parameters=parameters,
        )
        state = np.zeros(parameters.size)
        state[: parameters.local_size].reshape(60, 2)[:, 0] = 3
        state[-1] = 1.25
        projected = np.asarray(prior.project_gauge(jnp.asarray(state)))
        assert abs(projected[: parameters.local_size].reshape(60, 2)[:, 0].mean()) < 2e-14
        assert projected[-1] == 1.25


def test_rotations_pay_the_same_physical_size_prior_at_different_numeric_scales():
    physical = np.tile(np.deg2rad([0.3, -0.2, 0.1]), (60, 1))
    noise_sigma = 0.04
    expected = 0.5 * noise_sigma**2 * np.sum((physical / np.deg2rad(0.5)) ** 2)
    with jax.enable_x64(True):
        for lever in (12, 48):
            parameters = Parameters(60, "rotations", lever=lever)
            scales = np.array([1 / lever] * 3 + [0.7, 1.1])
            prior = MotionPrior(
                60,
                np.zeros((60 * 5, 0)),
                knot_spacing=(10,) * 5,
                delta=0.02,
                parameters=parameters,
                rotation_scales=scales,
                noise_sigma=noise_sigma,
            )
            state = jnp.asarray((physical * lever).reshape(-1))
            # Constant rotations lie in a drift spline, but still pay the size prior.
            np.testing.assert_allclose(prior.value(state, 0.0), expected, rtol=2e-14)
            assert expected > 0


def test_innovation_refinement_holds_global_parameters_exactly():
    parameters = Parameters(60, ("translations", "centre", "roll"))
    rng = np.random.default_rng(914)
    with jax.enable_x64(True):
        prior = MotionPrior(
            60,
            np.zeros((60 * 5, 0)),
            knot_spacing=(10,) * 5,
            delta=0.02,
            parameters=parameters,
        )
        blocks = jnp.broadcast_to(jnp.eye(parameters.k), (60, parameters.k, parameters.k))
        rhs = jnp.asarray(rng.normal(size=parameters.size)).at[-2:].set(1000)
        step = prior.innovation_solve(blocks, jnp.ones(59), rhs)
        np.testing.assert_array_equal(step[-2:], 0)
        assert np.linalg.norm(step[: parameters.local_size]) > 0
        table, _ = parameters.physical(step, np.ones(5))
        np.testing.assert_array_equal(table[:, :3], 0)


@pytest.mark.parametrize("cone", [False, True])
def test_global_geometry_matches_independent_physical_landmarks(cone):
    n = 60
    detector = Detector(48, 48, 0.7, 1.1, center=(0.4, -0.3))
    parameters = Parameters(n, ("centre", "tilt", "roll"), cone=cone, lever=20)
    angles = np.linspace(0, 360, n, endpoint=False)
    nominal = np.tile(np.eye(4), (n, 1, 1))
    nominal[:, :3, :3] = Rotation.from_euler("z", angles[:, None], degrees=True).as_matrix()
    nominal[:, :3, 3] = [0.2, -0.1, 0.7]
    corrections = {"centre_u": 0.6, "centre_v": -0.4, "tilt": 0.3, "roll": 0.25}
    state = np.array([corrections[name] for name in parameters.globals])
    frame = np.tile([[0.4, 12, -0.3], [1, 0, 0], [0, 0, 1], [0, -20, 0.7]], (n, 1, 1))
    with jax.enable_x64(True):
        fitted, fitted_frames = parameters.geometry(
            jnp.asarray(nominal),
            parameters.expand(state),
            np.ones(parameters.full_dofs),
            detector=detector,
            native=(detector.du, detector.dv),
            frames=jnp.asarray(frame) if cone else None,
        )
    tilt, roll = corrections["tilt"] / parameters.lever, corrections["roll"] / parameters.lever
    tilted = nominal.copy()
    tilted[:, :3, :3] = Rotation.from_rotvec([-tilt, 0, 0]).as_matrix() @ nominal[:, :3, :3]
    centre = np.array([detector.center[0] + 0.6 * detector.du, 0, detector.center[1]])
    if cone:
        centre[2] -= 0.4 * detector.dv
        np.testing.assert_allclose(fitted, tilted, atol=1e-14)
        np.testing.assert_array_equal(np.asarray(fitted_frames)[:, 3], frame[:, 3])
        np.testing.assert_allclose(
            np.asarray(fitted_frames)[:, 0] - frame[:, 0],
            np.tile(centre - [detector.center[0], 0, detector.center[1]], (n, 1)),
            atol=1e-14,
        )
        expected_u = [np.cos(roll), 0, np.sin(roll)]
        expected_v = [-np.sin(roll), 0, np.cos(roll)]
        np.testing.assert_allclose(
            np.asarray(fitted_frames)[:, 1], np.tile(expected_u, (n, 1)), atol=1e-14
        )
        np.testing.assert_allclose(
            np.asarray(fitted_frames)[:, 2], np.tile(expected_v, (n, 1)), atol=1e-14
        )
    else:
        points = np.array([[0, 0, 0], [7, -3, 11], [-9, 4, -8]], np.float64)
        lab = np.einsum("nij,kj->nki", tilted[:, :3, :3], points) + tilted[:, None, :3, 3]
        # Physical projection onto a rolled detector basis, not the solver's pose transform.
        displacement = lab - centre
        expected = np.stack(
            [
                displacement @ np.array([np.cos(roll), 0, np.sin(roll)]) / detector.du,
                displacement @ np.array([-np.sin(roll), 0, np.cos(roll)]) / detector.dv,
            ],
            axis=-1,
        )
        actual = np.einsum("nij,kj->nki", np.asarray(fitted)[:, :3, :3], points)
        actual += np.asarray(fitted)[:, None, :3, 3]
        actual = (actual[..., [0, 2]] - detector.center) / [detector.du, detector.dv]
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_analytic_fixture_integrates_density_independently_of_a_voxel_projector():
    from scipy.integrate import quad

    origin = np.array([[1.3, -0.7, 2.4]])
    direction = np.array([[0.4, 0.8, 0.2]])
    direction /= np.linalg.norm(direction)
    expected = quad(lambda t: float(density(origin + t * direction)[0]), -80, 80, epsabs=1e-10)[0]
    np.testing.assert_allclose(line_integrals(origin, direction), expected, atol=1e-10)


def test_detector_space_score_counts_centre_and_roll_before_a_rigid_gauge():
    scan, truth, _, _ = make_case("centre")
    error, _, _ = score(scan_rays(scan), truth)
    np.testing.assert_allclose(error, 2.0, atol=1e-7)
    scan, truth, _, _ = make_case("roll")
    error, _, _ = score(scan_rays(scan), truth)
    assert error > 0.2  # translations/common object rotation cannot undo detector roll
