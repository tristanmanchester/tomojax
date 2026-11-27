"""Independent contracts of active blocks, global geometry and their exact border."""

from __future__ import annotations

from bench.fit_alignment import (
    density,
    line_integrals,
    make_case,
    scan_rays,
    score,
    solid_line_integrals,
    solid_phantom,
)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# check-public-imports: allow-private
from tomojax.alignment._joint._parameters import Parameters
from tomojax.alignment._joint._pose import MotionPrior  # check-public-imports: allow-private
from tomojax.alignment._joint._projection import (  # check-public-imports: allow-private
    fit_gauge,
    posed,
    rigid_gauge,
    shared_axis_patterns,
)
from tomojax.geometry import Detector


@pytest.mark.parametrize("cone", [False, True])
@pytest.mark.parametrize(
    "fit", ["center", "translations", "rotations", ("translations", "tilt", "roll")]
)
def test_held_parameters_have_no_state_and_are_exactly_zero_in_the_output(fit, cone):
    parameters = Parameters(60, fit, cone=cone, lever=17)
    state = np.linspace(-0.3, 0.7, parameters.size)
    scales = np.array([0.01, 0.02, 0.03, 0.7, 1.1, 0.9])[: parameters.full_dofs]
    table, calibration = parameters.physical(state, scales)
    selected = (fit,) if isinstance(fit, str) else fit
    expected_local = (2 if "rotations" in selected else 0) + (
        (3 if cone else 2) if "translations" in selected else 0
    )
    expected_shared = ((2 if cone else 1) if "center" in selected else 0) + sum(
        name in selected for name in ("tilt", "roll")
    )
    assert parameters.size == 60 * expected_local + expected_shared
    assert set(calibration) == set(selected) - {"translations", "rotations"}
    if "rotations" not in selected:
        np.testing.assert_array_equal(table[:, :3], 0)
    np.testing.assert_array_equal(table[:, 2], 0)  # each view's angle is taken as recorded
    if "translations" not in selected:
        np.testing.assert_array_equal(table[:, 3:], 0)
    if not cone:
        np.testing.assert_array_equal(table[:, 5], 0)


def test_shared_columns_broadcast_once_and_scores_sum_over_views():
    parameters = Parameters(60, ("translations", "center", "tilt", "roll"), lever=12)
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
    ("fit", "cone", "n", "independent"),
    [
        ("center", False, 9, False),
        (("translations", "center", "tilt", "roll"), False, 9, False),
        (("translations", "rotations", "center", "tilt", "roll"), True, 9, False),
        (("translations", "rotations", "center", "tilt", "roll"), True, 241, False),
        (("translations", "rotations", "center", "tilt", "roll"), True, 321, True),
    ],
)
def test_shared_pose_schur_factor_matches_a_dense_constrained_normal_system(
    fit, cone, n, independent
):
    from scipy.interpolate import make_interp_spline

    parameters = Parameters(n, fit, cone=cone, lever=17)
    rng = np.random.default_rng(294)
    with jax.enable_x64(True):
        gauge = rng.normal(size=(n * parameters.full_dofs, 2))
        spacing = (5,) * parameters.full_dofs if n == 9 else (5, 5, 5, 1, 2, 4)
        prior = MotionPrior(
            n,
            gauge,
            knot_spacing=spacing,
            delta=0.03,
            parameters=parameters,
            independent=independent,
            rotation_scales=np.full(parameters.full_dofs, 0.05),
            noise_sigma=0.02,
        )
        a = rng.normal(size=(n, parameters.k + 2, parameters.k))
        blocks = np.einsum("nij,nik->njk", a, a) + np.eye(parameters.k)
        weights = rng.uniform(0.1, 0.7, (n - 1, parameters.nlocal))
        if independent:
            assert prior.matrices.chain is not None  # exercise the implicit spline representation
            weights.fill(0)
        rhs = rng.normal(size=parameters.size)

        # Independent dense assembly: every view has its own local indices,
        # while all views contribute to the same global rows and columns.
        hessian = np.zeros((parameters.size, parameters.size))
        shared = list(range(parameters.local_size, parameters.size))
        for view in range(n):
            local = list(range(view * parameters.nlocal, (view + 1) * parameters.nlocal))
            hessian[np.ix_(local + shared, local + shared)] += blocks[view]
        if parameters.nlocal:
            # Every local column has first-difference coupling with its own link weights;
            # the drift spline (outside which translations are penalized) spans none of the
            # rotation columns, so Q leaves those as they are.
            d = np.kron(np.diff(np.eye(n), axis=0), np.eye(parameters.nlocal))
            laplacian = d.T @ np.diag(weights.reshape(-1)) @ d
            q = np.eye(parameters.local_size)
            for i, column in enumerate(parameters.local):
                if column >= 3:
                    sites = np.r_[np.arange(0, n - 1, spacing[column]), n - 1]
                    b = make_interp_spline(sites, np.eye(len(sites)), k=min(3, len(sites) - 1))(
                        np.arange(n)
                    )
                    basis = np.linalg.qr(b)[0]
                    q[i :: parameters.nlocal, i :: parameters.nlocal] -= basis @ basis.T
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


def test_center_and_translation_share_one_identifiable_parallel_offset():
    parameters = Parameters(60, ("center", "translations"))
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
    physical = np.tile(np.deg2rad([0.3, -0.2]), (60, 1))  # the axis's two wobble components
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
    parameters = Parameters(60, ("translations", "center", "roll"))
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
        step = prior.innovation_solve(blocks, jnp.ones((59, parameters.nlocal)), rhs)
        np.testing.assert_array_equal(step[-2:], 0)
        assert np.linalg.norm(step[: parameters.local_size]) > 0
        table, _ = parameters.physical(step, np.ones(5))
        np.testing.assert_array_equal(table[:, :3], 0)


@pytest.mark.parametrize("cone", [False, True])
def test_global_geometry_matches_independent_physical_landmarks(cone):
    n = 60
    detector = Detector(48, 48, 0.7, 1.1, center=(0.4, -0.3))
    parameters = Parameters(n, ("center", "tilt", "roll"), cone=cone, lever=20)
    angles = np.linspace(0, 360, n, endpoint=False)
    nominal = np.tile(np.eye(4), (n, 1, 1))
    nominal[:, :3, :3] = Rotation.from_euler("z", angles[:, None], degrees=True).as_matrix()
    nominal[:, :3, 3] = [0.2, -0.1, 0.7]
    corrections = {"center_u": 0.6, "center_v": -0.4, "tilt": 0.3, "roll": 0.25}
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
    center = np.array([detector.center[0] + 0.6 * detector.du, 0, detector.center[1]])
    if cone:
        center[2] -= 0.4 * detector.dv
        np.testing.assert_allclose(fitted, tilted, atol=1e-14)
        np.testing.assert_array_equal(np.asarray(fitted_frames)[:, 3], frame[:, 3])
        np.testing.assert_allclose(
            np.asarray(fitted_frames)[:, 0] - frame[:, 0],
            np.tile(center - [detector.center[0], 0, detector.center[1]], (n, 1)),
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
        displacement = lab - center
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


def test_solid_fixture_integrates_ellipsoid_chords_independently():
    # Riemann sum of each ellipsoid's inside test along a ray through one center.
    centers, shape, densities = solid_phantom()
    origin = centers[:1] + np.array([[0.3, -0.2, 0.1]])
    direction = np.array([[0.4, 0.8, 0.2]])
    direction /= np.linalg.norm(direction)
    t = np.arange(-80, 80, 1e-4)
    points = origin + t[:, None] * direction
    inside = [
        value * (np.einsum("ni,ij,nj->n", points - c, m, points - c) <= 1)
        for c, m, value in zip(centers, shape, densities, strict=True)
    ]
    expected = np.sum(inside) * 1e-4
    assert expected > 0.05
    np.testing.assert_allclose(solid_line_integrals(origin, direction), expected, atol=2e-5)


def test_detector_space_score_counts_center_and_roll_before_a_rigid_gauge():
    scan, truth, _, _ = make_case("center")
    error, _, _ = score(scan_rays(scan), truth)
    np.testing.assert_allclose(error, 2.0, atol=1e-7)
    scan, truth, _, _ = make_case("roll")
    error, _, _ = score(scan_rays(scan), truth)
    assert error > 0.2  # translations/common object rotation cannot undo detector roll


@pytest.mark.parametrize(
    ("fit", "cone"),
    [
        (("translations", "rotations"), False),
        (("translations", "rotations"), True),
        (("rotations",), False),
    ],
)
def test_rotation_increments_gradient_is_the_weighted_apply(fit, cone):
    # Robust translation weights and constant rotation weights both come from the value.
    views = 17
    parameters = Parameters(views, fit, cone=cone, lever=10)
    rng = np.random.default_rng(75)
    with jax.enable_x64(True):
        prior = MotionPrior(
            views,
            rng.normal(size=(views * parameters.full_dofs, 3)),
            knot_spacing=(5,) * parameters.full_dofs,
            delta=0.02,
            parameters=parameters,
            rotation_scales=np.full(parameters.full_dofs, 0.05),
            noise_sigma=0.03,
        )
        pose = jnp.asarray(rng.normal(scale=0.05, size=parameters.size))
        gradient = jax.grad(lambda p: prior.value(p, 0.4))(pose)
        np.testing.assert_allclose(
            gradient, prior.apply(pose, prior.weights(pose, 0.4)), rtol=2e-12, atol=2e-12
        )


@pytest.mark.parametrize("tilt", [0.0, -30.0])
@pytest.mark.parametrize(("name", "sign"), [("tilt", -1.0), ("roll", 1.0)])
def test_a_shared_tilt_or_roll_is_its_per_view_pattern_to_first_order(tilt, name, sign):
    # The poses a small shared tilt (a lab Rx(-t)) or parallel roll (about the beam through
    # the detector center) produce are the ones its per-view pattern produces: the tangent
    # comes from the geometry, not from a copy of its formula.
    import tomojax as tj
    from tomojax.geometry import stack_view_poses

    grid = tj.Grid(8, 8, 6, 1.0, 1.0, 1.0, vol_center=(0.4, -0.3, 1.5))
    detector = Detector(12, 9, 1.0, 1.0, (0.7, -0.4))
    geometry = tj.LaminographyGeometry(grid, detector, np.arange(0, 360, 40.0), tilt=tilt)
    nominal = np.asarray(stack_view_poses(geometry, 9), np.float64)
    fit = ("translations", "rotations", name)
    parameters = Parameters(9, fit, lever=6.0)
    scales = np.asarray([1 / 6.0] * 3 + [1.0, 1.0])
    pattern = shared_axis_patterns(nominal, scales, fit, detector.center)[:, 0].reshape(9, 5)
    eps = 1e-5
    with jax.enable_x64(True):
        state = np.zeros(parameters.size)
        state[-1] = eps * parameters.lever  # eps radians of the shared parameter
        rows = parameters.expand(jnp.asarray(state))
        shared = np.asarray(
            parameters.geometry(
                jnp.asarray(nominal), rows, jnp.asarray(scales), detector=detector, native=(1, 1)
            )[0]
        )

        def written(table):
            return np.asarray(posed(jnp.asarray(nominal), jnp.asarray(table), jnp.asarray(scales)))

        assert np.abs(shared - written(sign * eps * pattern)).max() < 1e-8  # second order in eps
        assert np.abs(shared - written(-sign * eps * pattern)).max() > 1e-6  # the sign matters
    # Per-view rotations hold each view's angle, so any spin in a pattern must be a constant
    # object rotation, which is the rigid gauge: none for a tilt or an axial roll.
    spin = pattern[:, 2]
    assert np.ptp(spin) < 1e-12
    if name == "tilt" or tilt == 0:
        assert np.abs(spin).max() < 1e-12


def test_the_constrained_solve_leaves_no_per_view_component_along_a_shared_pattern():
    import tomojax as tj
    from tomojax.geometry import stack_view_poses

    grid, detector = tj.Grid(8, 8, 6, 1.0, 1.0, 1.0), Detector(12, 9, 1.0, 1.0)
    nominal = np.asarray(
        stack_view_poses(tj.ParallelGeometry(grid, detector, np.arange(0, 360, 12.0)), 30)
    )
    fit = ("translations", "rotations", "tilt", "roll")
    parameters = Parameters(30, fit, lever=6.0)
    scales = np.asarray([1 / 6.0] * 3 + [1.0, 1.0])
    pattern = shared_axis_patterns(nominal, scales, fit, detector.center)
    local = pattern.reshape(30, 5, -1)[:, list(parameters.local)].reshape(parameters.local_size, -1)
    rng = np.random.default_rng(31)
    a = rng.normal(size=(30, parameters.k + 2, parameters.k))
    blocks = jnp.asarray(np.einsum("nij,nik->njk", a, a) + np.eye(parameters.k))
    rhs = jnp.asarray(rng.normal(size=parameters.size))
    weights = jnp.full((29, parameters.nlocal), 0.3)
    with jax.enable_x64(True):

        def along(gauge):
            prior = MotionPrior(
                30,
                gauge,
                knot_spacing=(5,) * 5,
                delta=0.02,
                parameters=parameters,
                rotation_scales=scales,
                noise_sigma=0.02,
            )
            step = np.asarray(prior.solve(prior.factor(blocks, weights), rhs))[
                : parameters.local_size
            ]
            return np.abs(local.T @ step).max() / np.linalg.norm(step)

        rigid = rigid_gauge(nominal, scales)
        assert along(np.column_stack([rigid, pattern])) < 1e-10
        assert along(rigid) > 1e-3  # without the patterns the per-view poses take part of them


@pytest.mark.parametrize("name", ["tilt", "roll"])
def test_a_shared_tilt_or_roll_is_a_per_view_direction_the_fit_gauge_removes(name):
    # R1b's runaway: a shared tilt or roll and a per-view pose pattern give the same
    # projections, and the neighbor-increment prior does not see the pattern, so only the
    # size prior held that direction. The pattern comes here from the geometry (no fit code):
    # a tilted axis, or the scene turned about the beam through the detector center for a
    # rolled detector. The fit's gauge must contain it, so the per-view poses cannot carry it.
    import tomojax as tj
    from tomojax.geometry import RotationAxisGeometry, stack_view_poses

    grid = tj.Grid(8, 8, 6, 1.0, 1.0, 1.0)
    detector = Detector(12, 9, 1.0, 1.0, (0.7, -0.4))
    angles = np.arange(0, 360, 12.0)
    n, eps = len(angles), 1e-3
    nominal = np.asarray(
        stack_view_poses(tj.ParallelGeometry(grid, detector, angles), n), np.float64
    )
    if name == "tilt":
        shared = RotationAxisGeometry(grid, detector, angles, (0.0, np.sin(eps), np.cos(eps)))
        moved = np.asarray(stack_view_poses(shared, n), np.float64)
    else:
        shared = RotationAxisGeometry(
            grid, detector, angles, (0, 0, 1), detector_roll=np.rad2deg(eps)
        )
        pivot = np.array([detector.center[0], 0.0, detector.center[1]])
        turn = np.eye(4)
        turn[:3, :3] = Rotation.from_rotvec(
            [0.0, eps, 0.0]
        ).as_matrix()  # the scene with the detector
        turn[:3, 3] = pivot - turn[:3, :3] @ pivot
        moved = turn @ nominal
    local = np.einsum("nji,njk->nik", nominal[:, :3, :3], moved[:, :3, :3])
    table = np.zeros((n, 6))
    table[:, [1, 0, 2]] = Rotation.from_matrix(local).as_euler("YXZ", degrees=True)
    table[:, [3, 5, 4]] = moved[:, :3, 3] - nominal[:, :3, 3]
    volume = np.random.default_rng(3).random((6, 8, 8)).astype(np.float32)
    blank = tj.Scan(np.zeros((n, 9, 12), np.float32), tj.ParallelGeometry(grid, detector, angles))
    by_poses = np.asarray(tj.project(blank.with_poses(table).geometry, volume))
    by_shared = np.asarray(tj.project(shared, volume))
    # The degeneracy is real: per-view poses (with no spin) give the shared change's data.
    assert np.abs(by_poses - by_shared).max() < 1e-3 * np.abs(by_shared).max()
    assert np.abs(table[:, 2]).max() < np.degrees(eps**2)  # no spin to first order
    # In the fit's units (object-frame rotation vector over its scale, lab x and z over theirs),
    # the pattern lies in the fit's gauge to first order in eps.
    lever = 6.0
    scales = np.asarray([1 / lever] * 3 + [1.0, 1.0])
    direction = np.column_stack(
        [Rotation.from_matrix(local).as_rotvec() / scales[:3], table[:, [3, 4]] / scales[3:]]
    ).reshape(-1)
    fit = ("translations", "rotations", "tilt", "roll")
    gauge, _ = fit_gauge(nominal, scales, fit, detector.center)
    coefficients = np.linalg.lstsq(gauge, direction, rcond=None)[0]
    left = np.linalg.norm(direction - gauge @ coefficients) / np.linalg.norm(direction)
    assert left < 10 * eps  # second order; R1b's rigid gauge leaves it whole (about 1)
