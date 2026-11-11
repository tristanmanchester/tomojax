"""The fixed joint fit avoids absent-prior work without changing its objective."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.alignment._joint import _refine, _solver, _tv  # check-public-imports: allow-private
from tomojax.alignment._joint._operators import (  # check-public-imports: allow-private
    Operators,
    photometric_sums,
)
from tomojax.alignment._joint._pose import MotionPrior  # check-public-imports: allow-private
from tomojax.alignment._joint._projection import rigid_gauge  # check-public-imports: allow-private
from tomojax.alignment._joint._solver import (
    Budget,
    _Level,
    solve,
)  # check-public-imports: allow-private
from tomojax.geometry import stack_view_poses


@pytest.mark.parametrize("weight", [0.0, 0.02])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_initial_support_and_optional_tv_work(monkeypatch, weight, interpolation):
    calls = []

    def spy(name, original):
        def checked(*args, **kwargs):
            calls.append(name)
            assert weight > 0, f"disabled image prior executed {name}"
            return original(*args, **kwargs)

        return checked

    names = ("energy", "conductance", "diffusion", "inverse_diagonal")
    for name in names:
        monkeypatch.setattr(_tv, name, spy(name, getattr(_tv, name)))
    grid, detector = tj.Grid(4, 3, 3, 0.7, 0.8, 0.9), tj.Detector(6, 1, 0.7, 0.9)
    geometry = tj.ParallelGeometry(grid, detector, np.arange(8) * 45.0)
    with jax.enable_x64(True):
        nominal = np.asarray(stack_view_poses(geometry, 8))
        ops = Operators(
            grid, detector, nominal, lever=2, native=(0.7, 0.9), interpolation=interpolation
        )
        linearize = ops.linearize

        def checked_initial(q, x, *args):
            # The thin detector sees only z=0; the supplied x0 is positive everywhere.
            np.testing.assert_array_equal(np.asarray(x)[[0, 2]], 0)
            return linearize(q, x, *args)

        monkeypatch.setattr(ops, "linearize", checked_initial)
        gauge = rigid_gauge(nominal, [0.5] * 3 + [0.7, 0.9])
        prior = MotionPrior(8, gauge, knot_spacing=(4,) * 5, delta=0.02)
        image = jnp.asarray(np.random.default_rng(54).uniform(0.1, 0.3, (3, 3, 4)), jnp.float32)
        q = ops.pad_poses(jnp.zeros((8, 5)))
        observations = np.asarray(ops.project(ops.coefficients(ops.nominal, q, ops.valid), image))
        fitted, _, _, record = solve(
            observations,
            None,
            prior,
            weight,
            0.01,
            ops=ops,
            budget=Budget(outer=1, inner=3),
            x0=0.7 * image,
            p0=None,
            pose_sigma=0.001,
            delta=1.0,
            scales=(1.0, 1.0, 1.0),
        )
        jax.block_until_ready(fitted)
    assert len(record["steps"]) == 1
    assert record["final_objective"] < record["initial_objective"]
    assert np.isfinite(fitted).all() and np.min(fitted) >= 0
    assert (record["final_image_prior"] > 0) == (weight > 0)
    assert set(calls) == (set(names) if weight else set())


def test_initializer_does_not_capture_observations(monkeypatch):
    grid, detector = tj.Grid(2, 2, 2, 1, 1, 1), tj.Detector(2, 2, 1, 1)
    geometry = tj.LaminographyGeometry(grid, detector, np.arange(8) * 45.0, tilt_deg=-30)
    with jax.enable_x64(True):
        nominal = np.asarray(stack_view_poses(geometry, 8))
        ops = Operators(
            grid,
            detector,
            nominal,
            lever=2,
            native=(1, 1),
        )
        q = ops.pad_poses(np.zeros((8, 5)))
        prior = MotionPrior(
            8, rigid_gauge(nominal, [0.5] * 3 + [1, 1]), knot_spacing=(4,) * 5, delta=0.02
        )
        y = ops.pad_images(np.random.default_rng(58).normal(size=(8, 2, 2)))
        precision = jnp.ones_like(y)
        level = _Level(
            ops,
            prior,
            y,
            precision,
            lam=0,
            eps=0.01,
            scales=(1, 1, 1),
            delta=1,
            inner=1,
        )
        original_jit, captured = jax.jit, {}

        def inspect(fn, *args, **kwargs):
            if fn.__name__ != "run":
                return original_jit(fn, *args, **kwargs)

            def run(*values):
                captured["initializer"] = jax.make_jaxpr(fn)(*values).consts
                return values[0]

            return run

        monkeypatch.setattr(jax, "jit", inspect)
        level.initial_volume(q, 1.0, 1)
        monkeypatch.setattr(jax, "jit", original_jit)
        constants = captured["initializer"]
        assert all(a is not y and a is not precision for a in constants)


@pytest.mark.parametrize(("predicted", "expected"), [(2.0, 0.5), (0.0, 0.0), (-2.0, 0.0)])
def test_non_descending_model_cannot_justify_less_damping(predicted, expected):
    assert _solver._gain_ratio(1.0, predicted) == expected


def test_per_view_gain_and_background_are_recovered_up_to_their_means():
    rng = np.random.default_rng(311)
    prediction = rng.uniform(0, 1, (6, 9, 7))
    gain = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0])
    background = rng.normal(0, 0.1, (6, 2))
    rows = np.linspace(-1, 1, 9)[None, :, None]
    observed = (
        gain[:, None, None] * prediction + background[:, :1, None] + background[:, 1:, None] * rows
    )
    precision = np.ones_like(prediction)
    precision[2, 3:5] = 0  # missing rows do not bias the fit
    observed[2, 3:5] = 1e3
    with jax.enable_x64(True):
        sums = photometric_sums(
            jnp.asarray(prediction), jnp.asarray(observed), jnp.asarray(precision)
        )
    fitted_gain, fitted_background = _solver._photometric(sums)
    np.testing.assert_allclose(fitted_gain, gain / gain.mean(), rtol=1e-9)
    np.testing.assert_allclose(fitted_background, background - background.mean(0), atol=1e-9)


@pytest.mark.parametrize("all_masked", [False, True])
def test_views_without_a_photometric_fit_keep_identity(all_masked):
    rng = np.random.default_rng(5)
    prediction = rng.uniform(0, 1, (3, 9, 7))
    rows = np.linspace(-1, 1, 9)[None, :, None]
    observed = 2.0 * prediction + 3.0 + 4.0 * rows  # gain 2, constant 3, ramp 4
    precision = np.ones_like(prediction)
    precision[1] = 0.0  # nothing measured: neither gain nor background is identifiable
    if all_masked:
        precision[:] = 0.0
    with jax.enable_x64(True):
        sums = photometric_sums(
            jnp.asarray(prediction), jnp.asarray(observed), jnp.asarray(precision)
        )
    gain, background = _solver._photometric(sums)
    assert gain[1] == 1.0 and np.all(background[1] == 0.0)
    if all_masked:
        assert np.all(gain == 1.0) and np.all(background == 0.0)


def test_a_uniform_prediction_still_fits_its_background():
    # A flat prediction leaves the gain unidentifiable but the background observable.
    rng = np.random.default_rng(6)
    prediction = rng.uniform(0, 1, (3, 9, 7))
    prediction[1] = 0.0
    rows = np.linspace(-1, 1, 9)[None, :, None]
    offsets = np.array([[3.0, 4.0], [5.0, 6.0], [3.0, 4.0]])
    observed = prediction + offsets[:, :1, None] + offsets[:, 1:, None] * rows
    with jax.enable_x64(True):
        sums = photometric_sums(
            jnp.asarray(prediction), jnp.asarray(observed), jnp.asarray(np.ones_like(prediction))
        )
    gain, background = _solver._photometric(sums)
    assert gain[1] == 1.0
    np.testing.assert_allclose(background[1] - background[0], [2.0, 2.0], atol=1e-9)


@pytest.mark.parametrize("masked", [False, True])
def test_held_pose_trial_equals_the_projected_objective(masked):
    # With no pose step, beta = 1 projects at the same poses that beta = 0 holds.
    # Large residuals put pixels in the robust tail, where sw differs from sqrt(precision).
    grid, detector = tj.Grid(5, 5, 3, 0.7, 0.8, 0.9), tj.Detector(7, 3, 0.7, 0.9)
    geometry = tj.ParallelGeometry(grid, detector, [30.0 * i for i in range(6)])
    rng = np.random.default_rng(31)
    with jax.enable_x64(True):
        nominal = np.asarray(stack_view_poses(geometry, 6))
        ops = Operators(
            grid, detector, nominal, lever=2, native=(0.7, 0.9), interpolation="cubic", batch_size=4
        )
        gauge = rigid_gauge(nominal, [0.5] * 3 + [0.7, 0.9])
        prior = MotionPrior(6, gauge, knot_spacing=(4,) * 5, delta=0.02)
        y = ops.pad_images(jnp.asarray(rng.normal(scale=0.5, size=(6, 3, 7)), jnp.float32))
        precision = None
        if masked:
            weights = rng.uniform(0, 2, (6, 3, 7)) * (rng.uniform(size=(6, 3, 7)) > 0.3)
            precision = ops.pad_precision(jnp.asarray(weights, jnp.float32))
        level = _Level(ops, prior, y, precision, lam=0.0, eps=0.0, scales=None, delta=0.05, inner=1)
        c = prior.project_gauge(jnp.asarray(rng.normal(scale=0.2, size=prior.size)))
        x = jnp.asarray(rng.uniform(0.1, 0.3, (3, 5, 5)), jnp.float32)
        dx = jnp.asarray(rng.normal(size=(3, 5, 5)), jnp.float32)
        _, _, sw, j, *_, residual = level.linearize(c, x)
        assert float(jnp.mean(sw**2 < 0.99 * (1 if precision is None else precision))) > 0.2
        step, _ = ops.weighted_forward(ops.prepare(ops.pad_poses(c)), dx, sw, j)
        for a in (1.0, 0.25):
            trial = partial(level.trial, x, c, dx, jnp.zeros_like(c), 1.0, (residual, step, sw))
            held, projected = trial(a, 0.0), trial(a, 1.0)
            np.testing.assert_allclose(held[0], projected[0], rtol=1e-6)
            np.testing.assert_array_equal(held[1], projected[1])


@pytest.mark.parametrize(("shift", "moves"), [(1e-5, False), (0.3, True)])
def test_refinement_stops_on_a_proposal_below_the_pose_tolerance(monkeypatch, shift, moves):
    # Alternating view shifts lie outside the motion prior's smooth space.
    grid, detector = tj.Grid(8, 8, 3, 0.7, 0.7, 0.9), tj.Detector(12, 3, 0.7, 0.9)
    geometry = tj.ParallelGeometry(grid, detector, [15.0 * i for i in range(12)])
    with jax.enable_x64(True):
        nominal = np.asarray(stack_view_poses(geometry, 12))
        ops = Operators(grid, detector, nominal, lever=4, native=(0.7, 0.9), batch_size=4)
        prior = MotionPrior(
            12, rigid_gauge(nominal, [0.25] * 3 + [0.7, 0.9]), knot_spacing=(6,) * 5, delta=0.02
        )
        x = jnp.asarray(np.random.default_rng(8).uniform(0.1, 0.5, (3, 8, 8)), jnp.float32)
        q = np.zeros((ops.padded_n, 5))
        q[: ops.n, 3] = shift * (-1.0) ** np.arange(ops.n)
        y = ops.project(ops.coefficients(ops.nominal, jnp.asarray(q), ops.valid), x)[: ops.n]
        evaluated = []
        value = _refine._Refinement.value
        monkeypatch.setattr(
            _refine._Refinement, "value", lambda self, c: evaluated.append(c) or value(self, c)
        )
        _, record = _refine.refine(x, np.zeros(prior.size), y, None, prior, ops=ops, sigma=1e-3)
    assert bool(record["steps"]) == bool(evaluated) == moves
    if not moves:
        assert record["status"] == "converged"
