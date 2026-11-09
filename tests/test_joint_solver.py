"""The fixed joint fit avoids absent-prior work without changing its objective."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.alignment._joint import _solver, _tv  # check-public-imports: allow-private
from tomojax.alignment._joint._fit import _gauge  # check-public-imports: allow-private
from tomojax.alignment._joint._operators import Operators  # check-public-imports: allow-private
from tomojax.alignment._joint._pose import MotionPrior  # check-public-imports: allow-private
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
        gauge = _gauge(nominal, lever=2, native=(0.7, 0.9))
        prior = MotionPrior(8, gauge, knot_spacing=(4,) * 5, delta=0.02)
        image = jnp.asarray(np.random.default_rng(54).uniform(0.1, 0.3, (3, 3, 4)), jnp.float32)
        q = ops.pad_poses(jnp.zeros((8, 5)))
        observations = np.asarray(ops.project(ops.coefficients(ops.nominal, q, ops.valid), image))
        fitted, _, record = solve(
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
            8, _gauge(nominal, lever=2, native=(1, 1)), knot_spacing=(4,) * 5, delta=0.02
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
