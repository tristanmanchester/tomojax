"""The fixed joint fit avoids absent-prior work without changing its objective."""

from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.alignment._joint import _refine, _solver  # check-public-imports: allow-private
from tomojax.alignment._joint._operators import (  # check-public-imports: allow-private
    Operators,
    profile,
    profiled_residual,
)
from tomojax.alignment._joint._pose import MotionPrior  # check-public-imports: allow-private
from tomojax.alignment._joint._projection import rigid_gauge  # check-public-imports: allow-private
from tomojax.alignment._joint._solver import (
    Budget,
    solve,
)  # check-public-imports: allow-private
from tomojax.geometry import stack_view_poses

from . import _joint_stacks as js


@pytest.mark.parametrize("weight", [0.0, 0.02])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_initial_support_and_optional_tv_work(monkeypatch, weight, interpolation):
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
        prior = MotionPrior(
            8,
            gauge,
            knot_spacing=(4,) * 5,
            delta=0.02,
            parameters=ops.parameters,
            rotation_scales=np.ones(5),
        )
        image = jnp.asarray(np.random.default_rng(54).uniform(0.1, 0.3, (3, 3, 4)), jnp.float32)
        q = ops.pad_poses(jnp.zeros((8, ops.dofs)))
        observations = np.asarray(ops.project(ops.coefficients(ops.nominal, q, ops.valid), image))
        fitted, _, _, record = solve(
            observations,
            None,
            prior,
            weight,
            0.01,
            ops=ops,
            budget=Budget(iterations=1, image_iterations=3),
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


def test_each_views_offset_and_ramp_are_eliminated_relative_to_the_level_the_views_share():
    # Per-view offsets with a nonzero common part: each view's deviation is profiled out
    # and reported; the level every view shares belongs to the data (an image can produce
    # it, so profiling it away would leave the image a free pedestal) and is not reported.
    grid, detector = tj.Grid(6, 6, 4, 0.7, 0.7, 0.7), tj.Detector(8, 5, 0.7, 0.7)
    geometry = tj.ParallelGeometry(grid, detector, np.arange(12) * 15.0)
    with jax.enable_x64(True):
        nominal = np.asarray(stack_view_poses(geometry, 12))
        ops = Operators(grid, detector, nominal, lever=2, native=(0.7, 0.7))
        prior = MotionPrior(
            12,
            rigid_gauge(nominal, [0.5] * 3 + [0.7, 0.7]),
            knot_spacing=(4,) * 5,
            delta=0.02,
            parameters=ops.parameters,
            rotation_scales=np.ones(5),
        )
        image = jnp.asarray(np.random.default_rng(9).uniform(0.1, 0.3, (4, 6, 6)), jnp.float32)
        q = ops.pad_poses(jnp.zeros((12, ops.dofs)))
        clean = np.asarray(ops.project(ops.coefficients(ops.nominal, q, ops.valid), image))[:12]
        offsets = np.random.default_rng(10).normal(0, 0.05, (12, 2))
        rows = np.linspace(-1, 1, 5)[None, :, None]
        observations = (clean + offsets[:, :1, None] + offsets[:, 1:, None] * rows).astype(
            np.float32
        )
        given = observations.copy()
        fitted, c, y, _ = solve(
            observations,
            None,
            prior,
            0.0,
            0.01,
            ops=ops,
            budget=Budget(iterations=2, image_iterations=3),
            x0=image,
            p0=None,
            pose_sigma=0.001,
            delta=1.0,
            scales=(1.0, 1.0, 1.0),
        )
        reported = np.asarray(
            ops.view_summary(ops.pad_poses(c), fitted, js.stack(ops, ops.pad_images(y)), None)[0]
        )
    # The nuisance is eliminated inside each solve: the data are never rescaled.
    np.testing.assert_array_equal(np.asarray(y), given)
    np.testing.assert_array_equal(observations, given)
    assert np.isfinite(fitted).all() and np.min(fitted) >= 0
    # Uniform precision: every view weighs the same, so the shared level is the plain mean.
    assert np.abs(offsets.mean(0)).max() > 0.015  # the injected common part is not small
    np.testing.assert_allclose(reported, offsets - offsets.mean(0), atol=0.01)


def _linearized(half):
    """A small tilted scan's linearization, outliers in the loss tail, with FP16 or FP32 storage."""
    # Non-uniform precision makes the profile oblique; outliers put pixels in the
    # loss tail, where the robust weights differ from the precision.
    grid, detector = tj.Grid(6, 6, 4, 0.7, 0.7, 0.7), tj.Detector(8, 6, 0.7, 0.7)
    geometry = tj.LaminographyGeometry(grid, detector, np.arange(10) * 36.0, tilt=20)
    rng = np.random.default_rng(12)
    nominal = np.asarray(stack_view_poses(geometry, 10))
    # Cubic sampling: linear interpolation's pose derivative has kinks at
    # voxel boundaries that a finite difference straddles.
    ops = Operators(grid, detector, nominal, lever=2, native=(0.7, 0.7), interpolation="cubic")
    ops._half = half
    x = jnp.asarray(rng.uniform(0.1, 0.3, ops.shape), jnp.float32)
    c = jnp.asarray(rng.normal(0, 0.05, 10 * ops.dofs))
    q = ops.pad_poses(c)
    y = np.asarray(ops.project(ops.coefficients(ops.nominal, q, ops.valid), x))[:10]
    y = y + rng.normal(0, 0.01, y.shape) + 0.3 * (rng.uniform(size=y.shape) < 0.1)
    precision = ops.pad_precision(jnp.asarray(rng.uniform(0.5, 1.5, y.shape), jnp.float32))
    y, delta = ops.pad_images(y), 0.05
    out = js.linearize(ops, q, x, y, precision, delta)
    return ops, x, c, y, precision, delta, out, rng


def test_linearization_is_the_gradient_of_the_profiled_robust_objective():
    # The chain rule alone: the step's weights and columns kept in FP32.
    # The JAX operators: this checks the objective's calculus, not CUDA column precision.
    with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
        ops, x, c, y, precision, delta, out, rng = _linearized(half=False)
        gx, _, _, _, score, _, _, _, _, residual, _ = out
        tail = np.mean(np.abs(np.asarray(residual)[:10]) > delta)
        assert tail > 0.05

        stacked = js.stack(ops, y), js.stack(ops, precision)

        def objective(dx, dc):
            return float(ops.objective(ops.pad_poses(c + dc), x + dx, *stacked, delta)[0])

        dx = jnp.asarray(rng.normal(size=ops.shape), jnp.float32)
        dc = jnp.asarray(rng.normal(size=c.shape))

        def step(direction):
            # The step that moves the residuals by 1% of delta, measured along the direction.
            # Larger steps meet the robust loss's curvature (about 3e-3 at 3%); smaller ones
            # the FP32 projector's rounding.
            probe = 1e-3
            q = ops.pad_poses(c + probe * direction[1])
            moved = js.linearize(ops, q, x + probe * direction[0], y, precision, delta)[-2]
            rate = float(jnp.max(jnp.abs(moved[:10] - residual[:10]))) / probe
            return 0.01 * delta / rate

        for direction, analytic in (
            ((dx, 0 * dc), float(jnp.vdot(gx.astype(jnp.float64), dx))),
            ((0 * dx, dc), float(jnp.vdot(score, dc))),
        ):
            h = step(direction)
            plus = objective(*(h * d for d in direction))
            minus = objective(*(-h * d for d in direction))
            assert (plus - minus) / (2 * h) == pytest.approx(analytic, rel=1e-3)


def test_translation_prior_scores_use_the_stored_columns_and_weights():
    with jax.enable_x64(True):
        ops, _, _, _, precision, _, out, _ = _linearized(half=True)
        weights, columns, residual = out[2], out[3], out[-2]
        scores = ops.translation_scores(
            *(js.stack(ops, a) for a in (weights, columns, residual, precision))
        )
        weighted = weights.astype(jnp.float32) * profiled_residual(residual, precision)
        selected = columns[:, ops.parameters.translation_columns].astype(jnp.float32)
        expected = jnp.einsum("vkij,vij->vk", selected, weighted)[: ops.n]
        np.testing.assert_allclose(scores, expected, rtol=2e-6, atol=1e-7)


def test_translation_scale_preserves_the_fp32_dot_in_full_and_partial_batches():
    # Opposite pixel pairs have exactly zero offset and row ramp. Thus only the
    # accumulation of the many signed FP32 products can change this score.
    grid, detector = tj.Grid(8, 8, 8, 1, 1, 1), tj.Detector(256, 128, 1, 1)
    n = 11
    geometry = tj.ParallelGeometry(grid, detector, np.arange(n) * (360 / n))
    ops = Operators(grid, detector, stack_view_poses(geometry, n), lever=4, native=(1, 1))
    shape = (ops.padded_n, detector.nv, detector.nu)
    rng = np.random.default_rng(17)
    pairs = rng.normal(size=(*shape[:2], detector.nu // 2)).astype(np.float32)
    residual = jnp.asarray(np.stack((pairs, -pairs), axis=-1).reshape(shape))
    weights = jnp.asarray(rng.uniform(0.1, 1, shape), jnp.float16)
    columns = jnp.asarray(rng.normal(size=(shape[0], ops.dofs, *shape[1:])), jnp.float16)
    with jax.enable_x64(True):
        scores = ops.translation_scores(*(ops.stack(a) for a in (weights, columns, residual)), None)
        expected = []
        for start in range(0, n, ops.batch):
            views = slice(start, min(start + ops.batch, n))
            selected = columns[views][:, ops.parameters.translation_columns].astype(jnp.float32)
            res = weights[views].astype(jnp.float32) * residual[views]
            expected.append(jnp.einsum("vkij,vij->vk", selected, res))
        np.testing.assert_array_equal(scores, jnp.concatenate(expected))


def test_fp16_step_storage_stays_within_its_rounding_of_fp32():
    # FP16 keeps 11 significant bits (4.9e-4 relative); sums over many rounded weights and
    # columns stay within a few parts in a thousand of the FP32 step (measured 3e-4 to 8e-4).
    # That the score and Gram matrix use the same stored columns is
    # test_pose_normal_blocks_use_the_same_columns_as_the_schur_operator.
    with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
        half = _linearized(half=True)[6]
        full = _linearized(half=False)[6]
    for k in (0, 4, 5):  # image gradient, pose score, pose Gram matrix
        a, b = np.asarray(half[k], np.float64), np.asarray(full[k], np.float64)
        assert np.linalg.norm(a - b) < 5e-3 * np.linalg.norm(b)
        assert np.linalg.norm(a - b) > 0  # the step really is stored in FP16


@pytest.mark.parametrize(("predicted", "expected"), [(2.0, 0.5), (0.0, 0.0), (-2.0, 0.0)])
def test_non_descending_model_cannot_justify_less_damping(predicted, expected):
    assert _solver._gain_ratio(1.0, predicted) == expected


def _detrended(images, weights):
    """Per view and column: weighted least-squares residual on [1, v] (independent oracle)."""
    rows = np.linspace(-1, 1, images.shape[-2])
    out = np.array(images, np.float64)
    for view in np.ndindex(images.shape[:-2]):
        w = np.sqrt(weights[view[0]]).ravel()
        design = np.stack([np.ones_like(rows), rows], 1).repeat(images.shape[-1], 0)
        if not w.any():
            continue
        fit = np.linalg.lstsq(design * w[:, None], images[view].ravel() * w, rcond=None)[0]
        out[view] -= (design @ fit).reshape(images.shape[-2:])
    return out


def test_profiled_residual_removes_each_views_weighted_offset_and_ramp():
    rng = np.random.default_rng(311)
    residual = rng.normal(size=(4, 9, 7))
    precision = rng.uniform(0.5, 2.0, residual.shape)
    precision[2, 3:5] = 0  # missing rows neither bias the fit nor receive it
    residual[2, 3:5] = 1e3
    precision[3] = 0  # a view without weight is left alone
    with jax.enable_x64(True):
        actual = np.asarray(profiled_residual(jnp.asarray(residual), jnp.asarray(precision)))
    expected = np.where(precision > 0, _detrended(residual, precision), 0)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    # An offset and a ramp added to a view leave its profiled residual unchanged.
    rows = np.linspace(-1, 1, 9)[None, :, None]
    shifted = residual + rng.normal(size=(4, 1, 1)) + rng.normal(size=(4, 1, 1)) * rows
    with jax.enable_x64(True):
        again = np.asarray(profiled_residual(jnp.asarray(shifted), jnp.asarray(precision)))
    np.testing.assert_allclose(again[:3], actual[:3], atol=1e-9)


def test_profile_projects_whitened_columns_view_by_view():
    rng = np.random.default_rng(7)
    s = rng.uniform(0.2, 1.5, (3, 8, 5))
    s[1, :, :] = 0.0
    columns = rng.normal(size=(3, 2, 8, 5))
    with jax.enable_x64(True):
        actual = np.asarray(profile(jnp.asarray(columns), jnp.asarray(s)))
        twice = np.asarray(profile(jnp.asarray(actual), jnp.asarray(s)))
    # Whitened columns s*c project to s * (c less its s**2-weighted fit).
    plain = np.where(s[:, None] > 0, columns / np.where(s[:, None] > 0, s[:, None], 1), 0)
    expected = s[:, None] * _detrended(plain, s**2)
    expected[1] = columns[1]
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(twice, actual, atol=1e-10)  # a projection


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
        prior = MotionPrior(
            6,
            gauge,
            knot_spacing=(4,) * 5,
            delta=0.02,
            parameters=ops.parameters,
            rotation_scales=np.ones(5),
        )
        y = ops.pad_images(jnp.asarray(rng.normal(scale=0.5, size=(6, 3, 7)), jnp.float32))
        precision = None
        if masked:
            weights = rng.uniform(0, 2, (6, 3, 7)) * (rng.uniform(size=(6, 3, 7)) > 0.3)
            precision = ops.pad_precision(jnp.asarray(weights, jnp.float32))
        level = js.placed_level(
            ops, prior, y, precision, lam=0.0, eps=0.0, scales=None, delta=0.05, inner=1
        )
        stacks = level.stacks
        c = prior.project_gauge(jnp.asarray(rng.normal(scale=0.2, size=prior.size)))
        x = jnp.asarray(rng.uniform(0.1, 0.3, (3, 5, 5)), jnp.float32)
        dx = jnp.asarray(rng.normal(size=(3, 5, 5)), jnp.float32)
        _, _, sw, j, *_, residual, shared = level.linearize(c, x)
        weights = sw.data.astype(jnp.float32)
        assert float(jnp.mean(weights**2 < 0.99 * (1 if precision is None else precision))) > 0.2
        q = ops.prepare(ops.pad_poses(c))
        step, _ = ops.weighted_forward(q, dx, sw, j, level.precision, stacks["r"])
        for a in (1.0, 0.25):
            held = (residual, step, sw, shared.moments, shared.of_image(dx))
            trial = partial(level.trial, x, c, dx, jnp.zeros_like(c), 1.0, held)
            held, projected = trial(a, 0.0), trial(a, 1.0)
            np.testing.assert_allclose(held[0], projected[0], rtol=1e-6)
            np.testing.assert_array_equal(held[1], projected[1])


@pytest.mark.parametrize(("shift", "moves"), [(1e-5, False), (0.3, True)])
def test_refinement_stops_on_a_proposal_below_the_pose_tolerance(shift, moves):
    # Alternating view shifts lie outside the motion prior's smooth space.
    grid, detector = tj.Grid(8, 8, 3, 0.7, 0.7, 0.9), tj.Detector(12, 3, 0.7, 0.9)
    geometry = tj.ParallelGeometry(grid, detector, [15.0 * i for i in range(12)])
    with jax.enable_x64(True):
        nominal = np.asarray(stack_view_poses(geometry, 12))
        ops = Operators(grid, detector, nominal, lever=4, native=(0.7, 0.9), batch_size=4)
        prior = MotionPrior(
            12,
            rigid_gauge(nominal, [0.25] * 3 + [0.7, 0.9]),
            knot_spacing=(6,) * 5,
            delta=0.02,
            parameters=ops.parameters,
            rotation_scales=np.ones(5),
        )
        x = jnp.asarray(np.random.default_rng(8).uniform(0.1, 0.5, (3, 8, 8)), jnp.float32)
        q = np.zeros((ops.padded_n, ops.dofs))
        q[: ops.n, ops.parameters.local.index(3)] = shift * (-1.0) ** np.arange(ops.n)
        y = ops.project(ops.coefficients(ops.nominal, jnp.asarray(q), ops.valid), x)[: ops.n]
        _, record, _ = _refine.refine(x, np.zeros(prior.size), y, None, prior, ops=ops, sigma=1e-3)
    assert bool(record["steps"]) == moves
    if not moves:
        assert record["status"] == "pose_steps_below_tolerance"


@pytest.mark.parametrize(("promised", "stops"), [(1e-9, True), (1e-3, False)])
def test_refinement_stops_when_no_trial_lowers_a_proposal_below_the_rounding(promised, stops):
    # At converged poses no trial passes the Armijo test, and what the step promises is below
    # the objective's FP32 rounding; raised damping would only promise less. The step reports
    # convergence after one line search (K11 67208 spent up to 30 trials a step there). A
    # promise above the rounding keeps raising the damping, as before.
    views, k = 6, 2
    gp = np.full(views * k, np.sqrt(promised / (views * k)))
    prior = SimpleNamespace(
        weights=lambda c, w: None,
        reduce_score=lambda v: np.zeros(views * k),
        apply=lambda c, w: np.zeros(views * k),
        expand=lambda v: np.reshape(v, (views, k)),
        innovation_solve=lambda damped, w, g: g,
    )
    problem = object.__new__(_refine._Refinement)
    problem.prior, problem.anchor, problem.c0 = prior, np.zeros((views, k, k)), np.zeros(views * k)
    problem.weight = 1.0
    problem.value = lambda c: 1.0  # flat: no trial lowers it
    stats = (gp, np.broadcast_to(np.eye(k), (views, k, k)), None)
    result = problem.step(np.zeros(views * k), stats, 1.0, 1.0)  # damping already raised
    assert result == ("pose_steps_below_tolerance" if stops else "line_search_stalled")


def _replicated_tent(n):
    """[1, 2, 1] / 4 with the edges replicated, as a dense matrix."""
    k = np.zeros((n, n))
    for i in range(n):
        for offset, tap in ((-1, 0.25), (0, 0.5), (1, 0.25)):
            k[i, min(max(i + offset, 0), n - 1)] += tap
    return k


def test_the_pedestal_costs_nothing_to_the_per_view_profile_and_is_held_by_the_shared_term():
    # c / sqrt(R^2 - r^2), R the detector's half-width, projects to pi c at every u inside R:
    # with its amplitude linear in z, every view sees one offset and one row ramp, the same in
    # every view. The per-view profile removes it whole; only the shared term costs it.
    n, nz, views = 16, 6, 24
    grid, detector = tj.Grid(n, n, nz, 1, 1, 1), tj.Detector(n, nz, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, views, endpoint=False))
    s, radius = 8, n / 2  # voxel means from 8 x 8 in-plane samples
    yy, xx = (np.mgrid[: n * s, : n * s] + 0.5) / s - radius
    inside = np.hypot(xx, yy) < radius
    disc = np.where(inside, 1 / np.sqrt(np.maximum(radius**2 - xx**2 - yy**2, 1e-9)), 0)
    disc = disc.reshape(n, s, n, s).mean((1, 3))
    pedestal = ((1 + 0.5 * np.linspace(-1, 1, nz))[:, None, None] * disc).astype(np.float32)
    seen = np.asarray(tj.project(geometry, pedestal))  # the residual against empty data
    # The level compares through K = [1, 2, 1] / 4 along v and u, edges replicated.
    kv, ku = (_replicated_tent(m) for m in (nz, n))
    seen = np.einsum("ij,vjk,lk->vil", kv, seen, ku)
    rows = np.linspace(-1, 1, nz)[:, None]
    b = np.array([seen.sum(), (rows * seen).sum()])
    gram = views * n * np.array([[nz, rows.sum()], [rows.sum(), (rows**2).sum()]])
    expected = 0.5 * b @ np.linalg.solve(gram, b)
    with jax.enable_x64(True):
        nominal = np.asarray(stack_view_poses(geometry, views))
        ops = Operators(grid, detector, nominal, lever=2, native=(1.0, 1.0))
        prior = MotionPrior(
            views,
            rigid_gauge(nominal, [0.5] * 3 + [1.0, 1.0]),
            knot_spacing=(4,) * 5,
            delta=0.02,
            parameters=ops.parameters,
            rotation_scales=np.ones(5),
        )
        y = ops.pad_images(jnp.zeros((views, nz, n), jnp.float32))
        level = js.placed_level(ops, prior, y, lam=0.0, eps=0.0, scales=None, delta=1e6, inner=1)
        c = jnp.zeros(prior.size)
        profiled, _ = ops.objective(ops.pad_poses(c), jnp.asarray(pedestal), level.y, None, 1e6)
        total = level.linearize(c, jnp.asarray(pedestal))[6]
    assert float(profiled) < 0.01 * 0.5 * float(np.sum(seen**2))
    assert float(total) - float(profiled) == pytest.approx(expected, rel=1e-5)


def test_the_shared_terms_and_the_prior_scale_are_formed_one_view_batch_at_a_time(monkeypatch):
    # The shared moments and their Gram matrix are a handful of numbers. Forming them from the
    # whole stack at once made several data-sized float64 arrays: on K11 (1153 views at
    # 320 x 270) the first linearisation ran out of memory at 765 MiB each. Every array they
    # are formed from must span one view batch, never the stack; so must the line search's
    # held value at unchanged poses, and the motion prior's error scale from the views' steps
    # (a stack-wide unit precision ran K11 67208, 4000 views, out of memory at 1.29 GiB).
    from tomojax.alignment._joint import _common  # check-public-imports: allow-private

    seen = []
    for module, name, which in (
        (_common, "moments", -1),
        (_common, "metric", -1),
        (_solver, "robust_value", 0),  # the line search's held value, at unchanged poses
    ):
        original = getattr(module, name)

        def recording(*args, _original=original, _which=which):
            seen.append(np.shape(args[_which])[0])
            return _original(*args)

        monkeypatch.setattr(module, name, recording)
    precision = Operators.batch_precision

    def batch_precision(self, *args):
        out = precision(self, *args)
        seen.append(np.shape(out)[0])
        return out

    monkeypatch.setattr(Operators, "batch_precision", batch_precision)
    n, nz, views = 8, 4, 10
    grid, detector = tj.Grid(n, n, nz, 1, 1, 1), tj.Detector(n, nz, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, views, endpoint=False))
    with jax.enable_x64(True):
        nominal = np.asarray(stack_view_poses(geometry, views))
        ops = Operators(grid, detector, nominal, lever=2, native=(1.0, 1.0), batch_size=4)
        prior = MotionPrior(
            views,
            rigid_gauge(nominal, [0.5] * 3 + [1.0, 1.0]),
            knot_spacing=(4,) * 5,
            delta=0.02,
            parameters=ops.parameters,
            rotation_scales=np.ones(5),
        )
        y = ops.pad_images(jnp.ones((views, nz, n), jnp.float32))
        level = js.placed_level(ops, prior, y, lam=0.0, eps=0.0, scales=None, delta=1.0, inner=1)
        out = level.linearize(jnp.zeros(prior.size), jnp.zeros(ops.shape, jnp.float32))
        sw, residual, common = out[2], out[8], out[9]
        level.held(residual, residual, sw, level.precision, jnp.float32(0.5), common.moments)
        _solver._pose_weight(level, (*out[:6], residual, common), 0.01)
    assert ops.batch == 4 and ops.padded_n == 12
    assert seen and max(seen) <= ops.batch, seen


def test_the_levels_gradient_includes_the_shared_offset_and_ramp():
    # Data with a common offset and ramp the image cannot fully explain, and poses that move
    # the views' moments: the level's gradient and score are the derivatives of the profiled
    # term plus 0.5 b^T G^-1 b, in the image and along the poses.
    grid, detector = tj.Grid(6, 6, 5, 0.7, 0.7, 0.7), tj.Detector(8, 6, 0.7, 0.7)
    geometry = tj.ParallelGeometry(grid, detector, np.arange(10) * 18.0)
    rng = np.random.default_rng(31)
    with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
        nominal = np.asarray(stack_view_poses(geometry, 10))
        ops = Operators(grid, detector, nominal, lever=2, native=(0.7, 0.7), interpolation="cubic")
        prior = MotionPrior(
            10,
            rigid_gauge(nominal, [0.5] * 3 + [0.7, 0.7]),
            knot_spacing=(4,) * 5,
            delta=0.02,
            parameters=ops.parameters,
            rotation_scales=np.ones(5),
        )
        x = jnp.asarray(rng.uniform(0.1, 0.3, ops.shape), jnp.float32)
        c = jnp.asarray(rng.normal(0, 0.05, prior.size))
        q = ops.pad_poses(c)
        clean = np.asarray(ops.project(ops.coefficients(ops.nominal, q, ops.valid), x))[:10]
        rows = np.linspace(-1, 1, 6)[:, None]
        y = ops.pad_images(jnp.asarray(clean + 0.2 + 0.1 * rows, jnp.float32))
        level = js.placed_level(ops, prior, y, lam=0.0, eps=0.0, scales=None, delta=1e6, inner=1)
        gx, _, _, _, score, _, data, *_, shared = level.linearize(c, x)
        assert float(shared.value()) > 0.1 * float(data)

        def objective(dx, dc):
            value, moments = ops.objective(ops.pad_poses(c + dc), x + dx, level.y, None, 1e6)
            return float(value) + 0.5 * float(moments @ level.inverse_gram @ moments)

        dx = jnp.asarray(rng.normal(size=ops.shape), jnp.float32)
        dc = jnp.asarray(rng.normal(size=c.shape))
        for direction, analytic, h in (
            ((dx, 0 * dc), float(jnp.vdot(gx.astype(jnp.float64), dx)), 1e-3),
            # At 1e-4 the float32 objective's rounding moves this difference by about 0.3%
            # (and by 3% at 1e-5); at 1e-3 it agrees with the score to 0.12%, filtered or not.
            ((0 * dx, dc), float(jnp.vdot(score, dc)), 1e-3),
        ):
            plus = objective(*(h * d for d in direction))
            minus = objective(*(-h * d for d in direction))
            assert (plus - minus) / (2 * h) == pytest.approx(analytic, rel=2e-3)
