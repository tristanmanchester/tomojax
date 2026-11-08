"""Known digital block means retain their exact image and pose operators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.alignment._joint._fit import (  # check-public-imports: allow-private
    _coarsened,
    _finite_pixels,
    _level,
)
from tomojax.alignment._joint._operators import Operators  # check-public-imports: allow-private
from tomojax.alignment._joint._projection import (  # check-public-imports: allow-private
    PackedMask,
    Projection,
    posed,
)
from tomojax.alignment._joint._refine import _EdgeObjective  # check-public-imports: allow-private
from tomojax.core.backend_policy import cuda_backend
from tomojax.core.cone import beam_frame
from tomojax.geometry import ConeBeam, ConeGeometry, Detector, Grid, stack_view_poses

pytestmark = pytest.mark.numerical


def _geometry():
    grid = Grid(5, 4, 3, 0.7, 0.9, 1.1, vol_center=(0.13, -0.19, 0.23))
    detector = Detector(4, 3, 1.6, 1.6, center=(0.17, -0.11))
    beam = ConeBeam(20, 33, detector_roll_deg=7, detector_pitch_deg=-3)
    geometry = ConeGeometry(grid, detector, [13.7, 71.3, 119.1], beam, tilt_deg=27)
    nominal = jnp.asarray(stack_view_poses(geometry, 3), jnp.float32)
    frames = jnp.repeat(jnp.asarray(beam_frame(beam, detector))[None], 3, axis=0)
    return grid, detector, nominal, frames


def _input_mask():
    # Two input pixels per observed axis; each input already averages two u rays.
    mask = np.ones((3, 6, 8), bool)
    mask[:, 0, 0] = False
    mask[0, 2:4, 2:4] = False
    mask[2, 3, 6] = False
    return mask


def _packed(mask):
    return PackedMask(jnp.packbits(mask, axis=-1, bitorder="little"), mask.shape[-2:])


@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    ("cone", "interpolation"), [(False, "linear"), (False, "cubic"), (True, "linear")]
)
@pytest.mark.parametrize("missing", [False, True])
def test_digital_means_match_original_rays_with_exact_transpose(
    cone, interpolation, missing, backend
):
    if backend == "cuda" and not cuda_backend():
        pytest.skip("requires CUDA")
    grid, detector, nominal, frames = _geometry()
    model = Projection(
        grid,
        detector,
        cone=cone,
        interpolation=interpolation,
        backend=backend,
        ray_binning=(4, 2),
    )
    pixel_mask = _packed(_input_mask()) if missing else None
    valid = jnp.asarray([True, False, True])
    cf = model.coefficients(nominal, valid, frames if cone else None, pixel_mask)
    rng = np.random.default_rng(76)
    shape = (grid.nz, grid.ny, grid.nx)
    x = jnp.asarray(rng.normal(size=shape), jnp.float32)
    y = jnp.asarray(rng.normal(size=(3, 3, 4)), jnp.float32)
    fine = Projection(
        grid,
        Detector(16, 6, 0.4, 0.8, center=detector.center),
        cone=cone,
        interpolation=interpolation,
        backend=backend,
    )
    original = np.asarray(
        fine.project(fine.coefficients(nominal, jnp.ones(3, bool), frames if cone else None), x)
    )
    supplied = original.reshape(3, 6, 8, 2).mean(-1)
    if missing:
        supplied = np.where(_input_mask(), supplied, np.nan)
    expected, _ = _coarsened(supplied, 2, missing=missing)
    expected *= np.asarray(valid)[:, None, None]
    actual = jax.jit(model.project)(cf, x)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=3e-6)
    initial = jnp.asarray(rng.normal(size=shape), jnp.float32)
    adjoint = jax.jit(model.back)(initial, cf, y) - initial
    np.testing.assert_allclose(jnp.vdot(actual, y), jnp.vdot(x, adjoint), rtol=4e-6, atol=4e-5)


@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    ("cone", "interpolation"), [(False, "linear"), (False, "cubic"), (True, "linear")]
)
def test_masked_digital_mean_pose_columns_follow_finite_differences(cone, interpolation, backend):
    if backend == "cuda" and not cuda_backend():
        pytest.skip("requires CUDA")
    grid, detector, nominal, frames = _geometry()
    model = Projection(
        grid,
        detector,
        cone=cone,
        interpolation=interpolation,
        backend=backend,
        ray_binning=(4, 2),
    )
    frames = frames if cone else None
    scales = jnp.asarray([0.071] * 3 + [0.3] * (model.dofs - 3), jnp.float32)
    rng = np.random.default_rng(77)
    q = jnp.asarray(rng.uniform(-0.05, 0.05, (3, model.dofs)), jnp.float32)
    x = jnp.asarray(rng.uniform(0.1, 1, (grid.nz, grid.ny, grid.nx)), jnp.float32)
    valid = jnp.asarray([True, False, True])
    mask = _packed(jnp.asarray(_input_mask()))

    @jax.jit
    def prediction(parameters):
        return model.project(
            model.coefficients(posed(nominal, parameters, scales), valid, frames, mask), x
        )

    cf = model.coefficients(posed(nominal, q, scales), valid, frames, mask)
    tangent = model.tangent(nominal, q, scales, frames)
    columns = jax.jit(model.columns)(cf, x, tangent)
    np.testing.assert_allclose(columns[:, 0], prediction(q), rtol=3e-6, atol=2e-6)
    np.testing.assert_array_equal(columns[0, :, 1, 1], 0)
    step = 1e-3
    for k in range(model.dofs):
        direction = jnp.zeros_like(q).at[:, k].set(step)
        finite = (prediction(q + direction) - prediction(q - direction)) / (2 * step)
        relative = float(jnp.linalg.norm(columns[:, k + 1] - finite) / jnp.linalg.norm(finite))
        assert relative < 0.035, (k, relative)


@pytest.mark.parametrize(("cone", "interpolation"), [(False, "cubic"), (True, "linear")])
def test_averaged_majorizer_contains_cross_ray_terms(cone, interpolation):
    grid = Grid(3, 3, 2, 0.7, 0.9, 1.1)
    detector = Detector(4, 3, 1.6, 1.6, center=(0.17, -0.11))
    _, _, nominal, frames = _geometry()
    model = Projection(
        grid,
        detector,
        cone=cone,
        interpolation=interpolation,
        backend="jax",
        ray_binning=(4, 2),
    )
    cf = model.coefficients(
        nominal, jnp.asarray([True, False, True]), frames if cone else None, _packed(_input_mask())
    )
    shape = (grid.nz, grid.ny, grid.nx)
    size = np.prod(shape)
    basis = jnp.eye(size, dtype=jnp.float32).reshape(size, *shape)
    matrix = (
        np.asarray(jax.jit(jax.vmap(lambda x: model.project(cf, x)))(basis)).reshape(size, -1).T
    )
    positive = (
        np.asarray(jax.jit(jax.vmap(lambda x: model.project(cf, x, absolute=True)))(basis))
        .reshape(size, -1)
        .T
    )
    assert np.all(positive >= np.abs(matrix) - 2e-6)
    sqrtw = jnp.asarray(np.random.default_rng(78).uniform(0.1, 1, (3, 3, 4)), jnp.float32)
    weights = np.asarray(sqrtw).ravel() ** 2
    normal = matrix.T @ (weights[:, None] * matrix)
    expected = positive.T @ (weights * positive.sum(axis=1))
    diagonal = np.asarray(
        jax.jit(model.majorizer)(cf, sqrtw, jnp.zeros(shape, jnp.float32))
    ).ravel()
    np.testing.assert_allclose(diagonal, expected, rtol=4e-6, atol=4e-6)
    assert np.linalg.eigvalsh(np.diag(diagonal) - normal).min() >= -5e-6


def test_level_composes_known_binning_without_inferring_a_pixel_psf():
    grid, _, nominal, _ = _geometry()
    detector = Detector(8, 6, 0.8, 0.8, center=(0.17, -0.11))
    measured = np.random.default_rng(79).uniform(0.1, 1, (3, 6, 8)).astype(np.float32)
    measured[~_input_mask()] = np.nan
    _, precision, _, operators, _ = _level(
        measured,
        np.asarray(nominal),
        grid,
        detector,
        detector.center,
        2,
        1,
        lever=5,
        native=(0.1, 0.2),  # pose units alone do not imply ray averaging
        ray_binning=(2, 1),
        pixel_mask=_finite_pixels(measured),
    )
    assert operators.projection.ray_binning == (4, 2)
    assert operators.projection.ray_detector == Detector(16, 6, 0.4, 0.8, detector.center)
    np.testing.assert_array_equal(
        jnp.unpackbits(operators.pixel_mask.bits[:3], axis=-1, count=8, bitorder="little"),
        _input_mask(),
    )
    assert precision[0, 1, 1] == 0


@pytest.mark.parametrize("cone", [False, True])
def test_masked_ray_means_survive_joint_batches_and_edge_refinement(cone):
    grid, detector, nominal, frames = _geometry()
    rng = np.random.default_rng(80)
    with jax.enable_x64(True):
        ops = Operators(
            grid,
            detector,
            nominal,
            lever=5,
            native=(0.4, 0.8),
            batch_size=2,
            frames=frames if cone else None,
            ray_binning=(4, 2),
            pixel_mask=_packed(_input_mask()),
        )
        q = ops.pad_poses(rng.normal(0, 0.03, (3, ops.dofs)))
        x = jnp.asarray(rng.uniform(0.1, 1, ops.shape), jnp.float32)
        measured = ops.pad_images(rng.normal(size=(3, 3, 4)))
        _, fraction = _coarsened(np.where(_input_mask(), 1.0, np.nan), 2, missing=True)
        precision = ops.pad_images(fraction)
        _, _, weights, columns, _, _, _, _ = ops.linearize(q, x, measured, precision, 10.0)
        p = jnp.asarray(rng.normal(size=3 * ops.dofs), jnp.float64)
        _, cross = ops.weighted_forward(q, x, weights, columns)
        back = ops.pose_back(q, weights, columns, p)
        np.testing.assert_allclose(jnp.vdot(p, cross), jnp.vdot(x, back), rtol=5e-6, atol=2e-5)
        edge = _EdgeObjective(ops)
        score, blocks, objective = edge.stats(q, x, measured, precision, 10.0)
        assert score.shape == (3 * ops.dofs,)
        assert blocks.shape == (3, ops.dofs, ops.dofs)
        assert np.isfinite(score).all() and np.isfinite(blocks).all()
        np.testing.assert_allclose(
            objective, edge.value(q, x, measured, precision, 10.0), rtol=2e-6
        )


@pytest.mark.parametrize("counts", [(0, 1), (1, 2.5), (True, False), (1,), (np.inf, 1)])
def test_unknown_or_invalid_ray_averages_are_not_assumed(counts):
    with pytest.raises(ValueError, match="ray_binning"):
        Projection(Grid(2, 2, 2, 1, 1, 1), Detector(2, 2, 1, 1), ray_binning=counts)


@pytest.mark.parametrize("missing", [False, True])
def test_coarsening_memory_map_matches_dense_fp32(tmp_path, missing):
    source = np.lib.format.open_memmap(tmp_path / "views.npy", "w+", dtype="f8", shape=(3, 6, 8))
    source[:] = np.random.default_rng(81).normal(size=source.shape)
    if missing:
        source[1, 2, 3] = np.nan
    actual = _coarsened(source, 2, missing=missing, batch_views=1)
    expected = _coarsened(source.astype(np.float32), 2, missing=missing, batch_views=3)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("width", [1, 7, 9])
def test_packed_membership_retains_odd_shapes_and_late_missing_slabs(width):
    measured = np.ones((5, 3, width), np.float32)
    assert _finite_pixels(measured, batch_views=2) is None
    measured[3, 1, -1] = np.nan
    packed = _finite_pixels(measured, batch_views=2)
    assert packed.bits.shape == (5, 3, (width + 7) // 8)
    np.testing.assert_array_equal(
        jnp.unpackbits(packed.bits, axis=-1, count=width, bitorder="little"),
        np.isfinite(measured),
    )


@pytest.mark.parametrize("width", [1, 7, 9])
def test_packed_odd_width_means_match_input_membership(width):
    grid, _, nominal, _ = _geometry()
    detector = Detector(width, 3, 1.6, 0.8)
    model = Projection(grid, detector, interpolation="cubic", ray_binning=(2, 1))
    mask = np.ones((3, 3, width), bool)
    mask[1, 1, -1] = False
    valid = jnp.asarray([True, True, False])
    cf = model.coefficients(nominal, valid, pixel_mask=_packed(mask))
    fine = Projection(grid, Detector(2 * width, 3, 0.8, 0.8), interpolation="cubic")
    x = jnp.asarray(np.random.default_rng(82).normal(size=(grid.nz, grid.ny, grid.nx)), jnp.float32)
    rays = fine.project(fine.coefficients(nominal, valid), x)
    expected = rays.reshape(3, 3, width, 2).mean(-1) * mask
    actual = model.project(cf, x)
    np.testing.assert_array_equal(actual, expected)
    y = jnp.ones(actual.shape, jnp.float32)
    back = model.back(jnp.zeros_like(x), cf, y)
    np.testing.assert_allclose(jnp.vdot(actual, y), jnp.vdot(x, back), rtol=3e-6, atol=3e-6)
