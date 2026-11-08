"""Independent geometric and operator contracts of the joint cone adapter."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# check-public-imports: allow-private
from tomojax.alignment._joint._projection import (
    Projection,
    cone_gauge,
    cone_parameter_scales,
    cropped_frames,
    posed,
    rigid_gauge,
)
from tomojax.core.backend_policy import cuda_backend
from tomojax.core.cone import beam_frame
from tomojax.geometry import ConeBeam, ConeGeometry, Detector, Grid, stack_view_poses

pytestmark = pytest.mark.numerical


def _geometry():
    grid = Grid(7, 6, 5, 0.7, 0.9, 1.1, vol_center=(0.13, -0.19, 0.23))
    detector = Detector(9, 7, 1.2, 0.8, center=(0.17, -0.11))
    beam = ConeBeam(20, 33, detector_roll_deg=4, detector_pitch_deg=2, axis_offset=0.1)
    geometry = ConeGeometry(grid, detector, [13.7, 71.3, 119.1], beam, tilt_deg=27)
    nominal = np.asarray(stack_view_poses(geometry, 3))
    frames = np.repeat(beam_frame(beam, detector)[None], 3, axis=0)
    return grid, detector, nominal, frames


@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    ("cone", "interpolation"), [(False, "linear"), (False, "cubic"), (True, "linear")]
)
def test_adapter_transpose_includes_mask_and_accumulation(cone, interpolation, backend):
    if backend == "cuda" and not cuda_backend():
        pytest.skip("requires CUDA")
    grid, detector, nominal, frames = _geometry()
    model = Projection(grid, detector, cone=cone, backend=backend, interpolation=interpolation)
    valid = jnp.asarray([True, False, True])
    q = jnp.asarray(np.random.default_rng(43).normal(size=(3, model.dofs)), jnp.float32)
    scales = jnp.asarray([0.013] * 3 + [0.07] * (model.dofs - 3))
    cf = model.coefficients(posed(jnp.asarray(nominal), q, scales), valid, frames if cone else None)
    rng = np.random.default_rng(1981)
    x = jnp.asarray(rng.normal(size=(grid.nz, grid.ny, grid.nx)), jnp.float32)
    y = jnp.asarray(rng.normal(size=(3, detector.nv, detector.nu)), jnp.float32)
    initial = jnp.asarray(rng.normal(size=x.shape), jnp.float32)
    forward = jax.jit(model.project)(cf, x)
    back = jax.jit(model.back)(initial, cf, y) - initial
    np.testing.assert_array_equal(forward[1], 0)
    left, right = jnp.vdot(forward, y), jnp.vdot(x, back)
    np.testing.assert_allclose(left, right, rtol=3e-6, atol=3e-5)


@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_six_cone_columns_follow_physical_finite_differences(backend):
    if backend == "cuda" and not cuda_backend():
        pytest.skip("requires CUDA")
    grid, detector, nominal, frames = _geometry()
    model = Projection(grid, detector, cone=True, backend=backend)
    scales = cone_parameter_scales(nominal, frames, grid, (detector.du, detector.dv))
    nominal, scales, frames = map(jnp.asarray, (nominal, scales, frames))
    q = jnp.asarray(np.random.default_rng(73).uniform(-0.03, 0.03, (3, 6)), jnp.float32)
    volume = jnp.asarray(
        np.random.default_rng(57).normal(size=(grid.nz, grid.ny, grid.nx)), jnp.float32
    )
    valid = jnp.asarray([True, False, True])

    @jax.jit
    def prediction(p):
        return model.project(model.coefficients(posed(nominal, p, scales), valid, frames), volume)

    cf = model.coefficients(posed(nominal, q, scales), valid, frames)
    tangent = model.tangent(nominal, q, scales, frames)
    columns = jax.jit(model.columns)(cf, volume, tangent)
    np.testing.assert_array_equal(columns[1], 0)
    np.testing.assert_allclose(columns[:, 0], model.project(cf, volume), rtol=2e-6, atol=2e-6)
    # Central differences need a finite step above float32 coordinate rounding.
    # Compare RMS error so rays crossing bilinear knots do not dominate the test.
    step = 2e-3
    for k in range(6):
        direction = jnp.zeros_like(q).at[:, k].set(step)
        difference = (prediction(q + direction) - prediction(q - direction)) / (2 * step)
        relative = float(
            jnp.linalg.norm(columns[:, 1 + k] - difference) / jnp.linalg.norm(difference)
        )
        assert relative < 0.04
    assert float(jnp.linalg.norm(columns[:, 6])) > 1e-2


def test_rigid_gauge_cancels_common_object_coordinates_in_six_dofs():
    grid, detector, nominal, frames = _geometry()
    scales = cone_parameter_scales(nominal, frames, grid, (detector.du, detector.dv))
    gauge = rigid_gauge(nominal, scales).reshape(3, 6, 6)
    points = np.random.default_rng(53).normal(size=(7, 3))
    initial = np.einsum("nij,kj->nki", nominal[:, :3, :3], points) + nominal[:, None, :3, 3]
    epsilon = 1e-5
    with jax.enable_x64(True):
        for k in range(6):
            action = np.eye(4)
            if k < 3:
                action[:3, :3] = Rotation.from_rotvec(np.eye(3)[k] * epsilon).as_matrix()
            else:
                action[k - 3, 3] = epsilon
            changed = np.asarray(
                posed(
                    jnp.asarray(nominal, jnp.float64), jnp.asarray(gauge[:, :, k] * epsilon), scales
                )
            )
            combined = changed @ action
            actual = (
                np.einsum("nij,kj->nki", combined[:, :3, :3], points) + combined[:, None, :3, 3]
            )
            np.testing.assert_allclose(actual, initial, rtol=0, atol=3e-9)


def test_cone_row_sum_is_a_normal_majorizer_not_an_approximate_adjoint():
    grid = Grid(3, 3, 2, 0.7, 0.8, 1.1)
    detector = Detector(4, 3, 0.9, 0.8)
    beam = ConeBeam(10, 17, detector_roll_deg=3)
    nominal = jnp.asarray(stack_view_poses(ConeGeometry(grid, detector, [17, 81], beam), 2))
    frames = jnp.repeat(jnp.asarray(beam_frame(beam, detector))[None], 2, axis=0)
    model = Projection(grid, detector, cone=True, backend="jax")
    cf = model.coefficients(nominal, jnp.asarray([True, False]), frames)
    voxels = grid.nx * grid.ny * grid.nz
    basis = jnp.eye(voxels, dtype=jnp.float32).reshape(voxels, grid.nz, grid.ny, grid.nx)
    dense = (
        np.asarray(jax.jit(jax.vmap(lambda x: model.project(cf, x)))(basis)).reshape(voxels, -1).T
    )
    sqrtw = (
        np.random.default_rng(14).uniform(0.1, 1, (2, detector.nv, detector.nu)).astype(np.float32)
    )
    normal = dense.T @ (sqrtw.reshape(-1, 1) ** 2 * dense)
    diagonal = np.asarray(
        jax.jit(model.diagonal)(cf, jnp.asarray(sqrtw), jnp.zeros((2, 3, 3), jnp.float32))
    ).ravel()
    np.testing.assert_allclose(diagonal, normal.sum(axis=1), rtol=3e-6, atol=3e-6)
    assert np.linalg.eigvalsh(np.diag(diagonal) - normal).min() > -3e-6
    assert model.diagonal_kind == "normal_row_sum"


def test_cubic_majorizer_bounds_the_signed_normal_matrix_and_initial_step():
    grid = Grid(3, 4, 2, 0.7, 0.8, 1.1)
    detector = Detector(5, 4, 0.9, 0.8, center=(0.13, -0.23))
    nominal = jnp.asarray(
        stack_view_poses(ConeGeometry(grid, detector, [17, 81], ConeBeam(10, 17)), 2)
    )
    model = Projection(grid, detector, interpolation="cubic", backend="jax")
    cf = model.coefficients(nominal, jnp.asarray([True, False]))
    count = grid.nx * grid.ny * grid.nz
    basis = jnp.eye(count, dtype=jnp.float32).reshape(count, grid.nz, grid.ny, grid.nx)
    dense = (
        np.asarray(jax.jit(jax.vmap(lambda x: model.project(cf, x)))(basis)).reshape(count, -1).T
    )
    absolute = (
        np.asarray(jax.jit(jax.vmap(lambda x: model.project(cf, x, absolute=True)))(basis))
        .reshape(count, -1)
        .T
    )
    assert dense.min() < -1e-4
    np.testing.assert_allclose(absolute, np.abs(dense), rtol=2e-6, atol=2e-6)
    sqrtw = (
        np.random.default_rng(19).uniform(0.1, 1, (2, detector.nv, detector.nu)).astype(np.float32)
    )
    normal = dense.T @ (sqrtw.reshape(-1, 1) ** 2 * dense)
    expected = absolute.T @ (sqrtw.ravel() ** 2 * absolute.sum(axis=1))
    zero = jnp.zeros((grid.nz, grid.ny, grid.nx), jnp.float32)
    diagonal = np.asarray(jax.jit(model.majorizer)(cf, jnp.asarray(sqrtw), zero)).ravel()
    np.testing.assert_allclose(diagonal, expected, rtol=3e-6, atol=3e-6)
    assert np.linalg.eigvalsh(np.diag(diagonal) - normal).min() >= -3e-6
    bound = float(jnp.max(jax.jit(model.majorizer)(cf, jnp.ones_like(sqrtw), zero)))
    assert bound >= np.linalg.eigvalsh(dense.T @ dense).max() - 3e-6
    proxy = model.diagonal(cf, jnp.asarray(sqrtw), zero)
    linear = Projection(grid, detector, interpolation="linear", backend="jax")
    np.testing.assert_array_equal(proxy, linear.diagonal(cf, jnp.asarray(sqrtw), zero))
    assert model.diagonal_kind == "linear_basis_diagonal_proxy"


def test_cone_parameter_units_include_magnification_and_depth():
    grid = Grid(9, 7, 5, 0.4, 0.5, 0.6)
    detector = Detector(12, 10, 0.5, 0.8)
    beam = ConeBeam(30, 60)
    nominal = np.eye(4)[None]
    frames = beam_frame(beam, detector)[None]
    scales = cone_parameter_scales(nominal, frames, grid, (0.5, 0.8))
    # At magnification two, object x/z translations are about half a detector pixel.
    np.testing.assert_allclose(scales[0, 3:5], [0.25, 0.4], rtol=0.02)
    # A deeper shift changes magnification: it is observable, and needs a larger
    # physical displacement than an equal detector-plane shift.
    assert scales.shape == (1, 6)
    assert scales[0, 5] > 5 * max(scales[0, 3:5])


def test_odd_cone_detector_binning_keeps_native_pixel_centres():
    detector = Detector(11, 9, 0.7, 1.1, center=(0.17, -0.23))
    beam = ConeBeam(30, 52, detector_roll_deg=21, detector_pitch_deg=-7, detector_yaw_deg=11)
    frame = beam_frame(beam, detector).astype(np.float64)
    cropped = cropped_frames(frame, detector, high_u=1, high_v=1)

    def centre_at(f, nu, nv, du, dv, u, v):
        return f[0] + (u - (nu - 1) / 2) * du * f[1] + (v - (nv - 1) / 2) * dv * f[2]

    for u, v in ((0, 0), (4, 3), (2, 1)):
        expected = np.mean(
            [
                centre_at(frame, 11, 9, 0.7, 1.1, 2 * u + i, 2 * v + j)
                for i in (0, 1)
                for j in (0, 1)
            ],
            axis=0,
        )
        actual = centre_at(cropped, 5, 4, 1.4, 2.2, u, v)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(frame, beam_frame(beam, detector))


def test_cone_similarity_gauge_preserves_continuous_gaussian_line_integrals():
    grid, detector, nominal, frames = _geometry()
    nominal, frames = nominal.astype(np.float64), frames.astype(np.float64)
    scales = cone_parameter_scales(nominal, frames, grid, (detector.du, detector.dv))
    gauge = cone_gauge(nominal, scales, frames)
    assert gauge.shape == (18, 7)
    assert np.linalg.matrix_rank(gauge) == 7
    factor = 1.07
    centres = np.asarray([[0.13, -0.17, 0.21], [-0.41, 0.23, 0.19]])
    widths = np.asarray([[0.7, 0.9, 1.1], [1.1, 0.5, 0.8]])
    u, v = np.meshgrid(np.arange(detector.nu), np.arange(detector.nv))
    pixels = (
        frames[:, None, None, 0]
        + (u[..., None] - (detector.nu - 1) / 2) * detector.du * frames[:, None, None, 1]
        + (v[..., None] - (detector.nv - 1) / 2) * detector.dv * frames[:, None, None, 2]
    )
    ray = pixels - frames[:, None, None, 3]
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)

    def line_integral(poses, centres, widths, amplitude):
        directions = np.einsum("n...i,nij->n...j", ray, poses[:, :3, :3])
        origin = np.einsum("ni,nij->nj", frames[:, 3] - poses[:, :3, 3], poses[:, :3, :3])
        out = np.zeros((len(poses), detector.nv, detector.nu))
        for centre, width in zip(centres, widths, strict=True):
            delta = origin - centre
            a = np.sum((directions / width) ** 2, axis=-1)
            b = np.sum(directions * delta[:, None, None] / width**2, axis=-1)
            c = np.sum((delta / width) ** 2, axis=-1)
            out += amplitude * np.sqrt(2 * np.pi / a) * np.exp(-0.5 * (c[:, None, None] - b**2 / a))
        return out

    expected = line_integral(nominal, centres, widths, 1.0)
    changed = nominal.copy()
    changed[:, :3, 3] = factor * nominal[:, :3, 3] + (1 - factor) * frames[:, 3]
    actual = line_integral(changed, factor * centres, factor * widths, 1 / factor)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    # The analytic gauge direction reproduces exactly this translation change.
    with jax.enable_x64(True):
        actual_poses = posed(
            jnp.asarray(nominal, jnp.float64),
            jnp.asarray((factor - 1) * gauge[:, -1].reshape(3, 6)),
            scales,
        )
    np.testing.assert_allclose(actual_poses, changed, rtol=0, atol=1e-12)
