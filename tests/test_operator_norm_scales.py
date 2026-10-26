"""Norm estimates must retain physical units without squaring away the signal."""

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core.operator_norm import estimate_normal_norm
from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import FistaConfig, fista_tv


@pytest.mark.numerical
@pytest.mark.parametrize("support_scale", [0.0, 1e-4, 1.0, 1e4])
@pytest.mark.parametrize("upper_bound", [False, True])
def test_ray_normal_norm_preserves_support_scale_without_absolute_floor(support_scale, upper_bound):
    # A single axis-aligned voxel gives A=[1] and M=[support_scale].
    # Both the spectral norm and the positive row-sum bound are M**2 exactly.
    grid, detector = Grid(1, 1, 1, 1.0, 1.0, 1.0), Detector(1, 1, 1.0, 1.0)
    actual = estimate_normal_norm(
        jnp.eye(4, dtype=jnp.float32)[None],
        jnp.ones((1, 1, 1)),
        None,
        jnp.full((1, 1, 1), support_scale),
        grid=grid,
        detector=detector,
        batch_size=1,
        iterations=20,
        unroll=1,
        checkpoint=False,
        gather_dtype="fp32",
        ray_integrator="joseph",
        upper_bound=upper_bound,
    )
    np.testing.assert_allclose(actual, support_scale**2, rtol=3e-6)


@pytest.mark.numerical
@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
@pytest.mark.parametrize("model", ["ray", "joseph"])
def test_auto_step_fista_recovers_same_one_voxel_image_in_any_length_units(scale, model):
    grid, detector = Grid(1, 1, 1, scale, scale, scale), Detector(1, 1, scale, scale)
    geometry = ParallelGeometry(grid, detector, [0.0])
    # For this axis-aligned one-voxel case both projector models integrate [scale].
    actual, info = fista_tv(
        geometry,
        grid,
        detector,
        jnp.full((1, 1, 1), 3 * scale),
        config=FistaConfig(
            iterations=1,
            tv_weight=0,
            projector_model=model,
            backend="jax",
            power_iterations=3,
        ),
    )
    np.testing.assert_allclose(actual, 3.0, rtol=3e-6)
    np.testing.assert_allclose(info["lipschitz"], scale**2, rtol=3e-6)


@pytest.mark.numerical
@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
@pytest.mark.parametrize("model", ["ray", "joseph"])
def test_spdhg_auto_steps_use_the_physical_data_norm_without_floors(scale, model):
    from tomojax.recon import SPDHGConfig, spdhg_tv

    grid, detector = Grid(1, 1, 1, scale, scale, scale), Detector(1, 1, scale, scale)
    geometry = ParallelGeometry(grid, detector, [0.0])
    _, info = spdhg_tv(
        geometry,
        grid,
        detector,
        jnp.zeros((1, 1, 1)),
        config=SPDHGConfig(iterations=0, tv_weight=0.1, projector_model=model, backend="jax"),
    )
    # One axis-aligned voxel has A=[scale], with the 5% spectral safety margin.
    np.testing.assert_allclose(info["A_norm"], 1.05 * scale, rtol=3e-6)
    np.testing.assert_allclose(info["sigma_data"], 0.99 / (1.05 * scale), rtol=3e-6)


def _dense_gradient_norm(shape):
    edges = []
    for index in np.ndindex(shape):
        for axis in range(3):
            neighbor = list(index)
            neighbor[axis] += 1
            if neighbor[axis] < shape[axis]:
                edge = np.zeros(shape)
                edge[index], edge[tuple(neighbor)] = -1, 1
                edges.append(edge.ravel())
    return np.linalg.norm(np.asarray(edges), 2) if edges else 0


@pytest.mark.numerical
@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 1, 1), (3, 2, 1), (4, 3, 2)])
@pytest.mark.parametrize("tv_weight", [0.0, 0.1])
def test_spdhg_uses_exact_active_gradient_norm_and_keeps_explicit_steps(shape, tv_weight):
    from tomojax.recon import SPDHGConfig, spdhg_tv

    grid, detector = Grid(*shape, 1.0, 1.0, 1.0), Detector(3, 2, 1.0, 1.0)
    geometry = ParallelGeometry(grid, detector, [0.0])
    _, info = spdhg_tv(
        geometry,
        grid,
        detector,
        jnp.zeros((1, 2, 3)),
        config=SPDHGConfig(iterations=0, tv_weight=tv_weight, tau=0.2, sigma_data=0.3),
    )
    expected_norm = _dense_gradient_norm(shape) if tv_weight else 0
    np.testing.assert_allclose(info["grad_norm"], expected_norm, rtol=3e-6)
    np.testing.assert_allclose(info["sigma_tv"], 0.99 / (expected_norm or 1.0), rtol=3e-6)
    assert info["tau"] == 0.2
    assert info["sigma_data"] == 0.3
    assert info["A_norm"] is None


@pytest.mark.numerical
@pytest.mark.parametrize("tv_weight", [0.0, 0.1])
def test_spdhg_zero_combined_operator_has_defined_auto_steps_and_retains_start(tv_weight):
    from tomojax.recon import SPDHGConfig, spdhg_tv

    grid, detector = Grid(1, 1, 1, 1.0, 1.0, 1.0), Detector(1, 1, 1.0, 1.0, (100.0, 100.0))
    geometry = ParallelGeometry(grid, detector, [0.0])
    actual, info = spdhg_tv(
        geometry,
        grid,
        detector,
        jnp.ones((1, 1, 1)),
        init_x=jnp.ones((1, 1, 1)),
        config=SPDHGConfig(iterations=3, tv_weight=tv_weight, backend="jax"),
    )
    np.testing.assert_array_equal(actual, 1)
    assert info["A_norm"] == 0
    assert info["grad_norm"] == 0
    for name in ["tau", "sigma_data", "sigma_tv"]:
        assert np.isfinite(info[name]) and info[name] > 0


@pytest.mark.numerical
def test_spdhg_rejects_nonfinite_estimated_norm_instead_of_returning_nan_iterates():
    from tomojax.recon import SPDHGConfig, spdhg_tv

    # A is representable, but A.T A is beyond FP32. Do not silently continue
    # with NaN automatic steps when physical units overflow the normal operator.
    scale = 1e20
    grid, detector = Grid(1, 1, 1, scale, scale, scale), Detector(1, 1, scale, scale)
    geometry = ParallelGeometry(grid, detector, [0.0])
    with pytest.raises(ValueError, match="projection norm must be finite"):
        spdhg_tv(
            geometry,
            grid,
            detector,
            jnp.zeros((1, 1, 1)),
            config=SPDHGConfig(iterations=1, projector_model="joseph", backend="jax"),
        )


@pytest.mark.numerical
@pytest.mark.parametrize("voxels", [1, 2])
def test_huber_fista_step_uses_exact_finite_grid_gradient_norm(voxels):
    grid, detector = Grid(voxels, 1, 1, 1.0, 1.0, 1.0), Detector(voxels, 1, 1.0, 1.0)
    initial = np.asarray([0.2, 0.8][:voxels], np.float32)
    data = np.asarray([0.5, 1.5][:voxels], np.float32)
    # A=I, and every edge lies in Huber's quadratic basin: D.T D has norm 2 for
    # two voxels and 0 for one, so the Lipschitz constant is 1 + 0.2 * norm / 2.
    lipschitz = 1 + 0.1 * (2 if voxels == 2 else 0)
    gradient = initial - data
    if voxels == 2:
        gradient = gradient + 0.1 * np.asarray([initial[0] - initial[1], initial[1] - initial[0]])
    actual, info = fista_tv(
        ParallelGeometry(grid, detector, [0.0]),
        grid,
        detector,
        data.reshape(1, 1, voxels),
        init_x=jnp.asarray(initial.reshape(1, 1, voxels)),
        config=FistaConfig(
            iterations=1,
            tv_weight=0.2,
            regulariser="huber_tv",
            huber_delta=2,
            projector_model="joseph",
            backend="jax",
        ),
    )
    np.testing.assert_allclose(info["lipschitz"], lipschitz, rtol=3e-6)
    np.testing.assert_allclose(actual.ravel(), initial - gradient / lipschitz, rtol=3e-6)
