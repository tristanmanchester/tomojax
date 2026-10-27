"""Private resident buffers preserve public axes, derivatives and solver semantics."""

from dataclasses import replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.core import cone
from tomojax.core.volume_layout import transpose_volume

pytestmark = pytest.mark.gpu


@pytest.fixture(params=[0, 60])
def fixture(request):
    if not cone.use_cuda_cone():
        pytest.skip("requires CUDA cone kernels")
    grid = tj.Grid(7, 5, 3, 0.7, 1.1, 0.9, vol_center=(0.2, -0.1, 0.3))
    detector = tj.Detector(11, 9, 1.7, 1.9, center=(0.3, -0.2))
    geometry = tj.ConeGeometry(
        grid,
        detector,
        np.linspace(0, 360, 11, endpoint=False),
        tj.ConeBeam(50, 100),
        tilt_deg=request.param,
    )
    rng = np.random.default_rng(12017)
    initial = rng.uniform(0.1, 1, size=(3, 5, 7)).astype(np.float32)
    images = rng.uniform(0.1, 2, size=(11, 9, 11)).astype(np.float32)
    return geometry, grid, detector, initial, images


@pytest.mark.parametrize("shape", [(37, 19, 23), (1, 3, 65), (63, 2, 1)])
def test_bounded_transpose_and_linear_derivatives(shape, fixture):
    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    actual = jax.jit(transpose_volume)(x)
    np.testing.assert_array_equal(actual, np.asarray(x).transpose(2, 1, 0))
    _, tangent = jax.jvp(transpose_volume, (x,), (x,))
    np.testing.assert_array_equal(tangent, actual)
    gradient = jax.grad(lambda value: jnp.vdot(transpose_volume(value), actual))(x)
    np.testing.assert_array_equal(gradient, x)
    np.testing.assert_array_equal(transpose_volume(actual), x)


def test_private_model_forward_and_pose_volume_pullbacks(fixture):
    geometry, grid, detector, initial, images = fixture
    model = cone.cone_model(geometry)
    private = replace(model, _z_fast=True)
    coeff = model.coefficients(geometry.poses(), grid)
    x, y = jnp.asarray(initial), jnp.asarray(images)

    def project(c, value, layout):
        return layout.project(value, c, grid=grid, detector=detector, backend="cuda")

    def native(c, value):
        return project(c, value, model)

    def packed(c, value):
        return project(c, value, private)

    xp = transpose_volume(x)
    np.testing.assert_array_equal(native(coeff, x), packed(coeff, xp))
    direction = jnp.ones_like(coeff) * 0.003
    _, first = jax.jvp(native, (coeff, x), (direction, x))
    _, second = jax.jvp(packed, (coeff, xp), (direction, xp))
    np.testing.assert_allclose(second, first, rtol=2e-6, atol=2e-6)
    c0, x0 = jax.grad(lambda c, v: jnp.vdot(native(c, v), y), argnums=(0, 1))(coeff, x)
    c1, x1 = jax.grad(lambda c, v: jnp.vdot(packed(c, v), y), argnums=(0, 1))(coeff, xp)
    np.testing.assert_allclose(c1, c0, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(transpose_volume(x1), x0, rtol=2e-5, atol=2e-5)
    lhs, rhs = jnp.vdot(packed(coeff, xp), y), jnp.vdot(xp, x1)
    np.testing.assert_allclose(lhs, rhs, rtol=2e-6)


@pytest.mark.parametrize("method", ["sirt", "mlem", "cgls"])
@pytest.mark.parametrize("iterations", [0, 3])
def test_public_solver_axes_support_and_regularization(method, iterations, fixture):
    geometry, grid, detector, initial, images = fixture
    module = importlib.import_module("tomojax.recon." + method)
    config_type = getattr(
        module, {"sirt": "SIRTConfig", "mlem": "MLEMConfig", "cgls": "CGLSConfig"}[method]
    )
    settings = dict(iterations=iterations, views_per_batch=4)
    if method == "cgls":
        settings.update(damping=0.07, gradient_damping=0.13)
    else:
        support = np.indices(initial.shape).sum(axis=0) % 3 != 0
        settings.update(subsets=3, support=support)
    solve = getattr(module, method)
    before = initial.copy()
    reference = solve(
        geometry,
        grid,
        detector,
        images,
        init_x=initial,
        config=config_type(backend="jax", **settings),
    )[0]
    actual = solve(
        geometry,
        grid,
        detector,
        images,
        init_x=initial,
        config=config_type(backend="cuda", **settings),
    )[0]
    assert actual.shape == initial.shape
    np.testing.assert_allclose(actual, reference, rtol=3e-4, atol=3e-5)
    np.testing.assert_array_equal(initial, before)


def test_streamed_cgls_keeps_resident_volume_axes(fixture, monkeypatch):
    from tomojax.recon import CGLSConfig, cgls

    geometry, grid, detector, initial, images = fixture
    cfg = CGLSConfig(
        iterations=3, views_per_batch=4, damping=0.07, gradient_damping=0.13, backend="cuda"
    )
    reference = cgls(geometry, grid, detector, images, init_x=initial, config=cfg)[0]
    module = importlib.import_module("tomojax.recon._cgls_jax")
    monkeypatch.setattr(module, "_placement", lambda *args, **kwargs: (None, True))
    actual, info = cgls(geometry, grid, detector, images, init_x=initial, config=cfg)
    assert info["formulation"] == "streamed_normal_equations"
    np.testing.assert_allclose(actual, reference, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("method", ["sirt", "mlem"])
def test_streamed_uncached_private_norms(method, fixture, monkeypatch):
    geometry, grid, detector, initial, images = fixture
    module = importlib.import_module("tomojax.recon." + method)
    config_type = module.SIRTConfig if method == "sirt" else module.MLEMConfig
    cfg = config_type(iterations=3, subsets=3, views_per_batch=4, backend="cuda")
    solve = getattr(module, method)
    reference = solve(geometry, grid, detector, images, init_x=initial, config=cfg)[0]
    if method == "mlem":
        monkeypatch.setattr(module, "_cache_sensitivities", lambda *args, **kwargs: False)
    else:
        original = module._storage

        def storage(*args, **kwargs):
            initial, stream, _ = original(*args, **kwargs)
            return initial, stream, False

        monkeypatch.setattr(module, "_storage", storage)
    monkeypatch.setattr(module, "should_stream", lambda *args, **kwargs: True)
    actual = solve(geometry, grid, detector, images, init_x=initial, config=cfg)[0]
    np.testing.assert_allclose(actual, reference, rtol=3e-5, atol=3e-6)
