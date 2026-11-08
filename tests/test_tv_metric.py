"""Physical covariance and global measurement statistics for TV normalization."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon._data_term import project_residual  # check-public-imports: allow-private
from tomojax.recon._tv_metric import resolve_tv_metric  # check-public-imports: allow-private


def test_metric_global_masked_statistics_units_and_derivatives():
    rng = np.random.default_rng(8)
    y = rng.normal(size=(19, 7, 9)).astype(np.float32)
    w = rng.uniform(size=y.shape).astype(np.float32)
    w[w < 0.3] = 0
    y[w == 0] = np.nan
    grid, detector = Grid(8, 5, 7, 0.2, 0.3, 0.4), Detector(9, 7, 0.3, 0.4)
    geometry = ParallelGeometry(grid, detector, np.arange(19))

    def resolve(values, weights, g=grid, d=detector):
        return resolve_tv_metric(
            geometry, g, d, values, weights=weights, tv_weight=5e-5, huber_delta=0.01
        )

    host = resolve(y, w)
    device = jax.jit(resolve)(jnp.asarray(y), jnp.asarray(w))
    np.testing.assert_allclose(host.weight, device.weight, rtol=1e-6)
    integer = np.arange(y.size, dtype=np.int32).reshape(y.shape)
    np.testing.assert_allclose(
        resolve(integer, None).weight, resolve(jnp.asarray(integer), None).weight, rtol=1e-6
    )
    assert host.observed_count == np.count_nonzero(w)
    for length, amplitude in [(1000.0, 1.0), (1.0, 3.0), (0.001, 2.0)]:
        g = replace(grid, vx=grid.vx * length, vy=grid.vy * length, vz=grid.vz * length)
        d = replace(detector, du=detector.du * length, dv=detector.dv * length)
        scaled = resolve(y * amplitude, w, g, d)
        np.testing.assert_allclose(scaled.weight, host.weight * length * amplitude, rtol=1e-6)
        np.testing.assert_allclose(scaled.delta, host.delta * amplitude / length, rtol=1e-6)
    np.testing.assert_allclose(resolve(y, 3 * w).weight, 3 * host.weight, rtol=1e-6)
    derivative = jax.jit(jax.grad(lambda values: resolve(values, jnp.asarray(w)).weight))(y)
    assert np.isfinite(derivative).all()
    assert np.all(np.asarray(derivative)[w == 0] == 0)
    flat = jnp.ones_like(jnp.asarray(y))
    zero, derivative = jax.value_and_grad(lambda a: resolve(a, None).weight)(flat)
    assert zero == 0 and np.isfinite(derivative).all()


@pytest.mark.parametrize("weighted", [False, True])
def test_profiled_residual_is_weighted_nuisance_minimum(weighted):
    y = jnp.arange(30, dtype=jnp.float32).reshape(2, 3, 5) / 7
    w = jnp.linspace(0, 1, 30).reshape(y.shape) if weighted else jnp.ones_like(y)
    result = project_residual(y, "per_view_contrast", weights=w if weighted else None)
    np.testing.assert_allclose(jnp.sum(w * result, axis=(-2, -1)), 0, atol=3e-6)
    np.testing.assert_allclose(
        project_residual(result, "per_view_contrast", weights=w), result, atol=3e-7
    )

    def loss(values):
        return 0.5 * jnp.sum(w * project_residual(values, "per_view_contrast", weights=w) ** 2)

    np.testing.assert_allclose(jax.grad(loss)(y), w * result, atol=5e-7)
    assert project_residual(y, "absolute") is y
    masked = jnp.where(w > 0, y, jnp.nan)
    np.testing.assert_allclose(
        project_residual(masked, "absolute", weights=w), jnp.where(w > 0, y, 0)
    )


@pytest.mark.parametrize(("model", "batch"), [("joseph", 3), ("ray", 1), ("ray", 3)])
def test_fista_contrast_matches_profiled_gradient_and_offset_invariance(model, batch):
    from tomojax.recon import FistaConfig, fista_tv
    from tomojax.recon.fista_tv import grad_data_term

    grid, detector = Grid(4, 3, 2, 0.7, 0.8, 0.9), Detector(5, 4, 0.6, 0.7)
    geometry = ParallelGeometry(grid, detector, np.linspace(0, 170, 7))
    y = np.random.default_rng(4).normal(size=(7, 4, 5)).astype(np.float32)
    cfg = FistaConfig(
        iterations=1,
        tv_weight=0,
        lipschitz=100,
        backend="jax",
        projector_model=model,
        views_per_batch=batch,
        data_term="per_view_contrast",
    )
    x, info = fista_tv(geometry, grid, detector, y, config=cfg)
    offset = np.arange(7, dtype=np.float32)[:, None, None]
    shifted, _ = fista_tv(geometry, grid, detector, y + offset, config=cfg)
    np.testing.assert_allclose(x, shifted, rtol=3e-5, atol=2e-8)
    assert info["background_profiled"] and info["effective_tv_weight"] == 0
    if model == "ray":
        gradient, _ = grad_data_term(
            geometry,
            grid,
            detector,
            jnp.asarray(y),
            jnp.zeros_like(x),
            data_term="per_view_contrast",
            views_per_batch=batch,
        )
        np.testing.assert_allclose(x, -gradient / 100, rtol=1e-6, atol=2e-8)
    else:
        streamed, _ = fista_tv(
            geometry, grid, detector, y, config=replace(cfg, stream_projections=True)
        )
        np.testing.assert_allclose(x, streamed, rtol=1e-6, atol=2e-8)


def test_metric_cone_segments_and_pose_wrappers_use_object_plane_area():
    from tomojax import Scan
    from tomojax.geometry import ConeBeam, ConeGeometry, ConeSegments

    grid, detector = Grid(4, 3, 2, 0.7, 0.8, 0.9), Detector(5, 4, 0.6, 0.7)
    cone = ConeGeometry(grid, detector, np.arange(19), ConeBeam(10, 20))
    segments = ConeSegments(
        (
            replace(cone, angles=np.arange(7)),
            replace(cone, angles=np.arange(12), beam=ConeBeam(10, 30)),
        )
    )
    data = np.ones((19, 4, 5), np.float32)
    area = detector.nu * detector.du * detector.nv * detector.dv
    for geometry, factor in [(cone, 1 / 4), (segments, (7 / 4 + 12 / 9) / 19)]:
        for wrapped in [geometry, Scan(data, geometry).with_poses(np.zeros((19, 6))).geometry]:
            metric = resolve_tv_metric(
                wrapped, grid, detector, data, tv_weight=5e-5, huber_delta=0.01
            )
            assert metric.length_scale**2 == pytest.approx(area * factor)
            assert metric.weight == 0
            assert metric.weight_over_contrast > 0


@pytest.mark.parametrize("regulariser", ["tv", "huber_tv"])
@pytest.mark.parametrize("weight", [0, 0.001, None])
@pytest.mark.parametrize("support", [None, np.asarray([0.5, 0, 0.7, 1], np.float32)])
def test_public_fista_physical_metric_covaries_with_units_and_amplitude(
    regulariser, weight, support
):
    from tomojax.recon import FistaConfig, fista_tv

    grid, detector = Grid(4, 3, 2, 0.7, 0.8, 0.9), Detector(5, 4, 0.6, 0.7)
    geometry = ParallelGeometry(grid, detector, np.linspace(0, 170, 7))
    data = np.random.default_rng(4).normal(size=(7, 4, 5)).astype(np.float32)
    cfg = FistaConfig(
        iterations=5,
        tv_weight=weight,
        regulariser=regulariser,
        support=support,
        backend="jax",
        data_term="per_view_contrast",
    )
    expected, info = fista_tv(geometry, grid, detector, data, config=cfg)
    if support is not None:
        np.testing.assert_array_equal(np.asarray(expected)[..., support == 0], 0)
        zero_cfg = replace(cfg, iterations=0, support=support.tolist(), lower_bound=2)
        start, _ = fista_tv(geometry, grid, detector, data, config=zero_cfg)
        np.testing.assert_array_equal(
            start, np.broadcast_to(np.where(support != 0, 2, 0), start.shape)
        )
    if weight is None:
        explicit, _ = fista_tv(
            geometry,
            grid,
            detector,
            data,
            config=replace(cfg, tv_weight=info["resolved_tv_weight"]),
        )
        np.testing.assert_allclose(explicit, expected, rtol=2e-5, atol=3e-7)
    length, amplitude = 1000, 3
    g = replace(grid, vx=grid.vx * length, vy=grid.vy * length, vz=grid.vz * length)
    d = replace(detector, du=detector.du * length, dv=detector.dv * length)
    scaled = replace(geometry, grid=g, detector=d)
    actual, record = fista_tv(scaled, g, d, data * amplitude, config=cfg)
    np.testing.assert_allclose(actual * length / amplitude, expected, rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(record["loss"], np.asarray(info["loss"]) * amplitude**2, rtol=2e-5)
    flat, _ = fista_tv(geometry, grid, detector, np.ones_like(data), config=cfg)
    assert np.isfinite(flat).all()


def test_noise_weight_raises_all_metric_ratios_without_changing_geometry():
    from tomojax.recon._tv_metric import (  # check-public-imports: allow-private
        TVMetric,
        raise_tv_weight,
    )

    metric = TVMetric((1.0, 2.0, 0.5), 2.0, 0.1, 4.0, 3.0, 0.5, 8.0, 8.0)
    raised = raise_tv_weight(metric, 3.0)
    np.testing.assert_allclose((raised.weight, raised.weight_over_contrast), (3.0, 6.0))
    assert raised.delta == metric.delta and raised.gradient_scale_xyz == metric.gradient_scale_xyz
    assert float(raise_tv_weight(metric, 1.0).weight) == metric.weight
    # The selected dimensionless strength is not differentiated through noise.
    assert float(jax.grad(lambda noise: raise_tv_weight(metric, noise).weight)(3.0)) == 0


@pytest.mark.parametrize("pixels", [1, 2])
def test_explicit_fista_weight_bypasses_noise_and_single_pixel_auto_is_zero(monkeypatch, pixels):
    import importlib

    from tomojax.recon import FistaConfig, fista_tv

    module = importlib.import_module("tomojax.recon.fista_tv")

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit weight must not estimate noise")

    monkeypatch.setattr(module, "estimate_projection_noise", forbidden)
    grid, detector = Grid(2, 2, 1, 1, 1, 1), Detector(pixels, 1, 1, 1)
    geometry = ParallelGeometry(grid, detector, [0, 90])
    for weight in [0.0, 0.001] + ([None] if pixels == 1 else []):
        volume, info = fista_tv(
            geometry,
            grid,
            detector,
            np.ones((2, 1, pixels), np.float32),
            config=FistaConfig(iterations=2, tv_weight=weight, backend="jax"),
        )
        assert np.isfinite(volume).all() and info["effective_tv_weight"] == 0
