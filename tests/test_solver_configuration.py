"""Iterative solver controls fail early and explicit step sizes are never discarded."""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.recon import FistaConfig, SPDHGConfig, fista_tv, spdhg_tv


def _problem(tilted=False):
    grid = Grid(2, 3, 2, 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.9, 1.2, (0.17, -0.23))
    angles = np.array([13.0, 68.0, 142.0])
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt_deg=30)
        if tilted
        else ParallelGeometry(grid, detector, angles)
    )
    data = np.random.default_rng(78).normal(size=(3, 3, 4)).astype(np.float32)
    return grid, detector, geometry, data


@pytest.mark.parametrize("solver", ["fista", "spdhg"])
@pytest.mark.parametrize("field", ["iterations", "views_per_batch", "projector_unroll"])
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_solvers_reject_invalid_counts_before_building_geometry(monkeypatch, solver, field, value):
    grid, detector, geometry, data = _problem()
    module = importlib.import_module(f"tomojax.recon.{solver}_tv")

    def unexpected_geometry(*args, **kwargs):
        pytest.fail("invalid configuration reached geometry construction")

    monkeypatch.setattr(module, "stack_view_poses", unexpected_geometry)
    config = (FistaConfig if solver == "fista" else SPDHGConfig)(**{field: value})
    with pytest.raises(ValueError, match=field):
        (fista_tv if solver == "fista" else spdhg_tv)(geometry, grid, detector, data, config=config)


@pytest.mark.parametrize("value", [-1.0, np.inf, np.nan])
@pytest.mark.parametrize(
    ("solver", "field"),
    [
        ("fista", "tv_weight"),
        ("fista", "lipschitz"),
        ("fista", "recon_rel_tol"),
        ("spdhg", "tv_weight"),
        ("spdhg", "theta"),
        ("spdhg", "tau"),
        ("spdhg", "sigma_data"),
        ("spdhg", "sigma_tv"),
    ],
)
def test_solvers_reject_invalid_scalars_before_building_geometry(monkeypatch, solver, field, value):
    grid, detector, geometry, data = _problem()
    module = importlib.import_module(f"tomojax.recon.{solver}_tv")

    def unexpected_geometry(*args, **kwargs):
        pytest.fail("invalid configuration reached geometry construction")

    monkeypatch.setattr(module, "stack_view_poses", unexpected_geometry)
    config = (FistaConfig if solver == "fista" else SPDHGConfig)(**{field: value})
    with pytest.raises(ValueError, match=field):
        (fista_tv if solver == "fista" else spdhg_tv)(geometry, grid, detector, data, config=config)


@pytest.mark.parametrize(
    ("solver", "field"),
    [
        ("fista", "views_per_batch"),
        ("fista", "projector_unroll"),
        ("fista", "lipschitz"),
        ("fista", "tv_prox_iterations"),
        ("fista", "power_iterations"),
        ("spdhg", "views_per_batch"),
        ("spdhg", "projector_unroll"),
        ("spdhg", "tau"),
        ("spdhg", "sigma_data"),
        ("spdhg", "sigma_tv"),
    ],
)
def test_solvers_reject_zero_sizes_and_steps(solver, field):
    grid, detector, geometry, data = _problem()
    config = (FistaConfig if solver == "fista" else SPDHGConfig)(**{field: 0})
    with pytest.raises(ValueError, match=field):
        (fista_tv if solver == "fista" else spdhg_tv)(geometry, grid, detector, data, config=config)


@pytest.mark.parametrize("field", ["tv_prox_iterations", "power_iterations", "recon_patience"])
@pytest.mark.parametrize("value", [-1, 1.5])
def test_fista_rejects_invalid_internal_iteration_counts(field, value):
    grid, detector, geometry, data = _problem()
    with pytest.raises(ValueError, match=field):
        fista_tv(geometry, grid, detector, data, config=FistaConfig(**{field: value}))


@pytest.mark.parametrize("value", [-1, 1.5])
def test_spdhg_rejects_invalid_logging_interval(value):
    grid, detector, geometry, data = _problem()
    with pytest.raises(ValueError, match="log_every"):
        spdhg_tv(geometry, grid, detector, data, config=SPDHGConfig(log_every=value))


@pytest.mark.parametrize("value", [-1.0, -1e-40, np.nan, np.inf, 1j])
@pytest.mark.parametrize("storage", ["host", "device", "memmap"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("iterations", [0, 1])
def test_spdhg_rejects_invalid_weight_tail_before_geometry(
    monkeypatch, tmp_path, value, storage, stream, iterations
):
    grid, detector, geometry, data = _problem()
    module = importlib.import_module("tomojax.recon.spdhg_tv")

    def unexpected_geometry(*args, **kwargs):
        pytest.fail("invalid weights reached geometry construction")

    monkeypatch.setattr(module, "stack_view_poses", unexpected_geometry)
    dtype = np.complex64 if np.iscomplexobj(value) else np.float32
    weights = np.ones(data.shape, dtype=dtype)
    weights[-1, -1, -1] = value  # Invalid even if this tail block is never visited.
    if storage == "device":
        weights = jnp.asarray(weights)
    elif storage == "memmap":
        mapped = np.memmap(tmp_path / "weights", mode="w+", dtype=dtype, shape=data.shape)
        mapped[:] = weights
        weights = mapped
    config = SPDHGConfig(iterations=iterations, views_per_batch=2, stream_projections=stream)
    with pytest.raises(ValueError, match="weights must be real, finite in FP32 and nonnegative"):
        spdhg_tv(geometry, grid, detector, data, weights=weights, config=config)


@pytest.mark.parametrize("value", [-1e-50, 1e40, "invalid"])
def test_spdhg_checks_host_weights_before_fp32_conversion(monkeypatch, value):
    grid, detector, geometry, data = _problem()
    module = importlib.import_module("tomojax.recon.spdhg_tv")
    monkeypatch.setattr(
        module, "stack_view_poses", lambda *args, **kwargs: pytest.fail("geometry constructed")
    )
    weights = np.ones(data.shape, dtype=object if isinstance(value, str) else np.float64)
    weights[-1, -1, -1] = value
    with pytest.raises(ValueError, match="weights must be real, finite in FP32 and nonnegative"):
        spdhg_tv(geometry, grid, detector, data, weights=weights, config=SPDHGConfig(iterations=0))


@pytest.mark.numerical
@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int32, np.uint32, np.float16, np.float32, jnp.bfloat16]
)
@pytest.mark.parametrize("device", [False, True])
def test_spdhg_accepts_real_zero_weight_masks(dtype, device):
    grid, detector, geometry, data = _problem()
    weights = np.zeros(data.shape, dtype=dtype)
    if np.issubdtype(weights.dtype, np.floating):
        weights[:] = -0.0
    volume, info = spdhg_tv(
        geometry,
        grid,
        detector,
        data,
        weights=jnp.asarray(weights) if device else weights,
        config=SPDHGConfig(
            iterations=1,
            tv_weight=0,
            tau=0.1,
            sigma_data=0.1,
            sigma_tv=0.1,
            log_every=1,
            projector_backend="jax",
        ),
    )
    np.testing.assert_array_equal(volume, 0)
    np.testing.assert_array_equal(info["loss"], 0)


@pytest.mark.parametrize("storage", ["memmap", "noncontiguous"])
def test_spdhg_host_weight_validation_is_bounded_and_does_not_upload(
    monkeypatch, tmp_path, storage
):
    shape = (3, 513, 1024)  # More than the validation chunk bound.
    grid = Grid(1, 1, 1, 1, 1, 1)
    detector = Detector(shape[2], shape[1], 1, 1)
    geometry = ParallelGeometry(grid, detector, [0, 60, 120])
    data = np.broadcast_to(np.float32(1), shape)
    if storage == "memmap":
        weights = np.memmap(tmp_path / "weights", mode="w+", dtype=np.float32, shape=shape)
        weights[:] = 1
    else:
        weights = np.ones((shape[0], shape[1], shape[2] * 2), np.float32)[:, :, ::2]
    weights[-1, -1, -1] = -1
    checked_sizes = []
    isfinite = np.isfinite

    def bounded_isfinite(array):
        checked_sizes.append(array.size)
        assert array.size <= 2**20
        return isfinite(array)

    monkeypatch.setattr(np, "isfinite", bounded_isfinite)
    monkeypatch.setattr(jnp, "asarray", lambda *args, **kwargs: pytest.fail("host stack uploaded"))
    module = importlib.import_module("tomojax.recon.spdhg_tv")
    monkeypatch.setattr(
        module, "stack_view_poses", lambda *args, **kwargs: pytest.fail("geometry constructed")
    )
    with pytest.raises(ValueError, match="weights must be real, finite in FP32 and nonnegative"):
        spdhg_tv(geometry, grid, detector, data, weights=weights, config=SPDHGConfig(iterations=0))
    assert len(checked_sizes) > 1
    assert sum(checked_sizes) == weights.size


@pytest.mark.parametrize(
    "dtype", [jnp.float4_e2m1fn, jnp.float6_e2m3fn, jnp.float8_e4m3fn, jnp.float8_e5m2]
)
def test_spdhg_rejects_extended_host_weight_formats_cleanly(monkeypatch, dtype):
    grid, detector, geometry, data = _problem()
    weights = np.ones(data.shape, dtype=dtype)
    module = importlib.import_module("tomojax.recon.spdhg_tv")
    monkeypatch.setattr(
        module, "stack_view_poses", lambda *args, **kwargs: pytest.fail("geometry constructed")
    )
    with pytest.raises(ValueError, match="standard NumPy floating or bfloat16 dtypes"):
        spdhg_tv(geometry, grid, detector, data, weights=weights, config=SPDHGConfig(iterations=0))


@pytest.mark.parametrize("dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2])
def test_spdhg_rejects_extended_device_weight_formats_cleanly(monkeypatch, dtype):
    grid, detector, geometry, data = _problem()
    weights = jnp.asarray(np.ones(data.shape, dtype=dtype))
    module = importlib.import_module("tomojax.recon.spdhg_tv")
    monkeypatch.setattr(
        module, "stack_view_poses", lambda *args, **kwargs: pytest.fail("geometry constructed")
    )
    with pytest.raises(ValueError, match="standard NumPy floating or bfloat16 dtypes"):
        spdhg_tv(geometry, grid, detector, data, weights=weights, config=SPDHGConfig(iterations=0))


def test_fista_rejects_unknown_gradient_mode():
    grid, detector, geometry, data = _problem()
    with pytest.raises(ValueError, match="grad_mode"):
        fista_tv(geometry, grid, detector, data, config=FistaConfig(grad_mode="unknown"))


@pytest.mark.numerical
@pytest.mark.parametrize("solver", ["fista", "spdhg"])
def test_zero_iteration_budget_keeps_initial_volume_and_logs_nothing(solver):
    grid, detector, geometry, data = _problem()
    initial = np.random.default_rng(99).normal(size=(2, 3, 2)).astype(np.float32)
    if solver == "fista":
        config = FistaConfig(iterations=0, tv_weight=0, lipschitz=10, recon_rel_tol=0)
    else:
        config = SPDHGConfig(iterations=0, tv_weight=0, tau=0.1, sigma_data=0.1, log_every=0)
    calls = []
    volume, info = (fista_tv if solver == "fista" else spdhg_tv)(
        geometry,
        grid,
        detector,
        data,
        init_x=initial,
        config=config,
        callback=lambda *args: calls.append(args),
    )
    np.testing.assert_array_equal(volume, initial)
    assert info["loss"] == []
    assert calls == []


@pytest.mark.numerical
@pytest.mark.parametrize("model", ["ray", "joseph"])
@pytest.mark.parametrize("warm_start", [False, True])
def test_fista_auto_step_is_finite_for_zero_operator(model, warm_start):
    grid, detector, geometry, data = _problem()
    initial = np.random.default_rng(99).normal(size=(2, 3, 2)).astype(np.float32)
    volume, info = fista_tv(
        geometry,
        grid,
        detector,
        data,
        init_x=initial if warm_start else None,
        config=FistaConfig(
            iterations=3,
            tv_weight=0,
            support=jnp.zeros((2, 3, 2)),
            projector_model=model,
            projector_backend="jax",
        ),
    )
    np.testing.assert_array_equal(volume, initial if warm_start else np.zeros((2, 3, 2)))
    np.testing.assert_allclose(info["loss"], np.full(3, 0.5 * np.sum(data**2)), rtol=2e-6)
    assert np.isfinite(info["lipschitz"]) and info["lipschitz"] > 0
