"""Iterative solver controls fail early and explicit step sizes are never discarded."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.recon import FISTAConfig, SPDHGConfig, fista, spdhg


def _problem(tilted=False):
    grid = Grid(2, 3, 2, 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.9, 1.2, (0.17, -0.23))
    angles = np.array([13.0, 68.0, 142.0])
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt=30)
        if tilted
        else ParallelGeometry(grid, detector, angles)
    )
    data = np.random.default_rng(78).normal(size=(3, 3, 4)).astype(np.float32)
    return grid, detector, geometry, data


@pytest.mark.parametrize(
    ("solver", "field"),
    [
        ("fista", "iterations"),
        ("spdhg", "iterations"),
        ("fista", "views_per_batch"),
        ("spdhg", "views_per_block"),
    ],
)
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_solvers_reject_invalid_counts(solver, field, value):
    grid, detector, geometry, data = _problem()
    with pytest.raises(ValueError, match=field):
        (fista if solver == "fista" else spdhg)(
            geometry,
            grid,
            detector,
            data,
            config=(FISTAConfig if solver == "fista" else SPDHGConfig)(**{field: value}),
        )


@pytest.mark.parametrize("value", [-1.0, np.inf, np.nan])
@pytest.mark.parametrize(
    ("solver", "field"),
    [
        ("fista", "tv_weight"),
        ("fista", "lipschitz"),
        ("fista", "tolerance"),
        ("spdhg", "tv_weight"),
        ("spdhg", "theta"),
        ("spdhg", "tau"),
        ("spdhg", "sigma_data"),
        ("spdhg", "sigma_tv"),
    ],
)
def test_solvers_reject_invalid_scalars(solver, field, value):
    grid, detector, geometry, data = _problem()
    with pytest.raises(ValueError, match=field):
        (fista if solver == "fista" else spdhg)(
            geometry,
            grid,
            detector,
            data,
            config=(FISTAConfig if solver == "fista" else SPDHGConfig)(**{field: value}),
        )


@pytest.mark.parametrize(
    ("solver", "field"),
    [
        ("fista", "views_per_batch"),
        ("fista", "lipschitz"),
        ("fista", "tv_prox_iterations"),
        ("fista", "power_iterations"),
        ("spdhg", "views_per_block"),
        ("spdhg", "tau"),
        ("spdhg", "sigma_data"),
        ("spdhg", "sigma_tv"),
    ],
)
def test_solvers_reject_zero_sizes_and_steps(solver, field):
    grid, detector, geometry, data = _problem()
    with pytest.raises(ValueError, match=field):
        (fista if solver == "fista" else spdhg)(
            geometry,
            grid,
            detector,
            data,
            config=(FISTAConfig if solver == "fista" else SPDHGConfig)(**{field: 0}),
        )


@pytest.mark.parametrize("field", ["tv_prox_iterations", "power_iterations", "patience"])
@pytest.mark.parametrize("value", [-1, 1.5])
def test_fista_rejects_invalid_internal_iteration_counts(field, value):
    grid, detector, geometry, data = _problem()
    with pytest.raises(ValueError, match=field):
        fista(geometry, grid, detector, data, config=FISTAConfig(**{field: value}))


@pytest.mark.parametrize("value", [-1, 1.5])
def test_spdhg_rejects_invalid_logging_interval(value):
    grid, detector, geometry, data = _problem()
    with pytest.raises(ValueError, match="log_every"):
        spdhg(geometry, grid, detector, data, config=SPDHGConfig(log_every=value))


@pytest.mark.parametrize("value", [-1.0, -1e-40, np.nan, np.inf, 1j])
@pytest.mark.parametrize("storage", ["host", "device", "memmap"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("iterations", [0, 1])
def test_spdhg_rejects_an_invalid_weight_tail(tmp_path, value, storage, stream, iterations):
    grid, detector, geometry, data = _problem()
    dtype = np.complex64 if np.iscomplexobj(value) else np.float32
    weights = np.ones(data.shape, dtype=dtype)
    weights[-1, -1, -1] = value  # Invalid even if this tail block is never visited.
    if storage == "device":
        weights = jnp.asarray(weights)
    elif storage == "memmap":
        mapped = np.memmap(tmp_path / "weights", mode="w+", dtype=dtype, shape=data.shape)
        mapped[:] = weights
        weights = mapped
    config = SPDHGConfig(iterations=iterations, views_per_block=2, stream=stream)
    with pytest.raises(ValueError, match="weights must be real, finite in FP32 and nonnegative"):
        spdhg(geometry, grid, detector, data, weights=weights, config=config)


@pytest.mark.parametrize("value", [-1e-50, 1e40, "invalid"])
def test_spdhg_checks_host_weights_before_fp32_conversion(value):
    grid, detector, geometry, data = _problem()
    weights = np.ones(data.shape, dtype=object if isinstance(value, str) else np.float64)
    weights[-1, -1, -1] = value
    with pytest.raises(ValueError, match="weights must be real, finite in FP32 and nonnegative"):
        spdhg(geometry, grid, detector, data, weights=weights, config=SPDHGConfig(iterations=0))


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
    volume, info = spdhg(
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
            backend="jax",
        ),
    )
    np.testing.assert_array_equal(volume, 0)
    np.testing.assert_array_equal(info["loss"], 0)


@pytest.mark.parametrize(
    "dtype", [jnp.float4_e2m1fn, jnp.float6_e2m3fn, jnp.float8_e4m3fn, jnp.float8_e5m2]
)
def test_spdhg_rejects_extended_host_weight_formats_cleanly(dtype):
    grid, detector, geometry, data = _problem()
    weights = np.ones(data.shape, dtype=dtype)
    with pytest.raises(ValueError, match="standard NumPy floating or bfloat16 dtypes"):
        spdhg(geometry, grid, detector, data, weights=weights, config=SPDHGConfig(iterations=0))


@pytest.mark.parametrize("dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2])
def test_spdhg_rejects_extended_device_weight_formats_cleanly(dtype):
    grid, detector, geometry, data = _problem()
    weights = jnp.asarray(np.ones(data.shape, dtype=dtype))
    with pytest.raises(ValueError, match="standard NumPy floating or bfloat16 dtypes"):
        spdhg(geometry, grid, detector, data, weights=weights, config=SPDHGConfig(iterations=0))


@pytest.mark.numerical
@pytest.mark.parametrize("solver", ["fista", "spdhg"])
def test_zero_iteration_budget_keeps_initial_volume_and_logs_nothing(solver):
    grid, detector, geometry, data = _problem()
    initial = np.random.default_rng(99).normal(size=(2, 3, 2)).astype(np.float32)
    if solver == "fista":
        config = FISTAConfig(iterations=0, tv_weight=0, lipschitz=10, tolerance=0)
    else:
        config = SPDHGConfig(iterations=0, tv_weight=0, tau=0.1, sigma_data=0.1, log_every=0)
    volume, info = (fista if solver == "fista" else spdhg)(
        geometry,
        grid,
        detector,
        data,
        init_x=initial,
        config=config,
    )
    np.testing.assert_array_equal(volume, initial)
    assert info["loss"] == []


@pytest.mark.numerical
@pytest.mark.parametrize("warm_start", [False, True])
def test_fista_auto_step_is_finite_for_zero_operator(warm_start):
    grid, detector, geometry, data = _problem()
    initial = np.random.default_rng(99).normal(size=(2, 3, 2)).astype(np.float32)
    volume, info = fista(
        geometry,
        grid,
        detector,
        data,
        init_x=initial if warm_start else None,
        config=FISTAConfig(
            iterations=3,
            tv_weight=0,
            support=jnp.zeros((2, 3, 2), dtype=bool),
            backend="jax",
        ),
    )
    np.testing.assert_array_equal(volume, np.zeros_like(initial))
    np.testing.assert_allclose(info["loss"], np.full(3, 0.5 * np.sum(data**2)), rtol=2e-6)
    assert np.isfinite(info["lipschitz"]) and info["lipschitz"] > 0


@pytest.mark.parametrize(
    ("name", "valid", "invalid", "field"),
    [
        ("FBPConfig", {"filter": " HANN "}, {"filter": "banana"}, "filter"),
        ("CGLSConfig", {"damping": 0}, {"damping": -0.01}, "damping"),
        (
            "FISTAConfig",
            {"huber_delta": 0, "regularizer": "tv"},
            {"huber_delta": 0, "regularizer": "huber_tv"},
            "huber_delta",
        ),
        ("SPDHGConfig", {"views_per_block": 1}, {"views_per_block": 0}, "views_per_block"),
        ("SIRTConfig", {"subsets": 2}, {"subsets": 2, "acceleration": True}, "acceleration"),
        ("MLEMConfig", {"iterations": 0}, {"iterations": -1}, "iterations"),
        ("FourierConfig", {"slices_per_batch": 1}, {"slices_per_batch": 0}, "slices_per_batch"),
    ],
)
def test_configs_reject_domain_errors_at_construction(name, valid, invalid, field):
    from tomojax import InvalidParameter, recon

    constructor = getattr(recon, name)
    constructor(**valid)
    with pytest.raises(InvalidParameter) as error:
        constructor(**invalid)
    assert error.value.path == field
