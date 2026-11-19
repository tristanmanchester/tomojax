"""The compact TV field and the in-place SPDHG update retain FP32 accuracy."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, LaminographyGeometry
from tomojax.recon import SPDHGConfig
from tomojax.recon._spdhg_flux import (
    pack_flux,
    unpack_flux,
)  # check-public-imports: allow-private
from tomojax.recon.spdhg_tv import (  # check-public-imports: allow-private
    _prepare_spdhg_runtime,
    _run_spdhg_scan,
    _sweep,
)


def test_packed_flux_rounds_within_one_ulp_of_each_group():
    # Signed extrema immediately below a power of two round up at 24 bits.
    # Include zero, a short final group and groups on different physical scales.
    rng = np.random.default_rng(6)
    values = rng.uniform(-1, 1, (3, 3, 1, 19)).astype(np.float32)
    values[0, :, :, :2] = [np.nextafter(np.float32(2), np.float32(0)), -2]
    values[:, 0, :, :8] = 0
    values[:, :, :, 8:16] *= 1e-5
    actual = np.stack(jax.jit(lambda p: unpack_flux(pack_flux(p)))(tuple(values)))
    for start in range(0, 19, 8):
        group = values[..., start : start + 8]
        spacing = np.spacing(np.max(abs(group), axis=(0, 2, 3), keepdims=True))
        assert np.all(abs(actual[..., start : start + 8] - group) <= spacing)


@pytest.mark.parametrize("regularizer", ["tv", "huber_tv"])
def test_sweep_matches_whole_volume_update(regularizer):
    # Four planes per slab, a partial tail, and a different spacing on each axis.
    shape = (11, 255, 257)
    rng = np.random.default_rng(3)
    x, data = (jnp.asarray(rng.normal(size=shape), jnp.float32) for _ in range(2))
    p = tuple(jnp.asarray(rng.normal(size=shape) * 0.02, jnp.float32) for _ in range(3))
    p = (p[0].at[:, :, -1].set(0), p[1].at[:, -1, :].set(0), p[2].at[-1, :, :].set(0))
    scales, sigma, weight, delta = (0.7, 1.3, 0.9), 0.37, 0.05, 0.11
    runtime = SimpleNamespace(
        config=SPDHGConfig(iterations=2, tv_weight=weight, theta=0.8),
        schedule=SimpleNamespace(selection_prob=0.25),
        step_sizes=SimpleNamespace(sigma_tv=sigma, tau=0.3),
        storage_scales=scales,
        metric=SimpleNamespace(gradient_scale_xyz=scales, weight=weight),
        regularizer=regularizer,
        huber_delta=delta,
    )

    @jax.jit
    def update(x, p, data):
        return _sweep(SimpleNamespace(x=x, flux=p, s_data=data), runtime, jnp.int32(0))

    actual, partial = update(x, pack_flux(p), data)
    gradients = tuple(
        c * jnp.diff(x, axis=i, append=jnp.take(x, jnp.array([-1]), axis=i))
        for i, c in zip((2, 1, 0), scales, strict=True)
    )
    q = jnp.stack([a + sigma * b for a, b in zip(p, gradients, strict=True)])
    if regularizer == "huber_tv":
        q *= weight / (weight + sigma * delta)
    expected = q / jnp.maximum(1, jnp.linalg.norm(q, axis=0) / weight)

    def divergence(q):
        return sum(
            c * jnp.diff(qj, axis=i, prepend=jnp.zeros_like(jnp.take(qj, jnp.array([0]), axis=i)))
            for qj, i, c in zip(q, (2, 1, 0), scales, strict=True)
        )

    tv = -divergence(expected)
    extrapolated = tv + 0.8 * (tv + divergence(p))
    np.testing.assert_allclose(unpack_flux(actual), expected, rtol=0, atol=4e-8)
    np.testing.assert_allclose(
        partial, x - 0.3 * (extrapolated - 0.8 / 0.25 * data), rtol=1e-6, atol=5e-7
    )


@pytest.mark.gpu
def test_compiled_solve_fits_in_four_and_a_half_volume_buffers():
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector = Grid(256, 256, 257, 1.0, 1.0, 1.0), Detector(256, 257, 1.0, 1.0)
    geometry = LaminographyGeometry(grid, detector, np.linspace(0, 360, 8, endpoint=False), 30)
    data = np.random.default_rng(4).random((8, 257, 256), dtype=np.float32)
    runtime = _prepare_spdhg_runtime(
        geometry,
        grid,
        detector,
        data,
        weights=None,
        init_x=None,
        config=SPDHGConfig(iterations=3, tv_weight=0.01, views_per_batch=4, stream=False),
        det_grid=None,
    )
    mem = _run_spdhg_scan.lower(grid, detector, runtime).compile().memory_analysis()
    volume = 4 * grid.nx * grid.ny * grid.nz
    # One accumulator plus bounded scratch must fit in a single allocator region.
    assert mem.temp_size_in_bytes <= 1.25 * volume + data.nbytes
    assert mem.temp_size_in_bytes + mem.output_size_in_bytes <= 4.5 * volume + data.nbytes


def test_insufficient_state_memory_names_host_solvers(monkeypatch):
    import importlib

    from tomojax.recon import spdhg_tv

    module = importlib.import_module("tomojax.recon.spdhg_tv")
    grid, detector = Grid(8, 8, 4, 1.0, 1.0, 1.0), Detector(8, 4, 1.0, 1.0)
    geometry = LaminographyGeometry(grid, detector, np.linspace(0, 360, 6, endpoint=False), 30)
    monkeypatch.setattr(module, "device_free_memory_bytes", lambda: 4 * 4 * 8 * 8 * 4)
    with pytest.raises(MemoryError, match=r"volume state .* fista, cgls and sirt"):
        spdhg_tv(geometry, grid, detector, np.zeros((6, 4, 8), np.float32))
