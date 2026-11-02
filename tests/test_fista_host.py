"""Host FISTA's local finite-step TV work agrees across slab boundaries."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.geometry import Detector, Grid
from tomojax.recon import FistaConfig
from tomojax.recon._cone_slabs import HostVolume  # check-public-imports: allow-private
from tomojax.recon._cuda_tv import (
    proximal_host,
    workspace_bytes,
)  # check-public-imports: allow-private
from tomojax.recon._slab_engines import CudaEngine  # check-public-imports: allow-private
from tomojax.recon._tv_ops import (  # check-public-imports: allow-private
    cuda_tv_workspace_bytes,
    gradient_norm_squared,
    tv_proximal,
)


@pytest.mark.gpu
@pytest.mark.parametrize("iterations", [0, 1, 3, 10])
def test_host_tv_halos_keep_global_step_and_constraints(monkeypatch, iterations):
    import cupy as cp

    data = np.random.default_rng(20261101).normal(size=(35, 9, 7)).astype(np.float32)
    output = np.empty_like(data)
    engine = CudaEngine(Grid(7, 9, 35, 1, 1, 1), Detector(7, 9, 1, 1))
    source, target = HostVolume(data, engine, 3), HostVolume(output, engine, 3)
    expected = np.asarray(
        tv_proximal(jnp.asarray(data), 0.2, iterations, lower_bound=-0.1, upper_bound=0.7)
    )
    cp.get_default_memory_pool().free_all_blocks()
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (1000, 1000))
    proximal_host(
        source,
        target.put,
        0.2,
        iterations=iterations,
        tau=1 / gradient_norm_squared(data.shape),
        lower=-0.1,
        upper=0.7,
    )
    source.close()
    target.close()
    np.testing.assert_allclose(output, expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(
        data, np.random.default_rng(20261101).normal(size=data.shape).astype(np.float32)
    )


@pytest.mark.gpu
@pytest.mark.parametrize(("warm", "limit", "tolerance"), [(False, None, None), (True, 200.0, 1.0)])
def test_host_fista_matches_resident_with_mapped_state(
    tmp_path, monkeypatch, warm, limit, tolerance
):
    from tomojax.geometry import LaminographyGeometry
    from tomojax.recon import (  # check-public-imports: allow-private
        FistaConfig,
        _parallel_slabs,
        fista_tv,
    )

    grid, det = Grid(9, 7, 5, 0.8, 1.1, 1.3), Detector(11, 9, 0.9, 1.2)
    geometry = LaminographyGeometry(grid, det, np.linspace(0, 270, 17), tilt_deg=53)
    rng = np.random.default_rng(20261101)
    data = rng.uniform(0, 1, (17, 9, 11)).astype(np.float32)
    initial = rng.uniform(0, 0.2, (5, 7, 9)).astype(np.float32) if warm else None
    cfg = FistaConfig(
        iterations=5,
        tv_weight=0.1,
        tv_prox_iterations=3,
        nonnegative=True,
        upper_bound=0.7,
        lipschitz=limit,
        recon_rel_tol=tolerance,
        recon_patience=1,
        views_per_batch=6,
    )
    expected, reference = fista_tv(geometry, grid, det, data, init_x=initial, config=cfg)
    mapped = np.memmap(tmp_path / "volume", np.float32, "w+", shape=(5, 7, 9))
    if warm:
        mapped[...] = initial
    monkeypatch.setattr(_parallel_slabs, "slab_depth", lambda *args: 2)
    if warm:
        actual, record = fista_tv(geometry, grid, det, data, init_x=mapped, config=cfg, out=mapped)
    else:
        result = tj.reconstruct(tj.Scan(data, geometry), "fista", config=cfg, out=mapped)
        actual, record = result.volume, result.info
    assert actual is mapped
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(record["loss"], reference["loss"], rtol=3e-5)
    np.testing.assert_allclose(record["lipschitz"], reference["lipschitz"], rtol=3e-5)
    assert record["effective_iterations"] == reference["effective_iterations"]


def test_host_output_rejected_on_unsupported_backend():
    from tomojax.geometry import ParallelGeometry
    from tomojax.recon import FistaConfig, fista_tv

    grid, det = Grid(2, 2, 1, 1, 1, 1), Detector(2, 1, 1, 1)
    geometry = ParallelGeometry(grid, det, [0, 90])
    with pytest.raises(ValueError, match="out requires"):
        fista_tv(
            geometry,
            grid,
            det,
            np.ones((2, 1, 2), np.float32),
            config=FistaConfig(backend="jax"),
            out=np.zeros((1, 2, 2), np.float32),
        )


def test_cuda_tv_workspace_budget_matches_both_allocation_branches(monkeypatch):
    assert workspace_bytes((560, 640, 640), 10) == 444416000
    assert workspace_bytes((300, 300, 300), 100) == 16 * 300**3
    assert cuda_tv_workspace_bytes((32, 32, 32), np.dtype(np.float32), 10) is None
    monkeypatch.setenv("TOMOJAX_CUDA_KERNELS", "0")
    assert cuda_tv_workspace_bytes((560, 640, 640), np.dtype(np.float32), 10) is None


@pytest.mark.parametrize("weight", [None, 0.0, 0.005])
@pytest.mark.parametrize("regulariser", ["tv", "huber_tv"])
@pytest.mark.parametrize("shape", [(), (8, 1, 1), (8, 8, 8)])
def test_fista_host_selection_counts_proximal_workspace(monkeypatch, weight, regulariser, shape):
    from tomojax.recon._fista_host import solve_on_host  # check-public-imports: allow-private

    grid, detector = Grid(8, 8, 8, 1, 1, 1), Detector(1, 1, 1, 1)
    runtime = SimpleNamespace(
        projector=("joseph", "cuda", 1),
        x0=None,
        split=None,
        stream=True,
        constraints=None,
        config=FistaConfig(tv_weight=weight, regulariser=regulariser, support=np.ones(shape)),
    )
    data = np.zeros((1, 1, 1), np.float32)
    # Four outer volumes, four reference-TV fields and four one-pixel batches.
    boundary = ((4 + 4 * (weight != 0)) * 4 * 8**3 + 16 + (weight != 0) * np.prod(shape)) / 0.85
    monkeypatch.setattr("tomojax.backends.device_free_memory_bytes", lambda: boundary + 1)
    assert solve_on_host(None, grid, detector, data, runtime) is None
    monkeypatch.setattr("tomojax.backends.device_free_memory_bytes", lambda: boundary - 1)
    # A support mask is resident-only: selecting host must fail before any allocation.
    with pytest.raises(ValueError, match="host volumes require"):
        solve_on_host(None, grid, detector, data, runtime)
