"""Matched Poisson updates, angular subsets and reconstruction accuracy."""

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

import tomojax as tj
from tomojax.geometry import (
    ConeBeam,
    ConeGeometry,
    ConeSegments,
    Detector,
    Grid,
    LaminographyGeometry,
)
from tomojax.recon import MLEMConfig, mlem


@pytest.fixture(params=["lamino", "cone"])
def problem(request):
    grid = Grid(3, 2, 2, 0.8, 1.1, 1.3)
    detector = Detector(5, 4, 0.7, 1.2, (0.17, -0.23))
    angles = [83, 13, 161, 49, 122]
    geometry = LaminographyGeometry(grid, detector, angles, tilt_deg=23)
    if request.param == "cone":
        geometry = ConeSegments(
            (
                ConeGeometry(grid, detector, angles[:2], ConeBeam(12, 24)),
                ConeGeometry(
                    grid, replace(detector, center=(-0.2, 0.4)), angles[2:], ConeBeam(15, 27)
                ),
            )
        )
    basis = np.eye(12, dtype=np.float32).reshape(12, 3, 2, 2)
    # Assemble only forwards; NumPy's dense transpose supplies an independent
    # reference for all normalizations and multiplicative subset updates.
    matrix = np.stack([np.asarray(tj.project(geometry, x)).ravel() for x in basis], axis=1)
    return grid, detector, geometry, matrix.astype(np.float64)


@pytest.mark.numerical
@pytest.mark.parametrize("subsets", [1, 2, 3])
@pytest.mark.parametrize("batch", [2, 4])
def test_mlem_dense_subset_updates_with_support(problem, subsets, batch):
    grid, detector, geometry, matrix = problem
    support = np.ones((3, 2, 2), bool)
    support[0] = False
    truth = np.linspace(0.2, 1.1, 12) * support.ravel()
    data = matrix @ truth
    expected = np.full(12, 0.7) * support.ravel()
    order = np.argsort(geometry.angles, kind="stable")
    for _ in range(3):
        for group in range(subsets):
            rows = (order[group::subsets, None] * 20 + np.arange(20)).ravel()
            a = matrix[rows]
            den = a @ expected
            ratio = np.divide(data[rows], den, out=np.zeros_like(den), where=den > 0)
            sensitivity = a.sum(axis=0)
            expected *= np.divide(a.T @ ratio, sensitivity, out=np.ones(12), where=sensitivity > 0)
    volume, info = mlem(
        geometry,
        grid,
        detector,
        data.reshape(5, 4, 5),
        init_x=np.full((3, 2, 2), 0.7, np.float32),
        config=MLEMConfig(iterations=3, subsets=subsets, support=support, views_per_batch=batch),
    )
    np.testing.assert_allclose(np.asarray(volume).ravel(), expected, rtol=2e-5, atol=2e-6)
    assert info["subset_updates"] == 3 * subsets


@pytest.mark.numerical
def test_mlem_public_workflow_improves_likelihood_and_image(problem):
    grid, detector, geometry, matrix = problem
    truth = np.linspace(0.2, 1.1, 12)
    data = matrix @ truth
    scan = tj.Scan(data.reshape(5, 4, 5).astype(np.float32), geometry)
    previous = np.ones(12)
    valid = data > 0

    def likelihood(p):
        return np.sum(p[valid] - data[valid] * np.log(p[valid]))

    for iterations in [1, 10, 100]:
        volume = np.asarray(tj.reconstruct(scan, "mlem", iterations=iterations).volume).ravel()
        old, new = matrix @ previous, matrix @ volume
        assert likelihood(new) <= likelihood(old) + 1e-5
        assert np.linalg.norm(volume - truth) < np.linalg.norm(previous - truth)
        previous = volume
    assert np.linalg.norm(previous - truth) / np.linalg.norm(truth) < 0.04
    zero, _ = mlem(
        geometry, grid, detector, np.zeros((5, 4, 5), np.float32), config=MLEMConfig(iterations=2)
    )
    np.testing.assert_array_equal(np.asarray(zero).ravel()[matrix.sum(axis=0) > 0], 0)
    if isinstance(geometry, LaminographyGeometry):
        initial = np.maximum(np.asarray(tj.reconstruct(scan, "fbp").volume), 0)
        expected, _ = mlem(
            geometry, grid, detector, scan.projections, init_x=initial,
            config=MLEMConfig(iterations=1),
        )  # fmt: skip
        warm = tj.reconstruct(scan, "mlem", iterations=1, warm_start=True)
        np.testing.assert_allclose(warm.volume, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.numerical
@pytest.mark.parametrize("subsets", [1, 2])
def test_unobserved_activity_is_zero_not_the_initial_constant(subsets):
    geometry = tj.ParallelGeometry(Grid(3, 1, 3, 1, 1, 1), Detector(1, 1, 1, 1), [0, 90, 0])
    volume, _ = mlem(
        geometry, geometry.grid, geometry.detector, np.array([2, 6, 2], np.float32)[:, None, None],
        init_x=np.full((3, 1, 3), 7, np.float32),
        config=MLEMConfig(iterations=100, subsets=subsets),
    )  # fmt: skip
    np.testing.assert_allclose(np.asarray(volume), [[[0, 2, 0]]] * 3, atol=1e-5)


@pytest.mark.numerical
def test_uncached_sensitivities_preserve_visibility_and_updates(monkeypatch):
    module = import_module("tomojax.recon.mlem")
    geometry = tj.ParallelGeometry(Grid(3, 2, 3, 1, 1, 1), Detector(2, 1, 1, 1), [90, 0, 0])
    data = np.array([2, 1, 3, 2, 3, 2], np.float32).reshape(3, 1, 2)
    for iterations in (0, 1, 3):
        config = MLEMConfig(iterations=iterations, subsets=2)
        expected = tj.reconstruct(tj.Scan(data, geometry), "mlem", config=config).volume
        with monkeypatch.context() as patch:
            patch.setattr(module, "_SENSITIVITY_CACHE_BYTES", 0)
            patch.setattr(module, "device_free_memory_bytes", lambda: 0)
            actual = tj.reconstruct(tj.Scan(data, geometry), "mlem", config=config).volume
        np.testing.assert_array_equal(np.asarray(actual)[:, :, [0, 2]], 0)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_large_sensitivity_cache_is_bounded_and_requires_reuse(monkeypatch):
    module = import_module("tomojax.recon.mlem")
    for free, fits in ((None, False), (2 * 2**30, False), (4 * 2**30, True)):
        monkeypatch.setattr(module, "device_free_memory_bytes", lambda free=free: free)
        assert module._cache_sensitivities(2**30, 4, 4, workspace_bytes=2 * 2**30) == fits
        assert not module._cache_sensitivities(2**30, 4, 1, workspace_bytes=0)
    assert module._cache_sensitivities(3 * 2**30, 4, 4, workspace_bytes=2**30)
    assert not module._cache_sensitivities(3 * 2**30, 4, 4, workspace_bytes=2 * 2**30)
    assert not module._cache_sensitivities(2**30, 4, 4, workspace_bytes=4 * 2**30)
    completed = []
    initial = type("Pending", (), {"block_until_ready": lambda self: completed.append(True)})()
    monkeypatch.setattr(module, "device_free_memory_bytes", lambda: 4 * 2**30 if completed else 0)
    assert module._cache_sensitivities(2**30, 4, 4, workspace_bytes=0, initial=initial)
    assert completed


@pytest.mark.numerical
def test_streamed_counts_match_resident_subsets_and_release_sources(problem, monkeypatch):
    module = import_module("tomojax.recon.mlem")
    host = import_module("tomojax.backends.host_stream")
    grid, detector, geometry, matrix = problem
    data = (matrix @ np.linspace(0.2, 1.1, 12)).reshape(5, 4, 5).astype(np.float32)
    config = MLEMConfig(iterations=3, subsets=3, views_per_batch=1)
    expected = mlem(geometry, grid, detector, data, config=config)[0]
    original, reads = host._read, []

    def read(key, start, weights_key, indices, *, size):
        reads.append(np.asarray(indices).copy())
        assert size == 1 and indices.shape == (1,)
        return original(key, start, weights_key, indices, size=size)

    monkeypatch.setattr(host, "_read", read)
    monkeypatch.setattr(module, "should_stream", lambda _: True)
    registered = set(host._SOURCES)
    for scale in (1, 2):
        actual = mlem(geometry, grid, detector, data * scale, config=config)[0]
        np.testing.assert_allclose(actual, expected * scale, rtol=2e-6, atol=2e-6)
        assert set(host._SOURCES) == registered
    assert set(np.concatenate(reads)) == set(range(5))


@pytest.mark.numerical
def test_single_plane_sensitivities_match_full_axial_norms(monkeypatch):
    module = import_module("tomojax.recon.mlem")
    monkeypatch.setattr(module, "_SENSITIVITY_CACHE_BYTES", 0)
    monkeypatch.setattr(module, "device_free_memory_bytes", lambda: 2**30)
    rng = np.random.default_rng(9)
    grid = Grid(5, 4, 3, 0.8, 1.1, 1.3, vol_origin=(1.3, -2.7, 2.5 - 2 * 1.3 / 2))
    detector = Detector(8, 3, 0.8, 1.3, (0.2, 2.5))
    geometry = tj.ParallelGeometry(grid, detector, [11, 89, 141, 63, 122])
    truth = rng.uniform(0.1, 1, (5, 4, 3)).astype(np.float32)
    scan = tj.Scan(np.asarray(tj.project(geometry, truth)), geometry)
    support = rng.uniform(size=truth.shape) > 0.3
    for iterations in (0, 3):
        cfg = MLEMConfig(iterations=iterations, subsets=3, support=support)
        actual = tj.reconstruct(scan, "mlem", config=cfg).volume
        with monkeypatch.context() as patch:
            patch.setattr(module, "independent_planes", lambda *_: False)
            expected = tj.reconstruct(scan, "mlem", config=cfg).volume
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.numerical
def test_separately_allocated_norms_match_inline_cache(problem, monkeypatch):
    module = import_module("tomojax.recon.mlem")
    grid, detector, geometry, matrix = problem
    data = (matrix @ np.linspace(0.2, 1.1, 12)).reshape(5, 4, 5).astype(np.float32)
    cfg = MLEMConfig(iterations=3, subsets=3, views_per_batch=2)
    expected = mlem(geometry, grid, detector, data, config=cfg)[0]
    monkeypatch.setattr(module, "_SENSITIVITY_CACHE_BYTES", 0)
    monkeypatch.setattr(module, "device_free_memory_bytes", lambda: 2**30)
    for _ in range(2):
        actual = mlem(geometry, grid, detector, data, config=cfg)[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.numerical
@pytest.mark.parametrize("free", [1, 2**40])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_host_cone_updates_match_dense_transpose_without_consuming_inputs(
    problem, monkeypatch, free, dtype
):
    grid, detector, geometry, matrix = problem
    if not isinstance(geometry, ConeSegments):
        pytest.skip("coupled volume partition is for cone geometry")
    host = import_module("tomojax.recon._mlem_host")
    monkeypatch.setattr(host, "volume_depth", lambda *_, **__: 1)
    monkeypatch.setattr(host, "device_free_memory_bytes", lambda: free)
    monkeypatch.setattr(host, "_MEASURED_BATCH_BYTES", 1)
    support = np.ones((3, 2, 2), bool)
    support[0] = False
    initial = np.full((3, 2, 2), 0.7, np.float32)
    initial[0] = np.nan  # excluded values must not enter the likelihood
    data = (matrix @ np.linspace(0.2, 1.1, 12)).reshape(5, 4, 5).astype(dtype)
    for iterations in (0, 2):
        expected = np.where(support, initial, 0).ravel().astype(np.float64)
        order = np.argsort(geometry.angles, kind="stable")
        for _ in range(iterations):
            for group in range(3):
                a = matrix[(order[group::3, None] * 20 + np.arange(20)).ravel()]
                measured = data[order[group::3]].ravel()
                prediction = a @ expected
                ratio = np.divide(
                    measured, prediction, out=np.zeros_like(measured), where=prediction > 0
                )
                norm = a.sum(axis=0)
                expected *= np.divide(a.T @ ratio, norm, out=np.ones(12), where=norm > 0)
        expected[matrix.sum(axis=0) == 0] = 0
        volume, info = mlem(
            geometry,
            grid,
            detector,
            data,
            init_x=initial,
            config=MLEMConfig(iterations=iterations, subsets=3, support=support, views_per_batch=2),
        )
        assert isinstance(volume, np.ndarray) and info["host_volume"]
        assert volume.flags.f_contiguous
        np.testing.assert_allclose(volume.ravel(), expected, rtol=2e-5, atol=2e-6)
    assert np.isnan(initial[0]).all()
    np.testing.assert_array_equal(initial[1:], np.float32(0.7))


def test_host_prediction_buffer_is_zero_and_survives_donated_updates():
    import jax.numpy as jnp

    host = import_module("tomojax.recon._mlem_host")
    values = jnp.full((2, 3, 5), 0.25, jnp.float32)
    prediction = host._prediction_buffer((7, 3, 5), next(iter(values.devices())))
    np.testing.assert_array_equal(np.asarray(prediction), 0)
    prediction = host._add_predictions(prediction, values, jnp.int32(3))
    prediction = host._divide_predictions(prediction, values * 2, jnp.int32(3))
    np.testing.assert_array_equal(np.asarray(prediction)[:, 0, 0], [0, 0, 0, 2, 2, 0, 0])


def test_packed_device_visibility_unions_overlapping_odd_width_slabs():
    import jax.numpy as jnp

    host = import_module("tomojax.recon._mlem_host")
    expected = np.zeros((9, 3, 9), bool)
    seen = jnp.zeros((2, 3, 9), jnp.uint8)
    rng = np.random.default_rng(381)
    for start in (0, 3, 5):
        visible = rng.random((9, 3, 4)) > 0.5
        seen = host._merge_visibility(seen, jnp.asarray(visible), jnp.int32(start))
        expected[:, :, start : start + 4] |= visible
    np.testing.assert_array_equal(np.asarray(seen), np.packbits(expected, axis=0))


@pytest.mark.numerical
def test_large_cone_warm_start_is_planned_before_device_initialization(monkeypatch):
    host = import_module("tomojax.recon._mlem_host")
    grid = Grid(7, 5, 9, 0.8, 1.1, 0.6, vol_center=(0.1, -0.2, 0.3))
    detector = Detector(11, 9, 0.9, 0.8)
    angles = np.linspace(0, 360, 12, endpoint=False)
    geometry = ConeSegments(
        tuple(
            ConeGeometry(grid, detector, angles, ConeBeam(20, 35, source_height=h))
            for h in (-0.7, 1.2)
        )
    )
    data = np.random.default_rng(925).uniform(0.1, 1, (24, 9, 11)).astype(np.float32)
    scan = tj.Scan(data, geometry)
    cfg = MLEMConfig(iterations=1, subsets=3, views_per_batch=3)
    monkeypatch.setattr(host, "device_free_memory_bytes", lambda: 2**40)
    assert host.volume_depth(grid, detector, 3) is None
    monkeypatch.setattr(host, "device_free_memory_bytes", lambda: 1000)
    assert host.volume_depth(grid, detector, 3) == 1
    initializer, owned, starts = host.initial_volume, [], []

    def initialize(*args, **kwargs):
        owned.append(initializer(*args, **kwargs))
        starts.append(owned[-1].copy())
        return owned[-1]

    monkeypatch.setattr(host, "initial_volume", initialize)
    actual = tj.reconstruct(scan, "mlem", config=cfg, warm_start=True)
    assert actual.volume is owned[0]
    assert isinstance(actual.volume, np.ndarray)
    # Cropped FDK half textures have their own peak scale. Compare the exact
    # EM updates from identical input, independently of that initializer rounding.
    monkeypatch.setattr(host, "device_free_memory_bytes", lambda: 2**40)
    expected, _ = mlem(geometry, grid, detector, data, init_x=starts[0], config=cfg)
    np.testing.assert_allclose(actual.volume, expected, rtol=4e-5, atol=2e-6)


def test_host_slab_planner_bounds_transfer_storage_independently_of_device_capacity(monkeypatch):
    host = import_module("tomojax.recon._mlem_host")
    monkeypatch.setattr(host, "device_free_memory_bytes", lambda: 8 * 2**30)
    grid, detector = Grid(1001, 1001, 1001, 1, 1, 1), Detector(768, 972, 1, 1)
    depth = host.volume_depth(grid, detector, 32, subset_views=900)
    assert depth is not None and depth > 1
    assert 4 * grid.nx * grid.ny * depth <= 128 * 2**20
