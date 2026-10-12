"""Post-filter axial packing, including physical coordinates and zero extension."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, ParallelGeometry, grid_volume_origin
from tomojax.recon import FBPConfig, FBPHostConfig, fbp, fbp_host


@pytest.mark.parametrize("nv", [1, 5, 6])
@pytest.mark.parametrize("axial_scale", [-1.0, 1.0])
def test_filtered_row_packing_preserves_physical_samples_and_zero_extension(nv, axial_scale):
    implementation = importlib.import_module("tomojax.recon.fbp")
    pallas = importlib.import_module("tomojax.recon._fbp_pallas")
    grid = Grid(7, 6, 8, 0.8, 1.2, 0.75, vol_origin=(-2.3, -3.4, -2.125))
    detector = Detector(9, nv, 0.9, 0.8, (0.17, 0.21))
    geometry = ParallelGeometry(grid, detector, [13, 31, 78, 117, 161])
    poses = np.array([geometry.pose_for_view(i) for i in range(5)], np.float32)
    poses[:, 0, 3] = 0.31
    poses[:, 2, 2], poses[:, 2, 3] = axial_scale, 0.31
    filtered = np.random.default_rng(142).normal(size=(5, nv, 9)).astype(np.float32)

    local_poses, packed, local_grid, local_detector = implementation._pack_fixed_rows(
        jnp.asarray(poses), jnp.asarray(filtered), grid=grid, detector=detector
    )
    # Independent physical-coordinate interpolation includes the two outside
    # zero samples; clamping to the first/last measured row would be incorrect.
    z = grid_volume_origin(grid)[2] + np.arange(grid.nz) * grid.vz
    pixels = (axial_scale * z + 0.31 - detector.center[1]) / detector.dv + (nv - 1) / 2
    expected = np.empty((5, grid.nz, 9))
    for view in range(5):
        for column in range(9):
            values = np.r_[0.0, filtered[view, :, column], 0.0]
            expected[view, :, column] = np.interp(pixels, np.arange(-1, nv + 1), values)
    np.testing.assert_allclose(packed, expected, rtol=3e-5, atol=2e-6)
    np.testing.assert_array_equal(np.asarray(local_poses)[:, :2], poses[:, :2])
    assert local_detector.nu == detector.nu and local_detector.center[0] == detector.center[0]
    assert local_detector.nv == grid.nz and local_detector.dv == grid.vz
    assert grid_volume_origin(local_grid) == (*grid_volume_origin(grid)[:2], 0.0)

    reference = implementation._backproject_voxels_jax(
        jnp.asarray(poses), jnp.asarray(filtered), grid=grid, detector=detector
    )
    actual = pallas.backproject_filtered_pallas(
        local_poses,
        packed,
        grid=local_grid,
        detector=local_detector,
        z_mode="integer",
        interpret=True,
    )
    np.testing.assert_allclose(actual, reference, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize(
    ("shape", "detector_shape", "views", "mode", "backend", "packed"),
    [
        ((512, 512, 16), (128, 16), 3, "fixed", "pallas", False),
        ((1024, 1024, 16), (128, 16), 3, "fixed", "pallas", True),
        ((1024, 1024, 33), (128, 16), 3, "fixed", "pallas", False),
        ((7, 6, 3), (512, 512), 129, "fixed", "pallas", True),
        ((1024, 1024, 16), (128, 16), 3, "integer", "pallas", False),
        ((1024, 1024, 16), (128, 16), 3, "general", "pallas", False),
        ((1024, 1024, 16), (128, 16), 3, "fixed", "jax", False),
    ],
)
def test_packing_is_bounded_and_keeps_compact_or_general_sampling(
    monkeypatch, shape, detector_shape, views, mode, backend, packed
):
    implementation = importlib.import_module("tomojax.recon.fbp")
    pallas = importlib.import_module("tomojax.recon._fbp_pallas")
    grid = Grid(*shape, 1.0, 1.0, 0.75)
    detector = Detector(*detector_shape, 1.0, 1.0)
    calls = []

    def record(poses, filtered, *, grid, detector, z_mode, accumulate):
        del poses, grid
        calls.append((filtered.shape, detector, z_mode))
        return accumulate

    monkeypatch.setattr(pallas, "backproject_filtered_pallas", record)
    # Trace the policy without allocating large test volumes or projection stacks.
    jax.eval_shape(
        lambda accum, poses, filtered: implementation._backproject_into(
            accum, poses, filtered, grid=grid, detector=detector, backend=backend, z_mode=mode
        ),
        jax.ShapeDtypeStruct(shape, jnp.float32),
        jax.ShapeDtypeStruct((views, 4, 4), jnp.float32),
        jax.ShapeDtypeStruct((views, detector.nv, detector.nu), jnp.float32),
    )
    if backend == "jax":
        assert not calls
    else:
        assert len(calls) == 1
        rows, local_detector, local_mode = calls[0]
        assert rows == (views, grid.nz if packed else detector.nv, detector.nu)
        assert local_mode == ("integer" if packed else mode)
        assert local_detector.nv == (grid.nz if packed else detector.nv)


@pytest.mark.gpu
@pytest.mark.parametrize("storage", ["device", "host"])
@pytest.mark.parametrize("axial_sampling", ["upsampled", "sparse"])
def test_large_fractional_fbp_recovers_independent_three_dimensional_gaussian(
    monkeypatch, storage, axial_sampling
):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    implementation = importlib.import_module("tomojax.recon.fbp")
    pack = implementation._pack_fixed_rows
    calls = []

    def record(*args, **kwargs):
        calls.append(args[1].shape)
        return pack(*args, **kwargs)

    monkeypatch.setattr(implementation, "_pack_fixed_rows", record)
    sparse = axial_sampling == "sparse"
    nz, pitch, origin_z = (9, 2.5, -10.23) if sparse else (33, 0.5, -8.23)
    grid = Grid(768, 768, nz, 1.0, 1.0, pitch, vol_origin=(-383.5, -383.5, origin_z))
    detector = Detector(1104, 32, 1.0, 1.0, (0.23, 0.21))
    angles = np.linspace(7, 187, 181, endpoint=False)
    geometry = ParallelGeometry(grid, detector, angles)
    theta = np.deg2rad(angles)
    u = detector.center[0] + np.arange(detector.nu) - (detector.nu - 1) / 2
    v = detector.center[1] + np.arange(detector.nv) - (detector.nv - 1) / 2
    variance = (8 * np.cos(theta)) ** 2 + (10 * np.sin(theta)) ** 2
    mean = 100 * np.cos(theta) + 70 * np.sin(theta)
    profile = np.sqrt(2 * np.pi) * 8 * 10 / np.sqrt(variance[:, None])
    profile = profile * np.exp(-0.5 * (u[None, :] - mean[:, None]) ** 2 / variance[:, None])
    data = (profile[:, None, :] * np.exp(-0.5 * ((v - 1.1) / 6) ** 2)[None, :, None]).astype(
        np.float32
    )
    if storage == "device" or sparse:
        actual = np.asarray(
            fbp(
                geometry,
                grid,
                detector,
                jnp.asarray(data) if storage == "device" else data,
                config=FBPConfig(backprojector="pallas"),
            )
        )
    else:
        actual = fbp_host(
            geometry,
            grid,
            detector,
            data,
            config=FBPHostConfig(slices_per_batch=33, views_per_batch=13, backprojector="pallas"),
        )
    x = np.arange(grid.nx) - 383.5
    z = origin_z + np.arange(grid.nz) * grid.vz
    truth = np.exp(-0.5 * (((x[:, None] - 100) / 8) ** 2 + ((x[None, :] + 70) / 10) ** 2))
    truth = truth[:, :, None] * np.exp(-0.5 * ((z - 1.1) / 6) ** 2)[None, None, :]
    assert calls, "the large FFT/output working set must exercise row packing"
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.008


@pytest.mark.parametrize("nv", [1, 5, 6])
@pytest.mark.parametrize("phase", [-5e-6, 5e-6])
def test_integer_row_packing_preserves_sampler_rounding(nv, phase):
    implementation = importlib.import_module("tomojax.recon.fbp")
    pallas = importlib.import_module("tomojax.recon._fbp_pallas")
    detector = Detector(9, nv, 0.9, 0.8, (0.17, 0.21))
    origin_z = detector.center[1] - (nv - 1) * detector.dv / 2 + (-1 + phase) * detector.dv
    grid = Grid(7, 6, 8, 0.8, 1.2, 1.6000001, vol_origin=(-2.3, -3.4, origin_z))
    geometry = ParallelGeometry(grid, detector, [13, 31, 78, 117, 161])
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(5)], jnp.float32)
    rows = np.random.default_rng(143).normal(size=(5, nv, 9)).astype(np.float32)
    local_poses, packed, local_grid, local_detector = implementation._pack_fixed_rows(
        poses, jnp.asarray(rows), grid=grid, detector=detector, integer=True
    )
    physical = grid_volume_origin(grid)[2] + np.arange(grid.nz) * grid.vz - detector.center[1]
    index = np.floor(physical / detector.dv + (nv - 1) / 2 + 0.5).astype(int)
    expected = np.zeros((5, grid.nz, 9), np.float32)
    valid = (index >= 0) & (index < nv)
    expected[:, valid] = rows[:, index[valid]]
    np.testing.assert_array_equal(packed, expected)
    reference = pallas.backproject_filtered_pallas(
        poses, jnp.asarray(rows), grid=grid, detector=detector, z_mode="integer", interpret=True
    )
    actual = pallas.backproject_filtered_pallas(
        local_poses,
        packed,
        grid=local_grid,
        detector=local_detector,
        z_mode="integer",
        interpret=True,
    )
    np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("route", ["accumulate", "resident"])
@pytest.mark.parametrize(
    ("nv", "views", "mode", "backend", "separable", "packed"),
    [
        (64, 32, "fixed", "pallas", True, True),
        (64, 32, "integer", "pallas", True, True),
        (64, 32, "general", "pallas", True, False),
        (64, 32, "fixed", "pallas", False, False),
        (64, 32, "fixed", "jax", True, False),
        (64, 8, "fixed", "pallas", True, False),  # exact 64 MiB boundary
        (64, 9, "fixed", "pallas", True, True),
        (16, 64, "fixed", "pallas", True, False),  # no fewer FFT rows
    ],
)
def test_pre_filter_packing_keeps_compact_general_and_reference_paths(
    monkeypatch, route, nv, views, mode, backend, separable, packed
):
    implementation = importlib.import_module("tomojax.recon.fbp")
    grid = Grid(17, 15, 16, 1.0, 1.0, 2.5)
    detector = Detector(512, nv, 1.0, 1.0)
    calls = []

    def filter_rows(rows, *args, detector, separable):
        del args, separable
        calls.append(("filter", rows.shape, detector.nv))
        return rows

    def backproject(accum, poses, filtered, *, grid, detector, backend, z_mode):
        del poses, grid, backend
        calls.append((z_mode, filtered.shape, detector.nv))
        return accum

    monkeypatch.setattr(implementation, "_filter_views", filter_rows)
    monkeypatch.setattr(implementation, "_backproject_into", backproject)
    sizes = (
        jax.ShapeDtypeStruct((views, 4, 4), jnp.float32),
        jax.ShapeDtypeStruct((views, nv, 512), jnp.float32),
        jax.ShapeDtypeStruct((views,), jnp.float32),
        jax.ShapeDtypeStruct((views, 6), jnp.float32),
        jax.ShapeDtypeStruct((4097,), jnp.float32),
        jax.ShapeDtypeStruct((), jnp.float32),
    )
    options = dict(grid=grid, detector=detector, backend=backend, separable=separable, z_mode=mode)
    if route == "resident":
        jax.eval_shape(
            lambda *args: implementation._run_fbp_streamed(*args, batch_size=views, **options),
            *sizes,
        )
    else:
        jax.eval_shape(
            lambda accum, *args: implementation._fbp_accumulate_batch(accum, *args, **options),
            jax.ShapeDtypeStruct((grid.nx, grid.ny, grid.nz), jnp.float32),
            *sizes,
        )
    expected_rows = grid.nz if packed else nv
    assert calls == [
        ("filter", (views, expected_rows, 512), expected_rows),
        ("integer" if packed else mode, (views, expected_rows, 512), expected_rows),
    ]
