"""Physical geometry, bounded execution and host-storage checks for slab FBP."""

from __future__ import annotations

from dataclasses import replace
import importlib

import jax
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.recon import FBPConfig, FBPHostConfig, fbp, fbp_host


def scan(kind="fractional"):
    grid = Grid(9, 7, 11, 0.8, 1.2, 0.7, vol_center=(1.2, -0.4, 0.31))
    detector = Detector(9, 10, 0.9, 1.0, (0.23, -0.19))
    if kind == "integer":
        grid = replace(grid, vz=1.0, vol_center=(1.2, -0.4, 0.0))
        detector = replace(detector, nv=11, center=(0.23, 0.0))
    elif kind == "outside":
        grid = replace(grid, vz=3.0, vol_origin=(-1.7, -2.1, -25.0))
    elif kind == "one_row":
        detector = replace(detector, nv=1)
    angles = np.array([0.0, 29.0, 73.0, 118.0, 163.0], dtype=np.float32)
    geometry = ParallelGeometry(grid, detector, angles)
    data = np.random.default_rng(132).normal(size=(5, detector.nv, detector.nu)).astype(np.float32)
    return grid, detector, geometry, data


@pytest.mark.parametrize("kind", ["integer", "fractional", "outside", "one_row"])
@pytest.mark.parametrize("depth", [1, 4, 20])
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_host_slabs_match_full_physical_reconstruction(kind, depth, backend):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, geometry, data = scan(kind)
    reference = np.asarray(
        fbp(geometry, grid, detector, data, config=FBPConfig(backprojector="jax"))
    )
    actual = fbp_host(
        geometry,
        grid,
        detector,
        data,
        config=FBPHostConfig(slices_per_batch=depth, views_per_batch=3, backprojector=backend),
    )
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, reference, atol=4e-6, rtol=5e-5)


@pytest.mark.parametrize(("depth", "nz", "count", "row_bound"), [(3, 11, 12, 4), (None, 33, 9, 9)])
def test_host_slabs_reuse_compilation_and_bound_device_shapes(
    monkeypatch, depth, nz, count, row_bound
):
    module = importlib.import_module("tomojax.recon.fbp")
    grid, detector, geometry, data = scan()
    grid = replace(grid, nz=nz)
    geometry = ParallelGeometry(grid, detector, geometry.angles)
    kernel = module._fbp_accumulate_batch
    calls = []
    before = kernel._cache_size()

    def record(accum, poses, projections, *args, **kwargs):
        calls.append((projections.shape, kwargs["grid"], kwargs["detector"]))
        return kernel(accum, poses, projections, *args, **kwargs)

    monkeypatch.setattr(module, "_fbp_accumulate_batch", record)
    cfg = FBPHostConfig(slices_per_batch=depth, views_per_batch=2, backprojector="jax")
    actual = fbp_host(geometry, grid, detector, data, config=cfg)
    assert len(calls) == count
    assert all(shape[0] == 2 and shape[1] <= row_bound for shape, _, _ in calls)
    assert all(slab.nz == (3 if depth is not None else 11) for _, slab, _ in calls)
    assert len({(slab, det) for _, slab, det in calls}) == 1
    after = kernel._cache_size()
    assert after - before <= 1
    repeated = fbp_host(geometry, grid, detector, data * 1.2, config=cfg)
    assert kernel._cache_size() == after
    np.testing.assert_allclose(repeated, actual * 1.2, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize(
    ("nz", "automatic"),
    [(1, 1), (5, 5), (16, 16), (17, 9), (22, 11), (31, 16), (33, 11), (46, 16), (65, 13)],
)
@pytest.mark.parametrize("requested", [None, 1, 4, 16, 40])
def test_automatic_axial_slabs_balance_tail_work_without_increasing_storage(
    monkeypatch, nz, automatic, requested
):
    module = importlib.import_module("tomojax.recon.fbp_host")
    grid, detector = Grid(3, 2, nz, 1, 1, 1), Detector(5, nz + 2, 1, 1)
    geometry = ParallelGeometry(grid, detector, [7, 67, 127])
    data = np.zeros((3, detector.nv, detector.nu), np.float32)
    calls = []

    def record(poses, projections, *args, **kwargs):
        del poses, args
        local_grid = kwargs["grid"]
        calls.append((local_grid, projections.shape))
        return np.full((local_grid.nx, local_grid.ny, local_grid.nz), len(calls), np.float32)

    monkeypatch.setattr(module, "_fbp_from_host", record)
    actual = fbp_host(
        geometry, grid, detector, data, config=FBPHostConfig(slices_per_batch=requested)
    )
    depth = automatic if requested is None else min(requested, nz)
    assert len(calls) == (nz + depth - 1) // depth
    assert all(slab.nz == depth and shape[1] == depth for slab, shape in calls)
    assert len({slab for slab, _ in calls}) == 1
    if requested is None:
        assert depth <= 16 and len(calls) == (nz + 15) // 16
    expected = np.broadcast_to(1 + np.arange(nz) // depth, actual.shape)
    np.testing.assert_array_equal(actual, expected)


def test_host_slabs_read_and_convert_only_bounded_view_batches():
    grid, detector, geometry, data = scan()
    reads = []

    class RecordedArray(np.ndarray):
        def __getitem__(self, key):
            views = key[0] if isinstance(key, tuple) else key
            reads.append(len(range(*views.indices(self.shape[0]))))
            return super().__getitem__(key)

    source = data.astype(np.float64).view(RecordedArray)
    actual = fbp_host(
        geometry,
        grid,
        detector,
        source,
        config=FBPHostConfig(slices_per_batch=3, views_per_batch=2),
    )
    assert reads and max(reads) <= 2
    expected = np.asarray(fbp(geometry, grid, detector, data))
    np.testing.assert_allclose(actual, expected, atol=4e-6, rtol=5e-5)


@pytest.mark.parametrize(
    ("first", "step", "nz"),
    [(-9.0, 1.0, 17), (-9.25, 2.5, 9), (-7.0, 1.0, 7), (9.0, 1.0, 7)],
)
def test_empty_axial_slabs_do_not_filter_or_read_views(monkeypatch, first, step, nz):
    module = importlib.import_module("tomojax.recon.fbp_host")
    detector = Detector(9, 5, 0.9, 0.8, (0.23, -0.19))
    grid = Grid(
        9,
        7,
        nz,
        0.8,
        1.2,
        step * detector.dv,
        vol_origin=(-2.1, -3.2, detector.center[1] + (first - 2) * detector.dv),
    )
    geometry = ParallelGeometry(grid, detector, [0, 29, 73, 118, 163])
    data = np.random.default_rng(135).normal(size=(5, 5, 9)).astype(np.float32)
    run = module._fbp_from_host
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs["grid"].nz)
        return run(*args, **kwargs)

    monkeypatch.setattr(module, "_fbp_from_host", record)
    actual = fbp_host(
        geometry, grid, detector, data, config=FBPHostConfig(slices_per_batch=4, views_per_batch=3)
    )
    reference = np.asarray(
        fbp(geometry, grid, detector, data, config=FBPConfig(backprojector="jax"))
    )
    expected_calls = sum(
        first + start * step < detector.nv and first + (start + min(4, nz - start) - 1) * step > -1
        for start in range(0, nz, 4)
    )
    assert len(calls) == expected_calls < (nz + 3) // 4
    np.testing.assert_allclose(actual, reference, atol=4e-6, rtol=5e-5)


@pytest.mark.parametrize("first", [-1.0, -0.75, 0.75, 1.0])
def test_empty_slab_support_preserves_fractional_single_row_edges(monkeypatch, first):
    module = importlib.import_module("tomojax.recon.fbp_host")
    detector = Detector(9, 1, 1, 1)
    grid = Grid(9, 7, 1, 1, 1, 1, vol_origin=(-4, -3, first))
    geometry = ParallelGeometry(grid, detector, [0, 29, 73, 118, 163])
    data = np.random.default_rng(136).normal(size=(5, 1, 9)).astype(np.float32)
    run = module._fbp_from_host
    calls = []

    def record(*args, **kwargs):
        calls.append(1)
        return run(*args, **kwargs)

    monkeypatch.setattr(module, "_fbp_from_host", record)
    actual = fbp_host(geometry, grid, detector, data)
    reference = np.asarray(
        fbp(geometry, grid, detector, data, config=FBPConfig(backprojector="jax"))
    )
    assert len(calls) == int(-1 < first < 1)
    np.testing.assert_allclose(actual, reference, atol=4e-6, rtol=5e-5)


@pytest.mark.parametrize("stride", [2, 2.5, 5, 5.25])
@pytest.mark.parametrize("depth", [1, 4])
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_axially_decimated_slabs_pack_only_requested_rows(monkeypatch, stride, depth, backend):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    module = importlib.import_module("tomojax.recon.fbp_host")
    detector = Detector(9, 35, 0.9, 0.8, (0.23, -0.19))
    grid = Grid(
        9,
        7,
        7,
        0.8,
        1.2,
        stride * detector.dv,
        vol_origin=(-2.1, -3.2, detector.center[1] + (2 - 17) * detector.dv),
    )
    geometry = ParallelGeometry(grid, detector, [0, 29, 73, 118, 163])
    data = np.random.default_rng(132).normal(size=(5, 35, 9)).astype(np.float32)
    run = module._fbp_from_host
    shapes = []

    def record(poses, projections, *args, **kwargs):
        shapes.append(projections.shape)
        assert projections.stride == stride
        assert kwargs["detector"].dv == grid.vz
        return run(poses, projections, *args, **kwargs)

    monkeypatch.setattr(module, "_fbp_from_host", record)
    actual = fbp_host(
        geometry,
        grid,
        detector,
        data,
        config=FBPHostConfig(slices_per_batch=depth, views_per_batch=3, backprojector=backend),
    )
    expected = np.asarray(
        fbp(geometry, grid, detector, data, config=FBPConfig(backprojector="jax"))
    )
    assert shapes and all(shape[1] == depth for shape in shapes)
    np.testing.assert_allclose(actual, expected, atol=4e-6, rtol=5e-5)


@pytest.mark.parametrize("stride", [0, 1, 3])
@pytest.mark.parametrize("start", [-3, 2, 10])
def test_strided_projection_window_zero_extends_without_reading_skipped_rows(stride, start):
    module = importlib.import_module("tomojax.recon._host_arrays")
    data = np.random.default_rng(751).normal(size=(5, 11, 7)).astype(np.float64)
    reads = []

    class RecordedArray(np.ndarray):
        def __getitem__(self, key):
            assert isinstance(key, tuple), "Select source rows before reading or converting views"
            reads.append(key)
            return super().__getitem__(key)

    source = data[:, :, ::2].view(RecordedArray)
    window = module.ProjectionRows(source, start, 4, stride=stride)
    actual = window[slice(1, 5, 2)]
    expected = np.zeros((2, 4, 4), np.float32)
    for i in range(4):
        row = start + i * stride
        if 0 <= row < data.shape[1]:
            expected[:, i] = data[1:5:2, row, ::2]
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
    assert len(reads) <= 1


@pytest.mark.parametrize("stride", [0, 0.75, 2.5])
@pytest.mark.parametrize("start", [-3.25, 2.0, 10.4])
def test_fractional_projection_rows_blend_only_bounded_support_rows(stride, start):
    module = importlib.import_module("tomojax.recon._host_arrays")
    data = np.random.default_rng(752).normal(size=(5, 11, 7)).astype(np.float64)
    reads = []

    class RecordedArray(np.ndarray):
        def __getitem__(self, key):
            assert isinstance(key, tuple)
            reads.append(key)
            return super().__getitem__(key)

    source = data[:, :, ::2].view(RecordedArray)
    actual = module.ProjectionRows(source, start, 4, stride=stride)[slice(1, 5, 2)]
    expected = np.zeros((2, 4, 4), np.float32)
    for i in range(4):
        position = start + i * stride
        low = int(np.floor(position))
        weight = np.float32(position - low)
        for row, fraction in ((low, 1 - weight), (low + 1, weight)):
            if 0 <= row < data.shape[1]:
                expected[:, i] += data[1:5:2, row, ::2].astype(np.float32) * fraction
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
    assert len(reads) <= 2
    for views, rows, _ in reads:
        assert len(range(*views.indices(data.shape[0]))) == 2
        assert len(rows) <= 4
        support = {int(np.floor(start + i * stride)) + j for i in range(4) for j in (0, 1)}
        assert set(rows).issubset(support)


@pytest.mark.parametrize(("stride", "first"), [(1, 2), (3, 2), (7, 2), (2.5, 2.27), (4.25, 2.27)])
def test_axial_decimation_recovers_independent_gaussian_attenuation(stride, first):
    n, nz, nv, views = 40, 8, 55, 80
    spacing, sigma, cv = 0.8, 4.0, -0.2
    oz = cv + (first - (nv - 1) / 2) * spacing
    grid = Grid(n, n, nz, spacing, spacing, stride * spacing, vol_origin=(-15.6, -15.6, oz))
    detector = Detector(n, nv, spacing, spacing, (0.23, cv))
    angles = np.linspace(0, 180, views, endpoint=False)
    geometry = ParallelGeometry(grid, detector, angles)
    x = np.arange(n) * spacing - 15.6
    z = np.arange(nz) * stride * spacing + oz
    u = (np.arange(n) - (n - 1) / 2) * spacing + detector.center[0]
    v = (np.arange(nv) - (nv - 1) / 2) * spacing + cv
    theta = np.deg2rad(angles)
    mean = 3 * np.cos(theta) + 2 * np.sin(theta)
    profile = np.sqrt(2 * np.pi) * sigma * np.exp(-0.5 * ((u - mean[:, None]) / sigma) ** 2)
    data = (profile[:, None] * np.exp(-0.5 * (v / 15) ** 2)[None, :, None]).astype(np.float32)
    truth = np.exp(-0.5 * (((x[:, None] - 3) / sigma) ** 2 + ((x[None, :] + 2) / sigma) ** 2))
    truth = truth[:, :, None] * np.exp(-0.5 * (z / 15) ** 2)
    actual = fbp_host(
        geometry,
        grid,
        detector,
        data,
        config=FBPHostConfig(slices_per_batch=3, views_per_batch=17),
    )
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.01


def test_host_streamer_waits_before_queuing_more_input_batches(monkeypatch):
    module = importlib.import_module("tomojax.recon.fbp")
    grid, detector, _, data = scan()
    events = []
    pending = []

    class RecordedArray(np.ndarray):
        def __getitem__(self, key):
            events.append(("read", key.start))
            return super().__getitem__(key)

    class PendingVolume:
        def __init__(self, index):
            self.index = index
            self.ready = False

        def block_until_ready(self):
            events.append(("ready", self.index))
            self.ready = True

    def step(accum, *args, **kwargs):
        if pending:
            assert accum.ready
        current = PendingVolume(len(pending))
        events.append(("step", current.index))
        pending.append(current)
        return current

    monkeypatch.setattr(module, "_fbp_accumulate_batch", step)
    module._fbp_from_host(
        np.repeat(np.eye(4)[None], 5, axis=0),
        data.view(RecordedArray),
        np.ones(5),
        np.zeros((5, 6)),
        np.ones(33),
        0,
        grid=grid,
        detector=detector,
        backend="jax",
        batch_size=2,
        separable=True,
    )
    assert events == [
        ("read", 0),
        ("step", 0),
        ("read", 2),
        ("ready", 0),
        ("step", 1),
        ("read", 4),
        ("ready", 1),
        ("step", 2),
    ]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 1e100])
def test_host_slabs_check_a_nonfinite_tail_batch_before_writing(invalid):
    grid, detector, geometry, data = scan()
    data = data.astype(np.float64)
    data[-1] = invalid
    out = np.full((grid.nx, grid.ny, grid.nz), 91.0, dtype=np.float32)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite in FP32"):
        fbp_host(
            geometry,
            grid,
            detector,
            data,
            config=FBPHostConfig(slices_per_batch=3, views_per_batch=2),
            out=out,
        )
    np.testing.assert_array_equal(out, 91.0)


def test_host_slabs_write_memmap_and_preserve_custom_scale(tmp_path):
    grid, detector, geometry, data = scan()
    source = np.memmap(tmp_path / "data.bin", mode="w+", dtype=np.float32, shape=data.shape)
    source[:] = data
    target = np.memmap(
        tmp_path / "volume.bin", mode="w+", dtype=np.float32, shape=(grid.nx, grid.ny, grid.nz)
    )
    cfg = FBPHostConfig(slices_per_batch=4, views_per_batch=2, scale=0.27, filter="hann")
    result = fbp_host(geometry, grid, detector, source, config=cfg, out=target)
    assert result is target
    target.flush()
    reloaded = np.memmap(tmp_path / "volume.bin", mode="r", dtype=np.float32, shape=target.shape)
    expected = np.asarray(
        fbp(geometry, grid, detector, data, config=FBPConfig(scale=0.27, filter="hann"))
    )
    np.testing.assert_allclose(reloaded, expected, atol=3e-6, rtol=3e-5)
    np.testing.assert_array_equal(source, data)


@pytest.mark.parametrize("mapping", [False, True, "source", "output", "both"])
def test_host_slabs_reject_overlapping_output_before_writing(tmp_path, mapping):
    grid, detector = Grid(5, 4, 3, 1, 1, 1), Detector(3, 4, 1, 1)
    geometry = ParallelGeometry(grid, detector, [0, 31, 73, 114, 155])
    if mapping:
        path = tmp_path / "same.bin"
        data = np.memmap(path, mode="w+", dtype=np.float32, shape=(5, 4, 3))
        out = np.memmap(path, mode="r+", dtype=np.float32, shape=data.shape)
        if mapping in {"source", "both"}:
            data = np.frombuffer(memoryview(data), dtype=data.dtype).reshape(data.shape)
        if mapping in {"output", "both"}:
            out = np.frombuffer(memoryview(out), dtype=out.dtype).reshape(out.shape)
    else:
        data = np.ones((5, 4, 3), dtype=np.float32)
        out = data.view()
    data[:] = 1
    with pytest.raises(ValueError, match="overlap|separate files"):
        fbp_host(geometry, grid, detector, data, out=out)
    np.testing.assert_array_equal(data, 1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("slices_per_batch", 0),
        ("views_per_batch", -1),
        ("scale", np.nan),
        ("backprojector", "invalid"),
    ],
)
def test_host_slabs_reject_invalid_configuration(field, value):
    grid, detector, geometry, data = scan()
    with pytest.raises(ValueError, match="fbp_host"):
        fbp_host(geometry, grid, detector, data, config=replace(FBPHostConfig(), **{field: value}))


@pytest.mark.parametrize("depth", [None, 1, 4, 20])
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_host_slabs_reconstruct_laminography_like_fbp(depth, backend):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, geometry, data = scan()
    angles = np.linspace(0.0, 360.0, 5, endpoint=False) + 7.0
    tilted = LaminographyGeometry(grid, detector, angles, tilt_deg=30)
    reference = np.asarray(fbp(tilted, grid, detector, data, config=FBPConfig(backprojector="jax")))
    actual = fbp_host(
        tilted,
        grid,
        detector,
        data,
        config=FBPHostConfig(slices_per_batch=depth, views_per_batch=2, backprojector=backend),
    )
    np.testing.assert_allclose(actual, reference, atol=4e-6, rtol=5e-5)


@pytest.mark.parametrize("tilted", [False, True])
def test_host_slabs_reject_nonfinite_samples(tilted):
    grid, detector, geometry, data = scan()
    if tilted:
        geometry = LaminographyGeometry(grid, detector, geometry.angles, tilt_deg=30)
    data[0, 4, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fbp_host(geometry, grid, detector, data)


@pytest.mark.parametrize("invalid", ["dtype", "shape", "readonly", "device_input"])
def test_host_slabs_reject_incompatible_storage(invalid):
    grid, detector, geometry, data = scan()
    out = np.empty((grid.nx, grid.ny, grid.nz), dtype=np.float32)
    if invalid == "dtype":
        out = out.astype(np.float64)
    elif invalid == "shape":
        out = out[:, :, :-1]
    elif invalid == "readonly":
        out.flags.writeable = False
    else:
        data = jax.device_put(data)
    with pytest.raises((ValueError, TypeError), match="fbp_host"):
        fbp_host(geometry, grid, detector, data, out=out)


@pytest.mark.parametrize("length_scale", [0.5, 2.0])
def test_host_slabs_recover_independent_gaussian_attenuation(length_scale):
    n, nz, views = 40, 7, 80
    spacing = 0.8 * length_scale
    grid = Grid(n, n, nz, spacing, spacing, spacing)
    detector = Detector(n, nz, spacing, spacing)
    angles = np.linspace(0, 180, views, endpoint=False)
    geometry = ParallelGeometry(grid, detector, angles)
    x = (np.arange(n) - (n - 1) / 2) * spacing
    z = (np.arange(nz) - (nz - 1) / 2) * spacing
    sigma = 4 * length_scale
    axial = np.exp(-0.5 * (z / (1.1 * length_scale)) ** 2)
    profile = np.sqrt(2 * np.pi) * sigma * np.exp(-0.5 * (x / sigma) ** 2)
    data = np.broadcast_to(axial[:, None] * profile, (views, nz, n)).astype(np.float32)
    truth = np.exp(-0.5 * (x[:, None, None] ** 2 + x[None, :, None] ** 2) / sigma**2) * axial
    actual = fbp_host(
        geometry,
        grid,
        detector,
        data,
        config=FBPHostConfig(slices_per_batch=3, views_per_batch=17),
    )
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.01


@pytest.mark.parametrize("input_on_host", [True, False])
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("integer", "integer"),
        ("fractional", "fixed"),
        ("translated", "general"),
        ("lamino", "general"),
    ],
)
def test_fbp_only_hoists_rows_for_the_exact_builtin_parallel_geometry(
    monkeypatch, input_on_host, backend, kind, expected
):
    """A custom parallel subclass may move z differently in each view."""
    import jax.numpy as jnp

    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    module = importlib.import_module("tomojax.recon.fbp")
    grid, detector, geometry, data = scan("integer" if kind == "integer" else "fractional")

    class Translated(ParallelGeometry):
        def pose_for_view(self, i):
            pose = np.asarray(super().pose_for_view(i))
            pose[2, 3] = 0.23 * i
            return tuple(map(tuple, pose))

    if kind == "translated":
        geometry = Translated(grid, detector, geometry.angles)
    elif kind == "lamino":
        geometry = LaminographyGeometry(grid, detector, geometry.angles, tilt_deg=17)
    reference = np.asarray(
        fbp(geometry, grid, detector, data, config=FBPConfig(backprojector="jax"))
    )
    plan = module._plan
    modes = []

    def record(*args, **kwargs):
        result = plan(*args, **kwargs)
        modes.append(result.z_mode)
        return result

    monkeypatch.setattr(module, "_plan", record)
    projections = data if input_on_host else jnp.asarray(data)
    volume = fbp(geometry, grid, detector, projections, config=FBPConfig(backprojector=backend))
    assert modes == [expected]
    assert np.isfinite(np.asarray(volume)).all()
    np.testing.assert_allclose(volume, reference, atol=4e-6, rtol=5e-5)
