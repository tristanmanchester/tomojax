"""FDK slab equivalence and host-storage failure contracts."""

from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import pytest

from tomojax.geometry import ConeBeam, ConeGeometry, Detector, Grid, grid_volume_origin
from tomojax.recon import FDKConfig, FDKHostConfig, fdk, fdk_host

CUDA = pytest.param("cuda", marks=pytest.mark.gpu)


def scan(*, z_center=0.0, nz=7, roll=0.0):
    grid = Grid(5, 4, nz, 0.8, 1.2, 1.1, vol_center=(0.3, -0.2, z_center))
    detector = Detector(7, 5, 0.9, 1.0, (0.2, -0.15))
    geometry = ConeGeometry(
        grid,
        detector,
        np.linspace(0, 360, 8, endpoint=False),
        ConeBeam(30, 45, detector_roll_deg=roll),
    )
    data = np.random.default_rng(132).normal(size=(8, detector.nv, detector.nu)).astype(np.float32)
    return geometry, grid, detector, data


@pytest.mark.parametrize("backend", ["jax", CUDA])
@pytest.mark.parametrize("z_center", [-20.0, 0.0, 20.0])
@pytest.mark.parametrize("depth", [1, 4, 30])
def test_host_slabs_preserve_zero_extension_beyond_measured_rows(backend, z_center, depth):
    geometry, grid, detector, data = scan(z_center=z_center, nz=17, roll=1.0)
    cfg = FDKConfig(backend=backend, views_per_batch=3)
    reference = np.asarray(fdk(geometry, grid, detector, data, config=cfg))
    actual = fdk_host(
        geometry,
        grid,
        detector,
        data,
        config=FDKHostConfig(slices_per_batch=depth, fdk=cfg),
    )
    np.testing.assert_allclose(actual, reference, atol=2e-3 if backend == "cuda" else 3e-6)
    if z_center != 0.0:
        np.testing.assert_array_equal(reference, 0.0)
        np.testing.assert_array_equal(actual, 0.0)
    else:
        assert np.max(np.abs(reference)) > 0.0
        np.testing.assert_array_equal(actual[:, :, [0, -1]], 0.0)


@pytest.mark.parametrize("depth", [0, -1, 1.5, 2.0, True, np.bool_(True), "2"])
def test_host_slabs_reject_invalid_depth_without_writing(depth):
    geometry, grid, detector, data = scan()
    out = np.full((grid.nx, grid.ny, grid.nz), 91.0, np.float32)
    with pytest.raises(ValueError, match="fdk_host.*slices_per_batch"):
        fdk_host(
            geometry,
            grid,
            detector,
            data,
            config=FDKHostConfig(slices_per_batch=depth, fdk=FDKConfig(backend="jax")),
            out=out,
        )
    np.testing.assert_array_equal(out, 91.0)


@pytest.mark.parametrize("angle", [np.nan, np.inf])
def test_host_slabs_reject_nonfinite_angles_without_writing(angle):
    geometry, grid, detector, data = scan()
    geometry = replace(geometry, angles=[angle, *geometry.angles[1:]])
    out = np.full((grid.nx, grid.ny, grid.nz), 91.0, np.float32)
    with pytest.raises(ValueError, match="finite rotation angle"):
        fdk_host(geometry, grid, detector, data, out=out)
    np.testing.assert_array_equal(out, 91.0)


@pytest.mark.parametrize("invalid", ["dtype", "shape", "readonly", "device_input", "complex"])
def test_host_slabs_reject_invalid_storage_before_writing(invalid):
    geometry, grid, detector, data = scan()
    out = np.full((grid.nx, grid.ny, grid.nz), 91.0, np.float32)
    if invalid == "dtype":
        out = out.astype(np.float64)
    elif invalid == "shape":
        out = out[:, :, :-1]
    elif invalid == "readonly":
        out.flags.writeable = False
    elif invalid == "device_input":
        data = jax.device_put(data)
    else:
        data = data.astype(np.complex64)
    with pytest.raises((ValueError, TypeError), match="fdk_host"):
        fdk_host(geometry, grid, detector, data, out=out)
    np.testing.assert_array_equal(out, 91.0)


@pytest.mark.parametrize("mapping", [False, True])
def test_host_slabs_reject_overlapping_storage_before_writing(tmp_path, mapping):
    geometry, grid, detector, data = scan()
    shape = (grid.nx, grid.ny, grid.nz)
    if mapping:
        path = tmp_path / "same.bin"
        data = np.memmap(path, mode="w+", dtype=np.float32, shape=data.shape)
        out = np.memmap(path, mode="r+", dtype=np.float32, shape=shape)
    else:
        storage = np.ones(data.size, np.float32)
        data = storage.reshape(data.shape)
        out = storage[: np.prod(shape)].reshape(shape)
    data[:] = 1.0
    with pytest.raises(ValueError, match="fdk_host.*(overlap|separate files)"):
        fdk_host(geometry, grid, detector, data, out=out)
    np.testing.assert_array_equal(data, 1.0)


@pytest.mark.parametrize("value", [np.nan, np.inf, np.finfo(np.float64).max])
def test_host_slabs_reject_sampled_nonfinite_fp32_values(value):
    geometry, grid, detector, data = scan()
    data = data.astype(np.float64)
    data[0, 2, 3] = value
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="fdk_host.*finite.*FP32"):
        fdk_host(geometry, grid, detector, data, config=FDKHostConfig(fdk=FDKConfig(backend="jax")))


@pytest.mark.parametrize("invalid", ["nx", "vz"])
def test_host_slabs_validate_grid_before_allocating_or_writing(invalid):
    geometry, grid, detector, data = scan()
    grid = replace(grid, **{invalid: 0})
    with pytest.raises(ValueError, match="fdk_host grid"):
        fdk_host(geometry, grid, detector, data)


@pytest.mark.parametrize("layout", ["fortran", "reversed"])
def test_host_slabs_preserve_writable_noncontiguous_output(layout):
    geometry, grid, detector, data = scan()
    shape = (grid.nx, grid.ny, grid.nz)
    out = np.empty(shape, np.float32, order="F")
    if layout == "reversed":
        out = np.empty(shape, np.float32)[::-1, :, ::-1]
    cfg = FDKConfig(backend="jax", views_per_batch=3)
    reference = np.asarray(fdk(geometry, grid, detector, data, config=cfg))
    result = fdk_host(
        geometry,
        grid,
        detector,
        data,
        config=FDKHostConfig(slices_per_batch=np.int64(2), fdk=cfg),
        out=out,
    )
    assert result is out
    np.testing.assert_allclose(out, reference, atol=3e-6)


@pytest.mark.parametrize("backend", ["jax", CUDA])
@pytest.mark.parametrize("length_scale", [0.5, 2.0])
def test_host_slabs_recover_independent_gaussian_attenuation(backend, length_scale):
    # Infinite Gaussian line integrals are independent of either voxel projector.
    n, nz, views = 28, 11, 60
    sigma = 3.0 * length_scale
    grid = Grid(n, n, nz, 0.8 * length_scale, 0.9 * length_scale, length_scale)
    detector = Detector(40, 40, length_scale, length_scale)
    beam = ConeBeam(72 * length_scale, 108 * length_scale, detector_roll_deg=1.0)
    geometry = ConeGeometry(grid, detector, np.linspace(0, 360, views, endpoint=False), beam)
    center, u_dir, v_dir = beam.detector_frame(detector)
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du
    v = (np.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv
    pixels = center + u[None, :, None] * u_dir + v[:, None, None] * v_dir
    data = []
    for pose in geometry.poses():
        source = pose[:3, :3].T @ (beam.source() - pose[:3, 3])
        rays = (pixels - pose[:3, 3]) @ pose[:3, :3] - source
        squared_distance = np.sum(source**2) - (rays @ source) ** 2 / np.sum(rays**2, axis=-1)
        data.append(np.sqrt(2 * np.pi) * sigma * np.exp(-0.5 * squared_distance / sigma**2))
    origin = grid_volume_origin(grid)
    axes = [
        origin[i] + np.arange(size) * spacing
        for i, (size, spacing) in enumerate(
            zip((n, n, nz), (grid.vx, grid.vy, grid.vz), strict=True)
        )
    ]
    coordinates = np.meshgrid(*axes, indexing="ij")
    truth = np.exp(-0.5 * sum(axis**2 for axis in coordinates) / sigma**2)
    actual = fdk_host(
        geometry,
        grid,
        detector,
        np.asarray(data, np.float32),
        config=FDKHostConfig(
            slices_per_batch=4, fdk=FDKConfig(backend=backend, views_per_batch=17)
        ),
    )
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.03


@pytest.mark.gpu
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_cuda_host_slabs_release_textures_outside_stream_callbacks():
    from concurrent.futures import ThreadPoolExecutor

    geometry, grid, detector, data = scan()
    config = FDKConfig(backend="cuda", views_per_batch=2)
    reference = np.asarray(fdk(geometry, grid, detector, data, config=config))

    def reconstruct(amplitude):
        return amplitude, fdk_host(
            geometry,
            grid,
            detector,
            data * amplitude,
            config=FDKHostConfig(slices_per_batch=1, fdk=config),
        )

    # Interleave many short launches on one GPU, including independent callers.
    # Every queued kernel must retain its textures until that kernel completes;
    # CUDA resource destructors must never run inside a stream host callback.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for amplitude, actual in pool.map(reconstruct, np.linspace(0.5, 2.0, 16)):
            np.testing.assert_allclose(actual, reference * amplitude, atol=2e-3)


@pytest.mark.gpu
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_cuda_fdk_waits_for_queued_kernel_when_event_record_fails(monkeypatch):
    from contextlib import contextmanager
    import importlib

    import cupy as cp

    module = importlib.import_module("tomojax.recon._fdk_cuda")
    cone = importlib.import_module("tomojax.core.cone")
    stream = cp.cuda.Stream(non_blocking=True)
    waits = []

    class ObservedStream:
        def __init__(self, stream):
            self.stream = stream
            self.ptr = stream.ptr

        def synchronize(self):
            waits.append(True)
            self.stream.synchronize()

    @contextmanager
    def observe_stream(context, buffer):
        with cp.cuda.Device(buffer.device.id), stream:
            yield ObservedStream(stream)

    class FailedEvent:
        def __init__(self, *, disable_timing):
            assert disable_timing

        def record(self, stream):
            raise RuntimeError("injected FDK event-record failure")

    geometry, grid, detector, data = scan()
    coeff = cp.asarray(
        np.asarray(cone.cone_coefficients(geometry.poses(), grid, detector, geometry.beam))
    )
    target = cp.zeros((grid.nx, grid.ny, grid.nz), cp.float32)
    images = cp.asarray(np.swapaxes(data, 1, 2))
    half = cp.empty((len(data), detector.nu, 2 * module._LEAD), cp.float16)
    # The injected launch uses a separate nonblocking stream; finish input
    # initialization on its producer stream before bypassing JAX's dependencies.
    cp.cuda.get_current_stream().synchronize()
    config = FDKConfig(backend="cuda", views_per_batch=8)
    with monkeypatch.context() as patches:
        patches.setattr(module, "xla_stream", observe_stream)
        patches.setattr(cp.cuda, "Event", FailedEvent)
        with pytest.raises(RuntimeError, match="injected FDK event-record failure"):
            # Run the actual launch on real device buffers, without adding a
            # deliberately failed effect token to JAX's process-global state.
            module._launch(
                None,
                (target, half),
                coeff,
                images,
                target,
                grid=grid,
                det=detector,
                scale=1.0,
            )
    assert waits == [True]
    assert np.max(np.abs(cp.asnumpy(target))) > 0.0
    # A failed launch must not leave dangling texture handles or poison the next.
    recovered = np.asarray(fdk(geometry, grid, detector, data, config=config))
    assert np.isfinite(recovered).all()
    assert np.max(np.abs(recovered)) > 0.0


@pytest.mark.parametrize("backend", ["jax", CUDA])
@pytest.mark.parametrize(
    "kind", ["tilted_axis", "pitched_detector", "yawed_detector", "axis_offset"]
)
def test_host_slabs_match_full_fdk_with_nonseparable_geometry(backend, kind):
    geometry, grid, detector, data = scan(nz=17, z_center=1.2, roll=1.0)
    if kind == "tilted_axis":
        geometry = replace(geometry, tilt_deg=30.0)
    else:
        field = {
            "pitched_detector": "detector_pitch_deg",
            "yawed_detector": "detector_yaw_deg",
            "axis_offset": "axis_offset",
        }[kind]
        value = 0.6 if kind == "axis_offset" else 2.0
        geometry = replace(geometry, beam=replace(geometry.beam, **{field: value}))
    config = FDKConfig(backend=backend, views_per_batch=3)
    reference = np.asarray(fdk(geometry, grid, detector, data, config=config))
    actual = fdk_host(
        geometry,
        grid,
        detector,
        data,
        config=FDKHostConfig(slices_per_batch=4, fdk=config),
    )
    # Match the established FDK slab tolerance: FP32 coefficient shifts on JAX,
    # and texture-unit interpolation weights quantized to 1/256 on CUDA.
    tolerance = np.max(np.abs(reference)) * (2e-3 if backend == "cuda" else 2e-5)
    np.testing.assert_allclose(actual, reference, atol=tolerance)
