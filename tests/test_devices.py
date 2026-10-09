"""Sharing a scan's views among several devices.

CPU test runs have four CPU devices (``JAX_NUM_CPU_DEVICES=4``, set in
conftest.py); with fewer devices these tests skip. On GPUs,
``test_cone_kernels_*`` checks the CUDA kernels on every GPU there is.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tomojax as tj

if len(jax.devices()) < 2:
    pytest.skip("needs two or more devices", allow_module_level=True)

N, VIEWS = 16, 37  # views that do not divide among the devices


def _phantom() -> np.ndarray:
    c = (np.arange(N) - (N - 1) / 2) / (N / 4)
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    volume = np.exp(-((x - 0.3) ** 2 + y**2 + (z + 0.2) ** 2) * 2)
    return (volume + 0.5 * (x**2 + y**2 < 0.5)).astype(np.float32)


def _geometry(kind: str) -> tj.geometry.ScanGeometry:
    grid = tj.Grid(N, N, N, 1.0, 1.0, 1.0)
    detector = tj.Detector(24, 20, 1.0, 1.0, (0.3, -0.4))
    angles = np.linspace(0, 360, VIEWS, endpoint=False)
    if kind == "parallel":
        return tj.ParallelGeometry(grid, detector, angles / 2)
    if kind == "lamino":
        return tj.LaminographyGeometry(grid, detector, angles, tilt_deg=30)
    if kind == "orbits":  # two circular orbits, the second higher, on one detector
        low = tj.ConeGeometry(grid, detector, angles[:20], tj.ConeBeam(48.0, 72.0))
        high = tj.ConeGeometry(
            grid, tj.Detector(24, 20, 1.0, 1.0, (0.3, 2.0)), angles[20:], tj.ConeBeam(50.0, 72.0)
        )
        return tj.geometry.ConeSegments((low, high))
    return tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(48.0, 72.0))


def _relative(a: object, b: object) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("kind", ["parallel", "lamino", "cone", "orbits"])
def test_projection_and_transpose_shared_among_devices_match_one_device(kind):
    geometry = _geometry(kind)
    volume = _phantom()
    images = np.random.default_rng(0).random((VIEWS, 20, 24)).astype(np.float32)

    projected = tj.project(geometry, volume, devices=jax.devices())
    backprojected = tj.backproject(geometry, images, devices=jax.devices())

    assert projected.shape == (VIEWS, 20, 24)
    # Batches of other views compile to other float32 arithmetic.
    np.testing.assert_allclose(projected, tj.project(geometry, volume), rtol=1e-5, atol=1e-5)
    # Only the order of the devices' sum differs.
    assert _relative(backprojected, tj.backproject(geometry, images)) < 1e-6


def test_the_shared_transpose_is_the_exact_adjoint():
    geometry, devices = _geometry("cone"), jax.devices()
    rng = np.random.default_rng(1)
    x = rng.random((N, N, N)).astype(np.float32)
    y = rng.random((VIEWS, 20, 24)).astype(np.float32)

    ax = np.asarray(tj.project(geometry, x, devices=devices), np.float64)
    aty = np.asarray(tj.backproject(geometry, y, devices=devices), np.float64)

    lhs, rhs = np.vdot(ax, y), np.vdot(x.astype(np.float64), aty)
    assert abs(lhs - rhs) < 1e-6 * abs(lhs)


@pytest.mark.parametrize("views", [VIEWS, 4 * 9])  # uneven and even shares
def test_results_come_back_on_the_first_device(views):
    geometry = _geometry("parallel")
    geometry = tj.ParallelGeometry(geometry.grid, geometry.detector, np.linspace(0, 180, views))
    devices = jax.devices()[::-1]

    projected = tj.project(geometry, _phantom(), devices=devices)
    backprojected = tj.backproject(geometry, projected, devices=devices)
    scan = tj.Scan(np.asarray(projected), geometry)
    volume = tj.reconstruct(scan, "cgls", iterations=2, devices=devices).volume

    for result in (projected, backprojected, volume):
        assert result.devices() == {devices[0]}


def test_one_device_is_used_as_given():
    geometry, last = _geometry("cone"), jax.devices()[-1]
    projected = tj.project(geometry, _phantom(), devices=last)
    assert projected.devices() == {last}
    np.testing.assert_allclose(projected, tj.project(geometry, _phantom()), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["cgls", "fista"])
def test_iterative_reconstruction_shared_among_devices_matches_one_device(method):
    geometry = _geometry("cone")
    scan = tj.Scan(np.asarray(tj.project(geometry, _phantom())), geometry)
    options = {"iterations": 8}
    if method == "fista":
        options |= {"tv_weight": 1e-3, "nonnegative": True}

    one = tj.reconstruct(scan, method, **options).volume
    shared = tj.reconstruct(scan, method, devices=jax.devices(), **options).volume

    assert _relative(shared, one) < 1e-4


def test_cgls_from_a_start_shared_among_devices_matches_one_device():
    from tomojax.recon import CGLSConfig, cgls

    geometry = _geometry("parallel")
    data = np.asarray(tj.project(geometry, _phantom()))
    start = 0.5 * _phantom()
    one, _ = cgls(geometry, geometry.grid, geometry.detector, data, init_x=start,
                  config=CGLSConfig(iterations=4))  # fmt: skip
    shared, _ = cgls(geometry, geometry.grid, geometry.detector, data, init_x=start,
                     config=CGLSConfig(iterations=4, devices=jax.devices()))  # fmt: skip
    assert _relative(shared, one) < 1e-4


@pytest.mark.parametrize("kind", ["parallel", "lamino", "cone"])
@pytest.mark.parametrize("on_host", [False, True])
def test_filtered_backprojection_shared_among_devices_matches_one_device(kind, on_host):
    geometry = _geometry(kind)
    data = np.asarray(tj.project(geometry, _phantom()))
    scan = tj.Scan(data if on_host else jax.numpy.asarray(data), geometry)

    one = tj.reconstruct(scan, "fbp").volume
    shared = tj.reconstruct(scan, "fbp", devices=jax.devices()).volume

    assert shared.devices() == {jax.devices()[0]}
    assert _relative(shared, one) < 1e-6  # weighted for the whole scan, summed in another order


def test_fdk_slabs_shared_among_devices_match_one_device():
    from tomojax.recon.fdk import FDKConfig, FDKHostConfig, fdk_host

    geometry = _geometry("cone")
    data = np.asarray(tj.project(geometry, _phantom()))
    args = (geometry, geometry.grid, geometry.detector, data)
    one = fdk_host(*args, config=FDKHostConfig(slices_per_batch=3))
    shared = fdk_host(
        *args, config=FDKHostConfig(slices_per_batch=3, fdk=FDKConfig(devices=jax.devices()))
    )
    np.testing.assert_array_equal(shared, one)  # the same slabs, each on one device


@pytest.mark.parametrize(
    ("kind", "smoothness"), [("cone", 0.0), ("parallel", 1e-2), ("orbits", 0.0)]
)
def test_pose_alignment_shared_among_devices_matches_one_device(kind, smoothness):
    from dataclasses import replace

    # check-public-imports: allow-private
    from tomojax._data.geometry_meta import AugmentedGeometry
    from tomojax.alignment.api import alignment_plan

    geometry = _geometry(kind)
    rng = np.random.default_rng(3)
    motion = np.zeros((VIEWS, 6), np.float32)
    motion[:, 3], motion[:, 5] = rng.normal(0, 0.4, (2, VIEWS))
    scan = tj.Scan(
        np.asarray(tj.project(AugmentedGeometry(geometry, motion, "detector"), _phantom())),
        geometry,
    )
    # A few conjugate-gradient steps: unconverged float32 CG magnifies the
    # rounding of any change in the order of its sums, devices or not.
    config = replace(
        alignment_plan("pose", geometry.grid).config,
        outer_iterations=2,
        gn_joint_iterations=3,
        w_trans=smoothness,
    )

    one = tj.align(scan, levels=(1,), config=config)
    shared = tj.align(scan, levels=(1,), config=config, devices=jax.devices())

    assert shared.volume.devices() == {jax.devices()[0]}
    np.testing.assert_allclose(shared.poses, one.poses, atol=1e-4)
    assert _relative(shared.volume, one.volume) < 1e-5


def test_bad_devices_are_refused():
    scan = tj.Scan(np.zeros((VIEWS, 20, 24), np.float32), _geometry("parallel"))
    with pytest.raises(ValueError, match="does not take devices"):
        tj.reconstruct(scan, "spdhg", devices=jax.devices())
    with pytest.raises(ValueError, match="at least one device"):
        tj.project(scan.geometry, _phantom(), devices=[])
    with pytest.raises(ValueError, match="twice"):
        tj.project(scan.geometry, _phantom(), devices=[jax.devices()[0]] * 2)


@pytest.mark.parametrize("solver", ["cgls", "fista"])
def test_shared_projections_are_not_streamed(solver):
    from tomojax.recon import CGLSConfig, FistaConfig, cgls, fista_tv

    geometry = _geometry("parallel")
    data = np.zeros((VIEWS, 20, 24), np.float32)
    run, config = (cgls, CGLSConfig) if solver == "cgls" else (fista_tv, FistaConfig)
    settings = config(stream_projections=True, devices=jax.devices())
    with pytest.raises(ValueError, match="cannot be streamed"):
        run(geometry, geometry.grid, geometry.detector, data, config=settings)


@pytest.mark.gpu
def test_cone_kernels_give_the_same_projections_on_every_gpu():
    gpus = jax.devices("gpu")
    geometry = _geometry("cone")
    volume = _phantom()
    reference = np.asarray(tj.project(geometry, volume))
    for gpu in gpus[1:]:
        here = tj.project(geometry, volume, devices=gpu)
        assert here.devices() == {gpu}
        np.testing.assert_array_equal(np.asarray(here), reference)
    shared = tj.backproject(geometry, reference, devices=gpus)
    assert _relative(shared, tj.backproject(geometry, reference)) < 1e-6
