"""Host-owned FISTA keeps the finite-iteration reference objective and image."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core.host_planes import HostPlaneOperator
from tomojax.core.joseph import plane_coefficients
from tomojax.core.joseph_regions import PlaneRegion
from tomojax.geometry import Detector, Grid, LaminographyGeometry, stack_view_poses
from tomojax.recon.fista_host import fista_tv_core_host
from tomojax.recon.fista_tv_core import FistaCoreConfig, fista_tv_core_arrays


@pytest.mark.numerical
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    ("depth", "iterations", "interpolation", "flags", "supported"),
    [
        (1, 0, "linear", (True, True, True), False),
        (1, 3, "linear", (True, True, True), True),
        (2, 3, "linear", (True, False, False), False),
        (3, 3, "linear", (False, False, False), True),
        (1, 3, "cubic", (False, True, True), "broadcast"),
        (2, 3, "cubic", (True, True, True), False),
        (2, 3, "linear", (True, True, True), "scalar"),
    ],
)
def test_host_fista_matches_array_core(backend, depth, iterations, interpolation, flags, supported):
    if backend == "cuda" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(32402)
    grid, detector = Grid(4, 5, 3, 0.7, 0.8, 1.2), Detector(7, 5, 0.9, 0.8)
    poses = stack_view_poses(LaminographyGeometry(grid, detector, [13, 97, 208], tilt_deg=35), 3)
    coefficients = plane_coefficients(poses, grid, detector)
    initial = np.asarray(rng.uniform(-0.1, 0.7, (4, 5, 3)), np.float32)
    data = np.asarray(rng.uniform(0, 2, (3, 5, 7)), np.float32)
    original, original_data = initial.copy(), data.copy()
    support_shape = {"scalar": (), "broadcast": (1, 5, 1)}.get(supported, initial.shape)
    support = jnp.asarray(rng.uniform(0, 1, support_shape), jnp.float32) if supported else None
    weights = np.array([0, 0.8, 1.1], np.float32)
    cfg = FistaCoreConfig(
        iterations=iterations,
        lipschitz=500,
        tv_weight=0.01,
        huber_delta=0.2,
        nonnegative=True,
        lower_bound=-0.05,
        upper_bound=0.6,
        support=support,
        ray_integrator="joseph_cubic" if interpolation == "cubic" else "joseph",
        forward_projector=backend,
        backprojector=backend,
        views_per_batch=2,
        compute_iteration_loss=flags[0],
        compute_final_data_loss=flags[1],
        compute_final_regulariser_value=flags[2],
    )
    expected = fista_tv_core_arrays(
        x0=jnp.asarray(initial),
        T_all=poses,
        det_grid=None,
        projections=jnp.asarray(data),
        grid=grid,
        detector=detector,
        cfg=cfg,
        view_weights=jnp.asarray(weights),
    )
    region = PlaneRegion(grid=grid, detector=detector, backend=backend, interpolation=interpolation)
    # Alternate a single-view detector group and a fully resident detector stack.
    operator = HostPlaneOperator(
        coefficients,
        region=region,
        slab_depth=depth,
        detector_bytes=detector.nu * detector.nv * 4 * (1 if depth == 1 else 3),
    )
    actual = fista_tv_core_host(operator, initial, projections=data, cfg=cfg, view_weights=weights)
    for name in ("x", "loss", "data_loss", "regulariser_value"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=4e-6, atol=3e-6
        )
    np.testing.assert_array_equal(initial, original)
    np.testing.assert_array_equal(data, original_data)
    assert actual.x.flags.f_contiguous
    assert int(actual.effective_iterations) == iterations
    assert operator.host_to_device_bytes > 0 and operator.device_to_host_bytes > 0
    if iterations == 0:
        tangent = jnp.asarray(rng.normal(0, 0.01, (3, 2, 14)), jnp.float32).at[:, :, 0].set(0)
        columns = np.empty((3, 3, detector.nv, detector.nu), np.float32)
        column_operator = HostPlaneOperator(
            coefficients, region=region, slab_depth=depth, detector_bytes=3 * data[0].nbytes
        )
        column_operator.columns(
            initial, tangent=tangent, out=columns, support=np.array(0.5, np.float32)
        )
        wanted = region.columns(coefficients, jnp.asarray(initial) * 0.5, tangent=tangent)
        np.testing.assert_allclose(columns, wanted, rtol=3e-5, atol=2e-6)


def test_detector_capacity_must_hold_one_window():
    region = PlaneRegion(grid=Grid(1, 1, 1, 1, 1, 1), detector=Detector(3, 2, 1, 1), backend="jax")
    with pytest.raises(ValueError, match="one detector window"):
        HostPlaneOperator(jnp.zeros((1, 14)), region=region, slab_depth=1, detector_bytes=23)


def test_host_plane_transfers_own_buffers_and_preserve_strided_padding():
    region = PlaneRegion(grid=Grid(1, 1, 1, 1, 1, 1), detector=Detector(3, 2, 1, 1), backend="jax")
    operator = HostPlaneOperator(jnp.zeros((1, 14)), region=region, slab_depth=1, detector_bytes=24)
    value = jnp.arange(6, dtype=jnp.float32).reshape(2, 3).T
    storage = np.full((3, 4), -17, np.float64)
    operator.get_into(value, storage[:, ::2])
    owned = operator.get(value)
    owned[:] = -3
    np.testing.assert_array_equal(storage[:, ::2], np.arange(6).reshape(2, 3).T)
    np.testing.assert_array_equal(storage[:, 1::2], -17)
    np.testing.assert_array_equal(np.asarray(value), np.arange(6).reshape(2, 3).T)
    assert operator.device_to_host_bytes == 48
    buffer = np.empty(32, np.float32)
    first = (-buffer.ctypes.data % 64) // buffer.itemsize
    source = buffer[first : first + 16]
    source[:] = np.arange(16)
    uploaded = operator.put(source).block_until_ready()
    source[:] = -5
    np.testing.assert_array_equal(uploaded, np.arange(16))


@pytest.mark.parametrize("masked", [False, True])
def test_volume_upload_reads_bounded_owned_plane_windows(masked):
    shape = (2049, 4096, 3)  # An xy plane exceeds half the host window budget.
    region = PlaneRegion(grid=Grid(*shape, 1, 1, 1), detector=Detector(1, 1, 1, 1), backend="jax")
    operator = HostPlaneOperator(jnp.zeros((1, 14)), region=region, slab_depth=3, detector_bytes=4)
    read = []
    buffer = np.empty((*shape[:2], 1), np.float32)

    class Source:
        def __getitem__(self, index):
            first, last = index[2].start, index[2].stop
            assert last - first == 1
            read.append(first)
            buffer.fill(first + 1)
            return buffer

    support = np.broadcast_to(np.float32(0.5), shape) if masked else None
    value = operator.put_volume(Source(), 0, 3, support)
    buffer.fill(-17)
    expected = np.arange(1, 4, dtype=np.float32) * (0.5 if masked else 1)
    np.testing.assert_array_equal(np.asarray(value[0, 0]), expected)
    np.testing.assert_array_equal(np.asarray(value[-1, -1]), expected)
    assert read == [0, 1, 2]
    assert operator.host_to_device_bytes == 4 * np.prod(shape)
