"""Independent dense and whole-update checks for shared-geometry transposes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
import tomojax.backends
from tomojax.core.cone import beam_frame, cone_backproject_pair, cone_project, frame_coefficients
from tomojax.recon import SIRTConfig, sirt


@pytest.mark.numerical
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_two_transposes_match_dense_basis_with_mixed_views_and_chunk_tail(backend):
    grid, detector = tj.Grid(5, 4, 3, 0.7, 1.1, 0.9), tj.Detector(7, 5, 1.3, 0.8, (0.2, -0.3))
    geometry = tj.ConeGeometry(grid, detector, np.linspace(0, 360, 33), tj.ConeBeam(30, 50))
    poses = np.asarray(geometry.poses(), np.float32).copy()
    angle = np.float32(0.71)
    rotation = np.eye(4, dtype=np.float32)
    rotation[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    poses[::2] = poses[::2] @ rotation
    frames = np.broadcast_to(beam_frame(geometry.beam, detector), (33, 4, 3))
    coeff = frame_coefficients(jnp.asarray(poses), frames, grid, detector)
    shape = (grid.nz, grid.ny, grid.nx)
    basis = jnp.eye(np.prod(shape), dtype=jnp.float32).reshape((-1, *shape))
    forward = jax.jit(jax.vmap(lambda x: cone_project(x, coeff, grid, detector, backend=backend)))
    matrix = np.asarray(forward(basis), np.float64).reshape((np.prod(shape), -1))
    rng = np.random.default_rng(45)
    images = rng.normal(size=(2, 33, detector.nv, detector.nu)).astype(np.float32)
    images[0, :, :2] = 0
    images[1, :, 3:] = 0
    initial = tuple(rng.normal(size=shape).astype(np.float32) for _ in range(2))
    actual = cone_backproject_pair(
        tuple(images), coeff, grid, detector, backend=backend, accumulate=initial
    )
    for channel in (0, 1):
        expected = (matrix @ images[channel].ravel()).reshape(shape) + initial[channel]
        scale = (np.abs(matrix) @ np.abs(images[channel]).ravel()).max() + np.abs(
            initial[channel]
        ).max()
        np.testing.assert_allclose(actual[channel], expected, rtol=0, atol=5e-7 * scale)


@pytest.mark.numerical
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_uncached_sirt_preserves_masked_subset_updates(monkeypatch, backend):
    grid, detector = tj.Grid(5, 4, 3, 0.7, 1.1, 0.9), tj.Detector(7, 5, 1.3, 0.8)
    geometry = tj.ConeGeometry(
        grid, detector, np.linspace(0, 180, 7), tj.ConeBeam(30, 50), tilt_deg=23
    )
    truth = np.random.default_rng(46).uniform(size=(3, 4, 5)).astype(np.float32)
    data = np.asarray(tj.project(geometry, truth))
    cfg = SIRTConfig(
        iterations=3,
        subsets=3,
        support=truth > 0.2,
        lower_bound=0,
        upper_bound=0.8,
        views_per_batch=2,
        backend=backend,
    )
    cached, _ = sirt(geometry, grid, detector, data, init_x=truth * 0.1, config=cfg)
    monkeypatch.setattr(tomojax.backends, "device_free_memory_bytes", lambda: 1)
    paired, _ = sirt(geometry, grid, detector, data, init_x=truth * 0.1, config=cfg)
    np.testing.assert_allclose(paired, cached, rtol=3e-6, atol=3e-6)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("nz", "dv", "negative_v"), [(33, 0.4, False), (65, 1.0, True), (67, 5.0, False)]
)
def test_paired_separable_transpose_preserves_each_rhs_across_tile_tails(nz, dv, negative_v):
    from tomojax.core.cone import cone_backproject, cone_model
    from tomojax.geometry import stack_view_poses

    grid = tj.Grid(13, 11, nz, 0.9, 1.1, 1.0)
    detector = tj.Detector(103, 89, 1.2, dv, (0.4, -0.3))
    geometry = tj.ConeGeometry(
        grid, detector, np.linspace(0, 360, 33, endpoint=False), tj.ConeBeam(300, 450)
    )
    frames = cone_model(geometry).frames(33).copy()
    if negative_v:
        frames[:, 2, :] *= -1
    coeff = frame_coefficients(stack_view_poses(geometry, 33), frames, grid, detector)
    rng = np.random.default_rng(1733)
    images = rng.normal(size=(2, 33, detector.nv, detector.nu)).astype(np.float32)
    images[0, :, : detector.nv // 3] = 0
    images[1, :, 2 * detector.nv // 3 :] = 0
    initial = rng.normal(size=(2, nz, grid.ny, grid.nx)).astype(np.float32)
    saved_images, saved_initial = images.copy(), initial.copy()
    actual = cone_backproject_pair(
        tuple(images), coeff, grid, detector, backend="cuda", accumulate=tuple(initial)
    )
    for channel in (0, 1):
        expected = np.asarray(
            cone_backproject(
                images[channel], coeff, grid, detector, backend="cuda", accumulate=initial[channel]
            )
        )
        delta = np.asarray(actual[channel]) - expected
        assert np.linalg.norm(delta) <= 1e-6 * np.linalg.norm(expected)
        assert np.max(np.abs(delta)) <= 1e-6 * np.max(np.abs(expected))
    np.testing.assert_array_equal(images, saved_images)
    np.testing.assert_array_equal(initial, saved_initial)
