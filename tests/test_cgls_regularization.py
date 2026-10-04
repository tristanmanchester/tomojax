"""Compare physical quadratic regularization against explicit FP64 edge matrices."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.core.projector import forward_project_view_T
from tomojax.forward import project_joseph
from tomojax.geometry import Detector, Grid, LaminographyGeometry
from tomojax.recon import CGLSConfig, cgls


def edge_matrix(shape, spacing):
    rows = []
    for index in np.ndindex(shape):
        for axis, step in enumerate(spacing):
            neighbor = list(index)
            neighbor[axis] += 1
            if neighbor[axis] < shape[axis]:
                row = np.zeros(shape)
                row[index] = -1 / step
                row[tuple(neighbor)] = 1 / step
                rows.append(row.ravel())
    return np.asarray(rows).reshape(-1, np.prod(shape))


@pytest.mark.parametrize("shape", [(3, 2, 4), (1, 3, 2), (2, 1, 1), (1, 1, 1)])
def test_quadratic_gradient_matches_physical_edges_and_free_boundaries(shape):
    # check-public-imports: allow-private
    from tomojax.recon._quadratic import gradient_energy, gradient_normal

    spacing = (0.6, 1.3, 2.1)
    matrix = edge_matrix(shape, spacing)
    normal = matrix.T @ matrix
    volume = np.random.default_rng(643).normal(size=shape).astype(np.float32)
    np.testing.assert_allclose(
        gradient_energy(jnp.asarray(volume), spacing),
        np.sum((matrix @ volume.ravel()) ** 2),
        rtol=3e-7,
    )
    np.testing.assert_allclose(
        gradient_normal(jnp.asarray(volume), spacing).ravel(),
        normal @ volume.ravel(),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        gradient_normal(jnp.abs(volume), spacing, absolute_weights=True).ravel(),
        np.abs(normal) @ np.abs(volume.ravel()),
        rtol=3e-7,
    )
    np.testing.assert_array_equal(gradient_normal(jnp.ones(shape), spacing), np.zeros(shape))
    assert float(gradient_energy(jnp.ones(shape), spacing)) == 0
    automatic = jax.grad(lambda v: 0.5 * gradient_energy(v, spacing))(jnp.asarray(volume))
    np.testing.assert_allclose(automatic, gradient_normal(jnp.asarray(volume), spacing), rtol=2e-6)


@pytest.mark.parametrize("backend", ["jax", "pallas"])
@pytest.mark.parametrize(
    ("model", "interpolation"), [("ray", "linear"), ("joseph", "linear"), ("joseph", "cubic")]
)
@pytest.mark.parametrize("warm_start", [False, True])
def test_gradient_regularized_cgls_matches_dense_augmented_system(
    backend, model, interpolation, warm_start
):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    shape, spacing = (3, 2, 2), (0.8, 1.2, 0.7)
    grid = Grid(*shape, *spacing)
    detector = Detector(5, 4, 0.7, 1.1, (0.13, -0.29))
    geometry = LaminographyGeometry(grid, detector, [13, 49, 83, 122, 161], tilt_deg=23)
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(5)])

    def forward(flat):
        volume = flat.reshape(shape)
        if model == "joseph":
            return project_joseph(
                volume, poses, grid, detector, backend="jax", interpolation=interpolation
            ).ravel()
        return jax.vmap(lambda t: forward_project_view_T(t, grid, detector, volume))(poses).ravel()

    matrix = np.asarray(jax.jacfwd(forward)(jnp.zeros(np.prod(shape))), np.float64)
    edges = edge_matrix(shape, spacing)
    rng = np.random.default_rng(897)
    data = rng.normal(size=(5, 4, 5)).astype(np.float32)
    initial = rng.normal(size=shape).astype(np.float32) if warm_start else None
    damping, gradient_damping = (0.4 if warm_start else 0), 1.3
    augmented = np.concatenate([matrix, damping * np.eye(np.prod(shape)), gradient_damping * edges])
    expected = np.linalg.lstsq(
        augmented,
        np.concatenate([data.ravel(), np.zeros(len(augmented) - data.size)]),
        rcond=None,
    )[0]
    actual, info = cgls(
        geometry,
        grid,
        detector,
        data,
        init_x=initial,
        config=CGLSConfig(
            iters=80,
            rtol=1e-7,
            damping=damping,
            gradient_damping=gradient_damping,
            views_per_batch=3,
            projector_backend=backend,
            projector_model=model,
            joseph_interpolation=interpolation,
        ),
    )
    np.testing.assert_allclose(np.asarray(actual).ravel(), expected, rtol=3e-4, atol=3e-5)
    residual = matrix.T @ (data.ravel() - matrix @ np.asarray(actual).ravel())
    residual -= (
        damping**2 * np.eye(np.prod(shape)) + gradient_damping**2 * edges.T @ edges
    ) @ np.asarray(actual).ravel()
    assert np.linalg.norm(residual) < 3e-6 * np.linalg.norm(matrix.T @ data.ravel())
    assert info["gradient_damping"] == gradient_damping
    assert info["normal_residual_is_recomputed"]
    assert info["effective_iters"] < 80


@pytest.mark.parametrize("value", [-1.0, np.inf, np.nan])
def test_gradient_damping_rejects_invalid_weights(value):
    grid, detector = Grid(2, 2, 2, 1.0, 1.0, 1.0), Detector(3, 3, 1.0, 1.0)
    geometry = LaminographyGeometry(grid, detector, [0, 45, 90], tilt_deg=23)
    with pytest.raises(ValueError, match="gradient_damping"):
        cgls(
            geometry,
            grid,
            detector,
            jnp.zeros((3, 3, 3)),
            config=CGLSConfig(gradient_damping=value),
        )
