"""Contrast CGLS solves the dense centered least-squares system, not tiled means."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import ConeBeam, ConeGeometry, Detector, Grid, LaminographyGeometry
from tomojax.recon import CGLSConfig, cgls

# check-public-imports: allow-private
from tomojax.recon._data_term import project_residual

# check-public-imports: allow-private
from tomojax.recon._projection import projection_operators, resolve_geometry_projector


@pytest.mark.numerical
@pytest.mark.parametrize("cone", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_contrast_cgls_dense_reference_and_constant_invariance(
    stream, interpolation, backend, cone
):
    if backend == "cuda" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    if cone and interpolation == "cubic":
        pytest.skip("cone supports linear interpolation")
    grid = Grid(3, 3, 2, 0.8, 1.1, 1.3)
    detector = Detector(5, 4, 0.9, 1.2, (0.17, -0.23))
    geometry = LaminographyGeometry(grid, detector, [13, 47, 83, 127, 163], tilt_deg=23)
    if cone:
        geometry = ConeGeometry(grid, detector, geometry.angles, ConeBeam(20, 35), tilt_deg=17)
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(5)])

    model, backend = resolve_geometry_projector(
        geometry, "joseph", backend, detector=detector, context="contrast dot test"
    )
    fp, adjoint = projection_operators(
        poses, grid, detector, None, backend, 3, model, interpolation
    )

    def forward(x):
        return fp(x.reshape(2, 3, 3))

    matrix = np.asarray(jax.jacfwd(forward)(jnp.zeros(18)), dtype=np.float64)
    centered = (matrix - matrix.mean(axis=(1, 2), keepdims=True)).reshape(-1, 18)
    rng = np.random.default_rng(317)
    data = rng.normal(size=(5, 4, 5)).astype(np.float32)
    target = (data - data.mean(axis=(1, 2), keepdims=True)).ravel()
    expected = np.linalg.solve(centered.T @ centered + 0.04 * np.eye(18), centered.T @ target)
    probe = jnp.asarray(rng.normal(size=18), dtype=jnp.float32)
    projected = project_residual(forward(probe), "per_view_contrast")
    transpose = adjoint(jnp.asarray(data), lambda r: project_residual(r, "per_view_contrast"))
    np.testing.assert_allclose(
        jnp.vdot(projected, data), jnp.vdot(probe, transpose.ravel()), rtol=2e-5
    )
    config = CGLSConfig(
        iterations=150,
        damping=0.2,
        rtol=1e-6,
        views_per_batch=3,
        backend=backend,
        projector_model="joseph",
        joseph_interpolation=interpolation,
        stream_projections=stream,
        data_term="per_view_contrast",
    )
    results = []
    for i, shifted in enumerate(
        [data, data + np.linspace(-0.4, 0.7, 5, dtype=np.float32)[:, None, None]]
    ):
        volume, info = cgls(
            geometry,
            grid,
            detector,
            shifted,
            config=config,
            init_x=np.full((2, 3, 3), 0.1, np.float32) if i else None,
        )
        np.testing.assert_allclose(np.asarray(volume).ravel(), expected, rtol=2e-4, atol=2e-5)
        assert info["data_term"] == "per_view_contrast" and info["background_unresolved"]
        results.append(np.asarray(volume))
    np.testing.assert_allclose(*results, rtol=2e-4, atol=2e-5)


def test_contrast_cancellation_envelope_bounds_signed_dense_operator():
    # check-public-imports: allow-private
    from tomojax.recon._cgls_stopping import cancellation_magnitude

    rng = np.random.default_rng(318)
    matrix = rng.normal(size=(3, 4, 5, 7))
    centered = matrix - matrix.mean(axis=(1, 2), keepdims=True)
    data, prediction = rng.normal(size=(2, 3, 4, 5))
    size = np.abs(data) + np.abs(prediction)
    bound = (
        np.abs(matrix).reshape(-1, 7).T
        @ cancellation_magnitude(data, prediction, "per_view_contrast").ravel()
    )
    exact = np.abs(centered).reshape(-1, 7).T @ size.ravel()
    assert np.all(bound >= exact)
    np.testing.assert_array_equal(cancellation_magnitude(data, prediction, "absolute"), size)
