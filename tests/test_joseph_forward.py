"""Public Joseph projection and fused first-order loss contract."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_joseph_projector import matrix, problem
from tomojax.forward import joseph_l2_value_and_grad, project_joseph


@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_public_joseph_matches_physical_matrix_and_loss_derivative(backend, interpolation):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    g, d, poses = problem()
    rng = np.random.default_rng(10)
    x = jnp.asarray(rng.normal(size=(g.nx, g.ny, g.nz)), jnp.float32)
    y = jnp.asarray(rng.normal(size=(len(poses), d.nv, d.nu)), jnp.float32)
    poses = jnp.asarray(poses)
    a = matrix(np.asarray(poses), g, d, interpolation)

    def forward(v, t):
        return project_joseph(v, t, g, d, backend=backend, interpolation=interpolation)

    pred = jax.jit(forward)(x, poses)
    np.testing.assert_allclose(
        np.asarray(pred).ravel(), a @ np.asarray(x).ravel(), rtol=2e-5, atol=8e-6
    )
    actual = jax.jit(
        lambda v, t, y: joseph_l2_value_and_grad(
            v, t, y, g, d, backend=backend, interpolation=interpolation
        )
    )(x, poses, y)
    expected = jax.jit(
        jax.value_and_grad(lambda v, t: 0.5 * jnp.sum((forward(v, t) - y) ** 2), argnums=(0, 1))
    )(x, poses)
    for av, ev in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(av, ev, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(
        np.asarray(actual[1][0]).ravel(),
        a.T @ (a @ np.asarray(x).ravel() - np.asarray(y).ravel()),
        rtol=3e-5,
        atol=3e-5,
    )
    # A target predicted by this same model must have only roundoff-level loss.
    zero, (gx, gt) = jax.jit(
        lambda v, t, y: joseph_l2_value_and_grad(
            v, t, y, g, d, backend=backend, interpolation=interpolation
        )
    )(x, poses, pred)
    assert float(zero) <= 1e-12 * float(jnp.sum(pred**2))
    assert np.linalg.norm(gx) < 1e-5 * max(np.linalg.norm(x), 1)
    assert np.isfinite(np.asarray(gt)).all()


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_reference_supports_second_volume_derivatives(interpolation):
    g, d, poses = problem()
    x = jnp.ones((g.nx, g.ny, g.nz), jnp.float32)
    direction = jnp.linspace(-1, 1, x.size).reshape(x.shape)

    def gradient(v):
        return joseph_l2_value_and_grad(
            v, poses, jnp.zeros((6, d.nv, d.nu)), g, d, interpolation=interpolation
        )[1][0]

    _, hessian_vector = jax.jit(lambda v, dv: jax.jvp(gradient, (v,), (dv,)))(x, direction)
    a = matrix(poses, g, d, interpolation)
    np.testing.assert_allclose(
        np.asarray(hessian_vector).ravel(),
        a.T @ (a @ np.asarray(direction).ravel()),
        rtol=3e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize(
    "bad",
    [
        "volume_shape",
        "pose_shape",
        "empty",
        "complex_volume",
        "complex_pose",
        "spacing",
        "origin",
        "center",
        "backend",
    ],
)
def test_public_projection_rejects_invalid_static_inputs(bad):
    g, d, poses = problem()
    volume = np.zeros((g.nx, g.ny, g.nz), np.float32)
    backend = "jax"
    if bad == "volume_shape":
        volume = volume[:, :, :-1]
    elif bad == "pose_shape":
        poses = poses[:, :3, :3]
    elif bad == "empty":
        poses = poses[:0]
    elif bad == "complex_volume":
        volume = volume.astype(np.complex64)
    elif bad == "complex_pose":
        poses = poses.astype(np.complex64)
    elif bad == "spacing":
        d = replace(d, du=0)
    elif bad == "origin":
        g = replace(g, vol_origin=(np.nan, 0, 0))
    elif bad == "center":
        d = replace(d, det_center=(0, np.inf))
    else:
        backend = "automatic"
    with pytest.raises(ValueError, match="Joseph projection"):
        project_joseph(volume, poses, g, d, backend=backend)


@pytest.mark.parametrize("bad", ["shape", "complex"])
def test_public_loss_rejects_invalid_target(bad):
    g, d, poses = problem()
    y = np.zeros((6, d.nv, d.nu), np.float32)
    y = y[:5] if bad == "shape" else y.astype(np.complex64)
    with pytest.raises(ValueError, match="target must be real with shape"):
        joseph_l2_value_and_grad(np.zeros((g.nx, g.ny, g.nz)), poses, y, g, d)


def test_explicit_cuda_never_silently_falls_back():
    if jax.default_backend() == "gpu":
        pytest.skip("requires CPU to test explicit CUDA rejection")
    g, d, poses = problem()
    with pytest.raises(ValueError, match="requires an NVIDIA CUDA"):
        project_joseph(np.zeros((g.nx, g.ny, g.nz)), poses, g, d, backend="pallas")


@pytest.mark.parametrize("interpolation", ["nearest", "CUBIC", "automatic"])
def test_invalid_interpolation_is_rejected(interpolation):
    g, d, poses = problem()
    x = np.zeros((g.nx, g.ny, g.nz), np.float32)
    y = np.zeros((len(poses), d.nv, d.nu), np.float32)
    with pytest.raises(ValueError, match="interpolation"):
        project_joseph(x, poses, g, d, interpolation=interpolation)
    with pytest.raises(ValueError, match="interpolation"):
        joseph_l2_value_and_grad(x, poses, y, g, d, interpolation=interpolation)


@pytest.mark.parametrize("interpolation", ["cubic", "nearest"])
def test_cgls_rejects_invalid_or_inapplicable_interpolation(interpolation):
    from tomojax.geometry import ParallelGeometry
    from tomojax.recon import CGLSConfig, cgls

    g, d, poses = problem()
    geometry = ParallelGeometry(g, d, np.arange(len(poses)))
    with pytest.raises(ValueError, match="interpolation"):
        cgls(
            geometry,
            g,
            d,
            np.zeros((len(poses), d.nv, d.nu)),
            config=CGLSConfig(joseph_interpolation=interpolation),
        )


def test_cubic_outer_lobe_retains_small_weights_near_its_roots():
    from decimal import Decimal, localcontext

    # check-public-imports: allow-private
    from tomojax.core._plane_interpolation import cubic_weight

    radii = np.array([1 + 2**-10, 1 + 2**-20, 2 - 2**-10, 2 - 2**-20, 2 - 2**-23], np.float32)
    with localcontext() as context:
        context.prec = 70
        # Evaluate the expanded polynomial at high precision independently of
        # the factored FP32 implementation. Float64 also cancels at these roots.
        expected = np.array(
            [
                float(
                    (
                        -(Decimal(float(r)) ** 3)
                        + 5 * Decimal(float(r)) ** 2
                        - 8 * Decimal(float(r))
                        + 4
                    )
                    / 2
                )
                for r in radii
            ]
        )
    actual = np.asarray(jax.jit(cubic_weight)(jnp.asarray(radii)))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0)
    assert np.all(actual < 0)
