from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tomojax.forward import ProjectionArrayGeometryInput, project_parallel_reference_from_input
from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import SPDHGConfig, spdhg_tv

# check-public-imports: allow-private
from tomojax.recon._tv_ops import (
    div3,
    grad3,
    huber_tv_grad,
    isotropic_tv_value,
    prox_huber_tv_conj,
)


def test_forward_projector_preserves_constant_view_symmetry() -> None:
    volume = jnp.ones((2, 2, 2), dtype=jnp.float32)

    projections = project_parallel_reference_from_input(
        volume,
        ProjectionArrayGeometryInput(
            theta_rad=jnp.asarray([0.0, jnp.pi / 2.0], dtype=jnp.float32),
            dx_px=jnp.zeros((2,), dtype=jnp.float32),
            dz_px=jnp.zeros((2,), dtype=jnp.float32),
            detector_shape=(2, 2),
        ),
    )

    assert projections.shape == (2, 2, 2)
    assert bool(jnp.all(jnp.isfinite(projections)))
    np.testing.assert_allclose(np.asarray(projections[0]), np.asarray(projections[1]))
    np.testing.assert_allclose(np.asarray(projections), np.full((2, 2, 2), 2.0))


def test_tv_operators_preserve_adjoint_and_prox_invariants() -> None:
    volume = jnp.arange(27, dtype=jnp.float32).reshape(3, 3, 3) / 10.0
    px = jnp.linspace(-0.5, 0.5, 27, dtype=jnp.float32).reshape(3, 3, 3)
    py = jnp.cos(px)
    pz = jnp.sin(px)

    gx, gy, gz = grad3(volume)
    grad_inner = jnp.vdot(gx, px) + jnp.vdot(gy, py) + jnp.vdot(gz, pz)
    div_inner = jnp.vdot(volume, div3(px, py, pz))

    np.testing.assert_allclose(float(grad_inner + div_inner), 0.0, atol=1e-6)
    assert float(isotropic_tv_value(jnp.ones((3, 3, 3), dtype=jnp.float32))) == 0.0
    np.testing.assert_allclose(
        np.asarray(huber_tv_grad(jnp.ones((3, 3, 3), dtype=jnp.float32), delta=0.1)),
        np.zeros((3, 3, 3), dtype=np.float32),
    )

    qx, qy, qz = prox_huber_tv_conj(
        px * 4.0,
        py * 4.0,
        pz * 4.0,
        sigma=0.5,
        lam=0.25,
        delta=0.1,
    )
    dual_norm = jnp.sqrt(qx * qx + qy * qy + qz * qz)
    assert float(jnp.max(dual_norm)) <= 0.25 + 1e-6


def test_spdhg_one_iteration_smoke_is_finite_and_positive() -> None:
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=np.asarray([0.0, 90.0], dtype=np.float32),
    )
    projections = jnp.ones((2, 2, 2), dtype=jnp.float32)

    volume, info = spdhg_tv(
        geometry,
        grid,
        detector,
        projections,
        config=SPDHGConfig(
            iters=1,
            lambda_tv=0.0,
            views_per_batch=1,
            seed=123,
            tau=0.1,
            sigma_data=0.1,
            sigma_tv=0.1,
            checkpoint_projector=False,
            positivity=True,
            log_every=1,
        ),
    )

    assert volume.shape == (2, 2, 2)
    assert bool(jnp.all(jnp.isfinite(volume)))
    assert float(jnp.min(volume)) >= 0.0
    assert info["loss"] == [4.0]
    assert info["A_norm"] is None
    assert info["regulariser"] == "tv"
    assert info["views_per_batch"] == 1
    assert info["num_blocks"] == 2
