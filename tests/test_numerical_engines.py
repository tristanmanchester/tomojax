from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import SPDHGConfig, spdhg_tv

# check-public-imports: allow-private
from tomojax.recon._tv_ops import (
    div3,
    grad3,
    isotropic_tv_value,
    prox_huber_tv_conj,
)


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


def test_tv_components_follow_physical_axes_in_zyx_storage() -> None:
    z, y, x = jnp.indices((3, 4, 5), dtype=jnp.float32)
    gx, gy, gz = grad3(7 * z + 3 * y + 2 * x)
    np.testing.assert_array_equal(gx[:, :, :-1], 2)
    np.testing.assert_array_equal(gy[:, :-1], 3)
    np.testing.assert_array_equal(gz[:-1], 7)
    np.testing.assert_array_equal(gx[:, :, -1], 0)
    np.testing.assert_array_equal(gy[:, -1], 0)
    np.testing.assert_array_equal(gz[-1], 0)
    # Solver-private (x, y, z) storage gives the same components and divergence.
    u, p = 7 * z + 3 * y + 2 * x * x, jnp.sin(z + 2 * y - x)
    private = [g.transpose(2, 1, 0) for g in grad3(u.transpose(2, 1, 0), z_fast=True)]
    for got, want in zip(private, grad3(u), strict=True):
        np.testing.assert_array_equal(got, want)
    div = div3(*(q.transpose(2, 1, 0) for q in (p, 2 * p, 3 * p)), z_fast=True)
    np.testing.assert_array_equal(div.transpose(2, 1, 0), div3(p, 2 * p, 3 * p))


def test_spdhg_one_iteration_smoke_is_finite_and_positive() -> None:
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        angles=np.asarray([0.0, 90.0], dtype=np.float32),
    )
    projections = jnp.ones((2, 2, 2), dtype=jnp.float32)

    volume, info = spdhg_tv(
        geometry,
        grid,
        detector,
        projections,
        config=SPDHGConfig(
            iterations=1,
            tv_weight=0.0,
            views_per_batch=1,
            seed=123,
            tau=0.1,
            sigma_data=0.1,
            sigma_tv=0.1,
            nonnegative=True,
            log_every=1,
        ),
    )

    assert volume.shape == (2, 2, 2)
    assert bool(jnp.all(jnp.isfinite(volume)))
    assert float(jnp.min(volume)) >= 0.0
    assert info["loss"] == [4.0]
    assert info["A_norm"] is None
    assert info["regularizer"] == "tv"
    assert info["views_per_batch"] == 1
    assert info["num_blocks"] == 2
