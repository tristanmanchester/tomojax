from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

import tomojax.recon.fista_tv as fista_mod
from tomojax.recon.spdhg_tv import _prox_fstar_l2


def test_prox_fstar_l2_matches_closed_form_for_nonuniform_weights():
    u = jnp.array([2.0, -1.0, 0.5, 3.0], dtype=jnp.float32)
    sigma = 0.5
    y = jnp.array([1.0, -2.0, 4.0, 7.0], dtype=jnp.float32)
    w = jnp.array([3.0, 0.25, 10.0, 0.0], dtype=jnp.float32)

    got = _prox_fstar_l2(u, sigma, y, w)
    expected = jnp.where(
        w > 0,
        (u - sigma * y) * w / (sigma + w),
        0.0,
    ).astype(u.dtype)

    assert jnp.allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_fista_reports_objective_at_primal_iterate_not_momentum(monkeypatch: pytest.MonkeyPatch):
    target = jnp.array([[[1.0]]], dtype=jnp.float32)

    def fake_grad_data_term(
        geometry,
        grid,
        detector,
        projections,
        x,
        *,
        views_per_batch=None,
        projector_unroll=1,
        checkpoint_projector=True,
        gather_dtype="fp32",
        grad_mode="auto",
        T_all=None,
        vol_mask=None,
    ):
        resid = x - target
        loss = 0.5 * jnp.vdot(resid, resid).real
        return resid, loss

    monkeypatch.setattr(fista_mod, "grad_data_term", fake_grad_data_term)
    monkeypatch.setattr(
        fista_mod,
        "data_term_value",
        lambda geometry, grid, detector, projections, x, **kwargs: fake_grad_data_term(
            geometry, grid, detector, projections, x, **kwargs
        )[1],
    )
    monkeypatch.setattr(fista_mod, "power_method_L", lambda *args, **kwargs: 2.0)

    class DummyGeom:
        def pose_for_view(self, i: int):
            return jnp.eye(4, dtype=jnp.float32)

    grid = SimpleNamespace(nx=1, ny=1, nz=1)
    detector = SimpleNamespace()
    projections = jnp.zeros((1, 1, 1), dtype=jnp.float32)

    _, info = fista_mod.fista_tv(
        DummyGeom(),
        grid,
        detector,
        projections,
        iters=2,
        lambda_tv=0.0,
        tv_prox_iters=1,
    )

    # With L=2 and x0=0, the iterates are:
    # k=0: z0=0      -> x1 = 0.5
    # k=1: z1=0.5    -> x2 = 0.75
    # The true objective at the second primal iterate is 0.5 * (0.75 - 1)^2 = 0.03125.
    # The buggy implementation reported f(z1)=0.5 * (0.5 - 1)^2 = 0.125 instead.
    assert info["loss"][1] == pytest.approx(0.03125, rel=1e-6, abs=1e-6)
