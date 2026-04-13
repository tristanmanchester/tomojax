from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.projector import forward_project_view_T, get_detector_grid_device
import tomojax.recon.fista_tv as fista_mod
from tomojax.recon.spdhg_tv import _estimate_norm_A2, _prox_fstar_l2


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


def _make_small_parallel_case(nx=6, ny=6, nz=6, n_views=4):
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=nx, nv=nz, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = np.linspace(0.0, 180.0, n_views, endpoint=False)
    geometry = ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas)
    T_all = jnp.stack(
        [jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32) for i in range(n_views)],
        axis=0,
    )
    return geometry, grid, detector, T_all


def _reference_power_method_L(
    geometry,
    grid,
    detector,
    projections_shape,
    *,
    iters,
    views_per_batch,
    projector_unroll,
    checkpoint_projector,
    gather_dtype,
    grad_mode,
    T_all,
):
    n_views, nv, nu = projections_shape
    zero_proj = jnp.zeros((n_views, nv, nu), dtype=jnp.float32)
    v = jnp.ones((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    v = v / (jnp.linalg.norm(v.ravel()) + 1e-12)
    for _ in range(max(1, int(iters))):
        g, _ = fista_mod.grad_data_term(
            geometry,
            grid,
            detector,
            zero_proj,
            v,
            views_per_batch=views_per_batch,
            projector_unroll=projector_unroll,
            checkpoint_projector=checkpoint_projector,
            gather_dtype=gather_dtype,
            grad_mode=grad_mode,
            T_all=T_all,
        )
        v = g / (jnp.linalg.norm(g.ravel()) + 1e-12)
    g, _ = fista_mod.grad_data_term(
        geometry,
        grid,
        detector,
        zero_proj,
        v,
        views_per_batch=views_per_batch,
        projector_unroll=projector_unroll,
        checkpoint_projector=checkpoint_projector,
        gather_dtype=gather_dtype,
        grad_mode=grad_mode,
        T_all=T_all,
    )
    return max(float(jnp.vdot(v, g).real), 1e-6)


def _reference_estimate_norm_A2(
    geometry,
    grid,
    detector,
    projections_shape,
    T_all,
    *,
    views_per_batch,
    projector_unroll,
    checkpoint_projector,
    gather_dtype,
    key,
    power_iters,
    safety,
):
    n_views, nv, nu = projections_shape
    det_grid = get_detector_grid_device(detector)

    def A_apply(vol, T_chunk):
        vm_project = jax.vmap(
            lambda T, v: forward_project_view_T(
                T,
                grid,
                detector,
                v,
                use_checkpoint=checkpoint_projector,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
            ),
            in_axes=(0, None),
        )
        return vm_project(T_chunk, vol)

    b = int(max(1, min(views_per_batch, n_views)))
    m = (n_views + b - 1) // b

    def AtranA(v):
        def body(g_acc, i):
            i = jnp.int32(i)
            start = i * jnp.int32(b)
            remaining = jnp.maximum(0, jnp.int32(n_views) - start)
            valid = jnp.minimum(jnp.int32(b), remaining)
            shift = jnp.int32(b) - valid
            start_shifted = jnp.maximum(0, start - shift)

            T_chunk = jax.lax.dynamic_slice(T_all, (start_shifted, 0, 0), (b, 4, 4))
            pred_fun = lambda vol: A_apply(vol, T_chunk)
            proj = pred_fun(v)
            idx = jnp.arange(b)
            mask = (idx >= (jnp.int32(b) - valid))[:, None, None]
            proj = proj * mask

            _, vjp = jax.vjp(lambda vv: pred_fun(vv).ravel(), v)
            g_chunk = vjp(proj.ravel())[0]
            return g_acc + g_chunk, None

        g0 = jnp.zeros_like(v)
        g_final, _ = jax.lax.scan(body, g0, jnp.arange(m))
        return g_final

    v = jax.random.normal(key, (grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    v = v / (jnp.linalg.norm(v) + 1e-12)
    for _ in range(max(1, int(power_iters))):
        w = AtranA(v)
        v = w / (jnp.linalg.norm(w) + 1e-12)
    Aw = AtranA(v)
    return max(float(jnp.vdot(v, Aw).real) * float(safety ** 2), 1e-6)


def test_power_method_L_matches_python_loop_reference():
    geometry, grid, detector, T_all = _make_small_parallel_case()
    projections_shape = (T_all.shape[0], detector.nv, detector.nu)

    got = fista_mod.power_method_L(
        geometry,
        grid,
        detector,
        projections_shape,
        iters=3,
        views_per_batch=1,
        projector_unroll=1,
        checkpoint_projector=True,
        gather_dtype="fp32",
        grad_mode="stream",
        T_all=T_all,
    )
    expected = _reference_power_method_L(
        geometry,
        grid,
        detector,
        projections_shape,
        iters=3,
        views_per_batch=1,
        projector_unroll=1,
        checkpoint_projector=True,
        gather_dtype="fp32",
        grad_mode="stream",
        T_all=T_all,
    )

    assert got == pytest.approx(expected, rel=5e-5, abs=1e-5)


def test_estimate_norm_A2_matches_python_loop_reference():
    geometry, grid, detector, T_all = _make_small_parallel_case()
    projections_shape = (T_all.shape[0], detector.nv, detector.nu)
    key = jax.random.PRNGKey(7)

    got = _estimate_norm_A2(
        geometry,
        grid,
        detector,
        projections_shape,
        T_all,
        views_per_batch=2,
        projector_unroll=1,
        checkpoint_projector=True,
        gather_dtype="fp32",
        key=key,
        power_iters=4,
        safety=1.05,
    )
    expected = _reference_estimate_norm_A2(
        geometry,
        grid,
        detector,
        projections_shape,
        T_all,
        views_per_batch=2,
        projector_unroll=1,
        checkpoint_projector=True,
        gather_dtype="fp32",
        key=key,
        power_iters=4,
        safety=1.05,
    )

    assert got == pytest.approx(expected, rel=5e-5, abs=1e-5)
