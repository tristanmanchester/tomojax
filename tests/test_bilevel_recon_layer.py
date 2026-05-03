from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tomojax.align.geometry.geometry_applier import BaseGeometryArrays, apply_alignment_state
import tomojax.align.objectives.recon_layer as recon_layer_module
from tomojax.align.objectives.recon_layer import ReconLayer, ReconLayerConfig
from tomojax.align.model.state import AlignmentState, PoseState, SetupGeometryState
from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.recon.fista_tv import FistaConfig, fista_tv
from tomojax.recon.fista_tv_core import FistaCoreConfig, fista_tv_core_arrays
from tomojax.recon.fista_tv_core import projection_loss_arrays


def _case(size: int = 6, n_views: int = 5):
    grid = Grid(nx=size, ny=size, nz=size, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=size, nv=size, du=1.0, dv=1.0)
    thetas = np.linspace(0.0, 180.0, n_views, endpoint=False, dtype=np.float32)
    geometry = ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas)
    volume = jnp.zeros((size, size, size), dtype=jnp.float32)
    volume = volume.at[2:5, 1:4, 2:5].set(1.0)
    projections = jnp.stack(
        [
            forward_project_view(
                geometry,
                grid,
                detector,
                volume,
                i,
                gather_dtype="fp32",
            )
            for i in range(n_views)
        ],
        axis=0,
    )
    return grid, detector, geometry, volume, projections


def test_fista_core_arrays_matches_public_fista_for_small_fixed_l_case():
    grid, detector, geometry, _volume, projections = _case()
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))
    effective = apply_alignment_state(base, state)
    x0 = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)

    core = fista_tv_core_arrays(
        x0=x0,
        T_all=effective.pose_stack,
        det_grid=effective.det_grid,
        projections=projections,
        grid=grid,
        detector=detector,
        cfg=FistaCoreConfig(iters=2, lambda_tv=0.0, L=100.0, checkpoint_projector=False),
    )
    public, _info = fista_tv(
        geometry,
        grid,
        detector,
        projections,
        init_x=x0,
        config=FistaConfig(iters=2, lambda_tv=0.0, L=100.0, checkpoint_projector=False),
    )

    np.testing.assert_allclose(np.asarray(core.x), np.asarray(public), atol=1e-5, rtol=1e-5)


def test_fista_core_dynamic_l_override_matches_static_l():
    grid, detector, geometry, _volume, projections = _case(size=5, n_views=3)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))
    effective = apply_alignment_state(base, state)
    x0 = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    static = fista_tv_core_arrays(
        x0=x0,
        T_all=effective.pose_stack,
        det_grid=effective.det_grid,
        projections=projections,
        grid=grid,
        detector=detector,
        cfg=FistaCoreConfig(iters=2, lambda_tv=0.0, L=75.0, checkpoint_projector=False),
    )
    dynamic = fista_tv_core_arrays(
        x0=x0,
        T_all=effective.pose_stack,
        det_grid=effective.det_grid,
        projections=projections,
        grid=grid,
        detector=detector,
        cfg=FistaCoreConfig(iters=2, lambda_tv=0.0, L=1.0, checkpoint_projector=False),
        L_override=jnp.asarray(75.0, dtype=jnp.float32),
    )

    np.testing.assert_allclose(np.asarray(dynamic.x), np.asarray(static.x), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(dynamic.loss),
        np.asarray(static.loss),
        atol=1e-5,
        rtol=1e-6,
    )


def test_fista_core_views_per_batch_preserves_reconstruction_values():
    grid, detector, geometry, _volume, projections = _case(n_views=6)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))
    effective = apply_alignment_state(base, state)
    x0 = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)

    streamed = fista_tv_core_arrays(
        x0=x0,
        T_all=effective.pose_stack,
        det_grid=effective.det_grid,
        projections=projections,
        grid=grid,
        detector=detector,
        cfg=FistaCoreConfig(
            iters=2,
            lambda_tv=0.0,
            L=100.0,
            checkpoint_projector=False,
            views_per_batch=1,
        ),
    )
    batched = fista_tv_core_arrays(
        x0=x0,
        T_all=effective.pose_stack,
        det_grid=effective.det_grid,
        projections=projections,
        grid=grid,
        detector=detector,
        cfg=FistaCoreConfig(
            iters=2,
            lambda_tv=0.0,
            L=100.0,
            checkpoint_projector=False,
            views_per_batch=6,
        ),
    )

    np.testing.assert_allclose(np.asarray(streamed.x), np.asarray(batched.x), atol=1e-5, rtol=1e-5)


def test_fista_core_reports_final_data_loss_not_objective():
    grid, detector, geometry, _volume, projections = _case(size=5, n_views=3)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))
    effective = apply_alignment_state(base, state)
    x0 = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    cfg = FistaCoreConfig(
        iters=2,
        lambda_tv=0.01,
        L=100.0,
        checkpoint_projector=False,
    )

    result = fista_tv_core_arrays(
        x0=x0,
        T_all=effective.pose_stack,
        det_grid=effective.det_grid,
        projections=projections,
        grid=grid,
        detector=detector,
        cfg=cfg,
    )
    expected_data_loss = projection_loss_arrays(
        T_all=effective.pose_stack,
        grid=grid,
        detector=detector,
        volume=result.x,
        det_grid=effective.det_grid,
        projections=projections,
        cfg=cfg,
    )

    np.testing.assert_allclose(
        np.asarray(result.data_loss),
        np.asarray(expected_data_loss),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires real Pallas lowering")
def test_fista_core_pallas_backprojector_matches_jax_backprojector():
    grid, detector, geometry, _volume, projections = _case(size=5, n_views=3)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))
    effective = apply_alignment_state(base, state)
    x0 = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)

    def reconstruct(backprojector: str):
        cfg = FistaCoreConfig(
            iters=1,
            lambda_tv=0.001,
            L=5000.0,
            checkpoint_projector=False,
            projector_unroll=2,
            views_per_batch=0,
            backprojector=backprojector,
        )

        @jax.jit
        def run():
            result = fista_tv_core_arrays(
                x0=x0,
                T_all=effective.pose_stack,
                det_grid=effective.det_grid,
                projections=projections,
                grid=grid,
                detector=detector,
                cfg=cfg,
            )
            return result.x, result.loss, result.data_loss

        return run()

    jax_result = reconstruct("jax")
    pallas_result = reconstruct("pallas")

    np.testing.assert_allclose(np.asarray(pallas_result[1]), np.asarray(jax_result[1]))
    np.testing.assert_allclose(np.asarray(pallas_result[2]), np.asarray(jax_result[2]))
    np.testing.assert_allclose(np.asarray(pallas_result[0]), np.asarray(jax_result[0]), atol=1e-5, rtol=1e-5)


def test_recon_layer_unrolled_mode_returns_diagnostics():
    grid, detector, geometry, _volume, projections = _case()
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    layer = ReconLayer(
        base=base,
        grid=grid,
        detector=detector,
        config=ReconLayerConfig(
            iters=3,
            lambda_tv=0.001,
            regulariser="huber_tv",
            L=100.0,
            checkpoint_projector=False,
        ),
    )
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))

    result = layer.reconstruct(state=state, projections=projections)

    assert result.x.shape == (grid.nx, grid.ny, grid.nz)
    assert result.info["differentiation_mode"] == "unrolled"
    assert result.info["inner_regulariser"] == "huber_tv"
    assert result.info["effective_iters"] == 3


def test_recon_layer_passes_views_per_batch_to_core(monkeypatch):
    grid, detector, geometry, _volume, projections = _case()
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    captured: dict[str, int] = {}

    def fake_core(*, x0, T_all, det_grid, projections, grid, detector, cfg, view_weights=None):
        del T_all, det_grid, projections, grid, detector, view_weights
        captured["views_per_batch"] = int(cfg.views_per_batch)
        return recon_layer_module.FistaCoreResult(
            x=jnp.asarray(x0),
            loss=jnp.zeros((int(cfg.iters),), dtype=jnp.float32),
            data_loss=jnp.asarray(0.0, dtype=jnp.float32),
            regulariser_value=jnp.asarray(0.0, dtype=jnp.float32),
            effective_iters=jnp.asarray(int(cfg.iters), dtype=jnp.int32),
            status="ok",
        )

    monkeypatch.setattr(recon_layer_module, "fista_tv_core_arrays", fake_core)
    layer = ReconLayer(
        base=base,
        grid=grid,
        detector=detector,
        config=ReconLayerConfig(
            iters=1,
            lambda_tv=0.0,
            L=100.0,
            checkpoint_projector=False,
            views_per_batch=3,
        ),
    )
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))

    layer.reconstruct(state=state, projections=projections)

    assert captured["views_per_batch"] == 3


def test_recon_layer_is_differentiable_through_detector_center():
    grid, detector, geometry, _volume, projections = _case()
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    layer = ReconLayer(
        base=base,
        grid=grid,
        detector=detector,
        config=ReconLayerConfig(
            iters=2,
            lambda_tv=0.0,
            L=100.0,
            checkpoint_projector=False,
        ),
    )

    def objective(det_u: jnp.ndarray) -> jnp.ndarray:
        state = AlignmentState(
            setup=SetupGeometryState(det_u_px=det_u),
            pose=PoseState.zeros(projections.shape[0]),
        )
        return jnp.sum(layer.reconstruct(state=state, projections=projections).x)

    grad = jax.grad(objective)(jnp.asarray(0.0, dtype=jnp.float32))

    assert jnp.isfinite(grad)


def test_recon_layer_rejects_nonsmooth_tv_for_differentiable_reference_mode():
    grid, detector, geometry, _volume, projections = _case()
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    layer = ReconLayer(
        base=base,
        grid=grid,
        detector=detector,
        config=ReconLayerConfig(iters=1, lambda_tv=0.001, regulariser="tv"),
    )
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))

    with pytest.raises(ValueError, match="require huber_tv"):
        layer.reconstruct(state=state, projections=projections)


def test_recon_layer_implicit_mode_returns_cg_adjoint_diagnostics():
    grid, detector, geometry, _volume, projections = _case()
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    layer = ReconLayer(
        base=base,
        grid=grid,
        detector=detector,
        config=ReconLayerConfig(
            iters=2,
            lambda_tv=0.0,
            L=100.0,
            differentiation_mode="implicit",
            checkpoint_projector=False,
            implicit_cg_iters=8,
        ),
    )
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))

    result = layer.reconstruct(state=state, projections=projections)

    assert result.x.shape == (grid.nx, grid.ny, grid.nz)
    assert result.info["differentiation_mode"] == "implicit"
    assert result.info["implicit_gradient_status"] == "cg_adjoint"


def test_recon_layer_implicit_mode_has_finite_detector_center_gradient():
    grid, detector, geometry, _volume, projections = _case(size=5, n_views=4)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    layer = ReconLayer(
        base=base,
        grid=grid,
        detector=detector,
        config=ReconLayerConfig(
            iters=2,
            lambda_tv=0.0,
            L=100.0,
            differentiation_mode="implicit",
            checkpoint_projector=False,
            implicit_cg_iters=8,
        ),
    )

    def objective(det_u: jnp.ndarray) -> jnp.ndarray:
        state = AlignmentState(
            setup=SetupGeometryState(det_u_px=det_u),
            pose=PoseState.zeros(projections.shape[0]),
        )
        return jnp.sum(layer.reconstruct(state=state, projections=projections).x)

    grad = jax.grad(objective)(jnp.asarray(0.0, dtype=jnp.float32))

    assert jnp.isfinite(grad)
