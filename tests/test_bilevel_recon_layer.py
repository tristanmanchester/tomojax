from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tomojax.align.geometry_applier import BaseGeometryArrays, apply_alignment_state
from tomojax.align.recon_layer import ReconLayer, ReconLayerConfig
from tomojax.align.state import AlignmentState, PoseState, SetupGeometryState
from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.recon.fista_tv import FistaConfig, fista_tv
from tomojax.recon.fista_tv_core import FistaCoreConfig, fista_tv_core_arrays


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
