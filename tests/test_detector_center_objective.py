from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.align.detector_center import (
    projection_com_det_u_seed,
    train_heldout_view_indices,
)
from tomojax.align.geometry_blocks import GeometryCalibrationState, level_detector_grid
from tomojax.align.losses import L2OtsuLossSpec, build_loss_adapter
from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.recon.fbp import fbp


def _case(size: int = 8, n_views: int = 12, hidden_det_u_px: float = 2.0):
    grid = Grid(nx=size, ny=size, nz=size, vx=1.0, vy=1.0, vz=1.0)
    det_nom = Detector(nu=size, nv=size, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    det_true = Detector(
        nu=size,
        nv=size,
        du=1.0,
        dv=1.0,
        det_center=(float(hidden_det_u_px), 0.0),
    )
    thetas = np.linspace(0.0, 180.0, n_views, endpoint=False, dtype=np.float32)
    geom_nom = ParallelGeometry(grid=grid, detector=det_nom, thetas_deg=thetas)
    geom_true = ParallelGeometry(grid=grid, detector=det_true, thetas_deg=thetas)
    volume_np = np.zeros((size, size, size), dtype=np.float32)
    volume_np[1:5, 2:6, 2:5] = 1.0
    volume_np[5:7, 1:4, 5:7] = 0.7
    volume = jnp.asarray(volume_np)
    projections = jnp.stack(
        [
            forward_project_view(
                geom_true,
                grid,
                det_true,
                volume,
                i,
                gather_dtype="fp32",
            )
            for i in range(n_views)
        ],
        axis=0,
    )
    return grid, det_nom, geom_nom, volume, projections


def _candidate_loss(
    *,
    grid: Grid,
    detector: Detector,
    geometry: ParallelGeometry,
    volume: jnp.ndarray,
    projections: jnp.ndarray,
    candidate_det_u_px: float,
) -> float:
    adapter = build_loss_adapter(L2OtsuLossSpec(), projections)
    state = GeometryCalibrationState(det_u_px=float(candidate_det_u_px))
    det_grid = level_detector_grid(detector, state=state, factor=1)
    preds = jnp.stack(
        [
            forward_project_view(
                geometry,
                grid,
                detector,
                volume,
                i,
                gather_dtype="fp32",
                det_grid=det_grid,
            )
            for i in range(projections.shape[0])
        ],
        axis=0,
    )
    losses = adapter.per_view_loss(
        preds,
        projections,
        adapter.state.mask,
        view_indices=jnp.arange(projections.shape[0], dtype=jnp.int32),
    )
    return float(jnp.sum(losses))


def test_heldout_split_is_deterministic_and_non_empty():
    train, heldout = train_heldout_view_indices(12, holdout_stride=4)

    assert tuple(heldout) == (2, 6, 10)
    assert set(train).isdisjoint(set(heldout))
    assert len(train) + len(heldout) == 12


def test_projection_com_seed_is_finite_for_detector_center_case():
    _grid, _det, geom, _volume, projections = _case()
    adapter = build_loss_adapter(L2OtsuLossSpec(), projections)

    seed = projection_com_det_u_seed(projections, geom, adapter)

    assert seed.status == "ok"
    assert np.isfinite(seed.det_u_px)


@pytest.mark.parametrize("hidden_det_u_px", [-2.0, 2.0])
def test_true_volume_l2_otsu_detector_center_objective_minimizes_near_hidden_offset(
    hidden_det_u_px: float,
):
    grid, detector, geom, volume, projections = _case(hidden_det_u_px=hidden_det_u_px)
    candidates = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    losses = [
        _candidate_loss(
            grid=grid,
            detector=detector,
            geometry=geom,
            volume=volume,
            projections=projections,
            candidate_det_u_px=value,
        )
        for value in candidates
    ]

    assert candidates[int(np.argmin(losses))] == pytest.approx(hidden_det_u_px)


def test_wrong_geometry_fixed_volume_detector_center_objective_is_self_consistent_at_nominal():
    grid, detector, geom, _volume, projections = _case(hidden_det_u_px=2.0)
    nominal_recon = fbp(
        geom,
        grid,
        detector,
        projections,
        views_per_batch=1,
        gather_dtype="fp32",
        checkpoint_projector=False,
    )
    candidates = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    losses = [
        _candidate_loss(
            grid=grid,
            detector=detector,
            geometry=geom,
            volume=nominal_recon,
            projections=projections,
            candidate_det_u_px=value,
        )
        for value in candidates
    ]

    assert candidates[int(np.argmin(losses))] == pytest.approx(0.0)
