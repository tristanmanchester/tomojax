"""Train-fold reconstruction inside setup alignment."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment._geometry.geometry_applier import BaseGeometryArrays, apply_alignment_state

# check-public-imports: allow-private
from tomojax.alignment._model.state import AlignmentState

# check-public-imports: allow-private
from tomojax.alignment._objectives import fold_recon

# check-public-imports: allow-private
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import FistaConfig


@pytest.mark.numerical
def test_train_fold_keeps_the_requested_ray_integrator(monkeypatch):
    original = fold_recon.fista_tv

    def host_result(*args, **kwargs):
        volume, info = original(*args, **kwargs)
        return np.asarray(volume), info

    monkeypatch.setattr(fold_recon, "fista_tv", host_result)
    grid, detector = Grid(2, 1, 1, 0.8, 1.1, 1.3), Detector(4, 3, 0.7, 0.9)
    geometry = ParallelGeometry(grid, detector, [13.0, 71.0, 127.0])
    state = AlignmentState.zeros(n_views=3)
    effective = apply_alignment_state(BaseGeometryArrays.from_geometry(geometry, detector), state)

    def project(flat):
        return jax.vmap(
            lambda t: forward_project_view_T(
                t, grid, detector, flat.reshape(1, 1, 2), ray_integrator="exact"
            )
        )(effective.pose_stack)

    matrix = np.asarray(jax.jacfwd(project)(jnp.zeros(2)), np.float64)
    truth = np.asarray([0.2, 1.0])
    result, _ = fold_recon.reconstruct_train_fold_nograd(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=jnp.asarray(np.einsum("...i,i->...", matrix, truth), jnp.float32),
        state=state,
        train_idx=jnp.asarray([0, 1, 2]),
        train_mask=jnp.asarray([1, 1, 1]),
        init_x=None,
        level_factor=1,
        cfg=FistaConfig(
            iterations=200,
            tv_weight=0,
            regulariser="huber_tv",
            huber_delta=0.01,
            tv_prox_iterations=1,
            nonnegative=False,
            lipschitz=float(np.linalg.norm(matrix.reshape(-1, 2), 2) ** 2 * 1.01),
            ray_integrator="exact",
        ),
    )
    assert isinstance(result, jax.Array)
    np.testing.assert_allclose(np.asarray(result).ravel(), truth, rtol=2e-4, atol=2e-5)


@pytest.mark.numerical
def test_full_alignment_setup_volume_follows_the_coarse_motion(monkeypatch):
    # Level 1's setup volume seeds its pose fit, so it must use level 2's motion.
    import tomojax as tj
    from tomojax.alignment.api import AlignConfig

    volumes = []

    def record(**kwargs):
        result = fold_recon.reconstruct_train_fold_nograd(**kwargs)
        if np.size(kwargs["train_idx"]) == len(kwargs["projections"]):
            volumes.append(np.asarray(result[0]))
        return result

    monkeypatch.setattr(
        "tomojax.alignment._stages._setup_stage.reconstruct_train_fold_nograd", record
    )
    angles = np.linspace(0, 180, 60, endpoint=False)
    theta, u = np.deg2rad(angles)[:, None], np.arange(48) - 23.5
    centres = np.random.default_rng(4).uniform(-14, 14, (8, 2))
    config = AlignConfig(outer_iterations=3, iterations=20, ray_integrator="joseph", backend="jax")

    def setup_volume(shifts):
        # Independent analytic line integrals of Gaussians, shifted across the detector.
        rows = sum(
            np.exp(-0.5 * ((u - np.cos(theta) * x + np.sin(theta) * y - shifts) / 3) ** 2)
            for x, y in centres
        )
        data = np.repeat(rows[:, None], 3, axis=1).astype(np.float32)
        geometry = ParallelGeometry(Grid(40, 40, 3, 1, 1, 1), Detector(48, 3, 1, 1), angles)
        tj.align(tj.Scan(data, geometry), mode="full", levels=(2, 1), config=config)
        return volumes[-1]

    still = setup_volume(0.0)
    moved = setup_volume(np.where(np.arange(60) % 2, 2.0, -2.0)[:, None])
    assert np.linalg.norm(moved - still) < 0.2 * np.linalg.norm(still)
