from __future__ import annotations

# pyright: reportAny=false, reportUnknownMemberType=false
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from tomojax.bench import run_alignment_smoke
from tomojax.forward import project_parallel_reference
from tomojax.geometry import GeometryState
from tomojax.recon import reconstruct_average_reference


def test_projector_preserves_gauge_equivalent_detector_shifts() -> None:
    volume = jnp.zeros((4, 4, 4), dtype=jnp.float32)
    volume = volume.at[0, :, 1].set(1.0)
    geometry = GeometryState.zeros(2)
    pose = geometry.pose.with_updates(dx_px=np.array([1.0, 3.0], dtype=np.float64))
    geometry = GeometryState(setup=geometry.setup, pose=pose)

    canonical = run_alignment_smoke(project_parallel_reference(volume, geometry), geometry)

    before = project_parallel_reference(volume, geometry)
    after = project_parallel_reference(volume, canonical.canonicalized_geometry.state)
    np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=1e-6)


def test_reconstruct_average_reference_returns_stopped_preview_shape() -> None:
    projections = jnp.ones((3, 5, 4), dtype=jnp.float32)

    volume = reconstruct_average_reference(projections, depth=6)

    assert volume.shape == (5, 6, 4)
    np.testing.assert_allclose(np.asarray(volume), 1.0)


def test_run_alignment_smoke_reports_canonical_loss_preservation() -> None:
    truth_volume = jnp.ones((4, 4, 4), dtype=jnp.float32)
    geometry = GeometryState.zeros(3)
    pose = geometry.pose.with_updates(
        dx_px=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        dz_px=np.array([0.0, 1.0, 2.0], dtype=np.float64),
    )
    setup = geometry.setup.replace_parameter(
        "det_v_px",
        replace(geometry.setup.det_v_px, active=True),
    )
    geometry = GeometryState(setup=setup, pose=pose)
    observed = project_parallel_reference(truth_volume, geometry)
    mask = jnp.ones_like(observed)

    report = run_alignment_smoke(observed, geometry, mask=mask)

    assert report.valid_count == float(observed.size)
    np.testing.assert_allclose(report.canonical_loss, report.initial_loss, atol=1e-6)
    np.testing.assert_allclose(
        np.mean(report.canonicalized_geometry.state.pose.dx_px),
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.mean(report.canonicalized_geometry.state.pose.dz_px),
        0.0,
        atol=1e-12,
    )
