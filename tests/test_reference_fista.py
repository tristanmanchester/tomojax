from __future__ import annotations

# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
import csv
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.forward import (
    ResidualFilterConfig,
    core_projection_geometry_from_state,
    project_parallel_reference,
)
from tomojax.geometry import GeometryState
from tomojax.recon import (
    ReferenceFISTAConfig,
    build_det_u_gauge_mode,
    build_scout_support,
    centered_volume_support,
    fista_reconstruct_reference,
    project_det_u_gauge_component,
    reconstruct_backprojection_reference,
    reference_fista_diagnostic_artifacts,
    reference_fista_returned_quality,
    write_fista_trace_csv,
    write_fista_trace_recomputed_csv,
)

# check-public-imports: allow-private
from tomojax.recon._fista_reference import (
    _center_l2,
    _loss_and_explicit_gradient,
    _resolve_view_chunk_size,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_reference_fista_reduces_projection_loss_and_keeps_nonnegative() -> None:
    truth = _tiny_volume()
    geometry = GeometryState.zeros(2)
    projections = project_parallel_reference(truth, geometry)
    initial = jnp.zeros_like(truth)

    result = fista_reconstruct_reference(
        projections,
        geometry,
        initial_volume=initial,
        config=ReferenceFISTAConfig(iterations=6, step_size=5e-3, tv_weight=1e-3),
    )

    assert len(result.trace) == 6
    assert result.trace[-1].loss < result.trace[0].loss
    assert result.trace[-1].backend == "core_trilinear_ray_explicit_adjoint"
    assert float(jnp.min(result.volume)) >= 0.0
    assert float(jnp.sum(result.volume)) > 0.0


def test_reference_fista_warm_start_and_trace_csv(tmp_path: Path) -> None:
    truth = _tiny_volume()
    geometry = GeometryState.zeros(1)
    projections = project_parallel_reference(truth, geometry)
    warm_start = jnp.ones_like(truth) * 0.1

    result = fista_reconstruct_reference(
        projections,
        geometry,
        initial_volume=warm_start,
        config=ReferenceFISTAConfig(iterations=2, step_size=5e-3, non_negative=False),
    )
    trace_path = write_fista_trace_csv(result, tmp_path / "fista_trace.csv")

    assert trace_path.exists()
    assert result.volume.shape == warm_start.shape
    assert not np.allclose(np.asarray(result.volume), np.asarray(warm_start))
    with trace_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["iteration"] for row in rows] == ["0", "1"]
    assert rows[0]["backend"] == "core_trilinear_ray_explicit_adjoint"


def test_reference_backprojection_uses_geometry_and_preserves_shape() -> None:
    truth = _tiny_volume()
    geometry = GeometryState.zeros(2)
    projections = project_parallel_reference(truth, geometry)

    volume = reconstruct_backprojection_reference(projections, geometry, depth=truth.shape[1])

    assert volume.shape == truth.shape
    assert volume.dtype == jnp.float32
    assert float(jnp.max(volume)) > 0.0
    assert float(jnp.mean(volume)) <= float(jnp.max(projections))
    reprojection = project_parallel_reference(volume, geometry)
    assert reprojection.shape == projections.shape


def test_reference_fista_projects_candidate_and_momentum_into_support() -> None:
    truth = jnp.ones((6, 6, 6), dtype=jnp.float32)
    geometry = GeometryState.zeros(2)
    projections = project_parallel_reference(truth, geometry)
    support = centered_volume_support((6, 6, 6), kind="cylindrical", radius_fraction=0.35)
    warm_start = jnp.ones_like(truth)

    result = fista_reconstruct_reference(
        projections,
        geometry,
        initial_volume=warm_start,
        volume_support=support,
        config=ReferenceFISTAConfig(iterations=3, step_size=2e-3),
    )

    outside_support = jnp.where(support > 0, 0.0, result.volume)
    assert float(jnp.max(jnp.abs(outside_support))) == 0.0
    assert float(jnp.min(result.volume)) >= 0.0


def test_reference_fista_center_l2_penalty_enters_regulariser() -> None:
    geometry = GeometryState.zeros(1)
    observed = jnp.zeros((1, 4, 4), dtype=jnp.float32)
    warm_start = jnp.zeros((4, 4, 4), dtype=jnp.float32).at[3, :, :].set(1.0)

    result = fista_reconstruct_reference(
        observed,
        geometry,
        initial_volume=warm_start,
        config=ReferenceFISTAConfig(
            iterations=1,
            step_size=1.0e-3,
            tv_weight=0.0,
            center_l2_weight=2.0,
        ),
    )

    assert result.trace[0].regulariser > 0.0
    assert result.config.center_l2_weight == 2.0


def test_reference_fista_center_l2_uses_core_x_y_axes() -> None:
    axis0_offset = jnp.zeros((4, 4, 4), dtype=jnp.float32).at[3, :, :].set(1.0)
    axis2_offset = jnp.zeros((4, 4, 4), dtype=jnp.float32).at[:, :, 3].set(1.0)

    assert float(_center_l2(axis0_offset)) > 0.0
    assert float(_center_l2(axis2_offset)) == 0.0


def test_reference_fista_auto_batch_uses_memory_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = GeometryState.zeros(128)
    observed = jnp.zeros((128, 256, 256), dtype=jnp.float32)
    volume = jnp.zeros((256, 256, 256), dtype=jnp.float32)
    volume_shape = (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))
    core = core_projection_geometry_from_state(volume_shape, geometry, detector_shape=(256, 256))
    calls: list[dict[str, object]] = []

    def fake_estimate_views_per_batch_info(**kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        return SimpleNamespace(views_per_batch=2)

    monkeypatch.setattr(
        "tomojax.recon._fista_reference.estimate_views_per_batch_info",
        fake_estimate_views_per_batch_info,
    )

    resolved = _resolve_view_chunk_size(
        volume,
        observed,
        core=core,
        config=ReferenceFISTAConfig(views_per_batch=0),
    )

    assert resolved == 2
    assert calls[0]["n_views"] == 128
    assert calls[0]["grid_nxyz"] == (256, 256, 256)
    assert calls[0]["det_nuv"] == (256, 256)
    assert calls[0]["fallback_batch"] == 1


def test_centered_volume_support_generates_cylinder_and_sphere() -> None:
    cylinder = centered_volume_support((5, 7, 9), kind="cylindrical", radius_fraction=0.5)
    sphere = centered_volume_support((5, 7, 9), kind="spherical", radius_fraction=0.5)

    assert cylinder.shape == (5, 7, 9)
    assert sphere.shape == (5, 7, 9)
    assert cylinder.dtype == jnp.bool_
    assert sphere.dtype == jnp.bool_
    assert bool(cylinder[2, 3, 4])
    assert bool(sphere[2, 3, 4])
    assert bool(cylinder[2, 3, 0])
    assert bool(cylinder[2, 3, 8])
    assert not bool(cylinder[0, 0, 4])
    assert not bool(sphere[2, 3, 0])
    np.testing.assert_array_equal(np.asarray(cylinder[:, :, 0]), np.asarray(cylinder[:, :, 8]))
    assert int(jnp.sum(sphere)) < int(jnp.sum(cylinder))


@pytest.mark.parametrize(
    ("config", "mask"),
    [
        (
            ReferenceFISTAConfig(
                residual_filters=(ResidualFilterConfig(kind="raw"),),
                non_negative=False,
            ),
            None,
        ),
        (
            ReferenceFISTAConfig(
                residual_filters=(ResidualFilterConfig(kind="raw"),),
                non_negative=False,
            ),
            jnp.ones((2, 4, 4), dtype=jnp.float32).at[:, 0, :].set(0.0),
        ),
        (
            ReferenceFISTAConfig(
                residual_filters=(ResidualFilterConfig(kind="lowpass_gaussian", sigma_px=0.8),),
                non_negative=False,
            ),
            None,
        ),
        (
            ReferenceFISTAConfig(
                residual_filters=(ResidualFilterConfig(kind="lowpass_gaussian", sigma_px=0.8),),
                tv_weight=1.0e-3,
                center_l2_weight=0.2,
                non_negative=False,
            ),
            jnp.ones((2, 4, 4), dtype=jnp.float32).at[:, :, 0].set(0.0),
        ),
    ],
)
def test_reference_fista_explicit_gradient_matches_finite_difference(
    config: ReferenceFISTAConfig,
    mask: jnp.ndarray | None,
) -> None:
    geometry = GeometryState.zeros(2)
    truth = _tiny_volume()
    observed = project_parallel_reference(truth, geometry)
    volume: jnp.ndarray = truth * 0.7 + jnp.linspace(
        0.0, 0.2, truth.size, dtype=jnp.float32
    ).reshape(truth.shape)
    direction: jnp.ndarray = jnp.cos(jnp.arange(truth.size, dtype=jnp.float32)).reshape(truth.shape)
    direction = cast("jnp.ndarray", direction / jnp.linalg.norm(direction))
    volume_shape = (
        int(volume.shape[0]),
        int(volume.shape[1]),
        int(volume.shape[2]),
    )
    core = core_projection_geometry_from_state(
        volume_shape,
        geometry,
        detector_shape=(int(observed.shape[1]), int(observed.shape[2])),
    )

    loss, _data, _regulariser, explicit_gradient = _loss_and_explicit_gradient(
        volume,
        observed,
        core=core,
        mask=mask,
        config=config,
    )
    epsilon = jnp.asarray(1.0e-2, dtype=jnp.float32)
    plus, _plus_data, _plus_regulariser, _plus_grad = _loss_and_explicit_gradient(
        volume + epsilon * direction,
        observed,
        core=core,
        mask=mask,
        config=config,
    )
    minus, _minus_data, _minus_regulariser, _minus_grad = _loss_and_explicit_gradient(
        volume - epsilon * direction,
        observed,
        core=core,
        mask=mask,
        config=config,
    )

    finite_difference = (plus - minus) / (2.0 * epsilon)
    directional_gradient = jnp.sum(explicit_gradient * direction)

    assert float(loss) > 0.0
    np.testing.assert_allclose(
        float(directional_gradient),
        float(finite_difference),
        rtol=2.0e-2,
        atol=2.0e-3,
    )


def test_reference_fista_diagnostics_lock_scalar_gradient_contract(tmp_path: Path) -> None:
    diagnostics = reference_fista_diagnostic_artifacts()

    assert diagnostics.fista_gradient_checks["schema"] == "tomojax.fista_gradient_checks.v1"
    assert diagnostics.fista_gradient_checks["status"] == "passed"
    gradient_cases = cast("list[dict[str, object]]", diagnostics.fista_gradient_checks["cases"])
    assert {case["name"] for case in gradient_cases} == {
        "raw_valid_mask",
        "lowpass_boundary_mask",
        "dog_tv_center",
    }
    assert all(case["passed"] is True for case in gradient_cases)
    assert (
        cast("dict[str, object]", diagnostics.fista_gradient_checks["support_check"])["passed"]
        is True
    )

    assert diagnostics.adjoint_checks["schema"] == "tomojax.adjoint_checks.v1"
    assert diagnostics.adjoint_checks["status"] == "passed"
    assert diagnostics.geometry_jvp_vjp_checks["schema"] == ("tomojax.geometry_jvp_vjp_checks.v1")
    assert diagnostics.geometry_jvp_vjp_checks["status"] == "passed"
    assert diagnostics.loss_normalisation_report["schema"] == (
        "tomojax.loss_normalisation_report.v1"
    )
    assert diagnostics.loss_normalisation_report["current_contract"] == (
        "full_projection_array_size"
    )
    assert (
        diagnostics.loss_normalisation_report["reference_matches_array_pixel_normalisation"] is True
    )

    trace_path = write_fista_trace_recomputed_csv(
        diagnostics.fista_trace_recomputed_rows,
        tmp_path / "fista_trace_recomputed.csv",
    )
    with trace_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["trace_loss_point"] for row in rows] == ["momentum_point", "momentum_point"]
    assert all(row["recomputed_loss_point"] == "returned_final_volume" for row in rows)


def test_reference_fista_returned_quality_reports_candidate_loss() -> None:
    geometry = GeometryState.zeros(2)
    truth = _tiny_volume()
    observed = project_parallel_reference(truth, geometry)
    support = centered_volume_support(
        cast("tuple[int, int, int]", tuple(int(dim) for dim in truth.shape)),
        kind="cylindrical",
    )
    result = fista_reconstruct_reference(
        observed,
        geometry,
        initial_volume=None,
        volume_support=support,
        mask=jnp.ones_like(observed),
        config=ReferenceFISTAConfig(iterations=2, step_size=2e-3),
    )

    quality = reference_fista_returned_quality(
        result,
        observed,
        geometry,
        volume_support=support,
        mask=jnp.ones_like(observed),
    )

    assert quality.returned_candidate_loss >= 0.0
    assert quality.returned_data_loss >= 0.0
    assert quality.returned_regularizer >= 0.0
    assert quality.projected_gradient_norm >= 0.0
    assert quality.volume_rms >= 0.0
    assert quality.support_mass_fraction is not None
    assert quality.loss_normalisation == "full_projection_array_size"


def test_reference_fista_soft_support_anchor_gradient_matches_finite_difference() -> None:
    geometry = GeometryState.zeros(2)
    truth = _tiny_volume()
    observed = project_parallel_reference(truth, geometry)
    support = jnp.ones_like(truth).at[0, :, :].set(0.0)
    anchor = truth * 0.25
    config = ReferenceFISTAConfig(
        iterations=1,
        step_size=2e-3,
        residual_filters=(ResidualFilterConfig(kind="raw"),),
        soft_support=support,
        soft_support_outside_weight=0.3,
        low_frequency_anchor=anchor,
        low_frequency_anchor_weight=0.2,
        low_frequency_anchor_radius=1,
        non_negative=False,
    )
    volume = truth * 0.7 + 0.1
    direction = jnp.sin(jnp.arange(truth.size, dtype=jnp.float32)).reshape(truth.shape)
    direction = cast("jnp.ndarray", direction / jnp.linalg.norm(direction))
    core = core_projection_geometry_from_state(
        cast("tuple[int, int, int]", tuple(int(dim) for dim in truth.shape)),
        geometry,
        detector_shape=(int(observed.shape[1]), int(observed.shape[2])),
    )

    _loss, _data, _regularizer, gradient = _loss_and_explicit_gradient(
        volume,
        observed,
        core=core,
        mask=jnp.ones_like(observed),
        config=config,
    )
    epsilon = jnp.asarray(1.0e-2, dtype=jnp.float32)
    plus, _plus_data, _plus_regularizer, _plus_grad = _loss_and_explicit_gradient(
        volume + epsilon * direction,
        observed,
        core=core,
        mask=jnp.ones_like(observed),
        config=config,
    )
    minus, _minus_data, _minus_regularizer, _minus_grad = _loss_and_explicit_gradient(
        volume - epsilon * direction,
        observed,
        core=core,
        mask=jnp.ones_like(observed),
        config=config,
    )

    np.testing.assert_allclose(
        float(jnp.sum(gradient * direction)),
        float((plus - minus) / (2.0 * epsilon)),
        rtol=2.0e-2,
        atol=2.0e-3,
    )


def test_scout_support_builder_records_truth_free_provenance() -> None:
    geometry = GeometryState.zeros(2)
    truth = _tiny_volume()
    observed = project_parallel_reference(truth, geometry)

    scout = build_scout_support(
        observed,
        geometry,
        projection_valid_mask=jnp.ones_like(observed),
        volume_shape=cast("tuple[int, int, int]", tuple(int(dim) for dim in truth.shape)),
        smoothing_radius=1,
        dilation_radius=1,
    )

    assert scout.support_probability.shape == truth.shape
    assert scout.low_frequency_anchor.shape == truth.shape
    assert scout.provenance["uses_truth"] is False
    assert scout.provenance["geometry_source"] == "initial_metadata"
    assert float(jnp.min(scout.support_probability)) >= 0.0
    assert float(jnp.max(scout.support_probability)) <= 1.0


def test_det_u_gauge_mode_builder_and_penalty_are_truth_free() -> None:
    geometry = GeometryState.zeros(2)
    truth = _tiny_volume()
    observed = project_parallel_reference(truth, geometry)
    mode = build_det_u_gauge_mode(
        truth,
        geometry,
        projection_valid_mask=jnp.ones_like(observed),
        cg_max_iterations=2,
    )
    config = ReferenceFISTAConfig(
        iterations=1,
        step_size=2e-3,
        residual_filters=(ResidualFilterConfig(kind="raw"),),
        gauge_mode=mode.mode,
        gauge_reference=jnp.zeros_like(truth),
        gauge_mode_weight=0.4,
        non_negative=False,
    )
    core = core_projection_geometry_from_state(
        cast("tuple[int, int, int]", tuple(int(dim) for dim in truth.shape)),
        geometry,
        detector_shape=(int(observed.shape[1]), int(observed.shape[2])),
    )
    loss, _data, regularizer, gradient = _loss_and_explicit_gradient(
        truth,
        observed,
        core=core,
        mask=jnp.ones_like(observed),
        config=config,
    )

    assert mode.provenance["uses_truth"] is False
    assert mode.mode.shape == truth.shape
    assert mode.mode_norm >= 0.0
    assert float(loss) >= 0.0
    assert float(regularizer) >= 0.0
    assert float(jnp.sqrt(jnp.mean(gradient * gradient))) >= 0.0


def test_det_u_gauge_projection_removes_volume_component() -> None:
    geometry = GeometryState.zeros(2)
    truth = _tiny_volume()
    observed = project_parallel_reference(truth, geometry)
    mode = build_det_u_gauge_mode(
        truth,
        geometry,
        projection_valid_mask=jnp.ones_like(observed),
        cg_max_iterations=2,
    )
    contaminated = truth + 0.5 * mode.mode

    projected, report = project_det_u_gauge_component(
        contaminated,
        mode,
        reference=truth,
    )

    coeff_after = jnp.vdot(projected - truth, mode.mode).real / jnp.maximum(
        jnp.vdot(mode.mode, mode.mode).real,
        jnp.asarray(1.0e-12, dtype=jnp.float32),
    )
    assert abs(float(coeff_after)) <= 1.0e-5
    assert abs(report.coefficient_after) <= 1.0e-5
    assert report.transfer_ratio_after <= report.transfer_ratio_before
    assert report.provenance["uses_truth"] is False


def _tiny_volume() -> jnp.ndarray:
    volume = jnp.zeros((4, 4, 4), dtype=jnp.float32)
    volume = volume.at[1, :, 1].set(0.5)
    return volume.at[2, :, 3].set(0.8)
