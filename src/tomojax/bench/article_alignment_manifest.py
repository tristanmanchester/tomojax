"""Manifest helpers for article alignment benchmark runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from tomojax.bench.article_alignment_runs import (
    ArticleRunProfile,
    ArticleScenario,
    article_phantom_metadata,
    article_theta_span_deg,
)

_POSE_DOFS = {"alpha", "beta", "phi", "dx", "dz"}


def article_scenario_truth_payload(scenario: ArticleScenario) -> dict[str, float]:
    """Return the synthetic hidden setup perturbations for manifest/report output."""
    return {
        "det_u_px": float(scenario.hidden_det_u_px),
        "det_v_px": float(scenario.hidden_det_v_px),
        "detector_roll_deg": float(scenario.hidden_detector_roll_deg),
        "axis_rot_x_deg": float(scenario.hidden_axis_rot_x_deg),
        "axis_rot_y_deg": float(scenario.hidden_axis_rot_y_deg),
        "nominal_tilt_deg": float(scenario.nominal_tilt_deg),
        "true_tilt_deg": float(scenario.true_tilt_deg),
    }


def article_scenario_supplied_payload(scenario: ArticleScenario) -> dict[str, float]:
    """Return non-null user/metadata corrections supplied for a scenario."""
    supplied: dict[str, float] = {}
    for name in (
        "det_u_px",
        "det_v_px",
        "detector_roll_deg",
        "axis_rot_x_deg",
        "axis_rot_y_deg",
    ):
        value = getattr(scenario, f"supplied_{name}")
        if value is not None:
            supplied[name] = float(value)
    return supplied


def article_scenario_catalog_payload(scenario: ArticleScenario) -> dict[str, Any]:
    """Return the stable scenario taxonomy payload recorded with visual artifacts."""
    active_dofs = tuple(scenario.active_dofs or scenario.geometry_dofs)
    return {
        "scenario_category": scenario.scenario_category,
        "scenario_family": scenario.scenario_family,
        "expectation": scenario.expectation,
        "expected_status": list(scenario.expected_status),
        "headline_eligible": bool(scenario.headline_eligible),
        "phantom_key": scenario.phantom_key,
        "schedule": scenario.schedule,
        "expected_objective": scenario.expected_objective,
        "expected_optimizer": scenario.expected_optimizer,
        "expected_loss": scenario.expected_loss,
        "active_dofs": list(active_dofs),
        "active_pose_dofs": [dof for dof in active_dofs if dof in _POSE_DOFS],
        "active_geometry_dofs": list(scenario.geometry_dofs),
    }


def build_article_run_manifest(
    profile: ArticleRunProfile,
    scenarios: Sequence[ArticleScenario],
    *,
    suite_name: str = "default",
    generator: str = "scripts/generate_alignment_before_after_128.py",
) -> dict[str, Any]:
    """Build the top-level manifest for article before/after alignment runs."""
    return {
        "schema_version": 1,
        "generator": generator,
        "purpose": "geometry_block_before_after_taxonomy",
        "phantom": article_phantom_metadata(),
        "profile": asdict(profile),
        "scenario_set": suite_name,
        "suite_name": suite_name,
        "scenarios": [
            {
                "slug": s.slug,
                "title": s.title,
                "description": s.description,
                "scenario_category": s.scenario_category,
                "scenario_family": s.scenario_family,
                "suite_name": suite_name,
                "expectation": s.expectation,
                "expected_status": list(s.expected_status),
                "headline_eligible": bool(s.headline_eligible),
                "phantom_key": s.phantom_key,
                "schedule": s.schedule,
                "expected_objective": s.expected_objective,
                "expected_optimizer": s.expected_optimizer,
                "expected_loss": s.expected_loss,
                "geometry_type": s.geometry_type,
                "geometry_dofs": list(s.geometry_dofs),
                "active_dofs": list(s.active_dofs or s.geometry_dofs),
                "active_pose_dofs": [dof for dof in (s.active_dofs or ()) if dof in _POSE_DOFS],
                "active_geometry_dofs": list(s.geometry_dofs),
                "theta_span_deg": article_theta_span_deg(s),
                "n_views": int(profile.views),
                "hidden_truth": article_scenario_truth_payload(s),
                "supplied_corrections": article_scenario_supplied_payload(s),
            }
            for s in scenarios
        ],
        "gauge_notes": {
            "det_u_px": (
                "Canonical detector/ray-grid centre offset in native detector pixels under the "
                "detector_ray_grid_center gauge; not a standalone physical COR proof."
            ),
        },
    }


__all__ = [
    "article_scenario_catalog_payload",
    "article_scenario_supplied_payload",
    "article_scenario_truth_payload",
    "build_article_run_manifest",
]
