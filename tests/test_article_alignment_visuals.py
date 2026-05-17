from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import imageio.v3 as iio
import numpy as np

from tomojax.bench import article_visuals
from tomojax.bench.article_alignment_manifest import (
    article_scenario_catalog_payload,
    article_scenario_supplied_payload,
    article_scenario_truth_payload,
    build_article_run_manifest,
)
from tomojax.bench.article_alignment_results import (
    ArticleScenarioRunArtifacts,
    article_alignment_metadata,
    build_article_scenario_run_result,
)
from tomojax.bench.article_alignment_runs import (
    article_scenario_catalog_for_kind,
    article_theta_span_deg,
    diagnostic_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


article_runner = _load_module(
    "generate_alignment_before_after_128_under_test",
    ROOT / "scripts" / "generate_alignment_before_after_128.py",
)


def test_article_visual_scaling_handles_nonfinite_values(tmp_path: Path) -> None:
    image = np.asarray([[np.nan, -np.inf], [0.25, np.inf]], dtype=np.float32)

    gray = article_visuals.scale_shared_gray(image, np.nan, np.nan)
    diverging = article_visuals.scale_diverging(image, np.nan)
    loss = article_visuals.loss_panel(
        [
            {"geometry_block": "setup", "level_factor": 4, "geometry_loss_before": 1.0},
            {
                "geometry_block": "setup",
                "level_factor": 4,
                "geometry_loss_before": 1.0,
                "geometry_loss_after": np.nan,
            },
            {
                "geometry_block": "setup",
                "level_factor": 4,
                "geometry_loss_before": 1.0,
                "geometry_loss_after": 0.5,
                "geometry_accepted": True,
            },
        ]
    )

    assert gray.dtype == np.uint8
    assert diverging.dtype == np.uint8
    assert loss.dtype == np.uint8
    assert np.isfinite(gray).all()
    assert np.isfinite(diverging).all()
    assert np.isfinite(loss).all()

    out = tmp_path / "nonfinite_panel.png"
    iio.imwrite(out, gray)
    assert out.stat().st_size > 0


def test_scenario_finite_report_marks_nonfinite_alignment_volume() -> None:
    result = article_runner.ScenarioComputationResult(
        theta_span=180.0,
        naive_fbp=np.zeros((2, 2, 2), dtype=np.float32),
        calibrated_fbp=np.ones((2, 2, 2), dtype=np.float32),
        aligned_tv=np.full((2, 2, 2), np.nan, dtype=np.float32),
        provenance="estimated",
        supplied={},
        estimates={},
        metrics={
            "naive_volume_nmse": 0.1,
            "calibrated_volume_nmse": 0.2,
            "aligned_tv_volume_nmse": np.nan,
        },
        info={},
        diagnostics={},
        geometry_objectives=[],
        schedule_metadata={},
        executed_stages=[],
        solver_metadata={},
    )

    report = article_runner._scenario_finite_report(result)

    assert report["all_required_finite"] is False
    assert report["first_nonfinite"]["name"] == "aligned_tv"
    aligned_summary = next(v for v in report["volumes"] if v["name"] == "aligned_tv")
    assert aligned_summary["finite_fraction"] == 0.0


def test_article_alignment_run_contracts_live_behind_bench_module() -> None:
    profile = diagnostic_profile()
    scenarios = article_scenario_catalog_for_kind("default")

    assert profile.name == "diagnostic_32"
    assert profile.levels == (4, 2, 1)
    assert scenarios
    assert all(scenario.phantom_key == "phantom94" for scenario in scenarios)
    assert {scenario.slug for scenario in scenarios}
    assert all(article_theta_span_deg(scenario) in {180.0, 360.0} for scenario in scenarios)


def test_article_alignment_manifest_helpers_live_behind_bench_module() -> None:
    profile = diagnostic_profile()
    scenario = next(
        scenario
        for scenario in article_scenario_catalog_for_kind("default")
        if scenario.slug == "parallel_cor_u_m004"
    )

    truth = article_scenario_truth_payload(scenario)
    supplied = article_scenario_supplied_payload(scenario)
    catalog = article_scenario_catalog_payload(scenario)
    manifest = build_article_run_manifest(
        profile,
        [scenario],
        suite_name="article_headliners",
        generator="test-generator.py",
    )

    assert truth["det_u_px"] == scenario.hidden_det_u_px
    assert supplied == {}
    assert catalog["active_dofs"] == ["det_u_px"]
    assert catalog["active_geometry_dofs"] == ["det_u_px"]
    assert manifest["generator"] == "test-generator.py"
    assert manifest["suite_name"] == "article_headliners"
    assert manifest["phantom"]["selection"] == (
        "phantom_picker_128_10x10_center_biased_sphere_slot_94"
    )
    assert manifest["scenarios"] == [
        {
            "slug": scenario.slug,
            "title": scenario.title,
            "description": scenario.description,
            "scenario_category": scenario.scenario_category,
            "scenario_family": scenario.scenario_family,
            "suite_name": "article_headliners",
            "expectation": scenario.expectation,
            "expected_status": list(scenario.expected_status),
            "headline_eligible": scenario.headline_eligible,
            "phantom_key": scenario.phantom_key,
            "schedule": scenario.schedule,
            "expected_objective": scenario.expected_objective,
            "expected_optimizer": scenario.expected_optimizer,
            "expected_loss": scenario.expected_loss,
            "geometry_type": scenario.geometry_type,
            "geometry_dofs": ["det_u_px"],
            "active_dofs": ["det_u_px"],
            "active_pose_dofs": [],
            "active_geometry_dofs": ["det_u_px"],
            "theta_span_deg": 180.0,
            "n_views": profile.views,
            "hidden_truth": truth,
            "supplied_corrections": supplied,
        }
    ]


def test_article_alignment_run_results_live_behind_bench_module() -> None:
    profile = diagnostic_profile()
    scenario = next(
        scenario
        for scenario in article_scenario_catalog_for_kind("default")
        if scenario.slug == "parallel_cor_u_m004"
    )
    result = article_runner.ScenarioComputationResult(
        theta_span=180.0,
        naive_fbp=np.zeros((2, 2, 2), dtype=np.float32),
        calibrated_fbp=None,
        aligned_tv=None,
        provenance="naive_only",
        supplied={},
        estimates={},
        metrics={"naive_volume_nmse": 0.125},
        info={},
        diagnostics={},
        geometry_objectives=[],
        schedule_metadata={},
        executed_stages=[],
        solver_metadata={},
    )
    artifacts = ArticleScenarioRunArtifacts(
        visual_paths={"before_after_panel": "scenario/before_after.png"}
    )

    run_result = build_article_scenario_run_result(
        scenario,
        profile=profile,
        result=result,
        artifacts=artifacts,
        alignment_metadata=article_alignment_metadata(scenario, profile=profile, result=result),
        elapsed=1.25,
    )

    assert run_result.row["slug"] == "parallel_cor_u_m004"
    assert run_result.row["parameter_provenance"] == "naive_only"
    assert run_result.row["before_after_panel"] == "scenario/before_after.png"
    assert run_result.case_manifest["scenario_catalog"]["active_geometry_dofs"] == ["det_u_px"]
    assert run_result.case_manifest["phantom"]["selection"] == (
        "phantom_picker_128_10x10_center_biased_sphere_slot_94"
    )
